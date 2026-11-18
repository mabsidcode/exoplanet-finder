"""DR24 TCE catalogue retrieval, snapshotting and label mapping.

Source: NASA Exoplanet Archive table ``q1_q17_dr24_tce`` (Kepler Q1-Q17 Data
Release 24 Threshold Crossing Events), queried through the archive's TAP
service. Labels come from the ``av_training_set`` column, the Autovetter
training-set classification also used by Shallue & Vanderburg (2018):

    PC  planet candidate                 -> label 1
    AFP astrophysical false positive     -> label 0
    NTP non-transiting phenomenon        -> label 0
    UNK unknown-class sample             -> excluded
    (NULL, not in the training set)      -> excluded

A PC label means "planet candidate in the DR24 Autovetter training set", not
"confirmed planet". The model therefore learns to reproduce these catalogue
dispositions. Vetting outputs (av_pp_*, av_pred_class, av_vf_*) are never
downloaded, so they cannot leak into model inputs.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io_utils import atomic_write_bytes, atomic_write_json, bytes_digest, utc_now_iso

CATALOG_RELEASE = "Kepler Q1-Q17 DR24 TCE (NASA Exoplanet Archive table q1_q17_dr24_tce)"
TIME_REFERENCE = "BKJD = BJD_TDB - 2454833.0 (days)"

# Columns snapshotted. Units follow the archive's column documentation
# (https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html).
SNAPSHOT_COLUMNS = [
    "kepid",  # Kepler Input Catalog ID (host star)
    "tce_plnt_num",  # TCE number within the star
    "tce_period",  # days
    "tce_period_err",  # days
    "tce_time0bk",  # BKJD days, centre of first detected transit
    "tce_time0bk_err",  # days
    "tce_duration",  # hours
    "tce_duration_err",  # hours
    "tce_depth",  # ppm (used only for subgroup analysis, never as model input)
    "tce_model_snr",  # unitless (used only for subgroup analysis)
    "tce_num_transits",  # count
    "tce_rogue_flag",  # 0/1
    "tce_delivname",
    "av_training_set",  # label source
    "ra",
    "dec",
]

COLUMN_UNITS = {
    "tce_period": "days",
    "tce_time0bk": "BKJD days (BJD_TDB - 2454833.0)",
    "tce_duration": "hours",
    "tce_depth": "ppm",
    "tce_model_snr": "dimensionless",
}


def tce_id(kepid: int, plnt_num: int) -> str:
    """Stable example identifier: zero-padded KIC ID and TCE number."""
    return f"{int(kepid):09d}_{int(plnt_num):02d}"


def build_query(table: str) -> str:
    return f"select {','.join(SNAPSHOT_COLUMNS)} from {table} order by kepid, tce_plnt_num"


def fetch_catalog(cfg: dict[str, Any], snapshot: Path, provenance: Path, timeout: float = 300) -> pd.DataFrame:
    """Download the DR24 TCE table via TAP and write a snapshot + provenance."""
    c = cfg["catalog"]
    query = build_query(c["table"])
    params = {"query": query, "format": "csv"}
    resp = requests.get(c["tap_url"], params=params, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    df = pd.read_csv(io.BytesIO(content))
    missing = set(SNAPSHOT_COLUMNS) - set(df.columns)
    if missing or len(df) == 0:
        raise RuntimeError(f"Unexpected TAP response: missing columns {missing}, {len(df)} rows")
    atomic_write_bytes(snapshot, content)
    atomic_write_json(
        provenance,
        {
            "catalog_release": CATALOG_RELEASE,
            "tap_url": c["tap_url"],
            "query": query,
            "request_url": resp.url,
            "retrieved_utc": utc_now_iso(),
            "n_rows": int(len(df)),
            "sha256": bytes_digest(content),
            "column_units": COLUMN_UNITS,
            "time_reference": TIME_REFERENCE,
            "label_column": c["label_column"],
            "label_value_counts": {
                str(k): int(v)
                for k, v in df[c["label_column"]].fillna("NULL").value_counts().items()
            },
        },
    )
    return df


def load_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def label_examples(df: pd.DataFrame, cfg: dict[str, Any], provenance: dict[str, Any]) -> pd.DataFrame:
    """Attach identifiers, provenance and the training label to every TCE row.

    Rows whose disposition is not in the positive or negative sets keep
    ``label = -1`` and an ``exclusion_reason``; they are never silently dropped
    or promoted.
    """
    c = cfg["catalog"]
    pos, neg = set(c["positive"]), set(c["negative"])
    out = pd.DataFrame(
        {
            "tce_id": [tce_id(k, n) for k, n in zip(df["kepid"], df["tce_plnt_num"], strict=True)],
            "kepid": df["kepid"].astype(int),
            "tce_plnt_num": df["tce_plnt_num"].astype(int),
            "catalog_release": CATALOG_RELEASE,
            "original_disposition": df[c["label_column"]].fillna("NULL").astype(str),
            "period_days": df["tce_period"].astype(float),
            "epoch_bkjd": df["tce_time0bk"].astype(float),
            "duration_hours": df["tce_duration"].astype(float),
            "time_reference": TIME_REFERENCE,
            "catalog_depth_ppm": df["tce_depth"],
            "catalog_model_snr": df["tce_model_snr"],
            "tce_num_transits": df["tce_num_transits"],
            "tce_rogue_flag": df["tce_rogue_flag"],
            "catalog_source": provenance.get("request_url", ""),
            "catalog_retrieved_utc": provenance.get("retrieved_utc", ""),
        }
    )
    disp = out["original_disposition"]
    out["label"] = -1
    out.loc[disp.isin(pos), "label"] = 1
    out.loc[disp.isin(neg), "label"] = 0
    out["label_mapping"] = f"{'+'.join(sorted(pos))}=1; {'+'.join(sorted(neg))}=0"
    out["exclusion_reason"] = ""
    out.loc[out["label"] == -1, "exclusion_reason"] = "disposition_" + disp[out["label"] == -1]
    if out["tce_id"].duplicated().any():
        raise ValueError("Duplicate TCE identifiers in catalogue snapshot")
    return out


def labelled_examples(examples: pd.DataFrame) -> pd.DataFrame:
    return examples[examples["label"] >= 0].reset_index(drop=True)
