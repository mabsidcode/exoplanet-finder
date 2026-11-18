"""Consolidate per-star outputs into one dataset with a manifest.

Output in ``data/datasets/<experiment>/``:
* ``examples.csv``  every selected labelled TCE, including excluded or failed
  ones, with identifiers, catalogue provenance, label, split, light-curve
  source, preprocessing status, exclusion reason and diagnostics.
* ``views.npz``     arrays for usable examples, aligned with ``tce_ids``.
* ``manifest.json`` counts by split/class/status, unique stars, exclusions,
  failed downloads and the fingerprints needed to reproduce the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import label_examples, labelled_examples, load_snapshot
from .config import Paths
from .io_utils import atomic_path, atomic_write_json, file_digest, read_json, utc_now_iso
from .preprocess import PreprocessConfig
from .splits import attach_splits, check_no_star_overlap

DIAG_COLUMNS = [
    "depth_ppm",
    "noise_ppm",
    "snr_est",
    "n_in_transit",
    "n_transits_observed",
    "odd_even_sigma",
    "secondary_depth_ratio",
    "local_shape_ratio",
    "global_coverage",
    "local_coverage",
    "n_points",
    "time_span_days",
]


def load_catalog_examples(cfg: dict[str, Any], paths: Paths) -> pd.DataFrame:
    """All labelled TCEs (PC/AFP/NTP) with their fixed star-level split."""
    prov = read_json(paths.catalog_provenance)
    examples = label_examples(load_snapshot(paths.catalog_snapshot), cfg, prov)
    manifest = pd.read_csv(paths.split_manifest)
    return attach_splits(labelled_examples(examples), manifest)


def build(cfg: dict[str, Any], paths: Paths, kepids: list[int]) -> dict[str, Any]:
    exp = cfg["experiment_name"]
    out_dir = paths.dataset_dir(exp)
    pcfg = PreprocessConfig.from_dict(cfg["preprocess"])
    examples = load_catalog_examples(cfg, paths)
    examples = examples[examples["kepid"].isin(kepids)].copy()

    rows: dict[str, dict[str, Any]] = {}
    views_g, views_l, view_ids = [], [], []
    for kepid in sorted(kepids):
        lp = paths.ledger / f"{kepid:09d}.json"
        if not lp.exists():
            continue
        led = read_json(lp)
        lc = led.get("light_curve", {})
        common = {
            "lc_source_url": led.get("source_url", ""),
            "lc_retrieved_utc": led.get("retrieved_utc", ""),
            "lc_files": ";".join(f["filename"] for f in led.get("files", [])),
            "lc_data_release": ";".join(lc.get("data_releases", [])),
            "lc_flux_column": lc.get("flux_column", ""),
            "lc_quarters": ";".join(str(q) for q in lc.get("quarters", [])),
            "preprocess_fingerprint": led.get("preprocess_fingerprint", ""),
        }
        for e in led["examples"]:
            r = {**common, "status": e["status"], "exclusion_reason": e.get("reason", "")}
            if led.get("preprocess_fingerprint") != pcfg.fingerprint():
                r.update(status="stale_preprocessing", exclusion_reason="preprocess_fingerprint_mismatch")
            for c in DIAG_COLUMNS:
                r[c] = e.get("diagnostics", {}).get(c, np.nan)
            rows[e["tce_id"]] = r
        if led["status"] == "done" and led.get("preprocess_fingerprint") == pcfg.fingerprint():
            with np.load(paths.data / led["processed_file"]) as z:
                for i, tid in enumerate(z["tce_ids"]):
                    view_ids.append(str(tid))
                    views_g.append(z["global_view"][i])
                    views_l.append(z["local_view"][i])

    status = [rows.get(t, {}).get("status", "not_processed") for t in examples["tce_id"]]
    extra = pd.DataFrame([rows.get(t, {}) for t in examples["tce_id"]], index=examples.index)
    examples = examples.drop(columns=["exclusion_reason"]).join(extra)
    examples["status"] = status
    examples["exclusion_reason"] = examples["exclusion_reason"].fillna("not_processed")
    examples.loc[examples["status"] == "ok", "exclusion_reason"] = ""
    check_no_star_overlap(examples)

    if len(set(view_ids)) != len(view_ids):
        raise ValueError("Duplicate TCE IDs among processed views")
    ok_ids = set(examples.loc[examples["status"] == "ok", "tce_id"])
    keep_idx = [i for i, t in enumerate(view_ids) if t in ok_ids]
    view_ids = [view_ids[i] for i in keep_idx]
    g = np.stack([views_g[i] for i in keep_idx]) if keep_idx else np.zeros((0, 2, pcfg.global_num_bins))
    loc = np.stack([views_l[i] for i in keep_idx]) if keep_idx else np.zeros((0, 2, pcfg.local_num_bins))

    out_dir.mkdir(parents=True, exist_ok=True)
    with atomic_path(out_dir / "views.npz", suffix=".npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, tce_ids=np.array(view_ids), global_view=g.astype(np.float32), local_view=loc.astype(np.float32))
    with atomic_path(out_dir / "examples.csv") as tmp:
        examples.to_csv(tmp, index=False)

    usable = examples[examples["status"] == "ok"]
    manifest = {
        "experiment": exp,
        "built_utc": utc_now_iso(),
        "catalog_snapshot_sha256": file_digest(paths.catalog_snapshot),
        "split_manifest_sha256": file_digest(paths.split_manifest),
        "preprocess": pcfg.to_dict(),
        "preprocess_fingerprint": pcfg.fingerprint(),
        "views_sha256": file_digest(out_dir / "views.npz"),
        "n_selected_stars": len(kepids),
        "n_selected_examples": int(len(examples)),
        "n_usable_examples": int(len(usable)),
        "n_usable_stars": int(usable["kepid"].nunique()),
        "status_counts": examples["status"].value_counts().to_dict(),
        "exclusion_reasons": examples.loc[examples["status"] != "ok", "exclusion_reason"].value_counts().to_dict(),
        "failed_download_stars": int(
            examples.loc[examples["status"] == "failed_download", "kepid"].nunique()
        ),
        "by_split": _counts(usable, "split"),
        "by_split_disposition": usable.groupby(["split", "original_disposition"]).size().unstack(fill_value=0).to_dict("index"),
        "light_curve_data_releases": sorted(set(";".join(usable["lc_data_release"].astype(str)).split(";")) - {""}),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def _counts(df: pd.DataFrame, by: str) -> dict[str, Any]:
    out = {}
    for k, g in df.groupby(by):
        out[str(k)] = {
            "n_examples": int(len(g)),
            "n_stars": int(g["kepid"].nunique()),
            "n_pos": int((g["label"] == 1).sum()),
            "n_neg": int((g["label"] == 0).sum()),
            "prevalence": round(float(g["label"].mean()), 4),
        }
    return out


class ViewDataset:
    """In-memory dataset joined with example metadata (usable examples only)."""

    def __init__(self, dataset_dir: Path):
        self.dir = Path(dataset_dir)
        self.examples = pd.read_csv(self.dir / "examples.csv", dtype={"tce_id": str})
        with np.load(self.dir / "views.npz") as z:
            ids = [str(t) for t in z["tce_ids"]]
            self.global_view = z["global_view"]
            self.local_view = z["local_view"]
        self.manifest = read_json(self.dir / "manifest.json")
        meta = self.examples.set_index("tce_id").loc[ids].reset_index()
        self.meta = meta
        self.labels = meta["label"].to_numpy().astype(np.float32)

    def split_index(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.meta["split"].to_numpy() == split)
