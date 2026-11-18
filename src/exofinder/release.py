"""Release the validation-selected model and bundle demo light curves.

``release``: copies the release CNN (chosen by validation AP in ``evaluate``)
and the gradient-boosting baseline into ``models/release/`` with a JSON card.

``bundle_examples``: stores real Kepler light curves for the demo so it works
offline. Examples are chosen by fixed rules from the release model's *test*
predictions (made after the model was frozen), plus the Kepler-90 demo stars:

    1. test PC with the highest CNN score
    2. test PC with the lowest CNN score
    3. test AFP with the highest CNN score
    4. test AFP with the lowest CNN score
    5. test NTP with the highest CNN score
    6. Kepler-90 h  (DR24 TCE 011442793_01, label PC, demo hold-out)
    7. Kepler-90 i  (not a DR24 TCE; parameters from the NASA Exoplanet
       Archive Planetary Systems table, Shallue & Vanderburg 2018)

Each bundle holds TIME, PDCSAP_FLUX, SAP_QUALITY and quarter exactly as read
from the FITS files. Bundling re-runs the shared preprocessing on these
arrays and checks that the views match the training dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Paths
from .dataset import ViewDataset
from .inference import load_bundle, predict_candidate
from .io_utils import atomic_path, atomic_write_json, file_digest, read_json, utc_now_iso
from .lightcurve import BKJD_OFFSET, light_curve_from_raw_arrays, raw_arrays_from_fits
from .pipeline import USER_AGENT, download_file, list_star_files, star_url

EXAMPLES_DIR = Paths.from_config({}).root / "app" / "examples"
RELEASE_DIR = Paths.from_config({}).root / "models" / "release"

KEPLER90_I = {
    "id": "kepler-90i",
    "title": "Kepler-90 i (validated planet, not a DR24 TCE)",
    "kepid": 11442793,
    "tce_id": None,
    "period_days": 14.44912,
    "epoch_bjd": 2455644.3488,
    "duration_hours": 2.80,
    "parameter_source": "NASA Exoplanet Archive Planetary Systems table (default row, pl_name='Kepler-90 i'), "
    "reference Shallue & Vanderburg 2018, AJ 155, 94. pl_tranmid in BJD; pl_trandur in hours.",
    "parameter_uncertainties": {"period_days": 0.0002, "epoch_days": 0.0048, "duration_hours": 0.31},
    "catalogue_status": "Validated planet announced by Shallue & Vanderburg (2018) using a neural network search "
    "of Kepler data. It has no DR24 av_training_set label and was never part of this model's data.",
    "split": "not in dataset (demonstration only)",
}


def release(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    exp = cfg["experiment_name"]
    metrics = read_json(paths.report_dir(exp) / "metrics.json")
    name = metrics["release_model"]
    src = paths.model_dir(exp) / name / "best.pt"
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, RELEASE_DIR / "cnn.pt")
    if (paths.model_dir(exp) / "baseline_hgb.joblib").exists():
        shutil.copy2(paths.model_dir(exp) / "baseline_hgb.joblib", RELEASE_DIR / "baseline_hgb.joblib")
    m = metrics["models"][name]
    card = {
        "released_utc": utc_now_iso(),
        "experiment": exp,
        "run": name,
        "selection_rule": metrics["release_rule"],
        "decision_threshold": m["test"]["threshold"],
        "val": m["val"],
        "test": m["test"],
        "baseline_hgb_test": metrics["models"].get("baseline_hgb", {}).get("test"),
        "cnn_sha256": file_digest(RELEASE_DIR / "cnn.pt"),
        "dataset": metrics["dataset"],
        "score_semantics": "Model score in [0, 1]; not a calibrated probability.",
    }
    atomic_write_json(RELEASE_DIR / "release.json", card)
    print(f"Released {name} -> {RELEASE_DIR / 'cnn.pt'}")
    return card


def _fetch_star(kepid: int, cfg: dict[str, Any], paths: Paths) -> list[Path]:
    dl = cfg["download"]
    raw_dir = paths.raw_cache / f"{kepid:09d}"
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    names = list_star_files(s, dl["base_url"], kepid, dl["retries"], dl["timeout_s"])
    for n in names:
        download_file(s, star_url(dl["base_url"], kepid) + n, raw_dir / n, dl["retries"], dl["timeout_s"])
    return [raw_dir / n for n in names]


def _save_bundle(kepid: int, files: list[Path], cfg: dict[str, Any]) -> Path:
    raw = raw_arrays_from_fits(files, cfg["preprocess"]["flux_column"])
    out = EXAMPLES_DIR / f"kplr{kepid:09d}.npz"
    with atomic_path(out, suffix=".npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                time=raw["time"],
                flux=raw["flux"].astype(np.float32),
                quality=raw["quality"].astype(np.int32),
                segment=raw["segment"].astype(np.int16),
                files=np.array([f.name for f in files]),
            )
    return out


def load_bundled_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in ("time", "flux", "quality", "segment")} | {"files": [str(f) for f in z["files"]]}


def bundle_examples(cfg: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    exp = cfg["experiment_name"]
    metrics = read_json(paths.report_dir(exp) / "metrics.json")
    rel = metrics["release_model"]
    preds = pd.read_csv(paths.report_dir(exp) / "predictions.csv", dtype={"tce_id": str})
    t = preds[(preds["model"] == rel) & (preds["split"] == "test")]
    rules = [
        ("Test PC with the highest CNN score", t[t["original_disposition"] == "PC"].nlargest(1, "score")),
        ("Test PC with the lowest CNN score", t[t["original_disposition"] == "PC"].nsmallest(1, "score")),
        ("Test AFP with the highest CNN score", t[t["original_disposition"] == "AFP"].nlargest(1, "score")),
        ("Test AFP with the lowest CNN score", t[t["original_disposition"] == "AFP"].nsmallest(1, "score")),
        ("Test NTP with the highest CNN score", t[t["original_disposition"] == "NTP"].nlargest(1, "score")),
        ("Kepler-90 h: DR24 TCE reserved as a demonstration hold-out",
         preds[(preds["model"] == rel) & (preds["split"] == "demo_holdout")]),
    ]
    ds = ViewDataset(paths.dataset_dir(exp))
    idx = {tid: i for i, tid in enumerate(ds.meta["tce_id"])}
    bundle = load_bundle(RELEASE_DIR / "cnn.pt")
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for rule, rows in rules:
        for _, r in rows.iterrows():
            kepid = int(r["kepid"])
            files = _fetch_star(kepid, cfg, paths)
            path = _save_bundle(kepid, files, cfg)
            meta = ds.meta.iloc[idx[r["tce_id"]]]
            lc = light_curve_from_raw_arrays(load_bundled_raw(path), cfg["preprocess"]["quality_bitmask"])
            from .preprocess import CandidateParams

            params = CandidateParams(float(meta["period_days"]), float(meta["epoch_bkjd"]), float(meta["duration_hours"]))
            res, pc = predict_candidate(bundle, lc, params)
            i = idx[r["tce_id"]]
            diff = max(float(np.abs(pc.global_view - ds.global_view[i]).max()), float(np.abs(pc.local_view - ds.local_view[i]).max()))
            out.append({
                "id": r["tce_id"],
                "title": f"KIC {kepid} TCE {int(meta['tce_plnt_num'])} ({meta['original_disposition']})"
                + (" — Kepler-90 h" if kepid == 11442793 else ""),
                "selection_rule": rule,
                "kepid": kepid,
                "tce_id": r["tce_id"],
                "period_days": params.period_days,
                "epoch_bkjd": params.epoch_bkjd,
                "duration_hours": params.duration_hours,
                "catalogue_release": meta["catalog_release"],
                "catalogue_disposition": meta["original_disposition"],
                "training_label": int(meta["label"]),
                "split": meta["split"],
                "in_heldout_evaluation": meta["split"] in ("test", "demo_holdout"),
                "light_curve_file": path.name,
                "light_curve_source": star_url(cfg["download"]["base_url"], kepid),
                "source_files": [f.name for f in files],
                "bundled_utc": utc_now_iso(),
                "score_at_bundle": res["score"],
                "score_in_predictions_csv": float(r["score"]),
                "view_max_abs_diff_vs_dataset": diff,
            })
            print(f"{r['tce_id']}: score {res['score']:.4f} (predictions.csv {r['score']:.4f}), view diff {diff:.2e}")

    k = KEPLER90_I
    path = EXAMPLES_DIR / f"kplr{k['kepid']:09d}.npz"
    from .preprocess import CandidateParams

    lc = light_curve_from_raw_arrays(load_bundled_raw(path), cfg["preprocess"]["quality_bitmask"])
    params = CandidateParams(k["period_days"], k["epoch_bjd"] - BKJD_OFFSET, k["duration_hours"])
    res, _ = predict_candidate(bundle, lc, params)
    out.append({**k, "epoch_bkjd": params.epoch_bkjd, "selection_rule": "Historical inspiration (Shallue & Vanderburg 2018)",
                "catalogue_release": "NASA Exoplanet Archive Planetary Systems (ps)", "catalogue_disposition": "Validated planet",
                "training_label": None, "in_heldout_evaluation": False, "light_curve_file": path.name,
                "light_curve_source": star_url(cfg["download"]["base_url"], k["kepid"]),
                "bundled_utc": utc_now_iso(), "score_at_bundle": res["score"]})
    print(f"Kepler-90 i: score {res['score']:.4f} (threshold {bundle.threshold:.4f})")
    atomic_write_json(EXAMPLES_DIR / "examples.json", {"release_model_sha256": file_digest(RELEASE_DIR / "cnn.pt"),
                                                       "decision_threshold": bundle.threshold, "examples": out})
    return out
