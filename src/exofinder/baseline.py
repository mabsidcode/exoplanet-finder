"""scikit-learn baselines on interpretable light-curve features.

Features are computed by the shared preprocessing (``ProcessedCandidate.
diagnostics``) plus the candidate's own period and duration. No identifiers,
catalogue dispositions, catalogue depth/SNR or Autovetter outputs are used.

    log_period        log10(P / day)
    log_duration      log10(D / hour)
    log_duty_cycle    log10(D / P)
    log_depth         log10(local-view depth / ppm)
    log_noise         log10(robust out-of-transit scatter / ppm)
    log_snr           log10(depth / noise * sqrt(n_in_transit))
    log_n_transits    log10(1 + number of observed transits)
    odd_even_sigma    |odd depth - even depth| / its standard error (clipped 0..50)
    odd_even_missing  1 if odd/even depths could not both be measured
    secondary_ratio   depth near phase 0.5 / primary depth (clipped -5..5)
    shape_ratio       inner-half / outer-half mean of the local view (clipped -5..5)
    global_coverage, local_coverage   fraction of observed bins

Models (all fitted on the training split only, thresholds chosen on validation):
* ``logreg``: median imputation -> standardisation -> logistic regression with
  balanced class weights.
* ``hgb``: histogram gradient boosting with balanced class weights.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Paths
from .dataset import ViewDataset
from .io_utils import atomic_path, atomic_write_json, utc_now_iso
from .metrics import binary_metrics, select_threshold_max_f1

FEATURES = [
    "log_period",
    "log_duration",
    "log_duty_cycle",
    "log_depth",
    "log_noise",
    "log_snr",
    "log_n_transits",
    "odd_even_sigma",
    "odd_even_missing",
    "secondary_ratio",
    "shape_ratio",
    "global_coverage",
    "local_coverage",
]


def _safe_log10(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature table from candidate parameters and preprocessing diagnostics."""
    oe = df["odd_even_sigma"].astype(float)
    return pd.DataFrame(
        {
            "log_period": _safe_log10(df["period_days"]),
            "log_duration": _safe_log10(df["duration_hours"]),
            "log_duty_cycle": _safe_log10(df["duration_hours"] / 24.0 / df["period_days"]),
            "log_depth": _safe_log10(df["depth_ppm"]),
            "log_noise": _safe_log10(df["noise_ppm"]),
            "log_snr": _safe_log10(df["snr_est"]),
            "log_n_transits": np.log10(1 + df["n_transits_observed"].astype(float)),
            "odd_even_sigma": oe.clip(0, 50).fillna(0.0),
            "odd_even_missing": oe.isna().astype(float),
            "secondary_ratio": df["secondary_depth_ratio"].astype(float).clip(-5, 5),
            "shape_ratio": df["local_shape_ratio"].astype(float).clip(-5, 5),
            "global_coverage": df["global_coverage"].astype(float),
            "local_coverage": df["local_coverage"].astype(float),
        },
        index=df.index,
    )[FEATURES]


def make_models(seed: int = 0) -> dict[str, Any]:
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=5000),
        ),
        "hgb": HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, max_iter=300, learning_rate=0.05),
    }


def train_baselines(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    exp = cfg["experiment_name"]
    ds = ViewDataset(paths.dataset_dir(exp))
    X = make_features(ds.meta)
    y = ds.labels.astype(int)
    tr, va = ds.split_index("train"), ds.split_index("val")
    out_dir = paths.model_dir(exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in make_models().items():
        model.fit(X.iloc[tr], y[tr])
        val_scores = model.predict_proba(X.iloc[va])[:, 1]
        thr = select_threshold_max_f1(y[va], val_scores)
        meta = {
            "name": name,
            "features": FEATURES,
            "decision_threshold": thr,
            "threshold_rule": "maximise F1 on the validation split",
            "trained_utc": utc_now_iso(),
            "n_train": int(len(tr)),
            "val_metrics": binary_metrics(y[va], val_scores, thr),
            "dataset_views_sha256": ds.manifest["views_sha256"],
        }
        with atomic_path(out_dir / f"baseline_{name}.joblib", suffix=".joblib") as tmp:
            joblib.dump({"model": model, "meta": meta}, tmp)
        atomic_write_json(out_dir / f"baseline_{name}.json", meta)
        results[name] = {"val_ap": meta["val_metrics"]["average_precision"], "threshold": thr}
    return results
