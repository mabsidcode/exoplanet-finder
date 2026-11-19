"""Downloadable prediction report for one scored candidate (used by the demo app)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import __version__
from .io_utils import utc_now_iso
from .preprocess import CandidateParams

REPORT_FORMAT_VERSION = 1
CAVEAT = (
    "Model score for a candidate signal, measuring resemblance to Kepler DR24 planet-candidate dispositions. "
    "It is not a calibrated probability and not a planet confirmation."
)
EXAMPLE_FIELDS = (
    "id", "title", "kepid", "tce_id", "catalogue_release", "catalogue_disposition", "training_label", "split",
    "in_heldout_evaluation", "selection_rule", "light_curve_file", "light_curve_source", "parameter_source",
)


def build_prediction_report(
    result: dict[str, Any],
    params: CandidateParams,
    *,
    model_run: str | None,
    checkpoint: str,
    checkpoint_sha256: str | None,
    example: dict[str, Any] | None = None,
    upload: dict[str, Any] | None = None,
    epoch_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable report for one prediction.

    ``result`` is the dict returned by :func:`exofinder.inference.predict_candidate`. Exactly one of
    ``example`` (a bundled example's metadata) or ``upload`` (file name, size, SHA-256, time system,
    rows used, parser warnings) describes the input.
    """
    if (example is None) == (upload is None):
        raise ValueError("pass exactly one of example= or upload=")
    score, thr = float(result["score"]), float(result["threshold"])
    return {
        "report_format_version": REPORT_FORMAT_VERSION,
        "tool": "Exoplanet Finder",
        "exofinder_version": __version__,
        "generated_utc": utc_now_iso(),
        "input": {
            "source": "bundled_example" if example is not None else "upload",
            "example": None if example is None else {k: example.get(k) for k in EXAMPLE_FIELDS if k in example},
            "upload": upload,
        },
        "candidate": {
            "period_days": params.period_days,
            "epoch_bkjd": params.epoch_bkjd,
            "duration_hours": params.duration_hours,
            "epoch_as_entered": epoch_input or {"value": params.epoch_bkjd, "time_reference": "BKJD"},
            "time_reference_note": "BKJD = BJD_TDB - 2454833 days",
        },
        "model": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "run": model_run,
            "decision_threshold": thr,
            "threshold_rule": "maximise F1 on the validation split",
            "preprocess_version": result["preprocess_version"],
            "preprocess_fingerprint": result["preprocess_fingerprint"],
        },
        "result": {
            "score": score,
            "above_threshold": bool(score >= thr),
            "interpretation": (
                "above threshold: resembles DR24 planet-candidate signals"
                if score >= thr
                else "below threshold: resembles DR24 false-positive / non-transit signals"
            ),
            "score_semantics": result["score_semantics"],
        },
        "diagnostics": result["diagnostics"],
        "caveat": CAVEAT,
    }


def _clean(o: Any) -> Any:
    """Convert numpy values to Python ones and non-finite floats to None (strict JSON has no NaN)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def report_to_json(report: dict[str, Any]) -> str:
    """Strict JSON (NaN/Infinity written as null), indented for reading."""
    return json.dumps(_clean(report), indent=2, allow_nan=False)
