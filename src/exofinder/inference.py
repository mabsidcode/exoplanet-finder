"""Load a checkpoint and score a candidate with the shared preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .lightcurve import LightCurve
from .model import DualBranchCNN, ModelConfig
from .preprocess import (
    PREPROCESS_VERSION,
    CandidateParams,
    PreprocessConfig,
    ProcessedCandidate,
    preprocess_candidate,
)


class IncompatibleCheckpoint(RuntimeError):
    pass


@dataclass
class ModelBundle:
    model: DualBranchCNN
    preprocess: PreprocessConfig
    threshold: float
    meta: dict[str, Any]
    device: torch.device

    @property
    def use_local(self) -> bool:
        return self.model.cfg.use_local


def load_bundle(path: str | Path, device: str = "cpu") -> ModelBundle:
    ck = torch.load(Path(path), map_location=device, weights_only=True)
    if ck.get("format_version") != 1:
        raise IncompatibleCheckpoint(f"Unsupported checkpoint format {ck.get('format_version')}")
    if ck.get("preprocess_version") != PREPROCESS_VERSION:
        raise IncompatibleCheckpoint(
            f"Checkpoint expects preprocessing v{ck.get('preprocess_version')}, code provides v{PREPROCESS_VERSION}"
        )
    pcfg = PreprocessConfig.from_dict(ck["preprocess"])
    if pcfg.fingerprint() != ck.get("preprocess_fingerprint"):
        raise IncompatibleCheckpoint("Preprocessing fingerprint mismatch: stored configuration was altered")
    if "decision_threshold" not in ck:
        raise IncompatibleCheckpoint("Checkpoint has no validation-selected decision threshold (training unfinished?)")
    model = DualBranchCNN(ModelConfig.from_dict(ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    meta = {k: v for k, v in ck.items() if k not in ("state_dict",)}
    return ModelBundle(model, pcfg, float(ck["decision_threshold"]), meta, torch.device(device))


@torch.no_grad()
def score_views(bundle: ModelBundle, global_view: np.ndarray, local_view: np.ndarray) -> np.ndarray:
    g = torch.as_tensor(np.asarray(global_view, dtype=np.float32), device=bundle.device)
    loc = torch.as_tensor(np.asarray(local_view, dtype=np.float32), device=bundle.device)
    if g.ndim == 2:
        g, loc = g[None], loc[None]
    return torch.sigmoid(bundle.model(g, loc)).cpu().numpy()


def predict_candidate(bundle: ModelBundle, lc: LightCurve, params: CandidateParams) -> tuple[dict[str, Any], ProcessedCandidate]:
    """Preprocess with the checkpoint's own configuration, then score."""
    pc = preprocess_candidate(lc, params, bundle.preprocess, keep_intermediates=True)
    score = float(score_views(bundle, pc.global_view, pc.local_view)[0])
    result = {
        "score": score,
        "threshold": bundle.threshold,
        "above_threshold": bool(score >= bundle.threshold),
        "score_semantics": "Model score in [0, 1]; not a calibrated probability.",
        "period_days": params.period_days,
        "epoch_bkjd": params.epoch_bkjd,
        "duration_hours": params.duration_hours,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_fingerprint": bundle.preprocess.fingerprint(),
        "diagnostics": pc.diagnostics,
    }
    return result, pc
