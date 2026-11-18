import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from exofinder.inference import IncompatibleCheckpoint, load_bundle, score_views
from exofinder.metrics import binary_metrics, select_threshold_max_f1
from exofinder.model import DualBranchCNN, ModelConfig, count_parameters
from exofinder.preprocess import PREPROCESS_VERSION, PreprocessConfig


def _checkpoint(model, pcfg=None, **over):
    pcfg = pcfg or PreprocessConfig()
    ck = {
        "format_version": 1, "state_dict": model.state_dict(), "model_config": model.cfg.to_dict(),
        "preprocess": pcfg.to_dict(), "preprocess_version": PREPROCESS_VERSION,
        "preprocess_fingerprint": pcfg.fingerprint(), "decision_threshold": 0.4,
    }
    ck.update(over)
    return ck


def test_forward_shapes_and_global_only():
    torch.manual_seed(0)
    m = DualBranchCNN(ModelConfig())
    out = m(torch.randn(3, 2, 2001), torch.randn(3, 2, 201))
    assert out.shape == (3,)
    g = DualBranchCNN(ModelConfig(use_local=False))
    assert g(torch.randn(2, 2, 2001)).shape == (2,)
    assert count_parameters(g) < count_parameters(m)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = DualBranchCNN(ModelConfig()).eval()
    torch.save(_checkpoint(m), tmp_path / "ck.pt")
    b = load_bundle(tmp_path / "ck.pt")
    g, loc = np.random.default_rng(0).standard_normal((2, 2, 2001)).astype(np.float32), np.random.default_rng(1).standard_normal((2, 2, 201)).astype(np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(m(torch.from_numpy(g), torch.from_numpy(loc))).numpy()
    assert np.allclose(score_views(b, g, loc), ref, atol=1e-6)
    assert b.threshold == 0.4 and b.preprocess == PreprocessConfig()


def test_incompatible_checkpoints_rejected(tmp_path):
    m = DualBranchCNN(ModelConfig())
    torch.save(_checkpoint(m, preprocess_version="0.0.1"), tmp_path / "a.pt")
    with pytest.raises(IncompatibleCheckpoint):
        load_bundle(tmp_path / "a.pt")
    bad = _checkpoint(m)
    bad["preprocess"]["gap_days"] = 9.0  # altered without updating fingerprint
    torch.save(bad, tmp_path / "b.pt")
    with pytest.raises(IncompatibleCheckpoint):
        load_bundle(tmp_path / "b.pt")
    no_thr = _checkpoint(m)
    del no_thr["decision_threshold"]
    torch.save(no_thr, tmp_path / "c.pt")
    with pytest.raises(IncompatibleCheckpoint):
        load_bundle(tmp_path / "c.pt")


def test_binary_metrics_hand_computed():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.5, 0.2, 0.1, 0.4])
    m = binary_metrics(y, s, threshold=0.4)  # inclusive: score >= threshold is positive
    assert m["confusion"] == {"tp": 2, "fp": 2, "tn": 2, "fn": 0}
    assert m["precision"] == 0.5 and m["recall"] == 1.0 and m["accuracy"] == pytest.approx(4 / 6)
    assert m["prevalence"] == pytest.approx(1 / 3)
    assert m["average_precision"] == pytest.approx(average_precision_score(y, s))
    m2 = binary_metrics(y, s, threshold=0.41)
    assert m2["confusion"]["fn"] == 1


def test_threshold_selection_max_f1():
    y = np.array([0, 0, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    thr = select_threshold_max_f1(y, s)
    f1 = binary_metrics(y, s, thr)["f1"]
    for t in s:
        assert binary_metrics(y, s, t)["f1"] <= f1 + 1e-12
    with pytest.raises(ValueError):
        select_threshold_max_f1(np.zeros(3), np.ones(3))


def test_metrics_single_class():
    m = binary_metrics(np.zeros(4), np.linspace(0, 1, 4), 0.5)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["average_precision"])


def test_time_reversal_preserves_centre():
    g = np.zeros(2001)
    g[1000] = -1
    loc = np.zeros(201)
    loc[100] = -1
    assert np.argmin(g[::-1]) == 1000 and np.argmin(loc[::-1]) == 100
