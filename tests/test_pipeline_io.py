"""Pipeline bookkeeping: star selection, atomic writes, retry/resume without duplication."""

from pathlib import Path

import pandas as pd
import pytest
import requests

from exofinder import pipeline
from exofinder.config import load_config
from exofinder.io_utils import atomic_path, atomic_write_json, read_json

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "quickstart.yaml")


def _labelled():
    rows = []
    for k in range(1, 61):
        disp = ["PC", "AFP", "NTP"][k % 3]
        rows.append({"kepid": k, "original_disposition": disp, "period_days": float(k), "tce_id": f"{k:09d}_01"})
    rows.append({"kepid": 11442793, "original_disposition": "PC", "period_days": 331.6, "tce_id": "011442793_01"})
    return pd.DataFrame(rows)


def test_select_stars_stratified_includes_demo_and_is_deterministic():
    cfg = {**CFG, "download": {**CFG["download"], "max_stars": 13}}
    a = pipeline.select_stars(_labelled(), cfg)
    assert a == pipeline.select_stars(_labelled(), cfg)
    assert 11442793 in a and len(a) == 13
    cats = _labelled().set_index("kepid").loc[[k for k in a if k != 11442793], "original_disposition"]
    assert set(cats) == {"PC", "AFP", "NTP"}
    full = {**CFG, "download": {**CFG["download"], "max_stars": None}}
    assert len(pipeline.select_stars(_labelled(), full)) == 61


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "x.json"
    atomic_write_json(target, {"a": 1})
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_text("partial")
        raise RuntimeError("interrupted")
    assert read_json(target) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


class _Resp:
    def __init__(self, code):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _FlakySession:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = 0

    def get(self, url, timeout, stream=False):
        self.calls += 1
        code = self.codes.pop(0)
        if code == "conn":
            raise requests.ConnectionError("reset")
        return _Resp(code)


def test_get_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    s = _FlakySession(["conn", 503, 200])
    assert pipeline._get(s, "u", retries=3, timeout=1).status_code == 200
    assert s.calls == 3


def test_get_does_not_retry_404(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    s = _FlakySession([404, 200])
    with pytest.raises(requests.HTTPError):
        pipeline._get(s, "u", retries=3, timeout=1)
    assert s.calls == 1


def test_get_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    s = _FlakySession(["conn"] * 3)
    with pytest.raises(requests.ConnectionError):
        pipeline._get(s, "u", retries=2, timeout=1)


def test_completed_star_is_skipped_not_reprocessed(tmp_path, monkeypatch):
    """A 'done' ledger with matching fingerprint and TCE set is reused (resume without duplication)."""
    from exofinder.preprocess import PreprocessConfig

    data = tmp_path / "data"
    (data / "processed" / "ledger").mkdir(parents=True)
    (data / "processed" / "stars").mkdir(parents=True)
    (data / "processed" / "stars" / "000000001.npz").write_bytes(b"x")
    fp = PreprocessConfig.from_dict(CFG["preprocess"]).fingerprint()
    atomic_write_json(data / "processed" / "ledger" / "000000001.json",
                      {"status": "done", "preprocess_fingerprint": fp, "examples": [{"tce_id": "000000001_01"}]})
    monkeypatch.setattr(pipeline, "list_star_files", lambda *a, **k: pytest.fail("should not download"))
    led = pipeline.process_star(1, [{"tce_id": "000000001_01"}], CFG, str(data), keep_raw=False)
    assert led["skipped"] is True
