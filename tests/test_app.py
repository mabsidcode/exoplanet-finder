"""End-to-end tests of the Streamlit app with streamlit.testing (headless, offline, real bundled data).

The app is driven through its entry point, exactly as `streamlit run app/streamlit_app.py` runs it. Download
buttons are checked by capturing the bytes Streamlit stores for them.
"""

import base64
import json
import re

import numpy as np
import pytest

from exofinder.upload import light_curve_to_csv

from .conftest import EXAMPLES, RELEASE, ROOT

pytest.importorskip("streamlit.testing.v1")
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ENTRY = str(ROOT / "app" / "streamlit_app.py")
TIMEOUT = 180
K90H = {"period": 331.603, "epoch_bkjd": 140.48, "duration": 14.49}

pytestmark = [
    pytest.mark.app,
    pytest.mark.skipif(not (RELEASE / "cnn.pt").exists() or not (EXAMPLES / "examples.json").exists(),
                       reason="release model / bundled examples not present"),
]


@pytest.fixture
def downloads(monkeypatch):
    """Record every file Streamlit stores for a download button, keyed by media file id."""
    store: dict[str, bytes] = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def recording(self, path_or_data, mimetype, kind, filename=None):
        fid = original(self, path_or_data, mimetype, kind, filename)
        if isinstance(path_or_data, bytes):
            store[fid] = path_or_data
        return fid

    monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", recording)
    return store


def _download_bytes(store, button) -> bytes:
    fid = button.proto.url.rsplit("/", 1)[-1].split(".")[0]
    return store[fid]


def _run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=TIMEOUT)
    at.run()
    if page:
        at.switch_page(page).run()
    return at


def _no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]


def _score(at) -> float | None:
    for m in at.markdown:
        hit = re.search(r"class='big'>([\d.]+)<", m.value)
        if hit:
            return float(hit.group(1))
    return None


def _plotly_figures(at) -> list[dict]:
    return [json.loads(el.proto.spec) for el in at.main if getattr(el, "type", None) == "plotly_chart"]


def _array(v) -> np.ndarray:
    """Plotly >= 6 serialises numpy arrays as {"dtype": ..., "bdata": base64}."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).astype(float)
    return np.asarray(v, dtype=float)


def _assert_charts_have_data(at, n_expected):
    figs = _plotly_figures(at)
    assert len(figs) == n_expected
    for fig in figs:
        assert fig["data"], fig.get("layout", {}).get("title")
        for trace in fig["data"]:
            y = _array(trace.get("y", []))
            # NaN is allowed only as a line break (the trend line is split at data gaps)
            assert y.size > 0 and np.isfinite(y).mean() > 0.99, trace.get("name")


def _upload(at, data: bytes, tsys="BKJD", period=K90H["period"], epoch=K90H["epoch_bkjd"], duration=K90H["duration"]):
    at.radio[0].set_value("Upload a CSV").run()
    at.selectbox(key="upload_time_ref").set_value(tsys).run()
    at.file_uploader[0].upload("lightcurve.csv", data, "text/csv")
    at.number_input(key="upload_period").set_value(period)
    at.number_input(key="upload_epoch").set_value(epoch)
    at.number_input(key="upload_duration").set_value(duration)
    at.run()
    return at


# --------------------------------------------------------------------------- pages


def test_results_page_renders():
    at = _run("pages/evaluation.py")
    _no_exceptions(at)
    assert at.title[0].value == "Measured results"
    assert len(at.metric) == 4 and at.metric[2].value == "0.913"
    assert len(at.dataframe) == 3  # test metrics, seed variation, subgroups
    _assert_charts_have_data(at, 1)  # PR curve: release CNN + 2 baselines
    assert len(_plotly_figures(at)[0]["data"]) == 3
    assert sum(type(e).__name__ == "Image" for e in at.main) == 4  # confusion, learning curves, scores, errors
    assert not at.error and not at.warning


def test_about_page_renders():
    at = _run("pages/about.py")
    _no_exceptions(at)
    assert "Shallue" in at.markdown[-1].value or any("Shallue" in m.value for m in at.markdown)


def test_every_bundled_example_scores_and_draws_all_charts():
    meta = json.loads((EXAMPLES / "examples.json").read_text())
    at = _run()
    _no_exceptions(at)
    assert at.radio[0].value == "Bundled real Kepler example"
    assert at.selectbox[0].options == [e["title"] for e in meta["examples"]]
    for ex in meta["examples"]:
        at.selectbox[0].select(ex["title"]).run()
        _no_exceptions(at)
        assert not at.error
        assert _score(at) == pytest.approx(ex["score_at_bundle"], abs=5e-4)  # 3 decimals shown
        _assert_charts_have_data(at, 4)


# --------------------------------------------------------------------------- upload workflow + report


def test_sample_csv_upload_reproduces_bundled_kepler90h_and_report(downloads):
    at = _run()
    at.radio[0].set_value("Upload a CSV").run()
    _no_exceptions(at)
    sample = at.get("download_button")[0]
    assert sample.proto.label.startswith("Download a sample CSV")
    data = _download_bytes(downloads, sample)
    from exofinder.release import load_bundled_raw

    raw = load_bundled_raw(EXAMPLES / "kplr011442793.npz")
    assert data == light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])

    _upload(at, data)
    _no_exceptions(at)
    assert not at.error
    _assert_charts_have_data(at, 4)
    report_button = [b for b in at.get("download_button") if "prediction report" in b.proto.label][0]
    report = json.loads(_download_bytes(downloads, report_button))

    ex = next(e for e in json.loads((EXAMPLES / "examples.json").read_text())["examples"]
              if e["id"] == "011442793_01")
    assert report["result"]["score"] == pytest.approx(ex["score_at_bundle"], abs=1e-5)
    assert report["result"]["above_threshold"] is True
    assert report["input"]["source"] == "upload" and report["input"]["example"] is None
    up = report["input"]["upload"]
    assert up["filename"] == "lightcurve.csv" and up["size_bytes"] == len(data) and up["time_reference"] == "BKJD"
    from exofinder.io_utils import bytes_digest, file_digest

    assert up["sha256"] == bytes_digest(data)
    assert report["candidate"]["period_days"] == pytest.approx(K90H["period"])
    assert report["candidate"]["epoch_bkjd"] == pytest.approx(K90H["epoch_bkjd"])
    assert report["candidate"]["duration_hours"] == pytest.approx(K90H["duration"])
    card = json.loads((RELEASE / "release.json").read_text())
    assert report["model"]["checkpoint_sha256"] == file_digest(RELEASE / "cnn.pt") == card["cnn_sha256"]
    assert report["model"]["run"] == card["run"]
    assert report["model"]["decision_threshold"] == pytest.approx(card["decision_threshold"])
    assert report["diagnostics"]["n_transits_observed"] == 3
    assert "not a planet confirmation" in report["caveat"]


def test_bundled_example_report_download(downloads):
    at = _run()
    button = [b for b in at.get("download_button") if "prediction report" in b.proto.label][0]
    report = json.loads(_download_bytes(downloads, button))
    meta = json.loads((EXAMPLES / "examples.json").read_text())["examples"][-1]  # default selection
    assert report["input"]["source"] == "bundled_example" and report["input"]["upload"] is None
    assert report["input"]["example"]["id"] == meta["id"]
    assert report["result"]["score"] == pytest.approx(meta["score_at_bundle"], abs=1e-5)


def test_bjd_upload_and_switching_time_reference_keeps_epoch():
    from exofinder.lightcurve import BKJD_OFFSET
    from exofinder.release import load_bundled_raw

    raw = load_bundled_raw(EXAMPLES / "kplr011442793.npz")
    data = light_curve_to_csv(raw["time"] + BKJD_OFFSET, raw["flux"], raw["quality"], raw["segment"])
    at = _run()
    at.radio[0].set_value("Upload a CSV").run()
    at.number_input(key="upload_epoch").set_value(K90H["epoch_bkjd"] + BKJD_OFFSET).run()
    at.selectbox(key="upload_time_ref").set_value("BJD").run()  # regression: used to reset the epoch to 0
    assert at.number_input(key="upload_epoch").value == pytest.approx(K90H["epoch_bkjd"] + BKJD_OFFSET)
    _upload(at, data, tsys="BJD", epoch=K90H["epoch_bkjd"] + BKJD_OFFSET)
    _no_exceptions(at)
    assert not at.error
    assert _score(at) == pytest.approx(0.982, abs=1e-3)


@pytest.mark.parametrize(
    "label,data,kwargs,kind,fragment",
    [
        ("garbage", bytes(range(256)) * 4, {}, "error", "not UTF-8"),
        ("missing columns", b"time,brightness\n1,2\n", {}, "error", "Missing required column(s): flux"),
        ("bad quality", b"time,flux,quality\n" + b"140.0,1.0,inf\n" * 200, {}, "error", "Column 'quality'"),
        ("too few rows", b"time,flux\n" + b"".join(f"{140 + i * 0.02},1.0\n".encode() for i in range(20)), {},
         "error", "at least 100"),
        ("duration >= period", None, {"period": 0.5, "duration": 13.0}, "error", "Duration must be shorter"),
        ("long period", None, {"period": 5000.0}, "error", "low global coverage"),
        ("flat light curve", b"time,flux\n" + b"".join(f"{140 + i * 0.0204},1.0\n".encode() for i in range(5000)),
         {"period": 5.13, "epoch": 141.0, "duration": 3.0}, "error", "flat or no dip"),
        ("missing parameters", None, {"period": 0.0}, "info", "Enter a positive period"),
        ("epoch in wrong reference", None, {"epoch": K90H["epoch_bkjd"] + 2454833.0}, "warning",
         "more than 10 years"),
    ],
)
def test_bad_uploads_show_understandable_messages(label, data, kwargs, kind, fragment):
    if data is None:
        from exofinder.release import load_bundled_raw

        raw = load_bundled_raw(EXAMPLES / "kplr011442793.npz")
        data = light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])
    at = _upload(_run(), data, **kwargs)
    _no_exceptions(at)
    messages = [e.value for e in at.get(kind)]
    assert any(fragment in m for m in messages), messages
