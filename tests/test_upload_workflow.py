"""CSV upload workflow on REAL bundled Kepler light curves, input validation and the prediction report.

Numerical tolerances (documented in README "Demo app"):

* CSV with time in BKJD: model-input views must be bit-identical to the bundled-example workflow, and the
  score must agree within ``SCORE_TOL_BKJD`` (observed difference: exactly 0).
* CSV with time in BJD_TDB: adding and removing 2454833 d rounds timestamps at the ~1e-10 d level, so views agree
  within ``VIEW_TOL_BJD`` and scores within ``SCORE_TOL_BJD`` (observed: views <= 2.4e-7, scores 0).
* Against the score recorded when the example was bundled (``score_at_bundle``): ``SCORE_TOL_RECORDED``,
  allowing for float differences between machines and library versions.
"""

import json
import math

import numpy as np
import pytest

from exofinder.lightcurve import BKJD_OFFSET, LightCurveError, light_curve_from_raw_arrays
from exofinder.preprocess import CandidateParams
from exofinder.release import load_bundled_raw
from exofinder.report import build_prediction_report, report_to_json
from exofinder.upload import MIN_ROWS, light_curve_to_csv, parse_light_curve_csv

from .conftest import EXAMPLES, RELEASE

SCORE_TOL_BKJD = 1e-6
SCORE_TOL_BJD = 1e-5
VIEW_TOL_BJD = 1e-5
SCORE_TOL_RECORDED = 1e-5

pytestmark = pytest.mark.realdata


def _examples():
    p = EXAMPLES / "examples.json"
    return json.loads(p.read_text())["examples"] if p.exists() else []


@pytest.fixture(scope="module")
def bundle():
    if not (RELEASE / "cnn.pt").exists():
        pytest.skip("release model not present")
    from exofinder.inference import load_bundle

    return load_bundle(RELEASE / "cnn.pt")


def _params(ex):
    return CandidateParams(ex["period_days"], ex["epoch_bkjd"], ex["duration_hours"])


@pytest.mark.parametrize("ex", _examples(), ids=lambda e: e["id"])
def test_csv_upload_matches_bundled_example(bundle, ex):
    from exofinder.inference import predict_candidate

    raw = load_bundled_raw(EXAMPLES / ex["light_curve_file"])
    ref, ref_pc = predict_candidate(bundle, light_curve_from_raw_arrays(raw), _params(ex))
    assert ref["score"] == pytest.approx(ex["score_at_bundle"], abs=SCORE_TOL_RECORDED)

    # BKJD export -> upload parser -> same model
    csv = light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])
    lc, _ = parse_light_curve_csv(csv, "BKJD")
    up, up_pc = predict_candidate(bundle, lc, _params(ex))
    assert np.array_equal(up_pc.global_view, ref_pc.global_view)
    assert np.array_equal(up_pc.local_view, ref_pc.local_view)
    assert abs(up["score"] - ref["score"]) <= SCORE_TOL_BKJD
    assert up["above_threshold"] == ref["above_threshold"]

    # BJD_TDB export, epoch entered in BJD as a user would
    csv_bjd = light_curve_to_csv(raw["time"] + BKJD_OFFSET, raw["flux"], raw["quality"], raw["segment"])
    lc_bjd, warnings = parse_light_curve_csv(csv_bjd, "BJD")
    assert not any("outside the Kepler mission range" in w for w in warnings)
    p_bjd = CandidateParams(ex["period_days"], (ex["epoch_bkjd"] + BKJD_OFFSET) - BKJD_OFFSET, ex["duration_hours"])
    bjd, bjd_pc = predict_candidate(bundle, lc_bjd, p_bjd)
    assert np.abs(bjd_pc.local_view - ref_pc.local_view).max() <= VIEW_TOL_BJD
    assert np.abs(bjd_pc.global_view - ref_pc.global_view).max() <= VIEW_TOL_BJD
    assert abs(bjd["score"] - ref["score"]) <= SCORE_TOL_BJD


# --------------------------------------------------------------------------- invalid input


def _rows(n, flux="1.0", extra=""):
    return "".join(f"{140 + 0.02 * i:.5f},{flux}{extra}\n" for i in range(n))


BAD_CSVS = [
    ("empty", b"", "empty_file", "empty"),
    ("whitespace", b"  \n\n", "empty_file", "empty"),
    ("binary", bytes(range(256)) * 4, "bad_csv", "not UTF-8"),
    ("missing flux", b"time,brightness\n1,2\n", "missing_columns", "Missing required column(s): flux. Found: 'time', 'brightness'"),
    ("missing both", b"a,b\n1,2\n", "missing_columns", "flux, time"),
    ("semicolons", b"time;flux\n1;2\n", "missing_columns", "comma-separated"),
    ("header only", b"time,flux\n", "too_few_points", "no data rows"),
    ("all non-numeric", ("time,flux\n" + "x,y\n" * 200).encode(), "no_numeric_rows", "None of the 200 rows"),
    ("too few rows", ("time,flux\n" + _rows(MIN_ROWS - 1)).encode(), "too_few_points", f"at least {MIN_ROWS}"),
    ("negative flux", ("time,flux\n" + _rows(500, "-5.0")).encode(), "no_valid_data", "must be positive"),
    ("quality inf", ("time,flux,quality\n" + _rows(500, extra=",inf")).encode(), "bad_quality", "value 'inf'"),
    ("quality fraction", ("time,flux,quality\n" + _rows(500, extra=",0.5")).encode(), "bad_quality", "non-negative integer"),
    ("quality negative", ("time,flux,quality\n" + _rows(500, extra=",-1")).encode(), "bad_quality", "non-negative integer"),
    ("quality overflow", ("time,flux,quality\n" + _rows(500, extra=",1e30")).encode(), "bad_quality", "500 row(s)"),
    ("quality blank", ("time,flux,quality\n" + _rows(500, extra=",")).encode(), "bad_quality", "value ''"),
    ("segment text", ("time,flux,segment\n" + _rows(500, extra=",Q1")).encode(), "bad_segment", "integer segment ids"),
]


@pytest.mark.parametrize("data,reason,fragment", [c[1:] for c in BAD_CSVS], ids=[c[0] for c in BAD_CSVS])
def test_invalid_csv_is_rejected_with_readable_message(data, reason, fragment):
    with pytest.raises(LightCurveError) as e:
        parse_light_curve_csv(data, "BKJD")
    assert e.value.reason == reason
    assert fragment in str(e.value)
    assert "Traceback" not in str(e.value)


def test_upload_warnings_for_gaps_quality_and_wrong_time_reference():
    raw = load_bundled_raw(EXAMPLES / "kplr011442793.npz")
    csv = light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])
    _, ok = parse_light_curve_csv(csv, "BKJD")
    assert any("skipped" in w for w in ok) and any("quality bitmask" in w for w in ok)
    assert not any("mission range" in w for w in ok)
    _, wrong = parse_light_curve_csv(csv, "BJD")  # BKJD data declared as BJD
    assert any("outside the Kepler mission range" in w for w in wrong)


def test_bom_and_comment_lines_are_accepted():
    body = "# comment line\ntime,flux\n" + _rows(300)
    lc, _ = parse_light_curve_csv(b"\xef\xbb\xbf" + body.encode(), "BKJD")
    assert len(lc) == 300


# --------------------------------------------------------------------------- prediction report


def _result(score=0.9, threshold=0.671, **diag):
    return {"score": score, "threshold": threshold, "above_threshold": score >= threshold,
            "score_semantics": "Model score in [0, 1]; not a calibrated probability.",
            "preprocess_version": "1.0.0", "preprocess_fingerprint": "abc",
            "diagnostics": {"depth_ppm": np.float64(120.0), "n_points": np.int64(5000), **diag}}


def test_report_requires_exactly_one_input_description():
    p = CandidateParams(10.0, 140.0, 3.0)
    with pytest.raises(ValueError):
        build_prediction_report(_result(), p, model_run="r", checkpoint="c", checkpoint_sha256=None)
    with pytest.raises(ValueError):
        build_prediction_report(_result(), p, model_run="r", checkpoint="c", checkpoint_sha256=None,
                                example={"id": "x"}, upload={"filename": "y"})


def test_report_json_is_strict_and_complete():
    p = CandidateParams(14.44912, 811.3488, 2.8)
    upload = {"filename": "k90.csv", "size_bytes": 10, "sha256": "f" * 64, "time_reference": "BJD",
              "n_observations_after_cleaning": 5000, "warnings": ["w"]}
    rep = build_prediction_report(_result(0.5, odd_even_sigma=float("nan")), p, model_run="run", checkpoint="ck",
                                  checkpoint_sha256="0" * 64, upload=upload,
                                  epoch_input={"value": 2455644.3488, "time_reference": "BJD"})
    text = report_to_json(rep)
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["input"] == {"source": "upload", "example": None, "upload": upload}
    assert back["candidate"]["epoch_bkjd"] == pytest.approx(811.3488)
    assert back["candidate"]["epoch_as_entered"] == {"value": 2455644.3488, "time_reference": "BJD"}
    assert back["result"]["score"] == 0.5 and back["result"]["above_threshold"] is False
    assert back["result"]["interpretation"].startswith("below threshold")
    assert back["model"]["decision_threshold"] == 0.671 and back["model"]["checkpoint_sha256"] == "0" * 64
    assert back["diagnostics"] == {"depth_ppm": 120.0, "n_points": 5000, "odd_even_sigma": None}
    assert "not a planet confirmation" in back["caveat"]
    assert math.isclose(back["candidate"]["period_days"], 14.44912)
