"""Integration tests on REAL Kepler data bundled in app/examples (offline)."""

import json

import numpy as np
import pytest

from exofinder.lightcurve import light_curve_from_raw_arrays, read_kepler_fits
from exofinder.preprocess import CandidateParams, PreprocessConfig, preprocess_candidate
from exofinder.release import load_bundled_raw
from exofinder.upload import parse_light_curve_csv

from .conftest import EXAMPLES, RELEASE, ROOT

K90 = EXAMPLES / "kplr011442793.npz"
K90H = CandidateParams(331.603, 140.48, 14.49)  # DR24 TCE 011442793_01

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def raw():
    if not K90.exists():
        pytest.skip("bundled Kepler-90 light curve not present; run `exofinder bundle-examples`")
    return load_bundled_raw(K90)


def test_upload_csv_path_matches_training_path(raw):
    """Training path (bundled FITS arrays) and demo upload path (CSV) give identical views."""
    from exofinder.upload import light_curve_to_csv

    cfg = PreprocessConfig()
    a = preprocess_candidate(light_curve_from_raw_arrays(raw), K90H, cfg)
    csv = light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])
    lc_up, _ = parse_light_curve_csv(csv, "BKJD")
    b = preprocess_candidate(lc_up, K90H, cfg)
    assert np.array_equal(a.global_view, b.global_view)
    assert np.array_equal(a.local_view, b.local_view)


def test_bjd_upload_equivalent(raw):
    from exofinder.upload import light_curve_to_csv

    cfg = PreprocessConfig()
    a = preprocess_candidate(light_curve_from_raw_arrays(raw), K90H, cfg)
    csv = light_curve_to_csv(raw["time"] + 2454833.0, raw["flux"], raw["quality"], raw["segment"])
    lc_up, _ = parse_light_curve_csv(csv, "BJD")
    b = preprocess_candidate(lc_up, K90H, cfg)
    assert np.allclose(a.local_view, b.local_view, atol=1e-4)


def test_bundled_arrays_match_fits_reader(raw):
    files = [ROOT / "data" / "cache" / "raw" / "011442793" / f for f in raw["files"]]
    if not all(f.exists() for f in files):
        pytest.skip("raw FITS cache for Kepler-90 not present")
    a = read_kepler_fits(files)
    b = light_curve_from_raw_arrays(raw)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.flux, b.flux)


def test_release_model_scores_bundled_examples():
    if not (RELEASE / "cnn.pt").exists() or not (EXAMPLES / "examples.json").exists():
        pytest.skip("release model / examples not generated")
    from exofinder.inference import load_bundle, predict_candidate

    bundle = load_bundle(RELEASE / "cnn.pt")
    meta = json.loads((EXAMPLES / "examples.json").read_text())
    for ex in meta["examples"]:
        lc = light_curve_from_raw_arrays(load_bundled_raw(EXAMPLES / ex["light_curve_file"]))
        res, pc = predict_candidate(bundle, lc, CandidateParams(ex["period_days"], ex["epoch_bkjd"], ex["duration_hours"]))
        assert 0.0 <= res["score"] <= 1.0
        assert res["score"] == pytest.approx(ex["score_at_bundle"], abs=1e-5)
        if "score_in_predictions_csv" in ex:
            assert res["score"] == pytest.approx(ex["score_in_predictions_csv"], abs=1e-4)
            assert ex["view_max_abs_diff_vs_dataset"] < 1e-5
