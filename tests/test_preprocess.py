"""Scientific checks on the shared preprocessing, using SYNTHETIC light curves
with injected transits of known depth, period and epoch."""

import dataclasses

import numpy as np
import pytest

from exofinder.lightcurve import BKJD_OFFSET, KEPLER_DEFAULT_BITMASK, prepare_light_curve, to_bkjd
from exofinder.preprocess import (
    CandidateParams,
    PreprocessConfig,
    PreprocessingError,
    bin_median,
    phase_fold,
    preprocess_candidate,
)

from .conftest import synthetic_transit_lc


def test_quality_bitmask_matches_lightkurve():
    lk = pytest.importorskip("lightkurve")
    assert lk.utils.KeplerQualityFlags.DEFAULT_BITMASK == KEPLER_DEFAULT_BITMASK


def test_time_conversion_bjd_to_bkjd():
    assert to_bkjd(np.array([2455644.3488]), "BJD")[0] == pytest.approx(811.3488)
    assert to_bkjd(np.array([811.3488]), "BKJD")[0] == 811.3488
    assert BKJD_OFFSET == 2454833.0
    with pytest.raises(ValueError):
        to_bkjd(np.array([1.0]), "MJD")


def test_duration_hours_to_days():
    assert CandidateParams(10.0, 0.0, 12.0).duration_days == 0.5


def test_phase_fold_alignment():
    P, t0 = 3.7, 135.2
    t = t0 + P * np.array([-3, 0, 2, 10]) + np.array([0.0, 0.01, -0.02, 0.0])
    ph = phase_fold(t, P, t0)
    assert np.allclose(ph, [0.0, 0.01, -0.02, 0.0], atol=1e-9)
    ph_all = phase_fold(np.linspace(100, 200, 10_000), P, t0)
    assert ph_all.min() >= -P / 2 and ph_all.max() < P / 2


def test_transit_preserved_by_detrending(synth, pcfg):
    lc, params = synth
    pc = preprocess_candidate(lc, params, pcfg, keep_intermediates=True)
    # Injected depth 1000 ppm survives detrending of 5000 ppm variability
    assert pc.diagnostics["depth_ppm"] == pytest.approx(1000, rel=0.15)
    in_tr = np.abs(pc.phase) < 0.4 * params.duration_days
    assert np.median(pc.detrended[in_tr]) == pytest.approx(1 - 1e-3, abs=2e-4)
    # Variability removed out of transit
    assert np.std(pc.detrended[~in_tr]) < 3e-4
    # Transit centred in both views; local minimum = -1, median = 0
    g, loc = pc.global_view[0], pc.local_view[0]
    assert abs(int(np.argmin(loc)) - 100) <= 12
    half_width_bins = 0.5 * params.duration_days / (params.period_days / 2001)  # ~33 bins
    assert abs(int(np.argmin(g)) - 1000) <= half_width_bins
    assert np.mean(g[1000 - 25 : 1000 + 26]) == pytest.approx(-1.0, abs=0.15)  # flat transit bottom
    assert loc.min() == pytest.approx(-1.0) and abs(np.median(loc)) < 0.1
    assert pc.global_view.shape == (2, 2001) and pc.local_view.shape == (2, 201)


def test_downward_points_not_clipped_upward_outliers_removed(pcfg):
    t, f, s = synthetic_transit_lc(variability=0.0)
    f[100] *= 1.05  # upward spike (e.g. cosmic ray)
    lc = prepare_light_curve(t, f, s)
    pc = preprocess_candidate(lc, CandidateParams(5.0, 101.3, 4.0), pcfg, keep_intermediates=True)
    assert pc.diagnostics["n_upward_outliers_removed"] >= 1
    assert t[100] not in pc.time
    # every in-transit cadence kept
    n_in = int((np.abs(phase_fold(t, 5.0, 101.3)) < 2.0 / 24).sum())
    assert pc.diagnostics["n_in_transit"] == n_in


def test_missing_values_duplicates_and_unsorted():
    t, f, s = synthetic_transit_lc()
    f[5:15] = np.nan
    t2 = np.concatenate([t, t[:3]])
    f2 = np.concatenate([f, f[:3]])
    s2 = np.concatenate([s, s[:3]])
    order = np.random.default_rng(1).permutation(len(t2))
    lc = prepare_light_curve(t2[order], f2[order], s2[order])
    assert np.all(np.diff(lc.time) > 0)
    assert lc.info["cleaning"]["n_nonfinite_removed"] == 10
    assert lc.info["cleaning"]["n_duplicates_removed"] == 3
    assert np.allclose(np.median(lc.flux), 1.0, atol=0.01)


def test_gaps_create_segments_and_empty_bins_are_masked(pcfg):
    # Remove all data in a large phase range -> empty global bins flagged in mask
    t, f, s = synthetic_transit_lc(variability=0.0)
    ph = phase_fold(t, 5.0, 101.3)
    keep = ~((ph > 1.0) & (ph < 1.8))
    lc = prepare_light_curve(t[keep], f[keep], s[keep])
    pc = preprocess_candidate(lc, CandidateParams(5.0, 101.3, 4.0), pcfg)
    mask = pc.global_view[1]
    assert 0.75 < mask.mean() < 0.9
    assert np.isfinite(pc.global_view).all()


def test_insufficient_transit_coverage_rejected(pcfg):
    t, f, s = synthetic_transit_lc()
    with pytest.raises(PreprocessingError) as e:
        # epoch shifted so that no data fall in transit windows of a long period
        lc = prepare_light_curve(t, f, s)
        preprocess_candidate(lc, CandidateParams(500.0, 300.0, 4.0), pcfg)
    assert e.value.reason in ("insufficient_transit_coverage", "low_local_coverage")


def test_flat_curve_rejected(pcfg):
    t = np.arange(100, 150, 0.0204)
    lc = prepare_light_curve(t, np.full(len(t), 5000.0))
    with pytest.raises(PreprocessingError) as e:
        preprocess_candidate(lc, CandidateParams(5.0, 101.3, 4.0), pcfg)
    assert e.value.reason == "flat_or_no_dip"


def test_too_few_points_and_invalid_params(pcfg):
    lc = prepare_light_curve(np.arange(50.0), np.ones(50))
    with pytest.raises(PreprocessingError) as e:
        preprocess_candidate(lc, CandidateParams(5.0, 1.0, 4.0), pcfg)
    assert e.value.reason == "too_few_points"
    with pytest.raises(PreprocessingError) as e:
        CandidateParams(1.0, 0.0, 30.0).validate()
    assert e.value.reason == "invalid_parameters"


def test_bin_median_overlapping_bins():
    x = np.linspace(-1, 1, 2001)
    v, c, centers = bin_median(x, x, 5, 0.8, -1, 1)
    assert np.allclose(centers, [-0.6, -0.3, 0.0, 0.3, 0.6])
    assert np.allclose(v, centers, atol=1e-3) and (c > 0).all()


def test_config_fingerprint_changes_with_settings():
    a = PreprocessConfig()
    b = dataclasses.replace(a, spline_bkspace_days=2.0)
    assert a.fingerprint() != b.fingerprint()
    assert PreprocessConfig.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"nonsense": 1})


def test_deterministic(synth, pcfg):
    lc, params = synth
    a = preprocess_candidate(lc, params, pcfg)
    b = preprocess_candidate(lc, params, pcfg)
    assert np.array_equal(a.global_view, b.global_view) and np.array_equal(a.local_view, b.local_view)
