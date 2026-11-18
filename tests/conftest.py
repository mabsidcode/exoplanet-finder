"""Shared fixtures. Synthetic light curves are used only to test known physics
(injected transits with known depth and timing); all reported results use real
Kepler data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from exofinder.lightcurve import prepare_light_curve
from exofinder.preprocess import CandidateParams, PreprocessConfig

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "app" / "examples"
RELEASE = ROOT / "models" / "release"


def synthetic_transit_lc(
    period=5.0, epoch=101.3, duration_hours=4.0, depth=1e-3, noise=1e-4, variability=5e-3,
    span=(100.0, 190.0), cadence_min=29.4, seed=0, gaps=((130.0, 133.0),), quarters=2,
):
    """SYNTHETIC light curve: box transit + sinusoidal stellar variability + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(span[0], span[1], cadence_min / 1440.0)
    for g0, g1 in gaps:
        t = t[(t < g0) | (t > g1)]
    phase = np.mod(t - epoch + 0.5 * period, period) - 0.5 * period
    flux = 1.0 + variability * np.sin(2 * np.pi * t / 7.3) + noise * rng.standard_normal(len(t))
    flux[np.abs(phase) < duration_hours / 48.0] -= depth
    seg = np.minimum((t - span[0]) // ((span[1] - span[0]) / quarters), quarters - 1).astype(int)
    return t, flux * 1.0e4, seg  # e-/s-like scale


@pytest.fixture
def synth():
    t, f, s = synthetic_transit_lc()
    return prepare_light_curve(t, f, s), CandidateParams(5.0, 101.3, 4.0)


@pytest.fixture
def pcfg():
    return PreprocessConfig()
