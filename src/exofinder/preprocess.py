"""Candidate preprocessing: detrend, phase-fold and build global/local views.

This module is the single preprocessing implementation used for building the
training set and for inference in the demo. Its behaviour is fixed by
:data:`PREPROCESS_VERSION` plus a :class:`PreprocessConfig`; both are stored
inside every model checkpoint and checked at inference time.

Pipeline for one candidate (period P, epoch t0, duration D, all in days)
-----------------------------------------------------------------------
1. Segments: the light curve is split at quarter boundaries and at gaps longer
   than ``gap_days``. Each segment is detrended independently.
2. Transit mask: points with |phase| < ``transit_mask_half_width_durations``*D
   are excluded from the trend fit so the spline cannot absorb the transit.
3. Trend: a least-squares cubic B-spline with interior knots every
   ``spline_bkspace_days`` (knots are dropped where too few unmasked points
   exist), fit iteratively with symmetric ``spline_sigma_clip`` rejection.
   Clipping only decides which points inform the fit; no data are removed.
4. Detrended flux = flux / trend. Only *upward* outliers above
   ``upper_outlier_sigma`` robust standard deviations are removed, so genuine
   transit dips are never clipped.
5. Phase: phi = ((t - t0 + P/2) mod P) - P/2, in days, so the transit is at 0.
6. Views (geometry follows AstroNet, Shallue & Vanderburg 2018):
     global: 2001 bins of width P/2001 spanning [-P/2, P/2)
     local:  201 bins of width 0.16*D spanning [-4D, +4D] (clipped to +-P/2)
   Each bin holds the median of y = flux/trend - 1. Empty bins are filled by
   linear interpolation between observed bins and flagged in a coverage mask.
7. Normalisation, with m_L, m_G the medians of observed local/global bins:
     s = m_L - min(observed local bins)            (transit depth, > 0)
     local_norm  = (local  - m_L) / s   -> median 0, transit minimum -1
     global_norm = (global - m_G) / s   -> same scale, so secondary eclipses
                                           keep their depth relative to the
                                           primary transit
   Both are clipped to +-``view_clip``. If s < ``min_scale`` (no dip, flat
   input) the candidate is rejected rather than divided by ~0.

Each view is returned as a (2, n_bins) float32 array: channel 0 is the
normalised flux, channel 1 is the coverage mask (1 = observed bin).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

from .lightcurve import LightCurve

PREPROCESS_VERSION = "1.0.0"


class PreprocessingError(ValueError):
    """A candidate was rejected; ``reason`` is a short machine-readable code."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class PreprocessConfig:
    flux_column: str = "PDCSAP_FLUX"
    quality_bitmask: str = "default"
    gap_days: float = 0.75
    spline_bkspace_days: float = 1.5
    spline_sigma_clip: float = 3.0
    spline_max_iter: int = 5
    transit_mask_half_width_durations: float = 1.5
    upper_outlier_sigma: float = 5.0
    global_num_bins: int = 2001
    local_num_bins: int = 201
    local_num_durations: float = 4.0
    local_bin_width_durations: float = 0.16
    min_points: int = 100
    min_in_transit_points: int = 3
    min_global_coverage: float = 0.3
    min_local_coverage: float = 0.5
    min_scale: float = 1e-6
    view_clip: float = 10.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessConfig:
        d = {k: v for k, v in d.items() if k != "version"}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {"version": PREPROCESS_VERSION, **asdict(self)}

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateParams:
    period_days: float
    epoch_bkjd: float
    duration_hours: float

    @property
    def duration_days(self) -> float:
        return self.duration_hours / 24.0

    def validate(self) -> None:
        vals = (self.period_days, self.epoch_bkjd, self.duration_hours)
        if not all(np.isfinite(v) for v in vals):
            raise PreprocessingError("invalid_parameters", "period, epoch and duration must be finite")
        if self.period_days <= 0 or self.duration_hours <= 0:
            raise PreprocessingError("invalid_parameters", "period and duration must be positive")
        if self.duration_days >= self.period_days:
            raise PreprocessingError("invalid_parameters", "duration must be shorter than the period")


@dataclass
class ProcessedCandidate:
    global_view: np.ndarray  # (2, global_num_bins) float32: [flux, coverage]
    local_view: np.ndarray  # (2, local_num_bins) float32
    diagnostics: dict[str, Any]
    # Intermediate arrays for plotting (time-sorted, after cleaning).
    time: np.ndarray | None = None
    flux: np.ndarray | None = None
    trend: np.ndarray | None = None
    detrended: np.ndarray | None = None
    phase: np.ndarray | None = None
    global_centers: np.ndarray | None = None
    local_centers: np.ndarray | None = None


# ---------------------------------------------------------------- utilities


def robust_std(x: np.ndarray) -> float:
    """1.4826 * median absolute deviation (std-equivalent for Gaussian noise)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def phase_fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase in days in [-P/2, P/2), with transits of the given epoch at 0."""
    return np.mod(np.asarray(time, dtype=float) - epoch + 0.5 * period, period) - 0.5 * period


def transit_number(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Integer index of the nearest transit (0 = the epoch transit)."""
    return np.floor((np.asarray(time, dtype=float) - epoch) / period + 0.5).astype(np.int64)


def split_segments(time: np.ndarray, segment: np.ndarray, gap_days: float) -> np.ndarray:
    """Segment labels that change at input segment changes and at time gaps."""
    if len(time) == 0:
        return np.zeros(0, dtype=int)
    brk = np.concatenate([[False], (np.diff(time) > gap_days) | (np.diff(segment) != 0)])
    return np.cumsum(brk)


def _interior_knots(x: np.ndarray, bkspace: float, k: int = 3) -> np.ndarray:
    """Evenly spaced knots, dropping any that would leave < k+1 points in an interval."""
    if x[-1] - x[0] < 2 * bkspace:
        return np.zeros(0)
    cand = np.arange(x[0] + bkspace, x[-1] - 0.5 * bkspace, bkspace)
    knots: list[float] = []
    prev_idx = 0
    for c in cand:
        idx = int(np.searchsorted(x, c, side="right"))
        if idx - prev_idx >= k + 1:
            knots.append(c)
            prev_idx = idx
    while knots and len(x) - int(np.searchsorted(x, knots[-1], side="right")) < k + 1:
        knots.pop()
    return np.asarray(knots)


def _fit_once(x: np.ndarray, y: np.ndarray, bkspace: float, bbox: tuple[float, float]):
    """Return (callable model, method name).

    ``bbox`` spans the whole segment so that masked transits at a segment edge
    are interpolated by the spline basis rather than extrapolated.
    """
    n = len(x)
    if n >= 8:
        knots = _interior_knots(x, bkspace)
        try:
            spl = LSQUnivariateSpline(x, y, knots, k=3, bbox=list(bbox))
            return spl, "spline"
        except ValueError:
            pass
    if n >= 3:
        deg = min(2, n - 1)
        x0 = x.mean()
        coef = np.polyfit(x - x0, y, deg)
        return (lambda xx: np.polyval(coef, np.asarray(xx) - x0)), f"poly{deg}"
    med = float(np.median(y))
    return (lambda xx: np.full(np.shape(xx), med)), "constant"


def fit_trend(x: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, str]:
    """Iteratively sigma-clipped spline trend fitted to ``fit_mask`` points."""
    use = fit_mask.copy()
    bbox = (float(x[0]), float(x[-1]))
    model, method = _fit_once(x[use], y[use], cfg.spline_bkspace_days, bbox)
    for _ in range(cfg.spline_max_iter):
        resid = y - model(x)
        sigma = robust_std(resid[use])
        if not np.isfinite(sigma) or sigma == 0:
            break
        new_use = fit_mask & (np.abs(resid) <= cfg.spline_sigma_clip * sigma)
        if new_use.sum() < 3 or np.array_equal(new_use, use):
            break
        use = new_use
        model, method = _fit_once(x[use], y[use], cfg.spline_bkspace_days, bbox)
    return np.asarray(model(x), dtype=float), method


def bin_median(
    x: np.ndarray, y: np.ndarray, num_bins: int, bin_width: float, x_min: float, x_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median of y in ``num_bins`` (possibly overlapping) bins.

    Bin i covers [x_min + i*spacing, x_min + i*spacing + bin_width), with
    spacing = (x_max - x_min - bin_width) / (num_bins - 1), as in AstroNet.
    Returns (values with NaN for empty bins, counts, bin centres).
    """
    if bin_width <= 0 or x_max <= x_min or bin_width > x_max - x_min:
        raise ValueError("invalid binning parameters")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    starts = x_min + spacing * np.arange(num_bins)
    lo = np.searchsorted(xs, starts, side="left")
    hi = np.searchsorted(xs, starts + bin_width, side="left")
    counts = hi - lo
    values = np.full(num_bins, np.nan)
    for i in np.flatnonzero(counts):
        values[i] = np.median(ys[lo[i] : hi[i]])
    return values, counts, starts + 0.5 * bin_width


def _fill_empty(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    obs = np.isfinite(values)
    return np.interp(centers, centers[obs], values[obs])


# ---------------------------------------------------------------- main entry


def detrend(lc: LightCurve, params: CandidateParams, cfg: PreprocessConfig):
    """Return (time, flux, trend, detrended, info) with invalid points removed."""
    t, f = lc.time, lc.flux
    seg = split_segments(t, lc.segment, cfg.gap_days)
    phase = phase_fold(t, params.period_days, params.epoch_bkjd)
    masked = np.abs(phase) < cfg.transit_mask_half_width_durations * params.duration_days
    trend = np.full(len(t), np.nan)
    methods: dict[str, int] = {}
    dropped = 0
    for sid in np.unique(seg):
        idx = np.flatnonzero(seg == sid)
        fit_mask = ~masked[idx]
        if fit_mask.sum() < 3:
            dropped += 1
            continue
        trend[idx], method = fit_trend(t[idx], f[idx], fit_mask, cfg)
        methods[method] = methods.get(method, 0) + 1
    valid = np.isfinite(trend) & (trend > 0)
    t, f, trend, masked = t[valid], f[valid], trend[valid], masked[valid]
    det = f / trend
    oot_sigma = robust_std(det[~masked] - 1.0) if (~masked).any() else float("nan")
    upward = np.zeros(len(det), dtype=bool)
    if np.isfinite(oot_sigma) and oot_sigma > 0:
        upward = (det - 1.0) > cfg.upper_outlier_sigma * oot_sigma
    keep = ~upward
    info = {
        "n_segments": int(len(np.unique(seg))),
        "n_segments_dropped": int(dropped),
        "trend_methods": methods,
        "n_upward_outliers_removed": int(upward.sum()),
        "n_invalid_trend_removed": int((~valid).sum()),
    }
    return t[keep], f[keep], trend[keep], det[keep], info


def preprocess_candidate(
    lc: LightCurve, params: CandidateParams, cfg: PreprocessConfig, keep_intermediates: bool = False
) -> ProcessedCandidate:
    params.validate()
    if len(lc) < cfg.min_points:
        raise PreprocessingError("too_few_points", f"{len(lc)} valid observations < {cfg.min_points}")

    t, f, trend, det, dinfo = detrend(lc, params, cfg)
    if len(t) < cfg.min_points:
        raise PreprocessingError("too_few_points", f"{len(t)} observations remain after detrending")

    P, t0, D = params.period_days, params.epoch_bkjd, params.duration_days
    phase = phase_fold(t, P, t0)
    y = det - 1.0
    in_transit = np.abs(phase) < 0.5 * D
    n_in = int(in_transit.sum())
    if n_in < cfg.min_in_transit_points:
        raise PreprocessingError(
            "insufficient_transit_coverage", f"{n_in} in-transit observations < {cfg.min_in_transit_points}"
        )

    g_vals, g_counts, g_centers = bin_median(
        phase, y, cfg.global_num_bins, P / cfg.global_num_bins, -0.5 * P, 0.5 * P
    )
    half = min(0.5 * P, cfg.local_num_durations * D)
    l_vals, l_counts, l_centers = bin_median(
        phase, y, cfg.local_num_bins, cfg.local_bin_width_durations * D, -half, half
    )
    g_obs, l_obs = g_counts > 0, l_counts > 0
    g_cov, l_cov = float(g_obs.mean()), float(l_obs.mean())
    if l_cov < cfg.min_local_coverage:
        raise PreprocessingError("low_local_coverage", f"local coverage {l_cov:.2f} < {cfg.min_local_coverage}")
    if g_cov < cfg.min_global_coverage:
        raise PreprocessingError("low_global_coverage", f"global coverage {g_cov:.2f} < {cfg.min_global_coverage}")

    l_med = float(np.median(l_vals[l_obs]))
    g_med = float(np.median(g_vals[g_obs]))
    scale = l_med - float(np.min(l_vals[l_obs]))
    if not np.isfinite(scale) or scale < cfg.min_scale:
        raise PreprocessingError("flat_or_no_dip", f"local-view depth {scale:.2e} < {cfg.min_scale:.0e}")

    local_norm = np.clip((_fill_empty(l_vals, l_centers) - l_med) / scale, -cfg.view_clip, cfg.view_clip)
    global_norm = np.clip((_fill_empty(g_vals, g_centers) - g_med) / scale, -cfg.view_clip, cfg.view_clip)
    global_view = np.stack([global_norm, g_obs.astype(float)]).astype(np.float32)
    local_view = np.stack([local_norm, l_obs.astype(float)]).astype(np.float32)

    diagnostics = _diagnostics(t, phase, y, params, cfg, scale, n_in, g_cov, l_cov, local_norm, l_centers)
    diagnostics.update(dinfo)
    diagnostics["n_points"] = int(len(t))
    diagnostics["time_span_days"] = float(t[-1] - t[0])

    out = ProcessedCandidate(global_view=global_view, local_view=local_view, diagnostics=diagnostics)
    if keep_intermediates:
        out.time, out.flux, out.trend, out.detrended, out.phase = t, f, trend, det, phase
        out.global_centers, out.local_centers = g_centers, l_centers
    return out


def _diagnostics(t, phase, y, params, cfg, scale, n_in, g_cov, l_cov, local_norm, l_centers) -> dict[str, Any]:
    """Interpretable signal statistics, used by the baseline and for subgroup analysis."""
    P, t0, D = params.period_days, params.epoch_bkjd, params.duration_days
    in_tr = np.abs(phase) < 0.5 * D
    oot = np.abs(phase) >= cfg.transit_mask_half_width_durations * D
    noise = robust_std(y[oot]) if oot.sum() > 10 else float("nan")
    epochs = transit_number(t, P, t0)
    odd = in_tr & (epochs % 2 == 1)
    even = in_tr & (epochs % 2 == 0)
    oe_sigma = float("nan")
    if odd.sum() >= 2 and even.sum() >= 2 and np.isfinite(noise) and noise > 0:
        d_odd, d_even = -np.median(y[odd]), -np.median(y[even])
        oe_sigma = float(abs(d_odd - d_even) / (noise * np.sqrt(1 / odd.sum() + 1 / even.sum())))
    sec = np.abs(phase) > 0.5 * P - 0.5 * D
    sec_depth = float(-np.median(y[sec]) + np.median(y[oot])) if sec.sum() >= 3 and oot.sum() else float("nan")
    inner = np.abs(l_centers) < 0.25 * D
    edge = (np.abs(l_centers) >= 0.25 * D) & (np.abs(l_centers) < 0.5 * D)
    shape = float(np.mean(local_norm[inner]) / np.mean(local_norm[edge])) if inner.any() and edge.any() else float("nan")
    return {
        "depth_ppm": float(scale * 1e6),
        "noise_ppm": float(noise * 1e6),
        "snr_est": float(scale / noise * np.sqrt(n_in)) if np.isfinite(noise) and noise > 0 else float("nan"),
        "n_in_transit": int(n_in),
        "n_transits_observed": int(len(np.unique(epochs[in_tr]))),
        "odd_even_sigma": oe_sigma,
        "secondary_depth_ratio": float(sec_depth / scale) if np.isfinite(sec_depth) else float("nan"),
        "local_shape_ratio": shape,
        "global_coverage": g_cov,
        "local_coverage": l_cov,
    }
