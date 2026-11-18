"""Light-curve loading and cleaning shared by training and inference.

Every light curve, whether read from Kepler FITS files or from an uploaded CSV,
passes through :func:`prepare_light_curve`. That function is the single entry
point into the preprocessing pipeline, so training and the demo see identical
cleaning rules.

Conventions
-----------
* Time is Barycentric Kepler Julian Date, BKJD = BJD_TDB - 2454833.0, in days.
  This is the TIME column of Kepler light-curve files and the reference used
  by ``tce_time0bk`` in the DR24 TCE table.
* Flux is made relative by dividing each segment (a Kepler quarter, or a
  user-supplied segment) by its median, so the out-of-transit level is ~1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

BKJD_OFFSET = 2454833.0

# Lightkurve's KeplerQualityFlags.DEFAULT_BITMASK (lightkurve 2.x): attitude
# tweak, safe mode, coarse point, Earth point, desaturation event, manual
# exclude, detector anomaly, no data, thruster firing. Hard-coded so worker
# processes need not import lightkurve; tests/test_lightcurve.py checks it
# against the installed lightkurve.
KEPLER_DEFAULT_BITMASK = 1130799
QUALITY_BITMASKS = {"none": 0, "default": KEPLER_DEFAULT_BITMASK}


class LightCurveError(ValueError):
    """Raised when a light curve cannot be read or contains no usable data."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class LightCurve:
    time: np.ndarray  # BKJD days, strictly increasing
    flux: np.ndarray  # relative flux (segment median = 1)
    segment: np.ndarray  # integer segment id (Kepler quarter for FITS input)
    info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.time)


def to_bkjd(time: np.ndarray, system: str) -> np.ndarray:
    """Convert a time array to BKJD. ``system`` is 'BKJD' or 'BJD' (BJD_TDB)."""
    system = system.upper()
    t = np.asarray(time, dtype=float)
    if system == "BKJD":
        return t
    if system in ("BJD", "BJD_TDB"):
        return t - BKJD_OFFSET
    raise ValueError(f"Unsupported time system {system!r}; use 'BKJD' or 'BJD'")


def prepare_light_curve(
    time: Sequence[float] | np.ndarray,
    flux: Sequence[float] | np.ndarray,
    segment: Sequence[int] | np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> LightCurve:
    """Clean raw arrays into a :class:`LightCurve` (time already in BKJD).

    Steps: drop non-finite values; sort by time (stable); drop duplicate
    timestamps keeping the first occurrence; drop segments whose median flux is
    not positive; divide each segment by its median.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if t.shape != f.shape or t.ndim != 1:
        raise LightCurveError("bad_shape", "time and flux must be 1-D arrays of equal length")
    s = np.zeros(len(t), dtype=int) if segment is None else np.asarray(segment).astype(int)
    if s.shape != t.shape:
        raise LightCurveError("bad_shape", "segment must have the same length as time")
    stats: dict[str, Any] = {"n_input": int(len(t))}

    ok = np.isfinite(t) & np.isfinite(f)
    stats["n_nonfinite_removed"] = int((~ok).sum())
    t, f, s = t[ok], f[ok], s[ok]

    order = np.argsort(t, kind="stable")
    stats["was_unsorted"] = bool(np.any(order != np.arange(len(order))))
    t, f, s = t[order], f[order], s[order]

    if len(t):
        dup = np.concatenate([[False], np.diff(t) == 0])
        stats["n_duplicates_removed"] = int(dup.sum())
        t, f, s = t[~dup], f[~dup], s[~dup]
    else:
        stats["n_duplicates_removed"] = 0

    keep = np.ones(len(t), dtype=bool)
    rel = np.empty_like(f)
    dropped = []
    for seg in np.unique(s):
        idx = s == seg
        med = np.median(f[idx])
        if not np.isfinite(med) or med <= 0:
            keep[idx] = False
            dropped.append(int(seg))
            continue
        rel[idx] = f[idx] / med
    stats["segments_dropped_nonpositive_median"] = dropped
    t, rel, s = t[keep], rel[keep], s[keep]
    stats["n_clean"] = int(len(t))
    if len(t) == 0:
        raise LightCurveError("no_valid_data", "No finite, positive-flux observations remain")
    out_info = dict(info or {})
    out_info["cleaning"] = stats
    return LightCurve(time=t, flux=rel, segment=s, info=out_info)


def read_kepler_fits(
    paths: Sequence[str | Path], flux_column: str = "PDCSAP_FLUX", quality_bitmask: str | int = "default"
) -> LightCurve:
    """Read Kepler long-cadence light-curve files into a single LightCurve.

    Checks per file: Kepler telescope, long-cadence mode, barycentric TDB time
    with BJDREFI = 2454833. Observations are kept when TIME and the flux column
    are finite and ``SAP_QUALITY & bitmask == 0``.
    """
    mask = QUALITY_BITMASKS[quality_bitmask] if isinstance(quality_bitmask, str) else int(quality_bitmask)
    times, fluxes, segs, files = [], [], [], []
    for p in sorted(Path(x) for x in paths):
        with fits.open(p, memmap=False) as hdul:
            h0, h1 = hdul[0].header, hdul[1].header
            if h0.get("TELESCOP") != "Kepler":
                raise LightCurveError("bad_file", f"{p.name}: not a Kepler file")
            if h0.get("OBSMODE") != "long cadence":
                raise LightCurveError("bad_file", f"{p.name}: not long cadence ({h0.get('OBSMODE')})")
            if h1.get("TIMESYS") != "TDB" or int(h1.get("BJDREFI", -1)) != int(BKJD_OFFSET):
                raise LightCurveError("bad_time_reference", f"{p.name}: unexpected time reference")
            d = hdul[1].data
            t = np.asarray(d["TIME"], dtype=float)
            f = np.asarray(d[flux_column], dtype=float)
            q = np.asarray(d["SAP_QUALITY"], dtype=np.int64)
            good = np.isfinite(t) & np.isfinite(f) & ((q & mask) == 0)
            quarter = int(h0.get("QUARTER", -1))
            files.append(
                {
                    "filename": p.name,
                    "quarter": quarter,
                    "data_release": h0.get("DATA_REL"),
                    "pipeline_version": h0.get("PROCVER"),
                    "n_cadences": int(len(t)),
                    "n_good": int(good.sum()),
                }
            )
            times.append(t[good])
            fluxes.append(f[good])
            segs.append(np.full(int(good.sum()), quarter, dtype=int))
    if not files:
        raise LightCurveError("no_files", "No light-curve files supplied")
    info = {
        "source": "kepler_fits",
        "flux_column": flux_column,
        "quality_bitmask": mask,
        "files": files,
        "data_releases": sorted({str(f["data_release"]) for f in files}),
    }
    return prepare_light_curve(np.concatenate(times), np.concatenate(fluxes), np.concatenate(segs), info)


def raw_arrays_from_fits(paths: Sequence[str | Path], flux_column: str = "PDCSAP_FLUX") -> dict[str, np.ndarray]:
    """Extract TIME, flux, SAP_QUALITY and quarter for bundling demo examples."""
    t, f, q, s = [], [], [], []
    for p in sorted(Path(x) for x in paths):
        with fits.open(p, memmap=False) as hdul:
            d = hdul[1].data
            n = len(d)
            t.append(np.asarray(d["TIME"], dtype=float))
            f.append(np.asarray(d[flux_column], dtype=float))
            q.append(np.asarray(d["SAP_QUALITY"], dtype=np.int64))
            s.append(np.full(n, int(hdul[0].header.get("QUARTER", -1))))
    return {"time": np.concatenate(t), "flux": np.concatenate(f), "quality": np.concatenate(q), "segment": np.concatenate(s)}


def light_curve_from_raw_arrays(
    raw: dict[str, np.ndarray], quality_bitmask: str | int = "default", info: dict[str, Any] | None = None
) -> LightCurve:
    """Apply the same quality filtering as :func:`read_kepler_fits` to bundled arrays."""
    mask = QUALITY_BITMASKS[quality_bitmask] if isinstance(quality_bitmask, str) else int(quality_bitmask)
    q = np.asarray(raw["quality"], dtype=np.int64)
    t = np.asarray(raw["time"], dtype=float)
    f = np.asarray(raw["flux"], dtype=float)
    good = np.isfinite(t) & np.isfinite(f) & ((q & mask) == 0)
    return prepare_light_curve(t[good], f[good], np.asarray(raw["segment"])[good], info)
