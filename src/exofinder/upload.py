"""Parse user-uploaded light-curve CSV files into the shared LightCurve format.

Supported CSV (header row required, column names case-insensitive):
    time      required  days, in BKJD (BJD_TDB - 2454833) or BJD_TDB
    flux      required  positive flux in any unit (e.g. e-/s or normalised ~1)
    quality   optional  Kepler SAP_QUALITY integers; rows failing the default
                        Kepler bitmask are removed, exactly as in training
    segment   optional  integer segment id (e.g. Kepler quarter); each segment
                        is median-normalised and detrended separately

Anything else is ignored. Files are rejected above ``MAX_UPLOAD_BYTES`` or
``MAX_ROWS`` rows, or below ``MIN_ROWS`` usable rows.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .lightcurve import (
    KEPLER_DEFAULT_BITMASK,
    LightCurve,
    LightCurveError,
    prepare_light_curve,
    to_bkjd,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 1_000_000
MIN_ROWS = 100
KEPLER_BKJD_RANGE = (120.0, 1600.0)  # approximate span of Kepler Q0-Q17


def parse_light_curve_csv(data: bytes, time_system: str = "BKJD") -> tuple[LightCurve, list[str]]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise LightCurveError("file_too_large", f"File is {len(data) / 1e6:.1f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB")
    try:
        df = pd.read_csv(io.BytesIO(data), comment="#", nrows=MAX_ROWS + 1, float_precision="round_trip")
    except Exception as e:  # noqa: BLE001
        raise LightCurveError("bad_csv", f"Could not parse CSV: {e}") from e
    if len(df) > MAX_ROWS:
        raise LightCurveError("too_many_rows", f"More than {MAX_ROWS:,} rows")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"time", "flux"} - set(df.columns)
    if missing:
        raise LightCurveError("missing_columns", f"Missing required column(s): {', '.join(sorted(missing))}")
    warnings: list[str] = []
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(df["flux"], errors="coerce").to_numpy(dtype=float)
    n_bad = int((~np.isfinite(t) | ~np.isfinite(f)).sum())
    if n_bad:
        warnings.append(f"{n_bad} rows with missing or non-numeric time/flux were removed")
    keep = np.ones(len(df), dtype=bool)
    if "quality" in df.columns:
        q = pd.to_numeric(df["quality"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        bad_q = (q & KEPLER_DEFAULT_BITMASK) != 0
        keep &= ~bad_q
        if bad_q.any():
            warnings.append(f"{int(bad_q.sum())} rows flagged by the Kepler default quality bitmask were removed")
    seg = None
    if "segment" in df.columns:
        seg = pd.to_numeric(df["segment"], errors="coerce").fillna(-1).astype(int).to_numpy()[keep]
    t_bkjd = to_bkjd(t[keep], time_system)
    lc = prepare_light_curve(t_bkjd, f[keep], seg, info={"source": "upload", "time_system": time_system})
    if len(lc) < MIN_ROWS:
        raise LightCurveError("too_few_points", f"Only {len(lc)} usable rows; at least {MIN_ROWS} are needed")
    lo, hi = KEPLER_BKJD_RANGE
    if lc.time.min() < lo - 50 or lc.time.max() > hi + 50:
        warnings.append(
            f"Times span BKJD {lc.time.min():.1f}–{lc.time.max():.1f}, outside the Kepler mission range "
            f"(~{lo:.0f}–{hi:.0f}). Check the selected time reference."
        )
    if lc.info["cleaning"]["n_duplicates_removed"]:
        warnings.append(f"{lc.info['cleaning']['n_duplicates_removed']} duplicate timestamps were removed")
    if lc.info["cleaning"]["was_unsorted"]:
        warnings.append("Rows were not in time order and have been sorted")
    return lc, warnings


def light_curve_to_csv(time_bkjd: np.ndarray, flux: np.ndarray, quality: np.ndarray | None = None,
                       segment: np.ndarray | None = None) -> bytes:
    """Write the supported CSV format (time in BKJD)."""
    # float64 so float32 FITS values are written with enough digits to round-trip exactly
    cols = {"time": np.asarray(time_bkjd, dtype=np.float64), "flux": np.asarray(flux, dtype=np.float64)}
    if quality is not None:
        cols["quality"] = quality
    if segment is not None:
        cols["segment"] = segment
    buf = io.StringIO()
    buf.write("# Exoplanet Finder light-curve CSV. time: BKJD = BJD_TDB - 2454833 (days); flux: PDCSAP_FLUX (e-/s)\n")
    pd.DataFrame(cols).to_csv(buf, index=False, float_format="%.17g")  # exact float64 round trip
    return buf.getvalue().encode()
