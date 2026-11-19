"""Parse user-uploaded light-curve CSV files into the shared LightCurve format.

Supported CSV (header row required, column names case-insensitive):
    time      required  days, in BKJD (BJD_TDB - 2454833) or BJD_TDB
    flux      required  positive flux in any unit (e.g. e-/s or normalised ~1)
    quality   optional  Kepler SAP_QUALITY integers; rows failing the default
                        Kepler bitmask are removed, exactly as in training
    segment   optional  integer segment id (e.g. Kepler quarter); each segment
                        is median-normalised and detrended separately

Anything else is ignored. Rows with blank or non-numeric time/flux (Kepler gap
cadences) are skipped with a warning; blank or non-integer quality/segment
values reject the file, because guessing them would change which data are
used. Files are rejected above ``MAX_UPLOAD_BYTES`` or ``MAX_ROWS`` rows, or
below ``MIN_ROWS`` usable rows. Every rejection is a :class:`LightCurveError`
whose message is shown to the user.
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


MAX_FLAG_VALUE = 2**31 - 1  # Kepler SAP_QUALITY is a 32-bit integer


def _integer_column(df: pd.DataFrame, name: str, what: str, min_value: int) -> np.ndarray:
    """Return an optional integer column, or raise a LightCurveError that names the first bad row.

    A blank cell is an error too: silently treating it as "good quality" or as some segment would
    change which observations are used.
    """
    raw = df[name]
    x = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(x) & (x == np.round(x)) & (x >= min_value) & (x <= MAX_FLAG_VALUE)
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        value = "" if pd.isna(raw.iloc[i]) else str(raw.iloc[i])
        raise LightCurveError(
            f"bad_{name}",
            f"Column '{name}' must contain {what}; {int((~ok).sum()):,} row(s) do not "
            f"(first: data row {i + 1}, value '{value[:40]}'). Fix those rows or delete the '{name}' column.",
        )
    return x.astype(np.int64)


def parse_light_curve_csv(data: bytes, time_system: str = "BKJD") -> tuple[LightCurve, list[str]]:
    """Parse an uploaded CSV. Raises :class:`LightCurveError` with a user-readable message on bad input."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise LightCurveError("file_too_large", f"File is {len(data) / 1e6:.1f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB")
    if not data.strip():
        raise LightCurveError("empty_file", "The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data), comment="#", nrows=MAX_ROWS + 1, float_precision="round_trip",
                         encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise LightCurveError("bad_csv", "The file is not UTF-8 text. Upload a plain-text, comma-separated CSV.") from e
    except pd.errors.EmptyDataError as e:
        raise LightCurveError("bad_csv", "The file has no header row or columns.") from e
    except Exception as e:  # noqa: BLE001 - any parser failure is reported to the user
        raise LightCurveError("bad_csv", f"Could not parse the file as CSV: {e}") from e
    if len(df) > MAX_ROWS:
        raise LightCurveError("too_many_rows", f"More than {MAX_ROWS:,} rows")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"time", "flux"} - set(df.columns)
    if missing:
        found = ", ".join(f"'{c}'" for c in df.columns[:8]) + (" …" if len(df.columns) > 8 else "")
        sep_like = len(df.columns) == 1 and any(c in df.columns[0] for c in ";\t|")
        hint = " The file seems to use ';', '|' or tabs between columns; save it comma-separated." if sep_like else ""
        raise LightCurveError(
            "missing_columns",
            f"Missing required column(s): {', '.join(sorted(missing))}. Found: {found}.{hint}",
        )
    if len(df) == 0:
        raise LightCurveError("too_few_points", f"The file has a header but no data rows; at least {MIN_ROWS} are needed.")
    warnings: list[str] = []
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(df["flux"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(f)
    if not finite.any():
        raise LightCurveError("no_numeric_rows", f"None of the {len(df):,} rows has numeric 'time' and 'flux' values.")
    n_bad = int((~finite).sum())
    if n_bad:
        warnings.append(f"{n_bad:,} rows with missing (NaN) or non-numeric time/flux were skipped; "
                        "Kepler files contain such gap cadences")
    keep = np.ones(len(df), dtype=bool)
    if "quality" in df.columns:
        q = _integer_column(df, "quality", "non-negative integer Kepler SAP_QUALITY flags", 0)
        bad_q = (q & KEPLER_DEFAULT_BITMASK) != 0
        keep &= ~bad_q
        if bad_q.any():
            warnings.append(f"{int(bad_q.sum()):,} rows flagged by the Kepler default quality bitmask were removed")
    seg = None
    if "segment" in df.columns:
        seg = _integer_column(df, "segment", "integer segment ids (e.g. Kepler quarter numbers)", -MAX_FLAG_VALUE)[keep]
    t_bkjd = to_bkjd(t[keep], time_system)
    try:
        lc = prepare_light_curve(t_bkjd, f[keep], seg, info={"source": "upload", "time_system": time_system})
    except LightCurveError as e:
        if e.reason != "no_valid_data":
            raise
        raise LightCurveError(
            "no_valid_data",
            "No usable rows remain: 'flux' must be positive (e.g. Kepler PDCSAP_FLUX in e-/s, or flux normalised "
            "near 1), not magnitudes or flux differences centred on 0.",
        ) from e
    if len(lc) < MIN_ROWS:
        raise LightCurveError("too_few_points", f"Only {len(lc):,} usable row(s) remain after cleaning; "
                                                f"at least {MIN_ROWS} are needed.")
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
