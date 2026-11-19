"""Shared helpers for the Streamlit app: paths, cached loaders, charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EXAMPLES_DIR = ROOT / "app" / "examples"
RELEASE_DIR = ROOT / "models" / "release"
REPORT_DIR = ROOT / "reports" / "full"

ACCENT = "#8fb3ff"
MUTED = "#8a93a8"
WARN = "#f2b866"
BG = "#0a0e1a"
PANEL = "#131a2c"

REPRODUCE_CMD = """exofinder fetch-catalog --config configs/full.yaml
exofinder make-splits   --config configs/full.yaml
exofinder fetch         --config configs/full.yaml
exofinder build-dataset --config configs/full.yaml
exofinder train-baseline --config configs/full.yaml
exofinder train-cnn     --config configs/full.yaml
exofinder evaluate      --config configs/full.yaml
exofinder release       --config configs/full.yaml
exofinder bundle-examples --config configs/full.yaml"""

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: 0.01em;}
.card {background: #131a2c; border: 1px solid #222b42; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;}
.card h4 {margin: 0 0 0.5rem 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8a93a8;}
.card .big {font-size: 2.2rem; font-weight: 600; color: #e4e8f1; line-height: 1.1;}
.card .sub {color: #8a93a8; font-size: 0.9rem;}
.tag {display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.8rem; margin-right: 0.3rem;
      border: 1px solid #2c3755; color: #c9d3ea;}
.tag.cat {border-color: #6b5a2e; color: #f2d79b;}
.tag.model {border-color: #2e4a7a; color: #b8cdfa;}
.tag.warn {border-color: #7a5a2e; color: #f2b866;}
.small {color: #8a93a8; font-size: 0.85rem;}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_release_bundle():
    from exofinder.inference import load_bundle

    path = RELEASE_DIR / "cnn.pt"
    if not path.exists():
        return None, f"No released checkpoint found at `{path.relative_to(ROOT)}`."
    try:
        return load_bundle(path), None
    except Exception as e:  # noqa: BLE001 - surfaced to the user
        return None, f"Checkpoint at `{path.relative_to(ROOT)}` could not be loaded: {e}"


@st.cache_data
def release_checkpoint_sha256() -> str | None:
    from exofinder.io_utils import file_digest

    path = RELEASE_DIR / "cnn.pt"
    return file_digest(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def sample_csv_bytes(npz_name: str) -> bytes:
    from exofinder.upload import light_curve_to_csv

    raw = load_raw(str(EXAMPLES_DIR / npz_name))
    return light_curve_to_csv(raw["time"], raw["flux"], raw["quality"], raw["segment"])


@st.cache_data
def load_examples() -> dict | None:
    p = EXAMPLES_DIR / "examples.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_later_catalogue() -> dict:
    """Verified DR25 / Archive context for the bundled examples (scripts/check_dr25_koi.py); empty if absent."""
    p = EXAMPLES_DIR / "later_catalogue.json"
    return json.loads(p.read_text()) if p.exists() else {"examples": {}}


def later_catalogue_html(ctx: dict | None, unresolved_text: str) -> str | None:
    """Card body for one example's later-catalogue context, or None when there is nothing to show."""
    from html import escape

    if not ctx:
        return None
    lab = ctx.get("dr24_label")
    lines = [f"Original DR24 training label: <b>{escape(lab)}</b> "
             f"<span class='small'>(Kepler DR24 <code>av_training_set</code>; used for training and evaluation)</span>"
             if lab else "Original DR24 training label: <b>none</b> <span class='small'>(not a DR24 TCE)</span>"]
    d = ctx.get("dr25")
    if d:
        flags = f" · false-positive flags: {escape(', '.join(d['fp_flags']))}" if d["fp_flags"] else ""
        lines.append(
            f"DR25 KOI <code>{escape(d['kepoi_name'])}</code>: <b>{escape(d['disposition'])}</b>{flags}<br>"
            f"<span class='small'>Source: {escape(d['source']['table'])}, {escape(d['source']['column'])}; retrieved "
            f"{d['source']['retrieved_utc'][:10]}. Matched by {escape(d['match_evidence'])}.</span>")
    else:
        reason = ctx.get("unresolved_reason")
        lines.append(f"DR25 KOI: <b>{escape(unresolved_text)}</b>"
                     + (f" <span class='small'>({escape(reason)})</span>" if reason else ""))
    a = ctx.get("archive")
    if a:
        alias = f" (Archive name {escape(a['pl_name'])})" if a["pl_name"] != a["kepler_name"] else ""
        lines.append(
            f"NASA Exoplanet Archive status: <b>{escape(a['status'])}</b>, {escape(a['kepler_name'])}{alias}<br>"
            f"<span class='small'>Source: {escape(a['source']['table'])}; retrieved {a['source']['retrieved_utc'][:10]}. "
            f"{escape(a['cross_id'])}; {escape(a['evidence'])}. Discovery reference: "
            f"{escape(a['discovery_reference'].split(' (')[0])}.</span>")
    if ctx.get("disagreement"):
        lines.append(f"<span class='tag warn'>Labels disagree</span> {escape(ctx['disagreement'])}")
    return "".join(f"<div style='margin-bottom:0.6rem'>{x}</div>" for x in lines)


@st.cache_data
def load_release_card() -> dict | None:
    p = RELEASE_DIR / "release.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_raw(path: str) -> dict:
    from exofinder.release import load_bundled_raw

    return load_bundled_raw(Path(path))


def layout(fig: go.Figure, title: str, xlab: str, ylab: str, height: int = 370) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=60, r=20, t=45, b=95),
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlab,
        yaxis_title=ylab,
        showlegend=True,
        # legend below the axis title, so it never collides with the chart title on narrow screens
        legend=dict(orientation="h", y=-0.22, x=0, xanchor="left", yanchor="top", font=dict(size=11)),
        font=dict(size=12),
    )
    fig.update_xaxes(gridcolor="#1d2540", zeroline=False)
    fig.update_yaxes(gridcolor="#1d2540", zeroline=False)
    return fig


def _thin(n: int, max_points: int = 20000) -> slice:
    return slice(None, None, max(1, n // max_points))


def _break_at_gaps(t: np.ndarray, y: np.ndarray, gap_days: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN between points separated by more than ``gap_days`` so a line is not drawn across data gaps."""
    cut = np.flatnonzero(np.diff(t) > gap_days) + 1
    return np.insert(t.astype(float), cut, np.nan), np.insert(y.astype(float), cut, np.nan)


def raw_figure(pc) -> go.Figure:
    s = _thin(len(pc.time))
    fig = go.Figure()
    fig.add_scattergl(x=pc.time[s], y=pc.flux[s], mode="markers", marker=dict(size=2, color="#c9d3ea"), name="relative flux")
    tx, ty = _break_at_gaps(pc.time[s], pc.trend[s])
    fig.add_scattergl(x=tx, y=ty, mode="lines", line=dict(width=1.2, color=WARN), connectgaps=False,
                      name="fitted trend (transits masked)")
    return layout(fig, "Light curve after quality filtering", "Time (BKJD = BJD_TDB − 2454833, days)",
                  "Relative flux (segment median = 1)")


def folded_figure(pc, params) -> go.Figure:
    """Folded points plus a binned median. The y-range is set from the data's 1st–99.5th percentiles and the
    binned curve, so a shallow transit stays visible when a few points (e.g. another planet's transits) are deep."""
    D = params.duration_days
    win = min(0.5 * params.period_days, 4 * D)
    sel = np.abs(pc.phase) < win
    x, y = pc.phase[sel] * 24, (pc.detrended[sel] - 1) * 1e6
    fig = go.Figure()
    fig.add_scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color="#c9d3ea", opacity=0.45),
                      name="detrended observations")
    edges = np.linspace(-win * 24, win * 24, 65)  # 64 bins, 1/8 duration wide when the window is ±4 D
    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(y[idx == i]) if np.any(idx == i) else np.nan for i in range(len(centers))])
    fig.add_scatter(x=centers, y=med, mode="lines+markers", line=dict(width=2, color=ACCENT),
                    marker=dict(size=4), connectgaps=False, name="binned median")
    fig.add_vrect(x0=-12 * D, x1=12 * D, fillcolor=ACCENT, opacity=0.08, line_width=0,
                  annotation_text="expected transit", annotation_position="top left")
    if y.size:
        lo = min(np.percentile(y, 1), np.nanmin(med) if np.isfinite(med).any() else np.inf)
        hi = np.percentile(y, 99.5)
        pad = 0.15 * (hi - lo) if hi > lo else 1.0
        fig.update_yaxes(range=[lo - pad, hi + pad])
    return layout(fig, "Detrended, phase-folded flux near the transit", "Hours from transit centre", "Flux change (ppm)")


def view_figure(view: np.ndarray, centers: np.ndarray, kind: str, params) -> go.Figure:
    flux, mask = view[0], view[1].astype(bool)
    if kind == "global":
        x = centers / params.period_days
        xlab, title = "Phase (fraction of orbital period)", f"Global view — whole orbit, {len(flux)} bins"
    else:
        x = centers / params.duration_days
        xlab, title = "Time from transit centre (transit durations)", f"Local view — ±{abs(x[0]):.1f} durations, {len(flux)} bins"
    fig = go.Figure()
    fig.add_scatter(x=x, y=flux, mode="lines", line=dict(width=1.2, color=ACCENT), name="normalised flux")
    if (~mask).any():
        fig.add_scatter(x=x[~mask], y=flux[~mask], mode="markers", marker=dict(size=5, color=WARN, symbol="x"), name="empty bin (interpolated)")
    return layout(fig, title, xlab, "Normalised flux (median 0, transit −1)")
