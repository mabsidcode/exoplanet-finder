import json

import streamlit as st
from ui import (
    EXAMPLES_DIR,
    REPRODUCE_CMD,
    ROOT,
    folded_figure,
    load_examples,
    load_raw,
    load_release_bundle,
    raw_figure,
    setup_page,
    view_figure,
)

from exofinder.lightcurve import LightCurveError, light_curve_from_raw_arrays
from exofinder.preprocess import CandidateParams, PreprocessingError
from exofinder.upload import MAX_UPLOAD_BYTES, light_curve_to_csv, parse_light_curve_csv

setup_page()
st.title("Exoplanet Finder")
st.markdown(
    "Score how closely a **candidate transit signal** in a Kepler light curve resembles signals labelled "
    "*planet candidate* (PC) in the Kepler DR24 training set, rather than *astrophysical false positive* (AFP) "
    "or *non-transiting phenomenon* (NTP). You supply the candidate's period, epoch and duration; the app does "
    "**not** search for new periods. A high score is not a planet confirmation."
)

bundle, err = load_release_bundle()
if bundle is None:
    st.error(f"**Model unavailable.** {err}")
    st.markdown("Produce it with the full pipeline (several hours of downloading, see README):")
    st.code(REPRODUCE_CMD, language="bash")
    st.stop()

examples = load_examples()
source = st.radio("Light-curve source", ["Bundled real Kepler example", "Upload a CSV"], horizontal=True)

lc, params, ex = None, None, None
if source == "Bundled real Kepler example":
    if not examples:
        st.warning("No bundled examples found. Run `exofinder bundle-examples --config configs/full.yaml`.")
        st.stop()
    titles = [e["title"] for e in examples["examples"]]
    choice = st.selectbox("Example", titles, index=len(titles) - 1)
    ex = examples["examples"][titles.index(choice)]
    raw = load_raw(str(EXAMPLES_DIR / ex["light_curve_file"]))
    lc = light_curve_from_raw_arrays(raw, bundle.preprocess.quality_bitmask)
    params = CandidateParams(ex["period_days"], ex["epoch_bkjd"], ex["duration_hours"])
else:
    with st.expander("Supported CSV format", expanded=False):
        st.markdown(
            f"""
* Header row with columns **`time`** and **`flux`** (case-insensitive). Optional: **`quality`** (Kepler
  `SAP_QUALITY`; flagged rows are removed with the same bitmask as in training) and **`segment`**
  (e.g. quarter number; each segment is normalised and detrended separately).
* `time` in days, either **BKJD** (BJD_TDB − 2454833, the Kepler `TIME` column) or **BJD_TDB**.
* `flux` must be positive (e.g. Kepler `PDCSAP_FLUX` in e⁻/s, or flux normalised near 1).
* Limits: {MAX_UPLOAD_BYTES // 1024 // 1024} MB, 1,000,000 rows; at least 100 usable rows.
* The model was trained on Kepler long-cadence (~29.4 min) PDCSAP light curves. Other instruments or
  cadences are outside its training distribution.
"""
        )
        if examples:
            k90 = load_raw(str(EXAMPLES_DIR / "kplr011442793.npz"))
            st.download_button("Download a sample CSV (Kepler-90, real data)",
                               light_curve_to_csv(k90["time"], k90["flux"], k90["quality"], k90["segment"]),
                               "kepler90_pdcsap_bkjd.csv", "text/csv")
    up = st.file_uploader("Light-curve CSV", type=["csv"])
    c1, c2, c3, c4 = st.columns(4)
    tsys = c1.selectbox("Time reference", ["BKJD", "BJD"], help="BKJD = BJD_TDB − 2454833 days")
    period = c2.number_input("Period (days)", min_value=0.0, value=0.0, format="%.6f")
    epoch = c3.number_input(f"Transit epoch ({tsys}, days)", value=0.0, format="%.6f",
                            help="Time of a transit centre, in the same reference as the CSV time column")
    dur = c4.number_input("Duration (hours)", min_value=0.0, value=0.0, format="%.3f")
    if up is None:
        st.info("Upload a CSV and enter the candidate parameters to continue.")
        st.stop()
    if period <= 0 or dur <= 0:
        st.info("Enter a positive period and duration.")
        st.stop()
    try:
        lc, warnings = parse_light_curve_csv(up.getvalue(), tsys)
    except LightCurveError as e:
        st.error(f"Could not use this file ({e.reason}): {e}")
        st.stop()
    for w in warnings:
        st.warning(w)
    epoch_bkjd = epoch - 2454833.0 if tsys == "BJD" else epoch
    params = CandidateParams(period, epoch_bkjd, dur)

try:
    with st.spinner("Detrending, folding and scoring…"):
        from exofinder.inference import predict_candidate

        result, pc = predict_candidate(bundle, lc, params)
except PreprocessingError as e:
    st.error(f"**Input rejected — {e.reason.replace('_', ' ')}.** {e}")
    st.markdown("The model is only applied when the folded light curve contains enough information about the transit.")
    st.stop()

# ------------------------------------------------------------------ results
left, right = st.columns(2)
with left:
    if ex is not None:
        held = "yes — never used for training or model selection" if ex["in_heldout_evaluation"] else "no"
        if ex.get("tce_id"):
            body = (f"<span class='tag cat'>{ex['catalogue_disposition']}</span> {ex['catalogue_release']}<br>"
                    f"TCE <code>{ex['tce_id']}</code> · KIC {ex['kepid']} · training label {ex['training_label']}<br>"
                    f"Split: <b>{ex['split']}</b> · held out from training: {held}<br>"
                    f"<span class='small'>Bundled because: {ex['selection_rule']}</span>")
        else:
            body = (f"<span class='tag cat'>{ex['catalogue_disposition']}</span> {ex['catalogue_release']}<br>"
                    f"{ex['catalogue_status']}<br><span class='small'>Parameters: {ex['parameter_source']}</span>")
        st.markdown(f"<div class='card'><h4>Catalogue information (not model output)</h4>{body}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'><h4>Catalogue information</h4>User upload — no catalogue label.</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card'><h4>Candidate parameters used</h4>Period {params.period_days:.6f} d · "
        f"epoch {params.epoch_bkjd:.5f} BKJD · duration {params.duration_hours:.3f} h</div>", unsafe_allow_html=True)

with right:
    s, thr = result["score"], result["threshold"]
    verdict = "Above threshold — resembles DR24 planet-candidate signals" if s >= thr else "Below threshold — resembles DR24 false-positive / non-transit signals"
    st.markdown(
        f"<div class='card'><h4><span class='tag model'>Model output</span></h4><div class='big'>{s:.3f}</div>"
        f"<div class='sub'>CNN score (0–1) · decision threshold {thr:.3f} (chosen on validation data, max F1)</div>"
        f"<p style='margin-top:0.6rem'><b>{verdict}.</b></p>"
        f"<div class='small'>The score ranks signals; it is not a calibrated probability that a planet exists. "
        f"Confirming a planet needs further evidence (e.g. centroid and contamination checks, follow-up "
        f"spectroscopy or statistical validation).</div></div>", unsafe_allow_html=True)
    d = result["diagnostics"]
    st.markdown(
        f"<div class='card'><h4>Measured signal statistics</h4>Depth {d['depth_ppm']:.0f} ppm · noise {d['noise_ppm']:.0f} ppm · "
        f"SNR ≈ {d['snr_est']:.1f}<br>Transits observed {d['n_transits_observed']} · in-transit points {d['n_in_transit']} · "
        f"coverage global {d['global_coverage']:.0%} / local {d['local_coverage']:.0%}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Raw light curve", "Folded transit", "Global view (model input)", "Local view (model input)"])
with tabs[0]:
    st.plotly_chart(raw_figure(pc), width="stretch")
    st.caption(f"{len(pc.time):,} observations kept after quality filtering, detrending and upward-outlier removal "
               f"({d['n_segments']} segments).")
with tabs[1]:
    st.plotly_chart(folded_figure(pc, params), width="stretch")
with tabs[2]:
    st.plotly_chart(view_figure(pc.global_view, pc.global_centers, "global", params), width="stretch")
with tabs[3]:
    st.plotly_chart(view_figure(pc.local_view, pc.local_centers, "local", params), width="stretch")

report = {
    "tool": "Exoplanet Finder",
    "input": {"source": "bundled" if ex else "upload", "example": ex, "period_days": params.period_days,
              "epoch_bkjd": params.epoch_bkjd, "duration_hours": params.duration_hours},
    "model": {"checkpoint": "models/release/cnn.pt", "run": bundle.meta.get("experiment", {}).get("run"),
              "decision_threshold": result["threshold"], "preprocess_version": result["preprocess_version"],
              "preprocess_fingerprint": result["preprocess_fingerprint"]},
    "result": {k: result[k] for k in ("score", "above_threshold", "score_semantics")},
    "diagnostics": result["diagnostics"],
    "caveat": "Model score for a candidate signal. Not a planet confirmation.",
}
st.download_button("Download prediction report (JSON)", json.dumps(report, indent=2, default=float),
                   file_name=f"exofinder_report_{(ex or {}).get('id', 'upload')}.json", mime="application/json")
st.caption(f"Model: `{report['model']['run']}` · preprocessing v{result['preprocess_version']} "
           f"(fingerprint {result['preprocess_fingerprint']}) · code in `{ROOT.name}/src/exofinder`")
