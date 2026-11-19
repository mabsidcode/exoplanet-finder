import logging

import streamlit as st
from ui import (
    EXAMPLES_DIR,
    REPRODUCE_CMD,
    folded_figure,
    later_catalogue_html,
    load_examples,
    load_later_catalogue,
    load_raw,
    load_release_bundle,
    raw_figure,
    release_checkpoint_sha256,
    sample_csv_bytes,
    setup_page,
    view_figure,
)

from exofinder.io_utils import bytes_digest
from exofinder.lightcurve import BKJD_OFFSET, LightCurveError, light_curve_from_raw_arrays
from exofinder.preprocess import CandidateParams, PreprocessingError
from exofinder.report import build_prediction_report, report_to_json
from exofinder.upload import MAX_ROWS, MAX_UPLOAD_BYTES, MIN_ROWS, parse_light_curve_csv

log = logging.getLogger("exofinder.app")

# What to check when preprocessing rejects a candidate (reason codes from exofinder.preprocess).
REJECTION_HINTS = {
    "invalid_parameters": "Check that the period and duration are positive and the duration is shorter than the period.",
    "too_few_points": f"The light curve needs at least {MIN_ROWS} usable observations after cleaning and detrending.",
    "insufficient_transit_coverage": "Hardly any observations fall inside the transit window. Check the epoch, "
    "the duration (in hours) and the time reference.",
    "low_local_coverage": "The data cover too little of the transit neighbourhood. Check the epoch and period, "
    "or supply a longer light curve.",
    "low_global_coverage": "The data cover too little of the orbit, usually because the period is long compared "
    "with the light curve's time span.",
    "flat_or_no_dip": "No dip was found at the given ephemeris. Check the epoch, period and time reference.",
}

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
* Limits: {MAX_UPLOAD_BYTES // 1024 // 1024} MB, {MAX_ROWS:,} rows; at least {MIN_ROWS} usable rows.
* Lines starting with `#` are ignored. Rows with a blank or non-numeric `time`/`flux` (Kepler gap cadences)
  are skipped; blank or non-integer `quality`/`segment` values are rejected.
* The model was trained on Kepler long-cadence (~29.4 min) PDCSAP light curves. Other instruments or
  cadences are outside its training distribution.
"""
        )
        if (EXAMPLES_DIR / "kplr011442793.npz").exists():
            st.download_button("Download a sample CSV (Kepler-90, real data)", sample_csv_bytes("kplr011442793.npz"),
                               "kepler90_pdcsap_bkjd.csv", "text/csv")
            st.caption("To reproduce the bundled Kepler-90 h result: time reference BKJD, period 331.603 d, "
                       "epoch 140.48, duration 14.49 h.")
    up = st.file_uploader("Light-curve CSV", type=["csv"])
    c1, c2, c3, c4 = st.columns(4)
    # Fixed keys: the epoch label mentions the time reference, and without a key changing the reference would
    # create a new widget and silently reset the epoch to 0.
    tsys = c1.selectbox("Time reference", ["BKJD", "BJD"], key="upload_time_ref",
                        help="BKJD = BJD_TDB − 2454833 days. Applies to both the CSV time column and the epoch.")
    period = c2.number_input("Period (days)", min_value=0.0, value=0.0, format="%.6f", key="upload_period")
    epoch = c3.number_input(f"Transit epoch ({tsys}, days)", value=0.0, format="%.6f", key="upload_epoch",
                            help="Time of a transit centre, in the same reference as the CSV time column")
    dur = c4.number_input("Duration (hours)", min_value=0.0, value=0.0, format="%.3f", key="upload_duration")
    if up is None:
        st.info("Upload a CSV and enter the candidate parameters to continue.")
        st.stop()
    if period <= 0 or dur <= 0:
        st.info("Enter a positive period (days) and duration (hours) to score the candidate.")
        st.stop()
    data = up.getvalue()
    try:
        lc, warnings = parse_light_curve_csv(data, tsys)
    except LightCurveError as e:
        st.error(f"**Could not use this file.** {e}")
        st.caption(f"Error code: `{e.reason}`. Open *Supported CSV format* above for the expected layout.")
        st.stop()
    except Exception:  # noqa: BLE001 - never show a traceback for a bad upload
        log.exception("unexpected error while parsing an uploaded CSV")
        st.error("**Could not use this file.** It could not be read as a light-curve CSV. "
                 "Check it against *Supported CSV format* above.")
        st.stop()
    epoch_bkjd = epoch - BKJD_OFFSET if tsys == "BJD" else epoch
    if not (lc.time.min() - 3650 <= epoch_bkjd <= lc.time.max() + 3650):
        warnings.append(f"The epoch is more than 10 years from the light curve (BKJD {lc.time.min():.1f}–"
                        f"{lc.time.max():.1f}). Check that the epoch uses the selected time reference ({tsys}).")
    for w in warnings:
        st.warning(w)
    params = CandidateParams(period, epoch_bkjd, dur)
    upload_info = {"filename": up.name, "size_bytes": len(data), "sha256": bytes_digest(data), "time_reference": tsys,
                   "n_observations_after_cleaning": len(lc), "warnings": warnings}

try:
    with st.spinner("Detrending, folding and scoring…"):
        from exofinder.inference import predict_candidate

        result, pc = predict_candidate(bundle, lc, params)
except (PreprocessingError, LightCurveError) as e:
    detail = str(e).removeprefix(f"{e.reason}: ")
    st.error(f"**Input rejected: {e.reason.replace('_', ' ')}.** {detail[:1].upper() + detail[1:]}.")
    if e.reason in REJECTION_HINTS:
        st.markdown(REJECTION_HINTS[e.reason])
    st.caption("The model is only applied when the folded light curve contains enough information about the transit.")
    st.stop()
except Exception:  # noqa: BLE001 - numerical failures are reported, not shown as tracebacks
    log.exception("unexpected error while scoring a candidate")
    st.error("**Scoring failed** with an unexpected numerical error for this light curve and ephemeris. "
             "Check the inputs; if they are correct, please report the case.")
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
        later = load_later_catalogue()
        later_body = later_catalogue_html(later["examples"].get(ex["id"]), later.get("unresolved_text", ""))
        if later_body:
            st.markdown("<div class='card'><h4>Later catalogues (context only; not used for training or evaluation)</h4>"
                        f"{later_body}</div>", unsafe_allow_html=True)
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
    st.caption("Y-axis zoomed to the 1st–99.5th percentile of the points (and the binned curve); a few extreme points, "
               "such as transits of other planets in the same system, can fall outside it.")
with tabs[2]:
    st.plotly_chart(view_figure(pc.global_view, pc.global_centers, "global", params), width="stretch")
with tabs[3]:
    st.plotly_chart(view_figure(pc.local_view, pc.local_centers, "local", params), width="stretch")

model_run = bundle.meta.get("experiment", {}).get("run")
report = build_prediction_report(
    result, params, model_run=model_run, checkpoint="models/release/cnn.pt",
    checkpoint_sha256=release_checkpoint_sha256(), example=ex,
    upload=None if ex is not None else upload_info,
    epoch_input=None if ex is not None else {"value": epoch, "time_reference": tsys},
)
st.download_button("Download prediction report (JSON)", report_to_json(report),
                   file_name=f"exofinder_report_{ex['id'] if ex else 'upload'}.json", mime="application/json")
st.caption(f"Model: `{model_run}` · preprocessing v{result['preprocess_version']} "
           f"(fingerprint {result['preprocess_fingerprint']}) · code in `src/exofinder`")
