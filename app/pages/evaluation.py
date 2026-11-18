import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve
from ui import REPORT_DIR, REPRODUCE_CMD, layout, load_release_card, setup_page

setup_page()
st.title("Measured results")

mpath = REPORT_DIR / "metrics.json"
if not mpath.exists():
    st.warning("No evaluation found at `reports/full/metrics.json`. Status: the full experiment has not been evaluated yet.")
    st.code(REPRODUCE_CMD, language="bash")
    st.stop()

metrics = json.loads(mpath.read_text())
card = load_release_card()
rel = metrics["release_model"]
ds = metrics["dataset"]["by_split"]

st.markdown(
    "All numbers are computed from saved per-example predictions on a **test split of host stars never used for "
    "training, checkpoint selection or threshold choice**. Labels are Kepler DR24 `av_training_set` dispositions "
    "(PC = 1; AFP and NTP = 0), so these metrics measure agreement with that catalogue, not with confirmed planets."
)

c = st.columns(4)
t = metrics["models"][rel]["test"]
c[0].metric("Test examples (stars)", f"{t['n']:,} ({ds['test']['n_stars']:,})")
c[1].metric("PC prevalence", f"{t['prevalence']:.1%}")
c[2].metric("CNN average precision", f"{t['average_precision']:.3f}")
c[3].metric("CNN ROC-AUC", f"{t['roc_auc']:.3f}")

rows = []
for name, m in metrics["models"].items():
    if "test" not in m:
        continue
    x = m["test"]
    rows.append({"model": name + (" (release)" if name == rel else ""), "AP": x["average_precision"], "ROC-AUC": x["roc_auc"],
                 "threshold": x["threshold"], "precision": x["precision"], "recall": x["recall"], "F1": x["f1"],
                 "accuracy": x["accuracy"], "TP": x["confusion"]["tp"], "FP": x["confusion"]["fp"],
                 "TN": x["confusion"]["tn"], "FN": x["confusion"]["fn"]})
st.subheader("Test split, identical examples for every model")
st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, width="stretch")
st.caption("Average precision = Σₙ (Rₙ − Rₙ₋₁)·Pₙ over score thresholds (scikit-learn definition). Random ranking ≈ prevalence. "
           "Thresholds were selected on the validation split by maximising F1; a prediction is PC when score ≥ threshold. "
           f"Always predicting AFP/NTP would give accuracy {1 - t['prevalence']:.3f} with zero recall — so accuracy is secondary.")

pp = REPORT_DIR / "predictions.csv"
if pp.exists():
    preds = pd.read_csv(pp, dtype={"tce_id": str})
    test = preds[preds["split"] == "test"]
    fig = go.Figure()
    show = [rel] + [m for m in ("baseline_hgb", "baseline_logreg") if m in set(test["model"])]
    colors = {rel: "#8fb3ff", "baseline_hgb": "#f2b866", "baseline_logreg": "#9aa3b5"}
    for name in show:
        d = test[test["model"] == name]
        p, r, _ = precision_recall_curve(d["label"], d["score"])
        fig.add_scatter(x=r, y=p, mode="lines", name=f"{name} (AP {metrics['models'][name]['test']['average_precision']:.3f})",
                        line=dict(color=colors.get(name), width=2))
    fig.add_hline(y=t["prevalence"], line_dash="dash", line_color="#6c7489", annotation_text="chance = prevalence")
    st.plotly_chart(layout(fig, "Precision–recall curve, test split", "Recall", "Precision", height=420), width="stretch")

agg = metrics.get("cnn_seed_aggregate_test", {})
if agg:
    st.subheader("Training-seed variation (same split)")
    st.dataframe(pd.DataFrame([{"variant": v, "runs": a["n_runs"],
                                "AP mean": a["average_precision"]["mean"], "AP sd": a["average_precision"]["std"],
                                "ROC-AUC mean": a["roc_auc"]["mean"], "F1 mean": a["f1"]["mean"]} for v, a in agg.items()])
                 .style.format(precision=3, na_rep="n/a"), hide_index=True)

bs = metrics.get("bootstrap_test")
if bs:
    st.subheader("Uncertainty")
    lines = []
    for k, v in bs.items():
        if k == "difference":
            lines.append(f"- AP difference {v['models'][0]} − {v['models'][1]}: {v['ap_diff_mean']:+.3f} "
                         f"(95% star-bootstrap interval {v['ap_diff_ci95'][0]:+.3f} to {v['ap_diff_ci95'][1]:+.3f})")
        else:
            lines.append(f"- {k}: AP 95% interval {v['ap_ci95'][0]:.3f}–{v['ap_ci95'][1]:.3f}")
    st.markdown("\n".join(lines))

sub = metrics.get("subgroups_test", {}).get(rel)
if sub:
    st.subheader("Where the release CNN does well and poorly (test split)")
    st.dataframe(pd.DataFrame(sub).style.format(precision=3, na_rep="n/a"), hide_index=True, width="stretch")
    st.caption("Depth, duration and SNR terciles use edges computed on the training split. AP is shown only for subgroups "
               "with ≥30 examples and ≥5 of each class.")

st.subheader("Figures")
for f, cap in [("confusion_" + rel, "Confusion matrix at the validation-selected threshold"),
               ("learning_curves", "Training and validation learning curves"),
               ("score_hist_test", "Score distributions by catalogue disposition"),
               ("error_analysis_examples", "Rule-selected correct and incorrect examples (incl. failures)")]:
    p = REPORT_DIR / "figures" / f"{f}.png"
    if p.exists():
        st.image(str(p), caption=cap)

if card:
    st.caption(f"Release model `{card['run']}` selected by: {card['selection_rule']}. Released {card['released_utc']}.")
