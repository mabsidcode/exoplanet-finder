"""Evaluate every saved model on identical examples and write a report.

Procedure (fixed before looking at test results):
1. Score every usable example (train, val, test, demo_holdout) with every
   model: the two baselines and each finished CNN run.
2. Each model's decision threshold is the one it stored after training
   (validation max-F1). Test data are never used to choose thresholds,
   checkpoints or the released model.
3. Release model = the global+local CNN run with the highest validation AP.
4. Test metrics, star-level bootstrap confidence intervals, subgroup tables
   and error-analysis examples (chosen by fixed rules, below) are produced.

Outputs: reports/<experiment>/{predictions.csv, metrics.json, RESULTS.md,
figures/*.png}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from .baseline import make_features
from .config import Paths
from .dataset import ViewDataset
from .inference import load_bundle, score_views
from .io_utils import atomic_path, atomic_write_json, atomic_write_text, utc_now_iso
from .metrics import binary_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_BOOT = 1000
SUBGROUP_MIN = 30


def _discover(paths: Paths, exp: str) -> dict[str, dict[str, Any]]:
    mdir = paths.model_dir(exp)
    models: dict[str, dict[str, Any]] = {}
    for name in ("logreg", "hgb"):
        p = mdir / f"baseline_{name}.joblib"
        if p.exists():
            models[f"baseline_{name}"] = {"kind": "baseline", "path": p}
    for p in sorted(mdir.glob("cnn_*/best.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if "decision_threshold" in ck:
            variant = "global_local" if ck["model_config"]["use_local"] else "global_only"
            models[p.parent.name] = {"kind": "cnn", "path": p, "variant": variant}
    return models


def _score(models: dict[str, dict[str, Any]], ds: ViewDataset) -> pd.DataFrame:
    X = make_features(ds.meta)
    cols = ["tce_id", "kepid", "tce_plnt_num", "split", "label", "original_disposition", "period_days",
            "duration_hours", "depth_ppm", "snr_est", "catalog_model_snr"]
    rows = []
    for name, m in models.items():
        if m["kind"] == "baseline":
            obj = joblib.load(m["path"])
            scores = obj["model"].predict_proba(X)[:, 1]
            thr = obj["meta"]["decision_threshold"]
        else:
            b = load_bundle(m["path"])
            scores = np.concatenate([score_views(b, ds.global_view[i:i + 1024], ds.local_view[i:i + 1024])
                                     for i in range(0, len(ds.labels), 1024)])
            thr = b.threshold
        m["threshold"] = float(thr)
        df = ds.meta[cols].copy()
        df["model"] = name
        df["score"] = scores
        df["threshold"] = thr
        df["predicted"] = (df["score"] >= thr).astype(int)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _bootstrap(df: pd.DataFrame, models: list[str], seed: int = 0) -> dict[str, Any]:
    """Star-level bootstrap of test AP (resampling host stars with replacement)."""
    rng = np.random.default_rng(seed)
    piv = df.pivot_table(index=["tce_id", "kepid", "label"], columns="model", values="score").reset_index()
    stars = piv["kepid"].unique()
    groups = {k: np.flatnonzero(piv["kepid"].to_numpy() == k) for k in stars}
    draws = {m: [] for m in models}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[k] for k in rng.choice(stars, len(stars), replace=True)])
        y = piv["label"].to_numpy()[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for m in models:
            draws[m].append(average_precision_score(y, piv[m].to_numpy()[idx]))
    out = {m: {"ap_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n_boot": len(v)}
           for m, v in draws.items()}
    if len(models) >= 2:
        d = np.asarray(draws[models[0]]) - np.asarray(draws[models[1]])
        out["difference"] = {"models": models[:2], "ap_diff_mean": float(d.mean()),
                             "ap_diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def _subgroups(test: pd.DataFrame, train_meta: pd.DataFrame, model: str) -> list[dict[str, Any]]:
    t = test[test["model"] == model]
    out = []
    for col, label in [("depth_ppm", "Measured transit depth (ppm)"), ("duration_hours", "Duration (hours)"),
                       ("snr_est", "Measured SNR")]:
        edges = np.nanquantile(train_meta[col].astype(float), [1 / 3, 2 / 3])
        bins = pd.cut(t[col], [-np.inf, edges[0], edges[1], np.inf],
                      labels=[f"< {edges[0]:.3g}", f"{edges[0]:.3g} - {edges[1]:.3g}", f">= {edges[1]:.3g}"])
        for b, g in t.groupby(bins, observed=False):
            out.append(_sub_row(label, str(b), g))
    for disp, g in t.groupby("original_disposition"):
        out.append(_sub_row("Catalogue disposition", disp, g))
    return out


def _sub_row(group: str, value: str, g: pd.DataFrame) -> dict[str, Any]:
    y, s, p = g["label"].to_numpy(), g["score"].to_numpy(), g["predicted"].to_numpy()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return {
        "group": group, "value": value, "n": int(len(g)), "n_pos": n_pos, "n_neg": n_neg,
        "average_precision": float(average_precision_score(y, s)) if n_pos >= 5 and n_neg >= 5 and len(g) >= SUBGROUP_MIN else None,
        "recall": float(p[y == 1].mean()) if n_pos else None,
        "false_positive_rate": float(p[y == 0].mean()) if n_neg else None,
    }


def _select_examples(t: pd.DataFrame) -> pd.DataFrame:
    """Error-analysis examples chosen by fixed rules on the release model's test scores."""
    pos, neg = t[t["label"] == 1], t[t["label"] == 0]
    picks = [
        ("true positive: highest-scoring PC", pos[pos["predicted"] == 1].nlargest(2, "score")),
        ("true negative: lowest-scoring AFP", neg[(neg["predicted"] == 0) & (neg["original_disposition"] == "AFP")].nsmallest(1, "score")),
        ("true negative: lowest-scoring NTP", neg[(neg["predicted"] == 0) & (neg["original_disposition"] == "NTP")].nsmallest(1, "score")),
        ("false positive: highest-scoring AFP", neg[(neg["predicted"] == 1) & (neg["original_disposition"] == "AFP")].nlargest(2, "score")),
        ("false positive: highest-scoring NTP", neg[(neg["predicted"] == 1) & (neg["original_disposition"] == "NTP")].nlargest(1, "score")),
        ("false negative: lowest-scoring PC", pos[pos["predicted"] == 0].nsmallest(3, "score")),
        ("borderline: closest to threshold", t.iloc[(t["score"] - t["threshold"]).abs().argsort()[:2]]),
    ]
    frames = [df.assign(category=cat) for cat, df in picks if len(df)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def evaluate(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    exp = cfg["experiment_name"]
    ds = ViewDataset(paths.dataset_dir(exp))
    models = _discover(paths, exp)
    if not models:
        raise RuntimeError("No trained models found; run train-baseline and train-cnn first")
    rep = paths.report_dir(exp)
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preds = _score(models, ds)
    with atomic_path(rep / "predictions.csv") as tmp:
        preds.to_csv(tmp, index=False)

    metrics: dict[str, Any] = {"experiment": exp, "evaluated_utc": utc_now_iso(), "models": {},
                               "dataset": {k: ds.manifest[k] for k in ("n_usable_examples", "n_usable_stars", "by_split", "views_sha256")}}
    for name in models:
        p = preds[preds["model"] == name]
        thr = models[name]["threshold"]
        metrics["models"][name] = {
            "kind": models[name]["kind"],
            "variant": models[name].get("variant"),
            **{split: binary_metrics(p.loc[p["split"] == split, "label"], p.loc[p["split"] == split, "score"], thr)
               for split in ("val", "test") if (p["split"] == split).any()},
        }
    cnn = {n: m for n, m in models.items() if m["kind"] == "cnn"}
    gl = [n for n in cnn if cnn[n]["variant"] == "global_local"]
    release = max(gl, key=lambda n: metrics["models"][n]["val"]["average_precision"]) if gl else None
    metrics["release_model"] = release
    metrics["release_rule"] = "global+local CNN run with the highest validation average precision"

    agg = {}
    for variant in ("global_local", "global_only"):
        runs = [n for n in cnn if cnn[n]["variant"] == variant]
        if runs:
            agg[variant] = {"n_runs": len(runs), "runs": runs}
            for k in ("average_precision", "roc_auc", "precision", "recall", "f1", "accuracy"):
                v = np.array([metrics["models"][r]["test"][k] for r in runs], dtype=float)
                agg[variant][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                                   "min": float(v.min()), "max": float(v.max())}
    metrics["cnn_seed_aggregate_test"] = agg

    test = preds[preds["split"] == "test"]
    if test["label"].nunique() == 2:
        compare = [m for m in [release, "baseline_hgb", "baseline_logreg"] if m in models]
        metrics["bootstrap_test"] = _bootstrap(test[test["model"].isin(compare)], compare)
    train_meta = ds.meta[ds.meta["split"] == "train"]
    for m in [release, "baseline_hgb"]:
        if m in models:
            metrics.setdefault("subgroups_test", {})[m] = _subgroups(test, train_meta, m)
    demo = preds[preds["split"] == "demo_holdout"]
    metrics["demo_holdout"] = demo[["tce_id", "model", "label", "original_disposition", "score", "threshold", "predicted"]].to_dict("records")

    examples = pd.DataFrame()
    if release:
        examples = _select_examples(test[test["model"] == release])
        metrics["error_analysis_examples"] = examples[["category", "tce_id", "kepid", "original_disposition", "score",
                                                       "threshold", "period_days", "duration_hours", "depth_ppm", "snr_est"]].to_dict("records")
    _figures(preds, metrics, models, ds, examples, paths, exp, fig_dir)
    atomic_write_json(rep / "metrics.json", metrics)
    atomic_write_text(rep / "RESULTS.md", _report(metrics, models, ds, paths, exp))
    print(json.dumps({n: {s: round(metrics["models"][n][s]["average_precision"], 4) for s in ("val", "test") if s in metrics["models"][n]}
                      for n in models}, indent=2))
    print(f"Release model (by validation AP): {release}")
    return metrics


# ---------------------------------------------------------------- figures

COLORS = {"cnn": "#2a6fdb", "baseline_hgb": "#d9822b", "baseline_logreg": "#7a7a7a", "global_only": "#3aa57a"}


def _figures(preds, metrics, models, ds, examples, paths, exp, fig_dir: Path) -> None:
    test = preds[preds["split"] == "test"]
    release = metrics["release_model"]
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    if test["label"].nunique() == 2:
        prev = test[test["model"] == next(iter(models))]["label"].mean()
        for kind in ("pr", "roc"):
            fig, ax = plt.subplots(figsize=(6, 5))
            for name in models:
                t = test[test["model"] == name]
                is_rel = name == release
                if models[name]["kind"] == "cnn":
                    color = COLORS["cnn"] if models[name]["variant"] == "global_local" else COLORS["global_only"]
                    lw, alpha = (2.2, 1.0) if is_rel else (0.8, 0.5)
                else:
                    color, lw, alpha = COLORS[name], 1.6, 1.0
                ap = metrics["models"][name]["test"]["average_precision"]
                auc = metrics["models"][name]["test"]["roc_auc"]
                lab = f"{name}{' (release)' if is_rel else ''}: " + (f"AP={ap:.3f}" if kind == "pr" else f"AUC={auc:.3f}")
                if kind == "pr":
                    p, r, _ = precision_recall_curve(t["label"], t["score"])
                    ax.plot(r, p, color=color, lw=lw, alpha=alpha, label=lab)
                else:
                    f, tp, _ = roc_curve(t["label"], t["score"])
                    ax.plot(f, tp, color=color, lw=lw, alpha=alpha, label=lab)
            if kind == "pr":
                ax.axhline(prev, ls="--", color="k", lw=0.8, label=f"chance (prevalence {prev:.3f})")
                ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall, test split (n={len(t)})")
            else:
                ax.plot([0, 1], [0, 1], ls="--", color="k", lw=0.8)
                ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC, test split (n={len(t)})")
            ax.legend(fontsize=7, loc="lower left" if kind == "pr" else "lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{kind}_curve_test.png")
            plt.close(fig)

        for name in [release, "baseline_hgb", "baseline_logreg"]:
            if name not in models:
                continue
            c = metrics["models"][name]["test"]["confusion"]
            mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ax.imshow(mat, cmap="Blues")
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, str(v), ha="center", va="center", color="white" if v > mat.max() / 2 else "black", fontsize=12)
            ax.set_xticks([0, 1], ["pred AFP/NTP", "pred PC"])
            ax.set_yticks([0, 1], ["true AFP/NTP", "true PC"])
            ax.set_title(f"{name}\nthreshold={models[name]['threshold']:.3f} (val max-F1)", fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / f"confusion_{name}.png")
            plt.close(fig)

        if release:
            t = test[test["model"] == release]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bins = np.linspace(0, 1, 41)
            for disp, color in [("PC", "#2a6fdb"), ("AFP", "#d9822b"), ("NTP", "#7a7a7a")]:
                ax.hist(t.loc[t["original_disposition"] == disp, "score"], bins=bins, alpha=0.6, color=color,
                        label=f"{disp} (n={int((t['original_disposition'] == disp).sum())})", log=True)
            ax.axvline(models[release]["threshold"], color="k", ls="--", lw=1, label="threshold")
            ax.set(xlabel="CNN score", ylabel="count (log)", title="Test-split score distribution by catalogue disposition")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "score_hist_test.png")
            plt.close(fig)

    runs = [n for n in models if models[n]["kind"] == "cnn"]
    if runs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name in runs:
            h = pd.read_csv(models[name]["path"].parent / "history.csv")
            color = COLORS["cnn"] if models[name]["variant"] == "global_local" else COLORS["global_only"]
            axes[0].plot(h["epoch"], h["train_loss"], color=color, ls="-", lw=1, alpha=0.8)
            axes[0].plot(h["epoch"], h["val_loss"], color=color, ls="--", lw=1, alpha=0.8)
            axes[1].plot(h["epoch"], h["val_ap"], color=color, lw=1, alpha=0.8, label=name)
        axes[0].set(xlabel="epoch", ylabel="weighted BCE loss", title="Loss (solid: train, dashed: validation)")
        axes[1].set(xlabel="epoch", ylabel="validation AP", title="Validation average precision")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "learning_curves.png")
        plt.close(fig)

    if len(examples):
        idx = {t: i for i, t in enumerate(ds.meta["tce_id"])}
        n = len(examples)
        fig, axes = plt.subplots(n, 2, figsize=(10, 1.55 * n), squeeze=False)
        for r, ex in examples.iterrows():
            i = idx[ex["tce_id"]]
            g, loc = ds.global_view[i], ds.local_view[i]
            axes[r, 0].plot(np.linspace(-0.5, 0.5, g.shape[1]), g[0], ".", ms=1, color="#333")
            axes[r, 1].plot(np.linspace(-1, 1, loc.shape[1]), loc[0], ".-", ms=2, lw=0.5, color="#333")
            axes[r, 0].set_ylabel(f"{ex['tce_id']}\n{ex['original_disposition']} s={ex['score']:.2f}", fontsize=6.5)
            axes[r, 1].set_title(ex["category"], fontsize=7, loc="left", pad=2)
            for a in axes[r]:
                a.tick_params(labelsize=6)
        axes[-1, 0].set_xlabel("global view: phase (fraction of period)", fontsize=8)
        axes[-1, 1].set_xlabel("local view: phase (fraction of local window half-width)", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "error_analysis_examples.png")
        plt.close(fig)


# ---------------------------------------------------------------- report


def _report(metrics, models, ds, paths, exp) -> str:
    m = metrics["models"]
    rel = metrics["release_model"]
    L = [f"# Results: `{exp}` experiment", "",
         f"Generated by `exofinder evaluate` on {metrics['evaluated_utc']}. Every number below is computed from "
         f"`reports/{exp}/predictions.csv`; thresholds come from the validation split only.", ""]
    L += ["## Data", "", "| split | examples | stars | PC (1) | AFP+NTP (0) | prevalence |", "|---|---|---|---|---|---|"]
    for s, v in ds.manifest["by_split"].items():
        L.append(f"| {s} | {v['n_examples']} | {v['n_stars']} | {v['n_pos']} | {v['n_neg']} | {v['prevalence']:.3f} |")
    L += ["", f"Usable examples: {ds.manifest['n_usable_examples']} of {ds.manifest['n_selected_examples']} selected "
          f"({ds.manifest['n_usable_stars']} stars). Exclusions: `{json.dumps(ds.manifest['exclusion_reasons'])}`.", ""]

    L += ["## Test-split metrics (identical examples for every model)", "",
          "| model | AP | ROC-AUC | threshold | precision | recall | F1 | accuracy | TP | FP | TN | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in models:
        t = m[name].get("test")
        if not t:
            continue
        c = t["confusion"]
        star = " **(release)**" if name == rel else ""
        L.append(f"| {name}{star} | {_fmt(t['average_precision'])} | {_fmt(t['roc_auc'])} | {_fmt(t['threshold'])} | "
                 f"{_fmt(t['precision'])} | {_fmt(t['recall'])} | {_fmt(t['f1'])} | {_fmt(t['accuracy'])} | "
                 f"{c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} |")
    if "test" in m.get(next(iter(models)), {}):
        t0 = m[next(iter(models))]["test"]
        L += ["", f"Test prevalence is {t0['prevalence']:.3f} ({t0['n_pos']} PC of {t0['n']}). A classifier that always "
              f"predicts AFP/NTP would score accuracy {1 - t0['prevalence']:.3f} with zero recall, and a random "
              f"ranking has expected AP ≈ {t0['prevalence']:.3f}. That is why AP and the PR curve are the primary metrics.", ""]

    agg = metrics.get("cnn_seed_aggregate_test", {})
    if agg:
        L += ["## Variation across CNN training seeds (fixed split)", "", "| variant | runs | AP mean ± sd | ROC-AUC mean ± sd | F1 mean ± sd |", "|---|---|---|---|---|"]
        for v, a in agg.items():
            def ms(k, a=a):
                s = a[k]["std"]
                return f"{a[k]['mean']:.3f}" + (f" ± {s:.3f}" if s is not None else " (1 run)")
            L.append(f"| {v} | {a['n_runs']} | {ms('average_precision')} | {ms('roc_auc')} | {ms('f1')} |")
        L.append("")
    bs = metrics.get("bootstrap_test")
    if bs:
        L += ["## Uncertainty (star-level bootstrap of the test split, 1000 resamples)", ""]
        for k, v in bs.items():
            if k == "difference":
                L.append(f"- AP difference {v['models'][0]} − {v['models'][1]}: mean {v['ap_diff_mean']:+.3f}, "
                         f"95% interval [{v['ap_diff_ci95'][0]:+.3f}, {v['ap_diff_ci95'][1]:+.3f}]")
            else:
                L.append(f"- {k}: AP 95% interval [{v['ap_ci95'][0]:.3f}, {v['ap_ci95'][1]:.3f}]")
        L.append("")
    if metrics.get("subgroups_test"):
        L += ["## Subgroups (test split)", "", "Tercile edges are computed on the training split. AP is shown only when a "
              f"subgroup has ≥{SUBGROUP_MIN} examples and ≥5 of each class.", ""]
        for name, rows in metrics["subgroups_test"].items():
            L += [f"### {name}", "", "| group | value | n | PC | AFP+NTP | AP | recall | FPR |", "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['group']} | {r['value']} | {r['n']} | {r['n_pos']} | {r['n_neg']} | {_fmt(r['average_precision'])} | "
                         f"{_fmt(r['recall'])} | {_fmt(r['false_positive_rate'])} |")
            L.append("")
    if metrics.get("error_analysis_examples"):
        L += ["## Error analysis examples (release model, test split)", "",
              "Chosen by fixed rules: the 2 highest-scoring true positives; the lowest-scoring correctly rejected AFP and "
              "NTP; the 2 highest-scoring false-positive AFPs and the highest-scoring false-positive NTP; the 3 lowest-scoring "
              "false negatives; the 2 examples closest to the threshold.", "",
              "| category | TCE | disposition | score | period (d) | duration (h) | depth (ppm) | SNR |", "|---|---|---|---|---|---|---|---|"]
        for e in metrics["error_analysis_examples"]:
            L.append(f"| {e['category']} | {e['tce_id']} | {e['original_disposition']} | {e['score']:.3f} | {e['period_days']:.3f} | "
                     f"{e['duration_hours']:.2f} | {e['depth_ppm']:.0f} | {e['snr_est']:.1f} |")
        L += ["", "![examples](figures/error_analysis_examples.png)", ""]
    if metrics.get("demo_holdout"):
        L += ["## Demonstration hold-out (Kepler-90, never used for training, validation or the benchmark)", "",
              "| TCE | model | catalogue disposition | score | threshold | above threshold |", "|---|---|---|---|---|---|"]
        for d in metrics["demo_holdout"]:
            L.append(f"| {d['tce_id']} | {d['model']} | {d['original_disposition']} | {d['score']:.3f} | {d['threshold']:.3f} | {bool(d['predicted'])} |")
        L.append("")
    L += ["## Figures", ""]
    for f in ["pr_curve_test", "roc_curve_test", f"confusion_{rel}", "confusion_baseline_hgb", "confusion_baseline_logreg", "score_hist_test", "learning_curves"]:
        if (paths.report_dir(exp) / "figures" / f"{f}.png").exists():
            L.append(f"![{f}](figures/{f}.png)")
    return "\n".join(L) + "\n"
