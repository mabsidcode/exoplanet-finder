"""CNN training with early stopping on validation average precision.

* Loss: ``BCEWithLogitsLoss`` with ``pos_weight = n_neg / n_pos`` computed on
  the training split only.
* Optimiser: AdamW.
* Augmentation (training split only): with probability 0.5 an example is
  time-reversed, i.e. both views and their coverage masks are flipped along
  the phase axis. Both views have an odd number of bins centred on the
  transit, so the transit stays in the centre bin and the two views stay
  aligned. A transit's shape is symmetric to first order, so the label is
  unchanged.
* Selection: the epoch with the highest validation AP is kept (``best.pt``).
  Training stops after ``patience`` epochs without improvement.
* Resumable: ``last.pt`` holds model, optimiser, history and early-stopping
  state. Shuffling and augmentation use a generator seeded by (seed, epoch),
  so a resumed run continues exactly as an uninterrupted one would on CPU.
* After training, the decision threshold is chosen on validation (max F1) and
  stored in ``best.pt``.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .config import Paths
from .dataset import ViewDataset
from .io_utils import atomic_path, atomic_write_json, utc_now_iso
from .metrics import select_threshold_max_f1
from .model import DualBranchCNN, ModelConfig, count_parameters, select_device
from .preprocess import PREPROCESS_VERSION, PreprocessConfig

CHECKPOINT_FORMAT = 1
LABEL_MAPPING = {
    "positive_label": 1,
    "positive_dispositions": ["PC"],
    "negative_dispositions": ["AFP", "NTP"],
    "excluded_dispositions": ["UNK"],
    "catalog": "Kepler Q1-Q17 DR24 TCE av_training_set",
}


def environment_info() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def run_name(seed: int, use_local: bool) -> str:
    return f"cnn_{'global_local' if use_local else 'global_only'}_seed{seed}"


def _save(obj: dict[str, Any], path: Path) -> None:
    with atomic_path(path, suffix=".pt") as tmp:
        torch.save(obj, tmp)


@torch.no_grad()
def predict_scores(model: nn.Module, g: np.ndarray, loc: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(g), batch):
        gb = torch.from_numpy(np.ascontiguousarray(g[i : i + batch])).to(device)
        lb = torch.from_numpy(np.ascontiguousarray(loc[i : i + batch])).to(device)
        out.append(torch.sigmoid(model(gb, lb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_one(cfg: dict[str, Any], paths: Paths, seed: int, use_local: bool = True, resume: bool = True, log=print) -> dict[str, Any]:
    exp = cfg["experiment_name"]
    tcfg = cfg["train"]
    ds = ViewDataset(paths.dataset_dir(exp))
    tr, va = ds.split_index("train"), ds.split_index("val")
    y = ds.labels
    n_pos, n_neg = int(y[tr].sum()), int(len(tr) - y[tr].sum())
    if n_pos == 0 or len(va) == 0:
        raise RuntimeError("Training needs positives in train and a non-empty validation split")

    mcfg = ModelConfig.from_dict({**cfg["model"], "use_local": use_local})
    device = select_device(tcfg.get("device", "auto"))
    torch.manual_seed(seed)
    model = DualBranchCNN(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    pos_weight = torch.tensor(n_neg / n_pos, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_dir = paths.model_dir(exp) / run_name(seed, use_local)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, history, best_ap, bad = 0, [], -1.0, 0
    if resume and (run_dir / "last.pt").exists():
        st = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        start_epoch, history, best_ap, bad = st["epoch"] + 1, st["history"], st["best_ap"], st["bad_epochs"]
        log(f"Resumed {run_dir.name} at epoch {start_epoch}")
        if st.get("finished"):
            log(f"{run_dir.name} already finished")
            return st["summary"]

    pcfg = PreprocessConfig.from_dict(cfg["preprocess"])
    meta_base = {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": mcfg.to_dict(),
        "preprocess": pcfg.to_dict(),
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_fingerprint": pcfg.fingerprint(),
        "label_mapping": LABEL_MAPPING,
        "score_semantics": "sigmoid(logit); a ranking score, not a calibrated probability",
    }
    g_all = torch.from_numpy(ds.global_view)
    l_all = torch.from_numpy(ds.local_view)
    y_all = torch.from_numpy(y)
    bs = int(tcfg["batch_size"])
    t_start = time.time()
    log(f"{run_dir.name}: device={device} params={count_parameters(model):,} train={len(tr)} (pos {n_pos}) val={len(va)}")

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        model.train()
        gen = torch.Generator().manual_seed(seed * 100_003 + epoch)
        perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=gen)]
        flips = torch.rand(len(perm), generator=gen) < 0.5 if tcfg.get("augment_time_reversal", True) else torch.zeros(len(perm), dtype=torch.bool)
        t_ep, losses = time.time(), []
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            gb, lb, yb = g_all[idx].clone(), l_all[idx].clone(), y_all[idx]
            f = flips[i : i + bs]
            if f.any():
                gb[f] = torch.flip(gb[f], dims=[-1])
                lb[f] = torch.flip(lb[f], dims=[-1])
            gb, lb, yb = gb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(gb, lb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item() * len(idx))
        train_loss = float(np.sum(losses) / len(perm))

        val_scores = predict_scores(model, ds.global_view[va], ds.local_view[va], device)
        with torch.no_grad():
            logits = torch.logit(torch.from_numpy(val_scores).clamp(1e-7, 1 - 1e-7))
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(y[va]), pos_weight=pos_weight.cpu()))
        val_ap = float(average_precision_score(y[va], val_scores))
        val_auc = float(roc_auc_score(y[va], val_scores)) if 0 < y[va].sum() < len(va) else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_ap": val_ap,
                        "val_roc_auc": val_auc, "seconds": round(time.time() - t_ep, 2)})
        improved = val_ap > best_ap
        if improved:
            best_ap, bad = val_ap, 0
            _save({**meta_base, "state_dict": model.state_dict(), "best_epoch": epoch, "best_val_ap": val_ap}, run_dir / "best.pt")
        else:
            bad += 1
        log(f"  epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_AP={val_ap:.4f} "
            f"val_AUC={val_auc:.4f} {'*' if improved else ''} ({history[-1]['seconds']}s)")
        _save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
               "history": history, "best_ap": best_ap, "bad_epochs": bad, "finished": False}, run_dir / "last.pt")
        if bad >= int(tcfg["patience"]):
            log(f"  early stopping at epoch {epoch} (best val AP {best_ap:.4f})")
            break

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    val_scores = predict_scores(model, ds.global_view[va], ds.local_view[va], device)
    threshold = select_threshold_max_f1(y[va], val_scores)
    summary = {
        "run": run_dir.name,
        "seed": seed,
        "use_local": use_local,
        "device": str(device),
        "n_parameters": count_parameters(model),
        "n_train": int(len(tr)),
        "n_train_pos": n_pos,
        "n_val": int(len(va)),
        "pos_weight": n_neg / n_pos,
        "best_epoch": int(best["best_epoch"]),
        "best_val_ap": float(best["best_val_ap"]),
        "epochs_run": len(history),
        "decision_threshold": threshold,
        "threshold_rule": "maximise F1 on the validation split",
        "train_seconds": round(time.time() - t_start, 1),
        "finished_utc": utc_now_iso(),
        "dataset_manifest": {k: ds.manifest[k] for k in ("experiment", "views_sha256", "split_manifest_sha256", "catalog_snapshot_sha256")},
        "train_config": tcfg,
        "environment": environment_info(),
    }
    best.update({"decision_threshold": threshold, "threshold_rule": summary["threshold_rule"], "experiment": summary})
    _save(best, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    atomic_write_json(run_dir / "summary.json", summary)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    last.update(finished=True, summary=summary)
    _save(last, run_dir / "last.pt")
    log(f"{run_dir.name}: best epoch {summary['best_epoch']} val AP {summary['best_val_ap']:.4f} threshold {threshold:.4f}")
    return summary


def train_all(cfg: dict[str, Any], paths: Paths, seeds: list[int] | None = None, use_local: bool = True, resume: bool = True) -> list[dict[str, Any]]:
    seeds = seeds if seeds else cfg["train"]["seeds"]
    return [train_one(cfg, paths, s, use_local=use_local, resume=resume) for s in seeds]
