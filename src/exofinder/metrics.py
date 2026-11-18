"""Evaluation metrics with an explicit, validation-selected decision threshold.

Average precision (AP) follows scikit-learn's definition
    AP = sum_n (R_n - R_{n-1}) * P_n
where P_n and R_n are precision and recall at the n-th distinct score
threshold (no interpolation). A random classifier has AP equal to the positive
prevalence, which is the reference point for imbalanced data.

The decision rule is ``predict positive  <=>  score >= threshold``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def select_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data.

    Candidates are the distinct scores; ties in F1 resolve to the highest
    threshold (fewest positive predictions).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        raise ValueError("Cannot select an F1 threshold without positive examples")
    prec, rec, thr = precision_recall_curve(y, s)
    p, r = prec[:-1], rec[:-1]
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    best = np.flatnonzero(f1 == f1.max())
    return float(thr[best[-1]])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    both = n_pos > 0 and n_neg > 0
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / len(y) if len(y) else float("nan"),
        "average_precision": float(average_precision_score(y, s)) if n_pos else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "false_positive_rate": fp / n_neg if n_neg else float("nan"),
        "brier": float(np.mean((s - y) ** 2)) if len(y) else float("nan"),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
