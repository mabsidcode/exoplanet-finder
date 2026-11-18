"""Star-grouped train / validation / test splits.

The unit of grouping is the host star (KIC ID). Every TCE of a star, and every
derived or augmented view, inherits the star's split, so no star contributes
to more than one split.

Algorithm (deterministic for a given seed and catalogue snapshot):
1. Stars listed in ``demo_holdout_kepids`` go to the ``demo_holdout`` split.
2. Remaining stars are stratified by whether they host at least one positive
   (PC) TCE, so that both strata are spread across splits.
3. Within each stratum, stars are sorted by KIC ID, shuffled with
   ``numpy.random.default_rng(seed)`` and cut at the configured fractions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
DEMO_SPLIT = "demo_holdout"


def make_star_splits(labelled: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    s = cfg["split"]
    fractions = s["fractions"]
    if abs(sum(fractions[k] for k in SPLITS) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    holdout = {int(k) for k in cfg["catalog"].get("demo_holdout_kepids", [])}

    stars = (
        labelled.groupby("kepid")["label"].max().rename("has_positive").reset_index().sort_values("kepid")
    )
    rng = np.random.default_rng(int(s["seed"]))
    rows = []
    for kepid in stars.loc[stars["kepid"].isin(holdout), "kepid"]:
        rows.append((int(kepid), DEMO_SPLIT, "holdout"))
    rest = stars[~stars["kepid"].isin(holdout)]
    for stratum, group in rest.groupby("has_positive", sort=True):
        ids = group["kepid"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n = len(ids)
        n_train = int(round(fractions["train"] * n))
        n_val = int(round(fractions["val"] * n))
        for i, kepid in enumerate(ids):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            rows.append((int(kepid), split, f"has_positive={int(stratum)}"))
    manifest = pd.DataFrame(rows, columns=["kepid", "split", "stratum"]).sort_values("kepid")
    return manifest.reset_index(drop=True)


def attach_splits(examples: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = examples.merge(manifest[["kepid", "split"]], on="kepid", how="left", validate="many_to_one")
    if out["split"].isna().any():
        raise ValueError("Some examples have no split assignment")
    return out


def check_no_star_overlap(examples: pd.DataFrame) -> None:
    """Raise if any host star appears in more than one split."""
    per_star = examples.groupby("kepid")["split"].nunique()
    bad = per_star[per_star > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} stars appear in multiple splits, e.g. {bad.index[:5].tolist()}")


def split_summary(examples: pd.DataFrame) -> pd.DataFrame:
    g = examples.groupby("split")
    return pd.DataFrame(
        {
            "n_examples": g.size(),
            "n_stars": g["kepid"].nunique(),
            "n_pos": g["label"].sum(),
            "n_neg": g.size() - g["label"].sum(),
            "prevalence": g["label"].mean().round(4),
        }
    )
