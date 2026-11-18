from pathlib import Path

import pandas as pd
import pytest

from exofinder.catalog import SNAPSHOT_COLUMNS, label_examples, labelled_examples, tce_id
from exofinder.config import Paths, load_config
from exofinder.splits import attach_splits, check_no_star_overlap, make_star_splits

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs" / "quickstart.yaml")


def _toy_catalog():
    return pd.DataFrame(
        {
            "kepid": [1, 1, 2, 3, 3, 4, 5, 11442793],
            "tce_plnt_num": [1, 2, 1, 1, 2, 1, 1, 1],
            "tce_period": [1.0] * 8, "tce_time0bk": [130.0] * 8, "tce_duration": [2.0] * 8,
            "tce_depth": [100.0] * 8, "tce_model_snr": [10.0] * 8, "tce_num_transits": [5] * 8,
            "tce_rogue_flag": [0] * 8,
            "av_training_set": ["PC", "AFP", "NTP", "UNK", None, "PC", "AFP", "PC"],
        }
    )


def test_label_mapping_and_exclusions():
    ex = label_examples(_toy_catalog(), CFG, {})
    m = dict(zip(ex["original_disposition"], ex["label"], strict=False))
    assert m["PC"] == 1 and m["AFP"] == 0 and m["NTP"] == 0
    assert m["UNK"] == -1 and m["NULL"] == -1  # never promoted
    assert set(ex.loc[ex["label"] == -1, "exclusion_reason"]) == {"disposition_UNK", "disposition_NULL"}
    lab = labelled_examples(ex)
    assert set(lab["original_disposition"]) == {"PC", "AFP", "NTP"}
    assert ex["label_mapping"].iloc[0] == "PC=1; AFP+NTP=0"


def test_tce_id_stable():
    assert tce_id(11442793, 1) == "011442793_01"


def test_no_vetting_outputs_in_snapshot_columns():
    forbidden = {"av_pp_pc", "av_pp_afp", "av_pp_ntp", "av_pred_class", "av_vf_pc", "av_vf_afp", "av_vf_ntp"}
    assert not forbidden & set(SNAPSHOT_COLUMNS)


def test_star_split_no_overlap_and_holdout():
    rng_cat = pd.DataFrame(
        {"kepid": [k for k in range(200) for _ in range(1 + k % 3)], "label": [int(k % 4 == 0) for k in range(200) for _ in range(1 + k % 3)]}
    )
    rng_cat = pd.concat([rng_cat, pd.DataFrame({"kepid": [11442793], "label": [1]})])
    manifest = make_star_splits(rng_cat, CFG)
    ex = attach_splits(rng_cat, manifest)
    check_no_star_overlap(ex)
    assert set(ex.loc[ex["kepid"] == 11442793, "split"]) == {"demo_holdout"}
    frac = manifest[manifest["split"] != "demo_holdout"]["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    assert manifest.equals(make_star_splits(rng_cat, CFG))  # deterministic


def test_overlap_detector_catches_leak():
    bad = pd.DataFrame({"kepid": [1, 1], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        check_no_star_overlap(bad)


def test_committed_split_manifest_has_no_overlap():
    p = Paths.from_config(CFG)
    if not p.split_manifest.exists():
        pytest.skip("split manifest not generated")
    from exofinder.dataset import load_catalog_examples

    ex = load_catalog_examples(CFG, p)
    check_no_star_overlap(ex)
    assert ex.loc[ex["kepid"] == 11442793, "split"].eq("demo_holdout").all()
    assert len(ex) == ex["tce_id"].nunique()
