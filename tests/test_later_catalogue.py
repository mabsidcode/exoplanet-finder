"""Later-catalogue (DR25 KOI / NASA Archive) context on the bundled example cards.

Checks that the bundled context file is exactly what the documented rule produces from the cached catalogue rows
(offline), that every match is re-derivable by period AND epoch rather than by star ID, that the two sources are
never mixed up, that DR24 labels are untouched, and that the app cards show it.
"""

import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from .conftest import EXAMPLES, RELEASE, ROOT

CHECK = ROOT / "reports" / "full" / "dr25_koi_check"
CONTEXT = EXAMPLES / "later_catalogue.json"
UNRESOLVED = "Later-catalogue match unresolved."

pytestmark = pytest.mark.skipif(not CONTEXT.exists(), reason="later-catalogue context not generated")


@pytest.fixture(scope="module")
def ctx():
    return json.loads(CONTEXT.read_text())


@pytest.fixture(scope="module")
def bundled():
    return {e["id"]: e for e in json.loads((EXAMPLES / "examples.json").read_text())["examples"]}


@pytest.fixture(scope="module")
def check_module():
    spec = importlib.util.spec_from_file_location("check_dr25_koi", ROOT / "scripts" / "check_dr25_koi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_context_regenerates_exactly_from_cached_evidence(check_module, ctx, bundled):
    m = check_module
    metrics = json.loads((ROOT / "reports" / "full" / "metrics.json").read_text())
    snap = pd.read_csv(ROOT / "data" / "catalog" / "dr24_tce_snapshot.csv")
    snap["tce_id"] = snap["kepid"].map("{:09d}".format) + "_" + snap["tce_plnt_num"].map("{:02d}".format)
    rows = m.assess(m.targets(metrics, list(bundled.values()), snap.set_index("tce_id")),
                    pd.read_csv(CHECK / "dr25_koi.csv"), pd.read_csv(CHECK / "dr25_tce.csv"),
                    pd.read_csv(CHECK / "keplernames.csv"), pd.read_csv(CHECK / "ps.csv"))
    regenerated = json.loads(json.dumps(m.app_context(rows, json.loads((CHECK / "provenance.json").read_text())),
                                        sort_keys=True, default=str))
    assert regenerated == ctx


def test_cached_evidence_matches_recorded_sha256():
    import hashlib

    prov = json.loads((CHECK / "provenance.json").read_text())
    for name, p in prov.items():
        assert hashlib.sha256((CHECK / name).read_bytes()).hexdigest() == p["sha256"], name


def test_every_bundled_example_has_context_and_dr24_labels_are_unchanged(ctx, bundled):
    snap = pd.read_csv(ROOT / "data" / "catalog" / "dr24_tce_snapshot.csv")
    assert set(ctx["examples"]) == set(bundled)
    for eid, e in ctx["examples"].items():
        ex = bundled[eid]
        if ex.get("tce_id"):
            kepid, n = (int(x) for x in ex["tce_id"].split("_"))
            label = snap[(snap.kepid == kepid) & (snap.tce_plnt_num == n)]["av_training_set"].item()
            assert e["dr24_label"] == label == ex["catalogue_disposition"]
            assert ex["training_label"] == (1 if label == "PC" else 0)
        else:
            assert e["dr24_label"] is None


def _sep_hours(P, T0, P2, T2):
    n = np.arange(np.ceil((120 - T0) / P), np.floor((1600 - T0) / P) + 1)
    t = T0 + n * P
    return float(np.abs(t - (T2 + np.round((t - T2) / P2) * P2)).max() * 24)


def test_matches_use_period_and_epoch_not_star_id(ctx, bundled):
    """Independent re-derivation: the matched KOI is the ONLY KOI on the star with the same period and transit times."""
    koi = pd.read_csv(CHECK / "dr25_koi.csv")
    for eid, e in ctx["examples"].items():
        ex = bundled[eid]
        P, T0, D = ex["period_days"], ex["epoch_bkjd"], ex["duration_hours"]
        ok = [k["kepoi_name"] for _, k in koi[koi.kepid == ex["kepid"]].iterrows()
              if abs(k["koi_period"] / P - 1) <= 1e-3 and _sep_hours(P, T0, k["koi_period"], k["koi_time0bk"]) <= D / 2]
        if e["match_status"] == "matched":
            assert ok == [e["dr25"]["kepoi_name"]], eid
            row = koi[koi.kepoi_name == e["dr25"]["kepoi_name"]].iloc[0]
            assert e["dr25"]["disposition"] == row["koi_pdisposition"]
        else:
            assert ok == [] and e["dr25"] is None and e["display"] == UNRESOLVED, eid
            assert e["unresolved_reason"], eid


def test_archive_status_is_verified_and_attributed_separately(ctx, bundled):
    names = pd.read_csv(CHECK / "keplernames.csv")
    ps = pd.read_csv(CHECK / "ps.csv")
    for eid, e in ctx["examples"].items():
        a = e["archive"]
        if e["match_status"] == "unresolved" and bundled[eid].get("tce_id"):
            assert a is None, f"{eid}: no Archive status may be attached without a KOI match"
        if a is None:
            continue
        key = (names.kepid == bundled[eid]["kepid"]) & (
            names.koi_name == e["dr25"]["kepoi_name"] if e["dr25"] else names.kepler_name == a["kepler_name"])
        assert names[key]["pl_name"].tolist() == [a["pl_name"]]
        row = ps[(ps.pl_name == a["pl_name"]) & (ps.default_flag == 1)].iloc[0]
        assert row["soltype"] == "Published Confirmed" and a["status"] == "Confirmed planet"
        assert "Planetary Systems (ps)" in a["source"]["table"] and re.match(r"\d{4}-\d\d-\d\d", a["source"]["retrieved_utc"])
        if e["dr25"]:
            assert "q1_q17_dr25_koi" in e["dr25"]["source"]["table"]
            assert "koi_pdisposition" in e["dr25"]["source"]["column"]
            assert e["dr25"]["disposition"] in ("CANDIDATE", "FALSE POSITIVE")  # DR25 never says CONFIRMED here


def test_expected_findings(ctx):
    ex = ctx["examples"]
    assert ex["010587105_03"]["dr24_label"] == "AFP" and ex["010587105_03"]["dr25"]["disposition"] == "CANDIDATE"
    assert ex["010587105_03"]["archive"]["kepler_name"] == "Kepler-529 d"
    assert ex["008822421_01"]["dr25"]["disposition"] == "FALSE POSITIVE" and ex["008822421_01"]["archive"] is None
    assert ex["011442793_01"]["dr25"]["kepoi_name"] == "K00351.01" and ex["011442793_01"]["archive"]["pl_name"] == "KOI-351 h"
    assert {k for k, v in ex.items() if v["disagreement"]} == {"010587105_03", "008822421_01", "011442793_01"}
    assert {k for k, v in ex.items() if v["match_status"] == "unresolved"} == {"002445134_02", "008231830_01", "kepler-90i"}


@pytest.mark.app
@pytest.mark.skipif(not (RELEASE / "cnn.pt").exists(), reason="release model not present")
def test_cards_show_later_catalogue_context(ctx, bundled):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    by_title = {e["title"]: e for e in bundled.values()}
    for title in at.selectbox[0].options:
        at.selectbox[0].select(title).run()
        assert not at.exception
        eid = by_title[title]["id"]
        e = ctx["examples"][eid]
        cards = [m.value for m in at.markdown if "Later catalogues" in m.value]
        assert len(cards) == 1, eid
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cards[0]))
        assert f"Original DR24 training label: {e['dr24_label'] or 'none'}" in text
        if e["dr25"]:
            assert f"DR25 KOI {e['dr25']['kepoi_name']} : {e['dr25']['disposition']}" in text
            assert "q1_q17_dr25_koi" in text and f"retrieved {e['dr25']['source']['retrieved_utc'][:10]}" in text
            assert UNRESOLVED not in text
        else:
            assert UNRESOLVED in text
        if e["archive"]:
            assert f"NASA Exoplanet Archive status: Confirmed planet , {e['archive']['kepler_name']}" in text
            assert "Planetary Systems (ps)" in text
        else:
            assert "Confirmed planet" not in text
        assert ("Labels disagree" in text) == bool(e["disagreement"])
