"""Check the error-analysis explanations and the bundled demo examples against later Kepler catalogues.

This is later-catalogue *context* only. It never changes the DR24 labels, splits, models or recorded metrics.

Inputs
------
* the error-analysis examples recorded in ``reports/full/metrics.json`` (chosen by fixed rules at evaluation time)
  and the bundled demo examples in ``app/examples/examples.json``
* DR24 ephemerides from the committed snapshot ``data/catalog/dr24_tce_snapshot.csv`` (Kepler-90 i, which is not a
  DR24 TCE, uses its published ephemeris from ``examples.json``)
* NASA Exoplanet Archive TAP tables ``q1_q17_dr25_koi`` and ``q1_q17_dr25_tce`` for the same stars, the official
  Kepler cross-identification table ``keplernames`` (KIC + KOI -> Archive planet name), and the Planetary Systems
  table ``ps`` (default parameter sets) for those planet names. All are cached in ``reports/full/dr25_koi_check/`` with query, URL, retrieval time and SHA-256 (``--fetch`` refreshes them)

Outputs
-------
* ``reports/full/dr25_koi_check/results.json`` and ``RESULTS.md``: evidence and assessments
* ``app/examples/later_catalogue.json``: the context shown on the demo's example cards, available offline

Matching rule (uses only periods and epochs, never flags; a star-ID match alone is never enough)
------------------------------------------------------------------------------------------------
Written after a first look at the queried catalogue rows, so it is not a blind pre-registration; it is
deliberately conservative (anything ambiguous stays unresolved).
For a DR24 TCE with period P, epoch T0 and duration D, and each DR25 KOI (or TCE) on the same KIC star:

1. Period: ``r = P_dr25 / P``. Same period if ``|r - 1| <= 1e-3``. If ``r`` or ``1/r`` is within 0.1% of an integer
   2..5 the object is recorded as a *period alias*, which is never a match.
2. Epoch: for same-period objects, every DR24-predicted transit in the Kepler baseline (BKJD 120-1600) is compared
   with the nearest DR25-predicted transit. ``max_sep_h`` is the largest separation; ``phase_offset`` is the epoch
   offset as a fraction of the period (0.5 = the other object sits at the opposite phase, e.g. a secondary eclipse).
3. Compatible if ``max_sep_h <= 0.5 * D`` (transit centres stay within half a DR24 duration for the whole mission).
4. Exactly one compatible KOI = **matched**. No compatible KOI = **unresolved (no matching KOI)**. Several =
   **unresolved (ambiguous)**. A same-period KOI whose epoch is incompatible is reported as such; if it is within
   3 combined (propagated) standard errors the case is **unresolved (low-precision ephemeris)**.

Archive status (e.g. "confirmed planet") is shown only when verified independently of the KOI table's own
``koi_disposition`` column (which is recorded but never displayed as verified status):

a. ``keplernames`` has a row with the same KIC ID and the matched KOI's identifier (for Kepler-90 i, which has no
   KOI, the same KIC ID and its Kepler name), giving the Archive planet name ``pl_name``;
b. the ``ps`` default parameter set for that ``pl_name`` has ``soltype`` "Published Confirmed";
c. its period matches the KOI's (Kepler-90 i: the example's) within 0.1%, and, when that row gives a transit
   midpoint, its transit times stay within half the KOI duration over the mission. When the default row gives no
   midpoint, this is stated with the status.

Amendment, made after the first ``ps`` query and before writing the demo cards: the first version looked planets up
by ``kepler_name`` and required a midpoint. That missed Kepler-90 h (Archive name "KOI-351 h") and Kepler-529 d
(its default row, Valizadegan et al. 2022, gives a period but no midpoint). The cross-ID table replaces the name
lookup; the midpoint requirement became "check it when given, otherwise say so".

Flag meanings: koi_fpflag_* and koi_pdisposition from the NASA Exoplanet Archive column definitions
(https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html); minor flags in koi_comment from the
DR25 Robovetter source (https://github.com/nasa/kepler-robovetter, Kepler-RoboVetter-DR25.cpp).
``koi_pdisposition`` is the uniform, automated DR25 disposition (table last updated 2017-08-31);
``koi_disposition`` is the Exoplanet Archive's disposition at query time and may include later confirmations.

Usage:  python scripts/check_dr25_koi.py [--fetch]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "full" / "dr25_koi_check"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
BASELINE_BKJD = (120.0, 1600.0)
PERIOD_TOL = 1e-3
KOI_COLS = ("kepid,kepoi_name,kepler_name,koi_disposition,koi_pdisposition,koi_score,koi_fpflag_nt,koi_fpflag_ss,"
            "koi_fpflag_co,koi_fpflag_ec,koi_comment,koi_period,koi_period_err1,koi_time0bk,koi_time0bk_err1,"
            "koi_duration,koi_depth,koi_tce_plnt_num,koi_tce_delivname")
TCE_COLS = "kepid,tce_plnt_num,tce_period,tce_period_err,tce_time0bk,tce_time0bk_err,tce_duration,tce_depth,tce_model_snr"
KEPLERNAMES_COLS = "kepid,koi_name,kepler_name,pl_name"
NON_TCE_PLANETS = {"kepler-90i": "Kepler-90 i"}  # bundled examples that are not DR24 TCEs
PS_COLS = ("pl_name,hostname,default_flag,soltype,pl_orbper,pl_orbpererr1,pl_tranmid,pl_tranmiderr1,disc_year,"
           "disc_refname,pl_controv_flag,rowupdate")
BKJD_OFFSET = 2454833.0
APP_CONTEXT = ROOT / "app" / "examples" / "later_catalogue.json"
UNRESOLVED_TEXT = "Later-catalogue match unresolved."
FLAG_WORDS = {"koi_fpflag_nt": "not transit-like", "koi_fpflag_ss": "stellar eclipse",
              "koi_fpflag_co": "centroid offset: the signal comes from a nearby star",
              "koi_fpflag_ec": "ephemeris match: flux contamination from another source"}
FLAG_MEANINGS = {
    "koi_fpflag_nt": "Not Transit-Like: light curve not consistent with a transiting planet",
    "koi_fpflag_ss": "Stellar Eclipse: significant secondary event, transit shape or out-of-eclipse variability",
    "koi_fpflag_co": "Centroid Offset: signal source is a nearby star, from the image centroid",
    "koi_fpflag_ec": "Ephemeris Match: same period and epoch as another object; flux contamination",
}
MINOR_FLAG_MEANINGS = {  # from comments in Kepler-RoboVetter-DR25.cpp
    "SWEET_NTL": "sinusoidal signal at the TCE period larger than the transit depth (not transit-like)",
    "SWEET_EB": "significant sinusoidal out-of-eclipse variability smaller than the depth (stellar eclipse)",
    "DEEP_V_SHAPED": "too deep and too V-shaped for a planet (stellar eclipse)",
    "HAS_SEC_TCE": "a subsequent TCE on the same star at the same period was identified as this signal's secondary",
    "MOD_SEC_ALT": "significant secondary event in the model-shift test (alternate detrending)",
    "MOD_SEC_DV": "significant secondary event in the model-shift test (DV detrending)",
}

# Assessment of each explanation in docs/RESULTS_REPORT.md section 7 against the evidence printed by this script.
# Written after reading the evidence; "unresolved" rows are never used as support.
VERDICTS = {
    "011342416_01": ("no explanation proposed", "Correct PC; DR25 candidate, later confirmed (Kepler-1209 b)."),
    "007898352_02": ("no explanation proposed", "Correct PC; DR25 candidate, later confirmed (Kepler-302 b)."),
    "002445134_02": (
        "unresolved; indirect evidence consistent",
        "Hypothesis: secondary eclipse of an eclipsing binary. No DR25 KOI matches this TCE. The star's only KOI, "
        "K05983.01 (DR25 TCE 1), has the same period at phase offset +0.499 and is a FALSE POSITIVE with the Stellar "
        "Eclipse flag and HAS_SEC_TCE. The star's only subsequent DR25 TCE has this TCE's ephemeris, so it is the "
        "TCE the Robovetter identified as that secondary.",
    ),
    "004678875_02": (
        "unresolved; evidence suggests a different explanation",
        "Hypothesis: stellar variability. No DR25 KOI or TCE near 0.626 d. The star's KOI K03664.01 has exactly 3x "
        "this period (1.879 d) and is a FALSE POSITIVE eclipsing binary (Stellar Eclipse flag, SWEET_EB, "
        "HAS_SEC_TCE). The signal is more plausibly a 1/3-period harmonic of that binary than independent "
        "stellar variability; not verified.",
    ),
    "010587105_03": (
        "does not support",
        "Hypothesis: AFP label rests on centroid or contamination evidence. The matched KOI K00339.03 is a DR25 "
        "CANDIDATE with no false-positive flags and is now archive-CONFIRMED (Kepler-529 d). The later catalogue "
        "disagrees with the DR24 AFP label, so the high CNN score agrees with it. Why DR24 labelled it AFP was not "
        "investigated.",
    ),
    "010470206_01": (
        "supports",
        "Hypothesis: AFP label rests on centroid or contamination evidence. The matched KOI K00335.01 is a DR25 "
        "FALSE POSITIVE with the Centroid Offset flag (signal from a nearby star), which the light curve alone "
        "does not show. DR25 disposition score 0.394 (for false positives, higher means less confidence).",
    ),
    "008231830_01": (
        "unresolved; not testable with KOI flags",
        "Hypothesis: mostly noise. No DR25 KOI, and no DR25 TCE at all, on this star. Not being re-detected in "
        "DR25 is consistent with a spurious signal but is not vetting evidence.",
    ),
    "008822421_01": (
        "partly supports",
        "Hypothesis: residual stellar variability that detrending did not remove. The matched KOI K06188.01 is a "
        "DR25 FALSE POSITIVE, Not Transit-Like, SWEET_NTL: a sinusoid at the TCE period larger than the dip. DR25 "
        "agrees that variability dominates, but calls the signal itself not transit-like, so under the later "
        "catalogue the low score is agreement rather than a detrending failure.",
    ),
    "009047416_01": (
        "unresolved; not testable with KOI flags",
        "Hypothesis: residual variability and low SNR. The same-period KOI K05605.01 (DR25 CANDIDATE, no FP flags) "
        "agrees in epoch only within its large uncertainties (max separation 10.5 h), so it is not counted as a "
        "match. KOI flags do not test detrending in any case.",
    ),
    "005785279_01": (
        "unresolved; not testable with KOI flags",
        "Hypothesis: low SNR. The same-period KOI K04259.01 (DR25 CANDIDATE, no FP flags) drifts 1.9 h from this "
        "TCE's ephemeris over the mission, more than half the 1.37 h duration, so it is not counted as a match.",
    ),
    "006611330_01": (
        "no explanation proposed; later-catalogue context",
        "Described only (deep, short PC with an asymmetric local view). The matched KOI K01691.01 is a DR25 FALSE "
        "POSITIVE with the Stellar Eclipse flag (DEEP_V_SHAPED): the later catalogue disagrees with the DR24 PC label.",
    ),
    "010352333_01": ("no explanation proposed", "Borderline PC; DR25 candidate with no FP flags."),
    "011442793_01": ("no explanation proposed", "Bundled example (Kepler-90 h, demo hold-out)."),
    "kepler-90i": ("no explanation proposed", "Bundled example (Kepler-90 i, not a DR24 TCE)."),
}


def _tap(query: str) -> bytes:
    r = requests.get(TAP, params={"query": query, "format": "csv"}, timeout=60,
                     headers={"User-Agent": "exofinder-catalogue-check"})
    r.raise_for_status()
    return r.content


def fetch(kepids: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = ",".join(str(k) for k in sorted(kepids))
    queries = {
        "dr25_koi.csv": f"select {KOI_COLS} from q1_q17_dr25_koi where kepid in ({ids}) order by kepid,kepoi_name",
        "dr25_tce.csv": f"select {TCE_COLS} from q1_q17_dr25_tce where kepid in ({ids}) order by kepid,tce_plnt_num",
    }
    prov = {}

    def get(name: str, q: str) -> None:
        data = _tap(q)
        (OUT / name).write_bytes(data)
        prov[name] = {"tap_url": TAP, "query": q, "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                      "sha256": hashlib.sha256(data).hexdigest(), "n_rows": len(pd.read_csv(io.BytesIO(data)))}

    for name, q in queries.items():
        get(name, q)
    get("keplernames.csv", f"select {KEPLERNAMES_COLS} from keplernames where kepid in ({ids}) order by kepid,pl_name")
    names = sorted(set(pd.read_csv(OUT / "keplernames.csv")["pl_name"].dropna()))
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    get("ps.csv", f"select {PS_COLS} from ps where default_flag=1 and pl_name in ({quoted}) order by pl_name")
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")


def _err(x) -> float:
    x = float(x) if pd.notna(x) else np.nan
    return x if np.isfinite(x) and x > 0 else 0.0  # archive uses -1 / blank for unknown


def compare(P, T0, D_h, sP, sT, P2, T2, sP2, sT2) -> dict:
    """Period ratio, alias, and transit-time separation of a DR25 ephemeris against a DR24 one."""
    r = P2 / P
    alias = None
    for k in range(2, 6):
        if abs(r - k) <= PERIOD_TOL * k:
            alias = f"{k}x"
        elif abs(1 / r - k) <= PERIOD_TOL * k:
            alias = f"1/{k}x"
    out = {"period_ratio": r, "alias": alias, "same_period": abs(r - 1) <= PERIOD_TOL}
    if not out["same_period"]:
        return out
    n = np.arange(np.ceil((BASELINE_BKJD[0] - T0) / P), np.floor((BASELINE_BKJD[1] - T0) / P) + 1)
    t = T0 + n * P
    k = np.round((t - T2) / P2)
    sep = t - (T2 + k * P2)
    sigma = np.sqrt(sT**2 + (n * sP) ** 2 + sT2**2 + (k * sP2) ** 2)
    i = int(np.argmax(np.abs(sep)))
    out.update({
        "phase_offset": float(((T2 - T0) / P + 0.5) % 1 - 0.5),
        "max_sep_h": float(abs(sep[i]) * 24),
        "half_duration_h": 0.5 * D_h,
        "sep_over_sigma_at_max": float(abs(sep[i]) / sigma[i]) if sigma[i] > 0 else None,
        "compatible": bool(abs(sep[i]) * 24 <= 0.5 * D_h),
    })
    return out


def targets(metrics: dict, bundled: list[dict], snap: pd.DataFrame) -> list[dict]:
    """Error-analysis examples, then bundled examples not already among them, with their ephemerides."""
    out: list[dict] = []

    def add(tid, category, kepid, score, ex=None):
        if tid in snap.index:
            s = snap.loc[tid]
            eph = (float(s["tce_period"]), float(s["tce_time0bk"]), float(s["tce_duration"]),
                   _err(s["tce_period_err"]), _err(s["tce_time0bk_err"]))
            label = s["av_training_set"]
        else:  # Kepler-90 i: published ephemeris, no DR24 label
            u = ex.get("parameter_uncertainties", {})
            eph = (ex["period_days"], ex["epoch_bkjd"], ex["duration_hours"], u.get("period_days", 0.0),
                   u.get("epoch_days", 0.0))
            label = None
        out.append({"id": tid, "category": category, "kepid": int(kepid), "score": float(score), "dr24_label": label,
                    "eph": eph, "bundled": False})

    for e in metrics["error_analysis_examples"]:
        add(e["tce_id"], e["category"], e["kepid"], e["score"])
    for b in bundled:
        hit = next((t for t in out if t["id"] == b["id"]), None)
        if hit is None:
            add(b["id"], "bundled example", b["kepid"], b["score_at_bundle"], b)
            hit = out[-1]
        hit["bundled"] = True
    return out


def _refname(html: str) -> str:
    """'<a refstr=X href=URL ...>Name et al. 2022</a>' -> 'Name et al. 2022 (URL)'."""
    import html as _html
    import re

    m = re.search(r"href=(\S+)[^>]*>([^<]+)</a>", str(html))
    return _html.unescape(f"{m.group(2).strip()} ({m.group(1)})" if m else str(html))


def verify_archive_status(kepid: int, eph: tuple, names: pd.DataFrame, ps: pd.DataFrame, *,
                          koi_name: str | None = None, kepler_name: str | None = None) -> dict | None:
    """Confirmed-planet status via keplernames -> ps default row, with an ephemeris check (module docstring)."""
    rows = names[names["kepid"] == kepid]
    rows = rows[rows["koi_name"] == koi_name] if koi_name else rows[rows["kepler_name"] == kepler_name]
    if len(rows) != 1:
        return None
    pl_name = rows.iloc[0]["pl_name"]
    r = ps[(ps["pl_name"] == pl_name) & (ps["default_flag"] == 1)]
    if len(r) != 1 or r.iloc[0]["soltype"] != "Published Confirmed" or pd.isna(r.iloc[0]["pl_orbper"]):
        return None
    r = r.iloc[0]
    P, T0, D, sP, sT = eph
    ratio = float(r["pl_orbper"]) / P
    if abs(ratio - 1) > PERIOD_TOL:
        return None
    if pd.notna(r["pl_tranmid"]):
        c = compare(P, T0, D, sP, sT, float(r["pl_orbper"]), float(r["pl_tranmid"]) - BKJD_OFFSET,
                    _err(r["pl_orbpererr1"]), _err(r["pl_tranmiderr1"]))
        if not c["compatible"]:
            return None
        ev = f"transit times within {c['max_sep_h']:.2f} h over the mission (tolerance {c['half_duration_h']:.2f} h)"
    else:
        c, ev = None, "the Archive's default parameter set gives no transit epoch, so only the period was compared"
    return {"status": "Confirmed planet", "pl_name": pl_name, "kepler_name": rows.iloc[0]["kepler_name"],
            "soltype": r["soltype"], "discovery_reference": _refname(r["disc_refname"]),
            "disc_year": int(r["disc_year"]),
            "cross_id": f"keplernames: KIC {kepid}" + (f" + {koi_name}" if koi_name else f" + {kepler_name}")
                        + f" -> {pl_name}",
            "evidence": f"ps default period {float(r['pl_orbper']):.6g} d vs {P:.6g} d (ratio {ratio:.5f}); {ev}",
            "match": c}


def assess(tgts: list[dict], koi: pd.DataFrame, tce25: pd.DataFrame, names: pd.DataFrame,
           ps: pd.DataFrame) -> list[dict]:
    rows = []
    for ex in tgts:
        P, T0, D, sP, sT = ex["eph"]
        cands = []
        for _, k in koi[koi["kepid"] == ex["kepid"]].iterrows():
            c = compare(P, T0, D, sP, sT, float(k["koi_period"]), float(k["koi_time0bk"]),
                        _err(k["koi_period_err1"]), _err(k["koi_time0bk_err1"]))
            cands.append({"kepoi_name": k["kepoi_name"], **c})
        dr25_tces = []
        for _, t in tce25[tce25["kepid"] == ex["kepid"]].iterrows():
            c = compare(P, T0, D, sP, sT, float(t["tce_period"]), float(t["tce_time0bk"]),
                        _err(t["tce_period_err"]), _err(t["tce_time0bk_err"]))
            dr25_tces.append({"dr25_tce": f"{int(t['kepid']):09d}_{int(t['tce_plnt_num']):02d}",
                              "period_days": float(t["tce_period"]), "epoch_bkjd": float(t["tce_time0bk"]), **c})
        good = [c for c in cands if c.get("compatible")]
        if len(good) == 1:
            status = "matched"
        elif len(good) > 1:
            status = "unresolved: ambiguous (several compatible KOIs)"
        elif any(c["same_period"] and (c.get("sep_over_sigma_at_max") or np.inf) <= 3 for c in cands):
            status = "unresolved: low-precision ephemeris (same period, epochs agree only within 3 sigma)"
        elif not cands:
            status = "unresolved: no DR25 KOI on this star"
        else:
            status = "unresolved: no DR25 KOI with a compatible period and epoch"
        row = {
            "category": ex["category"], "tce_id": ex["id"], "kepid": ex["kepid"], "bundled_example": ex["bundled"],
            "dr24_label": ex["dr24_label"], "cnn_score": ex["score"],
            "dr24_ephemeris": {"period_days": P, "epoch_bkjd": T0, "duration_hours": D},
            "status": status, "koi_candidates": cands, "dr25_tces_on_star": dr25_tces,
        }
        if status == "matched":
            k = koi[koi["kepoi_name"] == good[0]["kepoi_name"]].iloc[0]
            row["matched_koi"] = {
                "kepoi_name": k["kepoi_name"], "kepler_name": k["kepler_name"] if pd.notna(k["kepler_name"]) else None,
                "dr25_disposition": k["koi_pdisposition"], "archive_disposition": k["koi_disposition"],
                "koi_score": float(k["koi_score"]),
                "flags": {f: int(k[f]) for f in FLAG_MEANINGS},
                "koi_comment": k["koi_comment"],
                "period_days": float(k["koi_period"]), "epoch_bkjd": float(k["koi_time0bk"]),
                "match_evidence": good[0],
            }
            koi_eph = (float(k["koi_period"]), float(k["koi_time0bk"]), float(k["koi_duration"]),
                       _err(k["koi_period_err1"]), _err(k["koi_time0bk_err1"]))
            row["archive_status_verified"] = verify_archive_status(ex["kepid"], koi_eph, names, ps,
                                                                   koi_name=k["kepoi_name"])
        elif ex["id"] in NON_TCE_PLANETS:
            row["archive_status_verified"] = verify_archive_status(ex["kepid"], ex["eph"], names, ps,
                                                                   kepler_name=NON_TCE_PLANETS[ex["id"]])
        else:
            row["archive_status_verified"] = None
        nearest = [c for c in cands if c["same_period"] or c["alias"]]
        if status != "matched" and nearest:
            k = koi[koi["kepoi_name"] == nearest[0]["kepoi_name"]].iloc[0]
            row["related_koi_not_matched"] = {
                "kepoi_name": k["kepoi_name"], "relation": nearest[0], "dr25_disposition": k["koi_pdisposition"],
                "archive_disposition": k["koi_disposition"], "flags": {f: int(k[f]) for f in FLAG_MEANINGS},
                "koi_comment": k["koi_comment"],
            }
        row["assessment"], row["assessment_note"] = VERDICTS.get(ex["id"], ("not assessed", ""))
        rows.append(row)
    return rows


def app_context(rows: list[dict], prov: dict) -> dict:
    """Card text for the bundled examples. Only rule-matched KOIs and ps-verified statuses are included."""
    src_koi = {"table": "NASA Exoplanet Archive q1_q17_dr25_koi",
               "column": "koi_pdisposition (DR25 Robovetter disposition; table last updated 2017-08-31)",
               "retrieved_utc": prov["dr25_koi.csv"]["retrieved_utc"], "sha256": prov["dr25_koi.csv"]["sha256"]}
    src_ps = {"table": "NASA Exoplanet Archive Planetary Systems (ps), default parameter set; cross-ID via keplernames",
              "retrieved_utc": prov["ps.csv"]["retrieved_utc"], "sha256": prov["ps.csv"]["sha256"]}
    out = {}
    for r in rows:
        if not r["bundled_example"]:
            continue
        e = {"dr24_label": r["dr24_label"], "match_status": "matched" if r["status"] == "matched" else "unresolved"}
        a = r["archive_status_verified"]
        e["archive"] = None if a is None else {**{k: v for k, v in a.items() if k != "match"}, "source": src_ps}
        m = r.get("matched_koi")
        if not m:
            e.update({"display": UNRESOLVED_TEXT, "unresolved_reason": r["status"].removeprefix("unresolved: "),
                      "dr25": None, "disagreement": None})
            out[r["tce_id"]] = e
            continue
        flags = [FLAG_WORDS[f] for f, v in m["flags"].items() if v]
        ev = m["match_evidence"]
        e["dr25"] = {"kepoi_name": m["kepoi_name"], "disposition": m["dr25_disposition"], "fp_flags": flags,
                     "koi_comment": m["koi_comment"], "source": src_koi,
                     "match_evidence": f"same star; period ratio {ev['period_ratio']:.5f}; transit times within "
                                       f"{ev['max_sep_h']:.2f} h over the mission (tolerance {ev['half_duration_h']:.2f} h)"}
        lab, disp = r["dr24_label"], m["dr25_disposition"]
        note = None
        if lab == "PC" and disp == "FALSE POSITIVE":
            why = "; ".join(flags) or "no flag recorded"
            extra = (" (its SWEET test found a sinusoid at this period larger than the dip, i.e. stellar variability)"
                     if "SWEET_NTL" in str(m["koi_comment"]) else "")
            note = (f"DR24 labelled this signal a planet candidate (PC), but the later DR25 vetting classifies it as a "
                    f"false positive: {why}{extra}.")
            if a:
                note += (f" The Archive's Planetary Systems table, however, lists it as the confirmed planet "
                         f"{a['kepler_name']}, so DR25's automated disposition is itself contradicted by later work.")
        elif lab in ("AFP", "NTP") and disp == "CANDIDATE":
            note = (f"DR24 labelled this signal {lab} (not a planet candidate), but the later DR25 vetting lists it as a "
                    f"planet candidate with no false-positive flags.")
            if a:
                note += f" The Archive's Planetary Systems table now lists it as the confirmed planet {a['kepler_name']}."
        e["disagreement"] = note
        out[r["tce_id"]] = e
    return {"generated_by": "scripts/check_dr25_koi.py",
            "note": "Later-catalogue context for the demo cards. Not used for training, model selection or evaluation; "
                    "the model was trained and evaluated on DR24 av_training_set labels.",
            "unresolved_text": UNRESOLVED_TEXT, "examples": out}


def _flags_text(m: dict) -> str:
    on = [f.replace("koi_fpflag_", "").upper() for f, v in m["flags"].items() if v]
    return ", ".join(on) if on else "none"


def markdown(rows: list[dict], prov: dict) -> str:
    L = ["# DR25 KOI cross-check of the error-analysis and bundled examples", "",
         "Generated by `scripts/check_dr25_koi.py`. Later-catalogue context only: DR24 labels, splits, models and "
         "metrics are unchanged. Sources: NASA Exoplanet Archive `q1_q17_dr25_koi` and `q1_q17_dr25_tce` "
         f"(retrieved {prov['dr25_koi.csv']['retrieved_utc']}) and the Planetary Systems table `ps` (retrieved "
         f"{prov['ps.csv']['retrieved_utc']}) with the `keplernames` cross-ID table for independently verified "
         "confirmed-planet status. Matching rule in the "
         "script docstring. The parenthesised disposition is the KOI table's own `koi_disposition`; the last column "
         "shows only `ps`-verified status.", "",
         "| DR24 TCE | category | DR24 label · CNN | status | DR25 KOI | match evidence | DR25 disposition (archive) "
         "| FP flags | koi_comment | confirmed in `ps` (verified) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r.get("matched_koi")
        if m:
            e = m["match_evidence"]
            ev = f"P ratio {e['period_ratio']:.5f}; max sep {e['max_sep_h']:.2f} h ≤ {e['half_duration_h']:.2f} h"
            koi = m["kepoi_name"] + (f" ({m['kepler_name']})" if m["kepler_name"] else "")
            disp = f"{m['dr25_disposition']} ({m['archive_disposition']})"
            a = r["archive_status_verified"]
            cells = [koi, ev, disp, _flags_text(m), f"`{m['koi_comment']}`",
                     f"{a['pl_name']}: {a['evidence']}" if a else "—"]
        else:
            notes = []
            for c in r["koi_candidates"]:
                if c["same_period"]:
                    notes.append(f"{c['kepoi_name']}: same period, phase offset {c['phase_offset']:+.3f}, "
                                 f"max sep {c['max_sep_h']:.1f} h")
                elif c["alias"]:
                    notes.append(f"{c['kepoi_name']}: period alias {c['alias']}")
                else:
                    notes.append(f"{c['kepoi_name']}: P ratio {c['period_ratio']:.3f}")
            rel = r.get("related_koi_not_matched")
            if rel:
                cells = ["— (not matched)", "; ".join(notes),
                         f"related {rel['kepoi_name']}: {rel['dr25_disposition']} ({rel['archive_disposition']})",
                         f"related: {_flags_text(rel)}", f"related: `{rel['koi_comment']}`", "—"]
                a = None
            else:
                a = r["archive_status_verified"]
                cells = ["—", "; ".join(notes) or "no KOI on star", "—", "—", "—",
                         f"{a['pl_name']}: {a['evidence']}" if a else "—"]
        cat = r["category"] + (" (bundled)" if r["bundled_example"] and r["category"] != "bundled example" else "")
        L.append(f"| `{r['tce_id']}` | {cat} | {r['dr24_label'] or 'none'} · {r['cnn_score']:.3g} | {r['status']} | "
                 + " | ".join(cells) + " |")
    L += ["", "## Does the evidence support each explanation?", "",
          "| DR24 TCE | assessment | note |", "|---|---|---|"]
    L += [f"| `{r['tce_id']}` | **{r['assessment']}** | {r['assessment_note']} |" for r in rows]
    L += ["", "Flag key: NT not transit-like, SS stellar eclipse, CO centroid offset, EC ephemeris match "
          "(contamination). Minor flags used in the interpretation:", ""]
    L += [f"* `{k}`: {v}" for k, v in MINOR_FLAG_MEANINGS.items()]
    L += ["* `CENT_*` comments are centroid-module minor flags; they are listed but not interpreted here.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="re-query the NASA Exoplanet Archive")
    args = ap.parse_args()
    metrics = json.loads((ROOT / "reports" / "full" / "metrics.json").read_text())
    bundled = json.loads((ROOT / "app" / "examples" / "examples.json").read_text())["examples"]
    snap = pd.read_csv(ROOT / "data" / "catalog" / "dr24_tce_snapshot.csv")
    snap["tce_id"] = snap["kepid"].map("{:09d}".format) + "_" + snap["tce_plnt_num"].map("{:02d}".format)
    snap = snap.set_index("tce_id")
    tgts = targets(metrics, bundled, snap)
    if args.fetch or not (OUT / "keplernames.csv").exists():
        fetch(sorted({t["kepid"] for t in tgts}))
    koi = pd.read_csv(OUT / "dr25_koi.csv")
    tce25 = pd.read_csv(OUT / "dr25_tce.csv")
    names = pd.read_csv(OUT / "keplernames.csv")
    ps = pd.read_csv(OUT / "ps.csv")
    prov = json.loads((OUT / "provenance.json").read_text())
    rows = assess(tgts, koi, tce25, names, ps)
    (OUT / "results.json").write_text(json.dumps({"matching_rule": __doc__.split("Matching rule")[1].split("Flag meanings")[0].strip(),
                                                  "flag_meanings": FLAG_MEANINGS,
                                                  "minor_flag_meanings": MINOR_FLAG_MEANINGS,
                                                  "sources": prov, "examples": rows}, indent=2, default=str) + "\n")
    (OUT / "RESULTS.md").write_text(markdown(rows, prov))
    APP_CONTEXT.write_text(json.dumps(app_context(rows, prov), indent=2, sort_keys=True) + "\n")
    for r in rows:
        m = r.get("matched_koi")
        a = r["archive_status_verified"]
        print(f"{r['tce_id']} {r['dr24_label'] or '-':>3} {r['status'][:45]:45s} "
              + (f"{m['kepoi_name']} {m['dr25_disposition']}/{m['archive_disposition']} {_flags_text(m)} "
                 if m else "") + f" ps:{a['pl_name'] if a else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
