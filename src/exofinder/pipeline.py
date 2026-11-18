"""Resumable download + preprocessing of Kepler light curves, one star at a time.

For each host star:
1. List its long-cadence light-curve files in the MAST Kepler archive.
2. Download missing files into ``data/cache/raw/<kepid>/`` (atomic writes,
   bounded retries with exponential back-off, MD5 recorded per file).
3. Read and clean the light curve once, then preprocess *every* labelled TCE
   of the star.
4. Atomically save the star's views to ``data/processed/stars/<kepid>.npz``,
   reopen it to verify shapes and finiteness, then write a JSON ledger with
   file provenance, checksums, per-TCE status and exclusion reasons.
5. Only after verification, delete the star's raw files from the managed cache
   unless the star is in the keep list (demo stars and a debug sample).

A star whose ledger reports ``done`` with the same preprocessing fingerprint
and TCE set is skipped, so interrupted runs resume without duplicating work.
Outputs are keyed by star and TCE ID, so retries overwrite rather than append.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Paths
from .io_utils import atomic_path, atomic_write_json, file_digest, read_json, utc_now_iso
from .lightcurve import LightCurveError, read_kepler_fits
from .preprocess import CandidateParams, PreprocessConfig, PreprocessingError, preprocess_candidate

USER_AGENT = "exofinder/0.1 (research portfolio project; python-requests)"
_FILE_RE = re.compile(r'href="(kplr\d{9}-\d{13}_llc\.fits)"')


def star_url(base_url: str, kepid: int) -> str:
    k = f"{int(kepid):09d}"
    return f"{base_url.rstrip('/')}/{k[:4]}/{k}/"


def _get(session: requests.Session, url: str, retries: int, timeout: float, stream: bool = False):
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout, stream=stream)
            if r.status_code == 404:
                r.raise_for_status()
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise
            if attempt == retries:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def list_star_files(session: requests.Session, base_url: str, kepid: int, retries: int, timeout: float) -> list[str]:
    r = _get(session, star_url(base_url, kepid), retries, timeout)
    return sorted(set(_FILE_RE.findall(r.text)))


def download_file(session: requests.Session, url: str, dest: Path, retries: int, timeout: float) -> dict[str, Any]:
    if dest.exists():
        return {"bytes": dest.stat().st_size, "md5": file_digest(dest, "md5"), "cached": True}
    r = _get(session, url, retries, timeout, stream=True)
    with atomic_path(dest) as tmp:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        expected = r.headers.get("Content-Length")
        if expected is not None and int(expected) != tmp.stat().st_size:
            raise OSError(f"Incomplete download of {url}")
    return {"bytes": dest.stat().st_size, "md5": file_digest(dest, "md5"), "cached": False}


def process_star(
    kepid: int,
    tces: list[dict[str, Any]],
    cfg: dict[str, Any],
    data_dir: str,
    keep_raw: bool,
) -> dict[str, Any]:
    """Download, preprocess and record one star. Runs in a worker process."""
    paths = Paths(root=Path(data_dir).parent, data=Path(data_dir), models=Path(), reports=Path())
    pcfg = PreprocessConfig.from_dict(cfg["preprocess"])
    dl = cfg["download"]
    ledger_path = paths.ledger / f"{kepid:09d}.json"
    npz_path = paths.processed_stars / f"{kepid:09d}.npz"
    tce_ids = sorted(t["tce_id"] for t in tces)

    if ledger_path.exists():
        old = read_json(ledger_path)
        if (
            old.get("status") == "done"
            and old.get("preprocess_fingerprint") == pcfg.fingerprint()
            and sorted(e["tce_id"] for e in old["examples"]) == tce_ids
            and npz_path.exists()
        ):
            old["skipped"] = True
            return old

    t_start = time.time()
    raw_dir = paths.raw_cache / f"{kepid:09d}"
    ledger: dict[str, Any] = {
        "kepid": int(kepid),
        "status": "started",
        "retrieved_utc": utc_now_iso(),
        "source_url": star_url(dl["base_url"], kepid),
        "preprocess_fingerprint": pcfg.fingerprint(),
        "preprocess": pcfg.to_dict(),
        "files": [],
        "examples": [],
        "skipped": False,
    }
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    try:
        names = list_star_files(session, dl["base_url"], kepid, dl["retries"], dl["timeout_s"])
        if not names:
            raise LightCurveError("no_files", "No long-cadence light-curve files listed at MAST")
        nbytes = 0
        for name in names:
            info = download_file(
                session, ledger["source_url"] + name, raw_dir / name, dl["retries"], dl["timeout_s"]
            )
            nbytes += info["bytes"] if not info["cached"] else 0
            ledger["files"].append({"filename": name, "url": ledger["source_url"] + name, **info})
        ledger["bytes_downloaded"] = nbytes
        ledger["raw_bytes"] = int(sum(f["bytes"] for f in ledger["files"]))
        t_dl = time.time()

        lc = read_kepler_fits([raw_dir / n for n in names], pcfg.flux_column, pcfg.quality_bitmask)
        file_meta = {f["filename"]: f for f in lc.info["files"]}
        for f in ledger["files"]:
            f.update({k: v for k, v in file_meta[f["filename"]].items() if k != "filename"})
        ledger["light_curve"] = {
            "data_releases": lc.info["data_releases"],
            "flux_column": lc.info["flux_column"],
            "quality_bitmask": lc.info["quality_bitmask"],
            "cadence": "long (~29.4 min)",
            "cleaning": lc.info["cleaning"],
            "quarters": sorted({f["quarter"] for f in lc.info["files"]}),
        }

        ok_ids, gviews, lviews = [], [], []
        for t in sorted(tces, key=lambda r: r["tce_id"]):
            rec: dict[str, Any] = {"tce_id": t["tce_id"]}
            params = CandidateParams(t["period_days"], t["epoch_bkjd"], t["duration_hours"])
            try:
                pc = preprocess_candidate(lc, params, pcfg)
                rec.update(status="ok", reason="", diagnostics=pc.diagnostics)
                ok_ids.append(t["tce_id"])
                gviews.append(pc.global_view)
                lviews.append(pc.local_view)
            except PreprocessingError as e:
                rec.update(status="rejected", reason=e.reason, message=str(e))
            except Exception as e:  # noqa: BLE001 - recorded, never silently dropped
                rec.update(status="error", reason=f"error_{type(e).__name__}", message=str(e)[:300])
            ledger["examples"].append(rec)

        g = np.stack(gviews) if gviews else np.zeros((0, 2, pcfg.global_num_bins), np.float32)
        loc = np.stack(lviews) if lviews else np.zeros((0, 2, pcfg.local_num_bins), np.float32)
        with atomic_path(npz_path, suffix=".npz") as tmp:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, tce_ids=np.array(ok_ids), global_view=g, local_view=loc)
        with np.load(npz_path) as chk:
            assert list(chk["tce_ids"]) == ok_ids
            assert chk["global_view"].shape == g.shape and np.isfinite(chk["global_view"]).all()
            assert chk["local_view"].shape == loc.shape and np.isfinite(chk["local_view"]).all()
        ledger["processed_file"] = str(npz_path.relative_to(paths.data))
        ledger["processed_sha256"] = file_digest(npz_path)
        ledger["status"] = "done"
        ledger["timing_s"] = {"download": round(t_dl - t_start, 2), "process": round(time.time() - t_dl, 2)}
    except LightCurveError as e:
        ledger.update(status="failed_read", reason=e.reason, message=str(e))
    except requests.RequestException as e:
        ledger.update(status="failed_download", reason="download_error", message=str(e)[:300])
    except Exception as e:  # noqa: BLE001
        ledger.update(
            status="failed_other", reason=f"error_{type(e).__name__}", message=traceback.format_exc()[-600:]
        )

    if ledger["status"] != "done":
        for t in tces:
            ledger["examples"].append(
                {"tce_id": t["tce_id"], "status": ledger["status"], "reason": ledger.get("reason", "")}
            )
    if ledger["status"] == "done" and dl.get("delete_raw_after_processing", True) and not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        ledger["raw_deleted"] = True
    else:
        ledger["raw_deleted"] = False
    atomic_write_json(ledger_path, ledger)
    return ledger


def select_stars(labelled: pd.DataFrame, cfg: dict[str, Any]) -> list[int]:
    """Stars to download for this experiment.

    ``max_stars: null`` selects every labelled star. Otherwise a representative
    sample: stars are grouped by their primary category (any PC > any AFP >
    NTP), and from each group stars are taken at evenly spaced quantiles of
    their longest TCE period (covering short and long periods). Demo stars
    (``keep_raw_kepids``) are always included.
    """
    dl = cfg["download"]
    keep = [int(k) for k in dl.get("keep_raw_kepids", [])]
    all_stars = sorted(labelled["kepid"].unique().tolist())
    if dl.get("max_stars") in (None, 0):
        return all_stars
    n_target = int(dl["max_stars"]) - len(keep)

    def cat(d: pd.Series) -> str:
        s = set(d)
        return "PC" if "PC" in s else ("AFP" if "AFP" in s else "NTP")

    stars = labelled.groupby("kepid").agg(
        category=("original_disposition", cat), max_period=("period_days", "max")
    ).reset_index()
    stars = stars[~stars["kepid"].isin(keep)]
    groups = sorted(stars["category"].unique())
    per = [n_target // len(groups) + (1 if i < n_target % len(groups) else 0) for i in range(len(groups))]
    chosen: list[int] = []
    for g, n in zip(groups, per, strict=True):
        sub = stars[stars["category"] == g].sort_values(["max_period", "kepid"]).reset_index(drop=True)
        idx = np.unique(np.linspace(0, len(sub) - 1, n).round().astype(int))
        chosen += sub.loc[idx, "kepid"].astype(int).tolist()
    return sorted(set(chosen) | set(keep))


def _dir_bytes(p: Path) -> int:
    total = 0
    if p.exists():
        for root, _, files in os.walk(p):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def run(
    examples: pd.DataFrame,
    kepids: list[int],
    cfg: dict[str, Any],
    paths: Paths,
    workers: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Process ``kepids`` in parallel worker processes; return run statistics."""
    dl = cfg["download"]
    workers = workers or int(dl.get("workers", 4))
    keep = {int(k) for k in dl.get("keep_raw_kepids", [])}
    debug_n = int(dl.get("keep_raw_debug_stars", 0))
    keep |= set(sorted(kepids)[:debug_n])  # deterministic debug sample
    by_star = {
        int(k): g[["tce_id", "period_days", "epoch_bkjd", "duration_hours"]].to_dict("records")
        for k, g in examples[examples["kepid"].isin(kepids)].groupby("kepid")
    }
    paths.ledger.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats = {"n_stars": len(kepids), "done": 0, "skipped": 0, "failed": 0, "bytes": 0, "peak_raw_bytes": 0}
    statuses: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_star, k, by_star[k], cfg, str(paths.data), k in keep): k for k in kepids
        }
        for i, fut in enumerate(as_completed(futs), 1):
            led = fut.result()
            statuses[led["status"]] = statuses.get(led["status"], 0) + 1
            if led.get("skipped"):
                stats["skipped"] += 1
            elif led["status"] == "done":
                stats["done"] += 1
                stats["bytes"] += led.get("bytes_downloaded", 0)
            else:
                stats["failed"] += 1
            if i % 10 == 0 or i == len(kepids):
                raw = _dir_bytes(paths.raw_cache)
                stats["peak_raw_bytes"] = max(stats["peak_raw_bytes"], raw)
                el = time.time() - t0
                new = stats["done"] + stats["failed"]
                rate = new / el if el > 0 and new else float("nan")
                remaining = len(kepids) - i
                eta_h = remaining / rate / 3600 if rate == rate and rate > 0 else float("nan")
                log(
                    f"[{i}/{len(kepids)}] done={stats['done']} skipped={stats['skipped']} "
                    f"failed={stats['failed']} downloaded={stats['bytes'] / 1e6:.1f} MB "
                    f"raw_cache={raw / 1e6:.1f} MB rate={rate * 3600:.0f} stars/h ETA={eta_h:.2f} h"
                )
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["statuses"] = statuses
    return stats
