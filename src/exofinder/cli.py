"""Command-line entry points: ``exofinder <command> --config configs/<name>.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import Paths, load_config


def _paths(cfg) -> Paths:
    return Paths.from_config(cfg)


def cmd_fetch_catalog(args, cfg):
    from .catalog import fetch_catalog, label_examples
    from .io_utils import read_json

    p = _paths(cfg)
    if p.catalog_snapshot.exists() and not args.force:
        print(f"Snapshot exists: {p.catalog_snapshot} (use --force to re-download)")
    else:
        fetch_catalog(cfg, p.catalog_snapshot, p.catalog_provenance)
        print(f"Saved {p.catalog_snapshot}")
    prov = read_json(p.catalog_provenance)
    ex = label_examples(pd.read_csv(p.catalog_snapshot), cfg, prov)
    print(json.dumps(prov["label_value_counts"], indent=2))
    lab = ex[ex["label"] >= 0]
    print(f"Labelled TCEs: {len(lab)} on {lab['kepid'].nunique()} stars "
          f"(positives {int(lab['label'].sum())}, negatives {int((lab['label'] == 0).sum())})")


def cmd_make_splits(args, cfg):
    from .catalog import label_examples, labelled_examples
    from .io_utils import atomic_path, atomic_write_json, file_digest, read_json, utc_now_iso
    from .splits import attach_splits, check_no_star_overlap, make_star_splits, split_summary

    p = _paths(cfg)
    if p.split_manifest.exists() and not args.force:
        print(f"Split manifest exists and is fixed: {p.split_manifest} (use --force to regenerate)")
        return
    prov = read_json(p.catalog_provenance)
    lab = labelled_examples(label_examples(pd.read_csv(p.catalog_snapshot), cfg, prov))
    manifest = make_star_splits(lab, cfg)
    ex = attach_splits(lab, manifest)
    check_no_star_overlap(ex)
    with atomic_path(p.split_manifest) as tmp:
        manifest.to_csv(tmp, index=False)
    summ = split_summary(ex)
    atomic_write_json(
        p.split_info,
        {
            "created_utc": utc_now_iso(),
            "seed": cfg["split"]["seed"],
            "fractions": cfg["split"]["fractions"],
            "grouping": "host star (kepid); stratified by star hosting >=1 PC",
            "demo_holdout_kepids": cfg["catalog"].get("demo_holdout_kepids", []),
            "catalog_snapshot_sha256": file_digest(p.catalog_snapshot),
            "split_manifest_sha256": file_digest(p.split_manifest),
            "summary": json.loads(summ.to_json(orient="index")),
        },
    )
    print(summ.to_string())


def cmd_fetch(args, cfg):
    from . import pipeline
    from .dataset import load_catalog_examples
    from .io_utils import atomic_write_json

    p = _paths(cfg)
    ex = load_catalog_examples(cfg, p)
    kepids = pipeline.select_stars(ex, cfg)
    if args.limit:
        kepids = kepids[: args.limit]
    out = p.dataset_dir(cfg["experiment_name"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"kepid": kepids}).to_csv(out / "selected_stars.csv", index=False)
    print(f"Processing {len(kepids)} stars ({ex['kepid'].isin(kepids).sum()} labelled TCEs) "
          f"with {args.workers or cfg['download']['workers']} workers")
    stats = pipeline.run(ex, kepids, cfg, p, workers=args.workers)
    atomic_write_json(out / "fetch_stats.json", stats)
    print(json.dumps(stats, indent=2))


def cmd_build_dataset(args, cfg):
    from .dataset import build

    p = _paths(cfg)
    sel = pd.read_csv(p.dataset_dir(cfg["experiment_name"]) / "selected_stars.csv")["kepid"].tolist()
    m = build(cfg, p, sel)
    print(json.dumps({k: m[k] for k in ["n_selected_examples", "n_usable_examples", "n_usable_stars",
                                        "status_counts", "exclusion_reasons", "by_split"]}, indent=2))


def cmd_train_baseline(args, cfg):
    from .baseline import train_baselines

    print(json.dumps(train_baselines(cfg, _paths(cfg)), indent=2))


def cmd_train_cnn(args, cfg):
    from .train import train_all

    train_all(cfg, _paths(cfg), seeds=args.seeds, use_local=not args.global_only, resume=not args.no_resume)


def cmd_evaluate(args, cfg):
    from .evaluate import evaluate

    evaluate(cfg, _paths(cfg))


def cmd_release(args, cfg):
    from .release import release

    release(cfg, _paths(cfg))


def cmd_bundle_examples(args, cfg):
    from .release import bundle_examples

    bundle_examples(cfg, _paths(cfg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="exofinder", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default="configs/quickstart.yaml")
        sp.set_defaults(fn=fn)
        return sp

    add("fetch-catalog", cmd_fetch_catalog, "Download and snapshot the DR24 TCE table").add_argument(
        "--force", action="store_true")
    add("make-splits", cmd_make_splits, "Create the fixed star-grouped split").add_argument(
        "--force", action="store_true")
    sp = add("fetch", cmd_fetch, "Download light curves and preprocess (resumable)")
    sp.add_argument("--workers", type=int, default=None)
    sp.add_argument("--limit", type=int, default=None, help="Process only the first N selected stars")
    add("build-dataset", cmd_build_dataset, "Consolidate processed views and write the manifest")
    add("train-baseline", cmd_train_baseline, "Train scikit-learn baselines")
    sp = add("train-cnn", cmd_train_cnn, "Train the dual-branch CNN")
    sp.add_argument("--seeds", type=int, nargs="*", default=None)
    sp.add_argument("--global-only", action="store_true", help="Ablation: global view only")
    sp.add_argument("--no-resume", action="store_true")
    add("evaluate", cmd_evaluate, "Evaluate saved models; write predictions, metrics, figures")
    add("release", cmd_release, "Copy the validation-selected model to models/release")
    add("bundle-examples", cmd_bundle_examples, "Bundle demo light curves for offline use")

    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(Path(args.config))
    args.fn(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
