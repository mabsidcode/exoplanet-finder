"""Configuration loading and the paths derived from it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    models: Path
    reports: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: Path | None = None) -> Paths:
        root = Path(root) if root else PROJECT_ROOT
        p = cfg.get("paths", {})
        return cls(
            root=root,
            data=root / p.get("data_dir", "data"),
            models=root / p.get("models_dir", "models"),
            reports=root / p.get("reports_dir", "reports"),
        )

    # Catalogue and split (committed, small)
    @property
    def catalog_snapshot(self) -> Path:
        return self.data / "catalog" / "dr24_tce_snapshot.csv"

    @property
    def catalog_provenance(self) -> Path:
        return self.data / "catalog" / "dr24_tce_snapshot.provenance.json"

    @property
    def split_manifest(self) -> Path:
        return self.data / "splits" / "split_manifest.csv"

    @property
    def split_info(self) -> Path:
        return self.data / "splits" / "split_info.json"

    # Download cache and processed outputs (git-ignored)
    @property
    def raw_cache(self) -> Path:
        return self.data / "cache" / "raw"

    @property
    def processed_stars(self) -> Path:
        return self.data / "processed" / "stars"

    @property
    def ledger(self) -> Path:
        return self.data / "processed" / "ledger"

    def dataset_dir(self, experiment: str) -> Path:
        return self.data / "datasets" / experiment

    def model_dir(self, experiment: str) -> Path:
        return self.models / experiment

    def report_dir(self, experiment: str) -> Path:
        return self.reports / experiment
