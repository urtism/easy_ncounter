from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def results_dir(self) -> Path:
        return Path(self.raw["outputs"]["results_dir"])

    @property
    def reports_dir(self) -> Path:
        return Path(self.raw["outputs"]["reports_dir"])

    @property
    def counts_input(self) -> Path:
        return Path(self.raw["inputs"]["counts"])

    @property
    def metadata_path(self) -> Path:
        return Path(self.raw["inputs"]["metadata"])

    @property
    def rscript(self) -> str:
        return str(self.raw.get("r", {}).get("executable", "Rscript"))


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    required = [
        ("inputs", "counts"),
        ("inputs", "metadata"),
        ("outputs", "results_dir"),
        ("outputs", "reports_dir"),
        ("analysis", "group_column"),
    ]
    missing = [f"{section}.{key}" for section, key in required if key not in raw.get(section, {})]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config keys: {joined}")

    return PipelineConfig(path=config_path, raw=raw)

