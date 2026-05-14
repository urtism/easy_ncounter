from __future__ import annotations

from pathlib import Path
import subprocess

from .config import PipelineConfig


def run_r_script(config: PipelineConfig, script_name: str) -> None:
    script_path = Path("scripts") / script_name
    command = [config.rscript, str(script_path), "--config", str(config.path)]
    subprocess.run(command, check=True)

