from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io import read_counts
from .qc import compute_qc_summary
from .runners import run_r_script


def main() -> None:
    parser = argparse.ArgumentParser(prog="easy-ncounter")
    parser.add_argument("command", choices=["prepare", "normalize", "differential", "report", "run"])
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command in {"prepare", "run"}:
        prepare(config)
    if args.command in {"normalize", "run"}:
        run_r_script(config, "normalize.R")
    if args.command in {"differential", "run"}:
        run_r_script(config, "differential_expression.R")
    if args.command in {"report", "run"}:
        run_r_script(config, "report.R")


def prepare(config) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    counts = read_counts(config.counts_input)
    metadata = _read_metadata(config.metadata_path)

    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    missing_metadata = sorted(set(sample_cols) - set(metadata["sample_id"]))
    if missing_metadata:
        joined = ", ".join(missing_metadata)
        raise ValueError(f"Missing metadata rows for samples: {joined}")

    counts.to_csv(config.results_dir / "counts_raw.csv", index=False)
    compute_qc_summary(counts).to_csv(config.results_dir / "qc_summary.csv", index=False)

    print(f"Wrote {config.results_dir / 'counts_raw.csv'}")
    print(f"Wrote {config.results_dir / 'qc_summary.csv'}")


def _read_metadata(path: Path):
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return __import__("pandas").read_csv(path, sep=sep)


if __name__ == "__main__":
    main()

