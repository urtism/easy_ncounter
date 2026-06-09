from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io import (
    metadata_from_samplesheet,
    read_counts,
    read_counts_from_samplesheet,
    read_delimited_table,
    read_rcc_metrics_from_samplesheet,
)
from .qc import background_correct_counts, compute_qc_summary, filter_endogenous_counts
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

    if config.samplesheet_path:
        counts = read_counts_from_samplesheet(config.samplesheet_path)
        metadata = metadata_from_samplesheet(config.samplesheet_path)
        rcc_metrics = read_rcc_metrics_from_samplesheet(config.samplesheet_path)
    else:
        counts = read_counts(config.counts_input)
        metadata = _read_metadata(config.metadata_path)
        rcc_metrics = None

    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    missing_metadata = sorted(set(sample_cols) - set(metadata["sample_id"]))
    if missing_metadata:
        joined = ", ".join(missing_metadata)
        raise ValueError(f"Missing metadata rows for samples: {joined}")

    qc_summary = compute_qc_summary(counts, config.raw.get("qc", {}), rcc_metrics)
    background_corrected = background_correct_counts(counts, qc_summary)
    filtered, filter_decisions = filter_endogenous_counts(
        background_corrected,
        qc_summary,
        metadata,
        config.raw["analysis"]["group_column"],
        config.raw.get("qc", {}),
    )
    probe_annotation = counts[["CodeClass", "Name"]].drop_duplicates()

    counts.to_csv(config.results_dir / "counts_raw.csv", index=False)
    metadata.to_csv(config.results_dir / "metadata.csv", index=False)
    probe_annotation.to_csv(config.results_dir / "probe_annotation.csv", index=False)
    qc_summary.to_csv(config.results_dir / "qc_summary.csv", index=False)
    qc_summary[["sample_id", "qc_status", "qc_warnings", "qc_fail_reasons"]].to_csv(
        config.results_dir / "sample_qc_decisions.csv", index=False
    )
    background_corrected.to_csv(config.results_dir / "counts_background_corrected.csv", index=False)
    filtered.to_csv(config.results_dir / "counts_filtered.csv", index=False)
    filter_decisions.to_csv(config.results_dir / "gene_filter_decisions.csv", index=False)
    print(f"Wrote {config.results_dir / 'counts_raw.csv'}")
    print(f"Wrote {config.results_dir / 'metadata.csv'}")
    print(f"Wrote {config.results_dir / 'probe_annotation.csv'}")
    print(f"Wrote {config.results_dir / 'qc_summary.csv'}")
    print(f"Wrote {config.results_dir / 'counts_background_corrected.csv'}")
    print(f"Wrote {config.results_dir / 'counts_filtered.csv'}")


def _read_metadata(path: Path):
    return read_delimited_table(path)


if __name__ == "__main__":
    main()
