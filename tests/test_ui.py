from pathlib import Path

import pandas as pd
import pytest
import yaml

from easy_ncounter.ui import (
    _delete_saved_analysis,
    _last_completed_step,
    _record_analysis_savepoint,
    _read_results_metadata,
    _saved_analysis_label,
    _selected_de_group_info,
    _write_saved_config,
)


def test_delete_saved_analysis_removes_selected_run(tmp_path: Path) -> None:
    workspace = tmp_path / "ui_runs"
    run_dir = workspace / "analysis_1"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "counts_normalized.csv").write_text("Name,sample_1\nGeneA,10\n")

    _delete_saved_analysis(workspace, run_dir)

    assert workspace.exists()
    assert not run_dir.exists()


def test_delete_saved_analysis_rejects_folder_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ui_runs"
    workspace.mkdir()
    outside_run = tmp_path / "analysis_1"
    outside_run.mkdir()

    with pytest.raises(ValueError, match="direct folder"):
        _delete_saved_analysis(workspace, outside_run)

    assert outside_run.exists()


def test_record_analysis_savepoint_tracks_last_completed_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "ui_runs" / "analysis_1"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {"results_dir": "results", "reports_dir": "reports"},
                "inputs": {"counts": "counts.csv", "metadata": "metadata.csv"},
                "analysis": {"group_column": "condition"},
            }
        ),
        encoding="utf-8",
    )

    _record_analysis_savepoint(config_path, "prepare")
    _record_analysis_savepoint(config_path, "normalize")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    savepoint = config["ui"]["savepoint"]
    assert savepoint["last_completed_step"] == "normalize"
    assert savepoint["completed_steps"] == ["prepare", "normalize"]
    assert _last_completed_step(config_path) == "normalize"


def test_saved_analysis_label_uses_savepoint_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "ui_runs" / "analysis_1"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "metadata.csv").write_text("sample_id,condition\ns1,A\n", encoding="utf-8")
    config_path = run_dir / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "outputs": {"results_dir": str(results_dir), "reports_dir": str(run_dir / "reports")},
                "inputs": {"counts": "counts.csv", "metadata": "metadata.csv"},
                "analysis": {"group_column": "condition"},
            }
        ),
        encoding="utf-8",
    )
    _record_analysis_savepoint(config_path, "differential")

    assert "statistical comparison" in _saved_analysis_label(run_dir)


def test_write_saved_config_quotes_yaml_scalars_that_r_reads_as_boolean(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"

    _write_saved_config(
        config_path,
        {
            "analysis": {
                "group_column": "RESPONDER",
                "reference_group": "R",
                "case_group": "N",
                "contrasts": [
                    {"comparison_id": "N_vs_R", "reference_group": "R", "case_group": "N"}
                ],
            }
        },
    )

    text = config_path.read_text(encoding="utf-8")
    assert "case_group: 'N'" in text
    assert "reference_group: R" in text


def test_results_metadata_rebuilds_composite_analysis_group_for_interactive_plots(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ui_runs" / "analysis_1"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "metadata.csv").write_text(
        "\n".join(
            [
                "sample_id,RESPONDER,TIME,condition",
                "s1,R,T12,ignored",
                "s2,N,T12,ignored",
                "s3,R,T0,ignored",
            ]
        ),
        encoding="utf-8",
    )
    _write_saved_config(
        run_dir / "pipeline.yaml",
        {
            "analysis": {
                "group_column": "__group__RESPONDER__TIME",
                "group_columns": ["RESPONDER", "TIME"],
                "reference_group": "RESPONDER=R | TIME=T12",
                "case_group": "RESPONDER=N | TIME=T12",
            }
        },
    )
    de = yaml.safe_load(
        """
        - reference_group: RESPONDER=R | TIME=T12
          case_group: RESPONDER=N | TIME=T12
        """
    )

    metadata = _read_results_metadata(results_dir)
    group_col, groups = _selected_de_group_info(
        pd.DataFrame(de),
        metadata,
        results_dir,
    )

    assert "__group__RESPONDER__TIME" in metadata.columns
    assert metadata["__group__RESPONDER__TIME"].tolist() == [
        "RESPONDER=R | TIME=T12",
        "RESPONDER=N | TIME=T12",
        "RESPONDER=R | TIME=T0",
    ]
    assert group_col == "__group__RESPONDER__TIME"
    assert groups == ["RESPONDER=R | TIME=T12", "RESPONDER=N | TIME=T12"]
