from pathlib import Path

import pytest
import yaml

from easy_ncounter.ui import (
    _delete_saved_analysis,
    _last_completed_step,
    _record_analysis_savepoint,
    _saved_analysis_label,
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
