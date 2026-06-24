from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easy_ncounter.ui import (
    _apply_metadata_filters,
    _apply_loaded_sample_ids_to_samplesheet,
    _counts_matrix_pre_qc,
    _comparison_group_labels,
    _contrast_filter_features,
    _contrast_color_map,
    _delete_saved_analysis,
    _duplicate_uploaded_file_names,
    _default_filter_sets,
    _float_default,
    _int_default,
    _last_completed_step,
    _metadata_filter_group_label,
    _metadata_template_for_download,
    _normalize_contrast_rows,
    _next_analysis_name,
    _option_index,
    _prepare_save_target,
    _qc_hover_reason,
    _record_analysis_savepoint,
    _rename_saved_analysis,
    _read_results_metadata,
    _saved_analysis_int,
    _saved_analysis_label,
    _selected_sample_from_plotly_event,
    _selected_de_group_info,
    _samplesheet_loaded_samples_frame,
    _summary_sample_list,
    _unique_columns,
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


def test_counts_matrix_pre_qc_flags_all_zero_samples() -> None:
    counts = pd.DataFrame(
        {
            "CodeClass": ["Endogenous", "Endogenous"],
            "Name": ["GENE1", "GENE2"],
            "sample_ok": [1, 2],
            "sample_zero": [0, 0],
        }
    )

    pre_qc = _counts_matrix_pre_qc(counts).set_index("sample_id")

    assert pre_qc.loc["sample_ok", "pre_qc_status"] == "OK"
    assert pre_qc.loc["sample_ok", "total_counts"] == 3
    assert pd.api.types.is_integer_dtype(pre_qc["total_counts"])
    assert pre_qc.loc["sample_zero", "pre_qc_status"] == "FAILED"
    assert pre_qc.loc["sample_zero", "pre_qc_reason"] == "all_counts_zero"


def test_duplicate_uploaded_file_names_detects_case_insensitive_duplicates() -> None:
    class Uploaded:
        def __init__(self, name: str) -> None:
            self.name = name

    duplicates = _duplicate_uploaded_file_names(
        [Uploaded("sample_1.RCC"), Uploaded("sample_2.RCC"), Uploaded("SAMPLE_1.rcc")]
    )

    assert duplicates == ["SAMPLE_1.rcc"]


def test_loaded_sample_id_edits_update_samplesheet() -> None:
    samplesheet = pd.DataFrame(
        {
            "RCC_FILE": ["rcc/original.RCC"],
            "RCC_FILE_NAME": ["original.RCC"],
            "SAMPLE_ID": ["old_id"],
        }
    )
    loaded = _samplesheet_loaded_samples_frame(samplesheet)
    loaded.loc[0, "sample_id"] = "new id"

    updated = _apply_loaded_sample_ids_to_samplesheet(samplesheet, loaded)

    assert updated.loc[0, "SAMPLE_ID"] == "new_id"


def test_metadata_template_adds_new_sample_id_column() -> None:
    metadata = pd.DataFrame({"sample_id": ["sample 1"], "condition": ["A"]})

    template = _metadata_template_for_download(metadata)

    assert template.columns.tolist() == ["sample_id", "new_sample_id", "condition"]
    assert template.loc[0, "sample_id"] == "sample_1"
    assert template.loc[0, "new_sample_id"] == ""


def test_delete_saved_analysis_rejects_folder_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ui_runs"
    workspace.mkdir()
    outside_run = tmp_path / "analysis_1"
    outside_run.mkdir()

    with pytest.raises(ValueError, match="direct folder"):
        _delete_saved_analysis(workspace, outside_run)

    assert outside_run.exists()


def test_rename_saved_analysis_moves_run_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ui_runs"
    run_dir = workspace / "analysis_1"
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline.yaml").write_text("project:\n  name: old\n", encoding="utf-8")

    renamed = _rename_saved_analysis(workspace, run_dir, "renamed analysis")

    assert renamed == workspace / "renamed_analysis"
    assert renamed.exists()
    assert not run_dir.exists()


def test_next_analysis_name_uses_available_copy_suffix(tmp_path: Path) -> None:
    workspace = tmp_path / "ui_runs"
    (workspace / "analysis_copy").mkdir(parents=True)

    assert _next_analysis_name(workspace, "analysis") == "analysis_copy_2"


def test_default_helpers_reject_invalid_values() -> None:
    assert _option_index(["a", "b"], "b") == 1
    assert _option_index(["a", "b"], "missing") == 0
    assert _int_default("3", 10, minimum=1) == 3
    assert _int_default("0", 10, minimum=1) == 10
    assert _float_default("2.5", 1.0) == 2.5
    assert _float_default("bad", 1.0) == 1.0


def test_prepare_save_target_copies_loaded_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "ui_runs" / "source"
    target = tmp_path / "ui_runs" / "target"
    (source / "input").mkdir(parents=True)
    (source / "input" / "counts.csv").write_text("Name,s1\nG1,1\n", encoding="utf-8")
    monkeypatch.setattr("easy_ncounter.ui.st_session_get", lambda key, default=None: str(source) if key == "loaded_analysis_dir" else default)

    _prepare_save_target(target, overwrite=False)

    assert (target / "input" / "counts.csv").exists()


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


def test_saved_analysis_int_uses_saved_gene_filter_values() -> None:
    saved_analysis = {"min_count": "1", "min_samples": 1, "top_variable_genes": 25}

    assert _saved_analysis_int(saved_analysis, "min_count", 10, minimum=0) == 1
    assert _saved_analysis_int(saved_analysis, "min_samples", 2, minimum=1) == 1
    assert _saved_analysis_int(saved_analysis, "top_variable_genes", 50, minimum=5) == 25
    assert _saved_analysis_int(saved_analysis, "missing", 10, minimum=0) == 10
    assert _saved_analysis_int({"min_samples": 0}, "min_samples", 2, minimum=1) == 2


def test_qc_hover_reason_includes_reason_and_value() -> None:
    reason = _qc_hover_reason(
        pd.Series(
            {
                "qc_warnings": "low_library_size",
                "qc_fail_reasons": "very_low_detected_endogenous_probes",
                "library_size": 1234,
                "detected_endogenous_probes": 4,
            }
        )
    )

    assert "WARN low_library_size: library_size=1234" in reason
    assert "FAIL very_low_detected_endogenous_probes: detected_endogenous_probes=4" in reason


def test_unique_columns_excludes_selected_metric_from_hover_columns() -> None:
    columns = _unique_columns(
        ["qc_warnings", "negative_control_mean", "negative_control_mean"],
        available=pd.Index(["qc_warnings", "negative_control_mean"]),
        exclude={"negative_control_mean"},
    )

    assert columns == ["qc_warnings"]


def test_selected_sample_from_plotly_event_reads_clicked_bar() -> None:
    event = {"selection": {"points": [{"x": "sample_1"}]}}

    assert _selected_sample_from_plotly_event(event) == "sample_1"


def test_selected_sample_from_plotly_event_reads_customdata_when_x_missing() -> None:
    event = {"selection": {"points": [{"customdata": ["sample_from_customdata"]}]}}

    assert _selected_sample_from_plotly_event(event) == "sample_from_customdata"


def test_selected_sample_from_plotly_event_reads_numpy_customdata() -> None:
    event = {"selection": {"points": [{"customdata": np.array(["sample_from_numpy"])}]}}

    assert _selected_sample_from_plotly_event(event) == "sample_from_numpy"


def test_selected_sample_from_plotly_event_reads_streamlit_state_object() -> None:
    class Event:
        def to_dict(self) -> dict:
            return {"selection": {"points": [{"customdata": np.array(["sample_from_state"])}]}}

    assert _selected_sample_from_plotly_event(Event()) == "sample_from_state"


def test_apply_metadata_filters_supports_is_and_is_not() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "condition": ["A", "B", "A"],
            "time": ["T0", "T0", "T1"],
        }
    )

    assert _apply_metadata_filters(metadata, [{"feature": "condition", "operator": "is", "value": "A"}]) == [
        "s1",
        "s3",
    ]
    assert _apply_metadata_filters(
        metadata,
        [
            {"feature": "condition", "operator": "is", "value": "A"},
            {"feature": "time", "operator": "is not", "value": "T0"},
        ],
    ) == ["s3"]


def test_default_filter_sets_create_group_vs_all_others() -> None:
    metadata = pd.DataFrame({"sample_id": ["s1", "s2"], "condition": ["A", "B"]})

    defaults = _default_filter_sets(metadata, {})

    assert defaults["reference"] == [{"feature": "condition", "operator": "is", "value": "A"}]
    assert defaults["case"] == [{"feature": "condition", "operator": "is not", "value": "A"}]


def test_comparison_group_labels_use_reference_case_names_for_custom_contrasts() -> None:
    assert _comparison_group_labels(["reference", "case"]) == ["Reference", "Case"]
    assert _comparison_group_labels(["condition=A", "condition=B"]) == ["condition=A", "condition=B"]


def test_metadata_filter_group_label_describes_selected_features() -> None:
    label = _metadata_filter_group_label(
        [
            {"feature": "condition", "operator": "is", "value": "A"},
            {"feature": "time", "operator": "is not", "value": "T0"},
        ],
        "Reference",
    )

    assert label == "condition=A + time!=T0"


def test_contrast_filter_features_collects_features_across_multiple_contrasts() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "time": ["T0", "T1"],
            "responder": ["R", "N"],
        }
    )
    contrasts = [
        {
            "reference_filters": [{"feature": "time", "operator": "is", "value": "T0"}],
            "case_filters": [{"feature": "responder", "operator": "is", "value": "R"}],
        },
        {
            "reference_filters": [{"feature": "time", "operator": "is", "value": "T1"}],
            "case_filters": [{"feature": "responder", "operator": "is not", "value": "N"}],
        },
    ]

    assert _contrast_filter_features(contrasts, metadata) == ["time", "responder"]


def test_summary_sample_list_accepts_lists_and_semicolon_strings() -> None:
    assert _summary_sample_list(["s1", "s2"]) == ["s1", "s2"]
    assert _summary_sample_list("s1;s2;") == ["s1", "s2"]


def test_contrast_color_map_uses_blue_and_red() -> None:
    colors = _contrast_color_map(["Reference", "Case"])

    assert colors == {"Reference": "#1F77B4", "Case": "#D62728"}


def test_normalize_contrast_rows_preserves_custom_sample_lists() -> None:
    contrasts = _normalize_contrast_rows(
        [
            {
                "comparison_id": "custom",
                "reference_group": "reference",
                "case_group": "case",
                "reference_samples": ["s1", "s2"],
                "case_samples": "s3;s4",
            }
        ]
    )

    assert contrasts == [
        {
            "comparison_id": "custom",
            "reference_group": "reference",
            "case_group": "case",
            "reference_samples": ["s1", "s2"],
            "case_samples": ["s3", "s4"],
            "reference_filters": [],
            "case_filters": [],
        }
    ]


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
