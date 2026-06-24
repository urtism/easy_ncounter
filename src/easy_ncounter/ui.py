from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from easy_ncounter.cli import prepare as prepare_pipeline
from easy_ncounter.config import load_config
from easy_ncounter.io import metadata_from_samplesheet, parse_probe_annotation, read_count_table, read_rcc_file
from easy_ncounter.runners import run_r_script


APP_TITLE = "easy-ncounter"
COMPOSITE_GROUP_SEPARATOR = " | "
COMPOSITE_GROUP_PREFIX = "__group__"
STEP_LABELS = {
    "configured": "saved configuration",
    "prepare": "processing/QC",
    "normalize": "normalization",
    "differential": "statistical comparison",
    "report": "report and plots",
}
YAML_AMBIGUOUS_SCALAR_RE = re.compile(
    r"^(?:|~|null|true|false|yes|no|on|off|y|n|na|nan|\.nan|[-+]?\.inf)$",
    re.IGNORECASE,
)


def main() -> None:
    try:
        import streamlit.web.cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install the UI with: "
            'python -m pip install -e ".[ui]"'
        ) from exc

    script_path = Path(__file__).resolve()
    extra_args = sys.argv[1:]
    sys.argv = ["streamlit", "run", str(script_path), "--server.headless", "true", *extra_args]
    raise SystemExit(stcli.main())


def _run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title=APP_TITLE, page_icon="EN", layout="wide")
    _inject_upload_page_styles()
    st.title(APP_TITLE)
    st.caption("Upload, process, and explore NanoString nCounter data.")

    workspace = Path("ui_runs")
    workspace.mkdir(exist_ok=True)
    entry_action = st.session_state.get("entry_action")
    if entry_action is None:
        _render_entry_page()
        return
    if entry_action in {"load_previous", "manage_saved"}:
        _render_saved_analysis_page(workspace, manage_only=entry_action == "manage_saved")
        return

    module_upload, module_qc, module_stats = st.tabs(
        ["1. Samples and metadata", "2. Processing and QC", "3. Analysis and plots"]
    )

    with module_upload:
        page = st.radio(
            "Page",
            ["Upload", "Metadata management"],
            horizontal=True,
            key="upload_page",
        )
        if page == "Upload":
            _render_upload_page()
        else:
            _render_metadata_page()
        _render_save_analysis_panel(workspace, key_prefix="upload")

    run_label = st.session_state.get("run_label", _default_run_label())
    run_dir = workspace / _safe_name(run_label)
    config_path = run_dir / "pipeline.yaml"
    results_dir = run_dir / "results"
    reports_dir = run_dir / "reports"

    with module_qc:
        page = st.radio(
            "Page",
            ["Processing steps", "QC evaluation"],
            horizontal=True,
            key="qc_page",
        )
        analysis_metadata = _included_metadata_preview()
        if not _metadata_ready():
            st.info("Complete module 1 first: sample and metadata upload.")
        elif st.session_state.get("input_mode") == "Final processed counts":
            st.info(
                "You loaded final processed counts: the Processing/QC module is skipped. "
                "You can go directly to Analysis and plots."
            )
            processing_settings = _render_processing_settings(analysis_metadata)
            st.write(f"Analysis folder: `{run_dir}`")
            if st.button(
                "Save final counts",
                type="primary",
                disabled=not bool(processing_settings),
            ):
                _materialize_current_run(run_dir, processing_settings, st.session_state.get("qc_settings", {}))
                _record_analysis_savepoint(config_path, "normalize")
                st.success("Final counts saved. You can now run the statistical comparison.")
            _render_step_outputs(
                "Prepared final-count files",
                [
                    results_dir / "counts_normalized.csv",
                    results_dir / "metadata.csv",
                    results_dir / "sample_qc.csv",
                ],
                key_prefix="processed-final",
            )
        elif page == "Processing steps":
            _render_processing_steps_workflow(config_path, run_dir, results_dir, reports_dir, analysis_metadata)
        else:
            _render_qc_evaluation(results_dir)
        _render_save_analysis_panel(workspace, key_prefix="qc")

    with module_stats:
        page = st.radio(
            "Page",
            ["Comparisons", "Data visualization"],
            horizontal=True,
            key="stats_page",
        )
        analysis_metadata = _included_metadata_preview()
        if not _metadata_ready():
            st.info("Complete module 1 first: sample and metadata upload.")
        elif page == "Comparisons":
            settings = _render_analysis_settings(analysis_metadata)
            settings_ready = bool(settings) and bool(settings.get("contrasts"))
            if not settings_ready:
                st.warning("Define at least one valid contrast with two distinct groups.")
            col_save, col_run = st.columns(2)
            if col_save.button("Save analysis", disabled=not settings_ready, key="save_stats_analysis"):
                qc_settings = st.session_state.get("qc_settings", {})
                _materialize_current_run(run_dir, settings, qc_settings)
                _record_analysis_savepoint(config_path, _last_completed_step(config_path))
                st.success("Analysis saved. You can reload it from Saved analyses.")
            if col_run.button("Run statistical comparison", type="primary", disabled=not settings_ready):
                qc_settings = st.session_state.get("qc_settings", {})
                _materialize_current_run(run_dir, settings, qc_settings)
                steps = ["differential", "report"]
                if not (results_dir / "counts_normalized.csv").exists():
                    steps = ["prepare", "normalize", *steps]
                _execute_steps(config_path, steps)
            _render_generated_files(results_dir, reports_dir, key_prefix="stats")
        else:
            _render_results_explorer(results_dir, reports_dir)
        _render_save_analysis_panel(workspace, key_prefix="analysis")


def _default_run_label() -> str:
    return "analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe or _default_run_label()


def _render_entry_page() -> None:
    import streamlit as st

    col_new, col_load = st.columns(2)
    if col_new.button("Nuova analisi", type="primary", use_container_width=True):
        _clear_current_analysis_state()
        st.session_state["entry_action"] = "new_analysis"
        st.session_state["run_label"] = _default_run_label()
        st.rerun()
    if col_load.button("Carica analisi precedente", use_container_width=True):
        st.session_state["entry_action"] = "load_previous"
        st.rerun()
    if col_load.button("Manage saved analysis", use_container_width=True):
        st.session_state["entry_action"] = "manage_saved"
        st.rerun()


def _inject_upload_page_styles() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        div[data-testid="stFileUploaderDropzone"] {
            min-height: 220px;
            aspect-ratio: 1 / 1;
            align-items: center;
            justify-content: center;
        }
        div[data-testid="stFileUploaderDropzone"] > div {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_back_to_entry_button(label: str = "Torna alla pagina precedente") -> None:
    import streamlit as st

    _, back_col = st.columns([5, 1])
    if back_col.button(label, use_container_width=True):
        st.session_state.pop("entry_action", None)
        st.rerun()


def _render_saved_analysis_page(workspace: Path, manage_only: bool = False) -> None:
    import streamlit as st

    saved_runs = _saved_analysis_dirs(workspace)
    st.subheader("Manage saved analysis" if manage_only else "Carica analisi precedente")
    _render_back_to_entry_button()

    message = st.session_state.pop("analysis_memory_message", None)
    if message:
        st.success(message)

    if st.session_state.get("input_mode") == "Saved analysis":
        st.success(f"Analisi caricata: {st.session_state.get('run_label', '')}")
        loaded = st.session_state.get("loaded_analysis_dir")
        if loaded:
            st.caption(f"Cartella analisi: `{Path(str(loaded)).name}`")
        metadata = st.session_state.get("edited_metadata")
        if isinstance(metadata, pd.DataFrame):
            _render_loaded_samples_summary(
                metadata,
                title="Campioni caricati",
                pre_qc=_saved_analysis_pre_qc(Path(str(loaded))) if loaded else None,
            )
        col_continue, col_new = st.columns(2)
        if col_continue.button("Continua con questa analisi", type="primary", use_container_width=True):
            st.session_state["entry_action"] = "new_analysis"
            st.rerun()
        if col_new.button("Inizia una nuova analisi", use_container_width=True):
            _clear_current_analysis_state()
            st.session_state["entry_action"] = "new_analysis"
            st.session_state["run_label"] = _default_run_label()
            st.rerun()
        return

    if not saved_runs:
        st.info("Non ci sono ancora analisi salvate.")
        if st.button("Inizia nuova analisi", type="primary", use_container_width=True):
            _clear_current_analysis_state()
            st.session_state["entry_action"] = "new_analysis"
            st.session_state["run_label"] = _default_run_label()
            st.rerun()
        return

    if manage_only:
        _render_saved_analysis_manager(workspace, saved_runs)
        return

    labels = [_saved_analysis_label(run_dir) for run_dir in saved_runs]
    selected_label = st.selectbox("Analisi salvata", labels, key="saved_analysis_select")
    selected_run = saved_runs[labels.index(selected_label)]
    savepoint = _saved_analysis_savepoint(selected_run)
    if savepoint:
        st.caption(f"Ultimo step completato: {_format_savepoint(savepoint)}")
    _render_loaded_samples_summary(
        _read_saved_metadata(selected_run),
        title="Campioni nell'analisi",
        pre_qc=_saved_analysis_pre_qc(selected_run),
    )

    confirm_delete = st.checkbox(
        f"Conferma eliminazione definitiva di {selected_run.name}",
        key=f"confirm_delete_{selected_run.name}",
    )
    col_load, col_delete = st.columns(2)
    if col_load.button("Carica analisi", type="primary", use_container_width=True):
        _load_saved_analysis(selected_run)
        st.rerun()
    if col_delete.button(
        "Elimina analisi",
        disabled=not confirm_delete,
        use_container_width=True,
    ):
        _delete_saved_analysis(workspace, selected_run)
        st.session_state["analysis_memory_message"] = f"Analisi '{selected_run.name}' eliminata."
        st.rerun()


def _render_saved_analysis_manager(workspace: Path, saved_runs: list[Path]) -> None:
    import streamlit as st

    selected_for_delete = []
    for run_dir in saved_runs:
        modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        title = f"{run_dir.name} - {modified}"
        with st.expander(title, expanded=False):
            selected = st.checkbox(
                "Seleziona per cancellazione multipla",
                key=f"select_delete_saved_{run_dir.name}",
            )
            if selected:
                selected_for_delete.append(run_dir)

            metadata = _read_saved_metadata(run_dir)
            if metadata.empty:
                st.caption("Nessun samplesheet/metadata disponibile.")
            else:
                st.dataframe(metadata, use_container_width=True, hide_index=True)

            rename_value = st.text_input(
                "Nuovo nome",
                value=run_dir.name,
                key=f"rename_saved_{run_dir.name}",
            )
            col_load, col_rename, col_delete = st.columns(3)
            if col_load.button("Carica", key=f"load_saved_{run_dir.name}", use_container_width=True):
                _load_saved_analysis(run_dir)
                st.rerun()
            if col_rename.button("Rename", key=f"rename_button_{run_dir.name}", use_container_width=True):
                try:
                    new_path = _rename_saved_analysis(workspace, run_dir, rename_value)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["analysis_memory_message"] = (
                        f"Analisi '{run_dir.name}' rinominata in '{new_path.name}'."
                    )
                    st.rerun()
            if col_delete.button("Delete", key=f"delete_saved_{run_dir.name}", use_container_width=True):
                _delete_saved_analysis(workspace, run_dir)
                st.session_state["analysis_memory_message"] = f"Analisi '{run_dir.name}' eliminata."
                st.rerun()

    if selected_for_delete:
        st.divider()
        confirm_bulk = st.checkbox("Conferma cancellazione delle analisi selezionate")
        if st.button(
            f"Delete selected ({len(selected_for_delete)})",
            type="primary",
            disabled=not confirm_bulk,
            use_container_width=True,
        ):
            deleted_names = [run.name for run in selected_for_delete]
            for run_dir in selected_for_delete:
                _delete_saved_analysis(workspace, run_dir)
            st.session_state["analysis_memory_message"] = (
                "Analisi eliminate: " + ", ".join(deleted_names)
            )
            st.rerun()


def _render_save_analysis_panel(workspace: Path, key_prefix: str) -> None:
    import streamlit as st

    if st.session_state.get("entry_action") is None:
        return
    metadata = _included_metadata_preview()
    if not isinstance(metadata, pd.DataFrame) or metadata.empty or "sample_id" not in metadata.columns:
        return

    st.divider()
    with st.expander("Save analysis", expanded=False):
        settings = _settings_for_background_save(metadata)
        qc_settings = st.session_state.get("qc_settings", {})
        if not isinstance(qc_settings, dict):
            qc_settings = {}
        if not settings:
            st.warning("Completa almeno una colonna metadata oltre a sample_id prima di salvare.")
            return

        current_label = str(st.session_state.get("run_label", _default_run_label()))
        loaded = st.session_state.get("loaded_analysis_dir")
        overwrite_target = Path(str(loaded)) if loaded else workspace / _safe_name(current_label)
        new_name = st.text_input(
            "Nome nuova analisi",
            value=_next_analysis_name(workspace, current_label),
            key=f"{key_prefix}_save_as_new_analysis_name",
        )
        col_new, col_overwrite = st.columns(2)
        if col_new.button(
            "Salva come nuova analisi",
            use_container_width=True,
            key=f"{key_prefix}_save_as_new_analysis",
        ):
            target = workspace / _safe_name(new_name)
            try:
                _prepare_save_target(target, overwrite=False)
            except ValueError as exc:
                st.error(str(exc))
                return
            _materialize_current_run(target, settings, qc_settings)
            _record_analysis_savepoint(target / "pipeline.yaml", _last_completed_step(target / "pipeline.yaml"))
            st.session_state["run_label"] = target.name
            st.session_state["loaded_analysis_dir"] = str(target)
            st.success(f"Analisi salvata come '{target.name}'.")
        if col_overwrite.button(
            "Sovrascrivi analisi in uso",
            use_container_width=True,
            key=f"{key_prefix}_overwrite_current_analysis",
        ):
            _prepare_save_target(overwrite_target, overwrite=True)
            _materialize_current_run(overwrite_target, settings, qc_settings)
            _record_analysis_savepoint(
                overwrite_target / "pipeline.yaml",
                _last_completed_step(overwrite_target / "pipeline.yaml"),
            )
            st.session_state["run_label"] = overwrite_target.name
            st.session_state["loaded_analysis_dir"] = str(overwrite_target)
            st.success(f"Analisi '{overwrite_target.name}' aggiornata.")


def _prepare_save_target(target: Path, overwrite: bool) -> None:
    loaded = st_session_get("loaded_analysis_dir")
    source = Path(str(loaded)) if loaded else None
    if target.exists() and not overwrite:
        raise ValueError(f"Esiste gia' un'analisi chiamata '{target.name}'. Scegli un altro nome.")
    if source is None or source.resolve() == target.resolve():
        return
    if target.exists():
        return
    if source.exists():
        shutil.copytree(source, target)


def _settings_for_background_save(metadata: pd.DataFrame) -> dict[str, object]:
    if not isinstance(metadata, pd.DataFrame) or "sample_id" not in metadata.columns:
        return {}
    analysis_settings = st_session_get("analysis_settings", {})
    if isinstance(analysis_settings, dict) and analysis_settings.get("contrasts"):
        return analysis_settings
    candidate_group_cols = [col for col in metadata.columns if col != "sample_id"]
    if not candidate_group_cols:
        return {}
    processing_settings = st_session_get("processing_settings", {})
    if not isinstance(processing_settings, dict):
        processing_settings = {}
    saved_config = _current_saved_config()
    saved_analysis = saved_config.get("analysis", {}) if isinstance(saved_config.get("analysis"), dict) else {}
    saved_normalization = (
        saved_config.get("normalization", {}) if isinstance(saved_config.get("normalization"), dict) else {}
    )
    group_col = str(
        processing_settings.get("group_column")
        or saved_analysis.get("group_column")
        or candidate_group_cols[0]
    )
    if group_col not in metadata.columns:
        group_col = candidate_group_cols[0]
    groups = [str(value) for value in metadata[group_col].dropna().unique()]
    reference_group = str(processing_settings.get("reference_group") or saved_analysis.get("reference_group") or "")
    case_group = str(processing_settings.get("case_group") or saved_analysis.get("case_group") or "")
    if reference_group not in groups:
        reference_group = groups[0] if groups else ""
    if case_group not in groups:
        case_group = groups[1] if len(groups) > 1 else reference_group
    return {
        "group_column": group_col,
        "group_columns": [group_col],
        "reference_group": reference_group,
        "case_group": case_group,
        "contrasts": saved_analysis.get("contrasts", []),
        "min_count": _int_default(processing_settings.get("min_count") or saved_analysis.get("min_count"), 10, 0),
        "min_samples": _int_default(
            processing_settings.get("min_samples") or saved_analysis.get("min_samples"),
            2,
            1,
        ),
        "top_variable_genes": _int_default(
            processing_settings.get("top_variable_genes") or saved_analysis.get("top_variable_genes"),
            50,
            5,
        ),
        "normalization_method": str(
            processing_settings.get("normalization_method")
            or saved_normalization.get("method")
            or "nanostringnorm"
        ),
    }


def _next_analysis_name(workspace: Path, current_label: str) -> str:
    base = _safe_name(current_label) or _default_run_label()
    candidate = f"{base}_copy"
    index = 2
    while (workspace / candidate).exists():
        candidate = f"{base}_copy_{index}"
        index += 1
    return candidate


def _render_analysis_memory(workspace: Path) -> None:
    import streamlit as st

    saved_runs = _saved_analysis_dirs(workspace)
    with st.expander("Saved analyses", expanded=False):
        message = st.session_state.pop("analysis_memory_message", None)
        if message:
            st.success(message)

        if not saved_runs:
            st.caption("No saved analysis found yet.")
        else:
            labels = [_saved_analysis_label(run_dir) for run_dir in saved_runs]
            selected_label = st.selectbox("Previous analysis", labels, key="saved_analysis_select")
            selected_run = saved_runs[labels.index(selected_label)]
            savepoint = _saved_analysis_savepoint(selected_run)
            if savepoint:
                st.caption(f"Savepoint: {_format_savepoint(savepoint)}")
            confirm_delete = st.checkbox(
                f"Confirm permanent deletion of {selected_run.name}",
                key=f"confirm_delete_{selected_run.name}",
            )
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("Load analysis", key="load_saved_analysis"):
                _load_saved_analysis(selected_run)
                st.rerun()
            if col_b.button("Start new analysis", key="start_new_analysis"):
                _clear_current_analysis_state()
                st.session_state["run_label"] = _default_run_label()
                st.rerun()
            if col_c.button(
                "Delete analysis",
                key="delete_saved_analysis",
                disabled=not confirm_delete,
            ):
                loaded = st.session_state.get("loaded_analysis_dir")
                deleting_current = (
                    st.session_state.get("run_label") == selected_run.name
                    or (loaded and Path(str(loaded)).resolve() == selected_run.resolve())
                )
                _delete_saved_analysis(workspace, selected_run)
                if deleting_current:
                    _clear_current_analysis_state()
                    st.session_state["run_label"] = _default_run_label()
                st.session_state["analysis_memory_message"] = (
                    f"Analysis '{selected_run.name}' deleted."
                )
                st.rerun()

        loaded = st.session_state.get("loaded_analysis_dir")
        if loaded:
            st.caption(f"Current saved analysis: `{Path(str(loaded)).name}`")


def _saved_analysis_dirs(workspace: Path) -> list[Path]:
    if not workspace.exists():
        return []
    runs = [
        path
        for path in workspace.iterdir()
        if path.is_dir() and ((path / "pipeline.yaml").exists() or (path / "results").exists())
    ]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def _delete_saved_analysis(workspace: Path, run_dir: Path) -> None:
    workspace_path = workspace.resolve()
    run_path = run_dir.resolve()
    if run_dir.is_symlink() or run_path.parent != workspace_path:
        raise ValueError("Saved analysis must be a direct folder inside the UI workspace.")
    if not run_path.is_dir():
        raise FileNotFoundError(f"Saved analysis not found: {run_dir}")
    shutil.rmtree(run_path)


def _rename_saved_analysis(workspace: Path, run_dir: Path, new_name: str) -> Path:
    workspace_path = workspace.resolve()
    run_path = run_dir.resolve()
    if run_dir.is_symlink() or run_path.parent != workspace_path:
        raise ValueError("Saved analysis must be a direct folder inside the UI workspace.")
    safe_name = _safe_name(new_name)
    target = workspace / safe_name
    target_path = target.resolve()
    if target_path == run_path:
        return run_dir
    if target_path.parent != workspace_path:
        raise ValueError("Rename target must stay inside the UI workspace.")
    if target.exists():
        raise ValueError(f"An analysis named '{safe_name}' already exists.")
    run_dir.rename(target)
    return target


def _saved_analysis_label(run_dir: Path) -> str:
    savepoint = _saved_analysis_savepoint(run_dir)
    if savepoint:
        status = _savepoint_step_label(savepoint)
    else:
        markers = []
        if (run_dir / "results" / "differential_expression.csv").exists():
            markers.append("DE")
        elif (run_dir / "results" / "counts_normalized.csv").exists():
            markers.append("normalized")
        elif (run_dir / "results" / "metadata.csv").exists():
            markers.append("metadata")
        status = ", ".join(markers) if markers else "configured"
    modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{run_dir.name} - {status} - {modified}"


def _saved_analysis_savepoint(run_dir: Path) -> dict[str, object]:
    config = _read_saved_config(run_dir)
    ui_state = config.get("ui", {}) if isinstance(config, dict) else {}
    savepoint = ui_state.get("savepoint", {}) if isinstance(ui_state, dict) else {}
    return savepoint if isinstance(savepoint, dict) else {}


def _savepoint_step_label(savepoint: dict[str, object]) -> str:
    step = str(savepoint.get("last_completed_step") or "configured")
    return STEP_LABELS.get(step, step)


def _format_savepoint(savepoint: dict[str, object]) -> str:
    label = _savepoint_step_label(savepoint)
    completed_at = savepoint.get("last_completed_at")
    if completed_at:
        return f"{label} ({completed_at})"
    return label


def _clear_current_analysis_state() -> None:
    import streamlit as st

    for key in [
        "input_mode",
        "counts_file",
        "metadata_file",
        "samplesheet_file",
        "rcc_files",
        "rlf_file",
        "edited_samplesheet",
        "loaded_analysis_dir",
        "excluded_sample_ids",
        "sample_id_renames",
        "upload_input_format",
        "metadata_signature",
        "processed_counts_signature",
        "samplesheet_signature",
        "processing_settings",
        "analysis_settings",
        "qc_settings",
        "processing_group_column",
        "processing_norm",
        "processing_top_variable_genes",
        "qc_mode",
        "qc_low_library_fraction",
        "qc_low_detected_fraction",
        "qc_low_positive_fraction",
        "qc_background_sd_multiplier",
        "qc_negative_control_mad_multiplier",
        "qc_min_fov_counted_fraction",
        "qc_binding_density_min",
        "qc_binding_density_max",
        "qc_fail_library_fraction",
        "qc_fail_detected_fraction",
        "qc_filter_min_count",
        "qc_filter_min_samples",
        "qc_strict_exclusion",
    ]:
        st.session_state.pop(key, None)
    _clear_metadata_state()


def _load_saved_analysis(run_dir: Path) -> None:
    import streamlit as st

    metadata = _read_saved_metadata(run_dir)
    config = _read_saved_config(run_dir)
    _clear_current_analysis_state()
    st.session_state["run_label"] = run_dir.name
    st.session_state["input_mode"] = "Saved analysis"
    st.session_state["loaded_analysis_dir"] = str(run_dir)
    st.session_state["edited_metadata"] = metadata
    st.session_state["metadata_preview"] = metadata
    analysis = config.get("analysis", {}) if isinstance(config.get("analysis"), dict) else {}
    normalization = config.get("normalization", {}) if isinstance(config.get("normalization"), dict) else {}
    st.session_state["processing_settings"] = {
        **analysis,
        "normalization_method": normalization.get("method", "nanostringnorm"),
    }
    st.session_state["analysis_settings"] = {
        **analysis,
        "normalization_method": normalization.get("method", "nanostringnorm"),
    }
    st.session_state["qc_settings"] = config.get("qc", {}) if isinstance(config, dict) else {}


def _read_saved_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "pipeline.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_saved_config(config_path: Path, config: dict[str, object]) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, Dumper=_QuotedStringDumper, sort_keys=False)


class _QuotedStringDumper(yaml.SafeDumper):
    pass


def _represent_config_string(dumper: yaml.SafeDumper, value: str):
    style = "'" if YAML_AMBIGUOUS_SCALAR_RE.match(value.strip()) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_QuotedStringDumper.add_representer(str, _represent_config_string)


def _record_analysis_savepoint(config_path: Path, step: str) -> None:
    config = _read_saved_config(config_path.parent)
    if not config:
        return

    ui_state = config.setdefault("ui", {})
    if not isinstance(ui_state, dict):
        ui_state = {}
        config["ui"] = ui_state

    savepoint = ui_state.setdefault("savepoint", {})
    if not isinstance(savepoint, dict):
        savepoint = {}
        ui_state["savepoint"] = savepoint

    completed_steps = savepoint.get("completed_steps", [])
    if not isinstance(completed_steps, list):
        completed_steps = []
    if step != "configured" and step not in completed_steps:
        completed_steps.append(step)

    savepoint["last_completed_step"] = step
    savepoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
    savepoint["completed_steps"] = completed_steps
    _write_saved_config(config_path, config)


def _last_completed_step(config_path: Path) -> str:
    config = _read_saved_config(config_path.parent)
    ui_state = config.get("ui", {}) if isinstance(config, dict) else {}
    savepoint = ui_state.get("savepoint", {}) if isinstance(ui_state, dict) else {}
    if not isinstance(savepoint, dict):
        return "configured"
    return str(savepoint.get("last_completed_step") or "configured")


def _current_saved_config() -> dict[str, object]:
    loaded = st_session_get("loaded_analysis_dir")
    if not loaded:
        return {}
    return _read_saved_config(Path(str(loaded)))


def _read_saved_metadata(run_dir: Path) -> pd.DataFrame:
    results_metadata = run_dir / "results" / "metadata.csv"
    if results_metadata.exists():
        return pd.read_csv(results_metadata)

    config = _read_saved_config(run_dir)
    inputs = config.get("inputs", {}) if isinstance(config, dict) else {}
    metadata_path = inputs.get("metadata")
    if metadata_path:
        path = _resolve_saved_path(run_dir, str(metadata_path))
        if path.exists():
            return _read_delimited_path(path)

    samplesheet_path = inputs.get("samplesheet")
    if samplesheet_path:
        path = _resolve_saved_path(run_dir, str(samplesheet_path))
        if path.exists():
            return metadata_from_samplesheet(path)

    return pd.DataFrame(columns=["sample_id"])


def _resolve_saved_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return run_dir / path


def _read_delimited_path(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _upload_signature(uploaded_files: list) -> tuple[tuple[str, int, str], ...]:
    signature = []
    for uploaded_file in uploaded_files:
        size = getattr(uploaded_file, "size", 0)
        file_type = getattr(uploaded_file, "type", "")
        signature.append((uploaded_file.name, int(size or 0), str(file_type)))
    return tuple(signature)


def _duplicate_uploaded_file_names(uploaded_files: list) -> list[str]:
    seen = set()
    duplicates = []
    for uploaded_file in uploaded_files:
        name = Path(str(getattr(uploaded_file, "name", ""))).name
        normalized_name = name.casefold()
        if normalized_name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(normalized_name)
    return duplicates


def _render_duplicate_upload_error(uploaded_files: list) -> bool:
    import streamlit as st

    duplicates = _duplicate_uploaded_file_names(uploaded_files)
    if not duplicates:
        return False
    st.error(
        "File duplicati caricati: "
        + ", ".join(duplicates)
        + ". Rimuovi i duplicati prima di continuare."
    )
    return True


def _clear_metadata_state() -> None:
    import streamlit as st

    for key in [
        "edited_metadata",
        "metadata_preview",
        "metadata_table",
        "processed_counts_metadata_table",
        "rcc_metadata_table",
        "samplesheet_metadata_table",
        "rcc_samplesheet_table",
        "uploaded_samplesheet_table",
    ]:
        st.session_state.pop(key, None)


def _render_upload_page() -> None:
    import streamlit as st

    _render_back_to_entry_button()

    if st.session_state.get("input_mode") == "Saved analysis":
        st.text_input(
            "Analysis name",
            value=st.session_state.get("run_label", _default_run_label()),
            disabled=True,
        )
        st.info("A saved analysis is loaded. You can edit metadata, change comparison settings, and rerun results.")
        metadata = st.session_state.get("edited_metadata")
        if isinstance(metadata, pd.DataFrame):
            loaded = st.session_state.get("loaded_analysis_dir")
            _render_loaded_samples_summary(
                metadata,
                pre_qc=_saved_analysis_pre_qc(Path(str(loaded))) if loaded else None,
            )
        if st.button("Switch to new upload"):
            _clear_current_analysis_state()
            st.session_state["run_label"] = _default_run_label()
            st.rerun()
        return

    input_modes = ["RCC + RLF", "Counts + metadata"]
    saved_input_mode = st.session_state.get("input_mode")
    input_mode = st.selectbox(
        "Input format",
        input_modes,
        index=input_modes.index(saved_input_mode) if saved_input_mode in input_modes else 0,
        key="upload_input_format",
    )
    run_label = st.text_input("Analysis name", value=st.session_state.get("run_label", _default_run_label()))
    st.session_state["run_label"] = run_label

    if input_mode == "RCC + RLF":
        upload_cols = st.columns(2)
        with upload_cols[0]:
            uploaded_rcc_files = st.file_uploader("File RCC", type=["rcc", "RCC"], accept_multiple_files=True)
        with upload_cols[1]:
            uploaded_rlf_file = st.file_uploader("Panel design / RLF", type=["rlf", "RLF"])
        rcc_files = uploaded_rcc_files or st.session_state.get("rcc_files", [])
        rlf_file = uploaded_rlf_file or st.session_state.get("rlf_file")
        if rcc_files:
            if _render_duplicate_upload_error(rcc_files):
                return
            signature = _upload_signature(rcc_files)
            if st.session_state.get("samplesheet_signature") != signature:
                _clear_metadata_state()
                st.session_state["edited_samplesheet"] = _build_samplesheet_from_rcc_uploads(rcc_files)
                st.session_state["samplesheet_signature"] = signature
            samplesheet = st.session_state["edited_samplesheet"]
            edited_summary = _render_loaded_samples_summary(
                _samplesheet_loaded_samples_frame(samplesheet),
                pre_qc=_rcc_upload_pre_qc(rcc_files, samplesheet),
                editable_sample_id=True,
                key="rcc_loaded_samples",
            )
            edited = _apply_loaded_sample_ids_to_samplesheet(samplesheet, edited_summary)
            st.session_state["input_mode"] = input_mode
            st.session_state["rcc_files"] = rcc_files
            st.session_state["rlf_file"] = rlf_file
            st.session_state["edited_samplesheet"] = edited
            st.success("RCC files loaded. Open Metadata management to define groups and covariates.")
    elif input_mode == "Counts + metadata":
        upload_cols = st.columns(2)
        with upload_cols[0]:
            uploaded_counts_file = st.file_uploader("Counts NanoString", type=["csv", "tsv", "txt"])
        with upload_cols[1]:
            uploaded_metadata_file = st.file_uploader("Experiment design / metadata", type=["csv", "tsv", "txt"])
        counts_file = uploaded_counts_file or st.session_state.get("counts_file")
        metadata_file = uploaded_metadata_file or st.session_state.get("metadata_file")
        if counts_file and metadata_file:
            if _render_duplicate_upload_error([counts_file, metadata_file]):
                return
            signature = _upload_signature([counts_file, metadata_file])
            if st.session_state.get("metadata_signature") != signature:
                _clear_metadata_state()
                st.session_state["metadata_signature"] = signature
            st.session_state["input_mode"] = input_mode
            st.session_state["counts_file"] = counts_file
            st.session_state["metadata_file"] = metadata_file
            counts = _normalize_counts_preview(_read_uploaded_table(counts_file))
            _render_loaded_samples_summary(
                _read_uploaded_table(metadata_file),
                pre_qc=_counts_matrix_pre_qc(counts),
            )
            st.success("Counts and metadata loaded. Open Metadata management to edit them.")
    elif input_mode == "Final processed counts":
        upload_cols = st.columns(2)
        with upload_cols[0]:
            counts_file = st.file_uploader("Final normalized/processed counts", type=["csv", "tsv", "txt"])
        with upload_cols[1]:
            metadata_file = st.file_uploader("Metadata", type=["csv", "tsv", "txt"])
        if counts_file:
            counts = _normalize_counts_preview(_read_uploaded_table(counts_file))
            template = _metadata_template_from_counts(counts)
            st.download_button(
                "Download metadata template",
                template.to_csv(index=False).encode("utf-8"),
                file_name="metadata_template.csv",
                mime="text/csv",
                key="processed_counts_metadata_template",
            )
            with st.expander("Final counts preview", expanded=False):
                st.dataframe(counts.head(20), use_container_width=True)

        if counts_file and metadata_file:
            signature = _upload_signature([counts_file, metadata_file])
            if st.session_state.get("processed_counts_signature") != signature:
                _clear_metadata_state()
                st.session_state["processed_counts_signature"] = signature
            st.session_state["input_mode"] = input_mode
            st.session_state["counts_file"] = counts_file
            st.session_state["metadata_file"] = metadata_file
            _render_loaded_samples_summary(
                _read_uploaded_table(metadata_file),
                pre_qc=_counts_matrix_pre_qc(counts),
            )
            st.success("Final counts and metadata loaded. You can go directly to Analysis and plots.")
    else:
        upload_cols = st.columns(2)
        with upload_cols[0]:
            samplesheet_file = st.file_uploader("Samplesheet nf-core", type=["csv", "tsv", "txt"])
        with upload_cols[1]:
            st.empty()
        if samplesheet_file:
            signature = _upload_signature([samplesheet_file])
            if st.session_state.get("samplesheet_signature") != signature:
                st.session_state["edited_samplesheet"] = _read_uploaded_table(samplesheet_file)
                st.session_state["samplesheet_signature"] = signature
                _clear_metadata_state()
            samplesheet = st.session_state["edited_samplesheet"]
            edited = _render_samplesheet_upload_editor(samplesheet, key="uploaded_samplesheet")
            st.session_state["input_mode"] = input_mode
            st.session_state["samplesheet_file"] = samplesheet_file
            st.session_state["edited_samplesheet"] = edited
            _render_loaded_samples_summary(_metadata_preview_from_samplesheet_frame(edited))
            st.success("Samplesheet loaded. Open Metadata management to define groups and covariates.")


def _render_metadata_page() -> None:
    import streamlit as st

    _render_back_to_entry_button()

    input_mode = st.session_state.get("input_mode")
    if not input_mode:
        st.info("Upload samples and design files first.")
        return

    if input_mode == "RCC + RLF":
        samplesheet = st.session_state.get("edited_samplesheet")
        metadata = _metadata_preview_from_samplesheet_frame(samplesheet)
        metadata = _filter_metadata_for_downstream_pages(metadata)
        metadata = _render_metadata_template_tools(metadata, key="rcc_metadata")
        edited = _render_metadata_editor(metadata, key="rcc_metadata")
        st.session_state["edited_metadata"] = edited
        st.session_state["metadata_preview"] = edited
    elif input_mode == "Samplesheet RCC":
        samplesheet = st.session_state.get("edited_samplesheet")
        metadata = _metadata_preview_from_samplesheet_frame(samplesheet)
        metadata = _filter_metadata_for_downstream_pages(metadata)
        metadata = _render_metadata_template_tools(metadata, key="samplesheet_metadata")
        edited = _render_metadata_editor(metadata, key="samplesheet_metadata")
        st.session_state["edited_metadata"] = edited
        st.session_state["metadata_preview"] = edited
    elif input_mode == "Counts + metadata":
        counts = _normalize_counts_preview(_read_uploaded_table(st.session_state["counts_file"]))
        metadata = st.session_state.get("edited_metadata")
        if not isinstance(metadata, pd.DataFrame):
            metadata = _read_uploaded_table(st.session_state["metadata_file"])
        metadata = _filter_metadata_for_downstream_pages(metadata)
        metadata = _render_metadata_template_tools(metadata, key="metadata")
        edited = _render_metadata_only_preview(counts, metadata, key="metadata")
        st.session_state["edited_metadata"] = edited
        st.session_state["metadata_preview"] = edited
    elif input_mode == "Saved analysis":
        metadata = st.session_state.get("edited_metadata")
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(columns=["sample_id"])
        metadata = _filter_metadata_for_downstream_pages(metadata)
        metadata = _render_metadata_template_tools(metadata, key="saved_analysis_metadata")
        edited = _render_metadata_editor(metadata, key="saved_analysis_metadata")
        st.session_state["edited_metadata"] = edited
        st.session_state["metadata_preview"] = edited
    else:
        counts = _normalize_counts_preview(_read_uploaded_table(st.session_state["counts_file"]))
        metadata = st.session_state.get("edited_metadata")
        if not isinstance(metadata, pd.DataFrame):
            metadata = _read_uploaded_table(st.session_state["metadata_file"])
        metadata = _filter_metadata_for_downstream_pages(metadata)
        metadata = _render_metadata_template_tools(metadata, key="processed_counts_metadata")
        edited = _render_processed_counts_preview(counts, metadata)
        st.session_state["edited_metadata"] = edited
        st.session_state["metadata_preview"] = edited


def _metadata_ready() -> bool:
    metadata = _included_metadata_preview()
    return isinstance(metadata, pd.DataFrame) and "sample_id" in metadata.columns and len(metadata) > 0


def _included_metadata_preview() -> pd.DataFrame:
    metadata = st_session_get("metadata_preview")
    if not isinstance(metadata, pd.DataFrame):
        return pd.DataFrame(columns=["sample_id"])
    if "sample_id" not in metadata.columns:
        return metadata
    try:
        return _filter_metadata_by_included_samples(metadata)
    except ValueError:
        return pd.DataFrame(columns=metadata.columns)


def _processing_defaults() -> tuple[dict[str, object], dict[str, object]]:
    saved_config = _current_saved_config()
    saved_analysis = saved_config.get("analysis", {}) if isinstance(saved_config.get("analysis"), dict) else {}
    saved_normalization = (
        saved_config.get("normalization", {}) if isinstance(saved_config.get("normalization"), dict) else {}
    )
    session_processing = st_session_get("processing_settings", {})
    session_qc = st_session_get("qc_settings", {})
    if not isinstance(session_processing, dict):
        session_processing = {}
    if not isinstance(session_qc, dict):
        session_qc = {}
    processing = {
        **saved_analysis,
        "normalization_method": saved_normalization.get("method"),
        **session_processing,
    }
    qc = {
        **(saved_config.get("qc", {}) if isinstance(saved_config.get("qc"), dict) else {}),
        **session_qc,
    }
    return processing, qc


def _render_processing_settings(metadata: pd.DataFrame) -> dict[str, object]:
    import streamlit as st

    st.subheader("Main parameters")
    candidate_group_cols = [col for col in metadata.columns if col != "sample_id"]
    if not candidate_group_cols:
        st.error("Add at least one metadata column to continue.")
        return {}
    processing_defaults, _ = _processing_defaults()
    saved_group = str(processing_defaults.get("group_column") or "")
    group_index = candidate_group_cols.index(saved_group) if saved_group in candidate_group_cols else 0
    group_col = st.selectbox(
        "Metadata column for QC/PCA color",
        candidate_group_cols,
        index=group_index,
        key="processing_group_column",
    )
    groups = [str(value) for value in metadata[group_col].dropna().unique()]
    reference_group = groups[0] if groups else ""
    case_group = groups[1] if len(groups) > 1 else reference_group

    method = st.selectbox(
        "Normalization",
        ["nanostringnorm", "library_size", "hk_geomean_all", "hk_geomean_geNorm"],
        index=_option_index(
            ["nanostringnorm", "library_size", "hk_geomean_all", "hk_geomean_geNorm"],
            processing_defaults.get("normalization_method"),
            0,
        ),
        key="processing_norm",
    )
    top_variable = st.number_input(
        "Top variable genes for report",
        min_value=5,
        value=_int_default(processing_defaults.get("top_variable_genes"), 50, minimum=5),
        step=5,
        key="processing_top_variable_genes",
    )

    settings = {
        "group_column": group_col,
        "group_columns": [group_col],
        "reference_group": reference_group,
        "case_group": case_group,
        "min_count": _int_default(processing_defaults.get("min_count"), 10, minimum=0),
        "min_samples": _int_default(processing_defaults.get("min_samples"), 2, minimum=1),
        "top_variable_genes": int(top_variable),
        "normalization_method": method,
    }
    st.session_state["processing_settings"] = settings
    return settings


def _render_qc_settings() -> dict[str, object]:
    import streamlit as st

    st.subheader("QC parameters")
    _, qc_defaults = _processing_defaults()
    qc_mode = st.selectbox(
        "QC mode",
        ["easy_strict", "bruker_like"],
        index=_option_index(["easy_strict", "bruker_like"], qc_defaults.get("qc_mode"), 0),
        key="qc_mode",
    )
    col_a, col_b, col_c = st.columns(3)
    low_library = col_a.slider(
        "Minimum library vs median",
        0.0,
        1.0,
        _float_default(qc_defaults.get("low_library_fraction"), 0.5),
        0.05,
        key="qc_low_library_fraction",
    )
    low_detected = col_b.slider(
        "Minimum detected probes vs median",
        0.0,
        1.0,
        _float_default(qc_defaults.get("low_detected_fraction"), 0.5),
        0.05,
        key="qc_low_detected_fraction",
    )
    low_positive = col_c.slider(
        "Minimum positive controls vs median",
        0.0,
        1.0,
        _float_default(qc_defaults.get("low_positive_fraction"), 0.5),
        0.05,
        key="qc_low_positive_fraction",
    )

    with st.expander("Advanced QC options"):
        background_sd = st.number_input(
            "Background: negative mean + N SD",
            0.0,
            10.0,
            _float_default(qc_defaults.get("background_sd_multiplier"), 2.0),
            0.25,
            key="qc_background_sd_multiplier",
        )
        negative_mad = st.number_input(
            "Negative control threshold: median + N MAD",
            0.0,
            10.0,
            _float_default(qc_defaults.get("negative_control_mad_multiplier"), 3.0),
            0.25,
            key="qc_negative_control_mad_multiplier",
        )
        min_fov = st.slider(
            "Minimum counted FOV",
            0.0,
            1.0,
            _float_default(qc_defaults.get("min_fov_counted_fraction"), 0.75),
            0.05,
            key="qc_min_fov_counted_fraction",
        )
        binding_min = st.number_input(
            "Minimum binding density",
            0.0,
            10.0,
            _float_default(qc_defaults.get("binding_density_min"), 0.05),
            0.05,
            key="qc_binding_density_min",
        )
        binding_max = st.number_input(
            "Maximum binding density",
            0.0,
            10.0,
            _float_default(qc_defaults.get("binding_density_max"), 2.25),
            0.05,
            key="qc_binding_density_max",
        )
        fail_library = st.slider(
            "FAIL minimum library vs median",
            0.0,
            1.0,
            _float_default(qc_defaults.get("fail_library_fraction"), 0.25),
            0.05,
            key="qc_fail_library_fraction",
        )
        fail_detected = st.slider(
            "FAIL detected probes vs median",
            0.0,
            1.0,
            _float_default(qc_defaults.get("fail_detected_fraction"), 0.25),
            0.05,
            key="qc_fail_detected_fraction",
        )
        filter_min_count = st.number_input(
            "Gene filter: minimum count",
            0,
            1000,
            _int_default(qc_defaults.get("filter_min_count"), 10, minimum=0),
            1,
            key="qc_filter_min_count",
        )
        filter_min_samples = st.number_input(
            "Gene filter: minimum samples",
            1,
            1000,
            _int_default(qc_defaults.get("filter_min_samples"), 2, minimum=1),
            1,
            key="qc_filter_min_samples",
        )
        strict_exclusion = st.checkbox(
            "Strict exclusion in Bruker-like mode",
            value=bool(qc_defaults.get("strict_exclusion", False)),
            help="When disabled, Bruker-like FOV, binding-density, and positive-control findings remain informative warnings.",
            key="qc_strict_exclusion",
        )

    qc_settings = {
        "qc_mode": qc_mode,
        "low_library_fraction": float(low_library),
        "low_detected_fraction": float(low_detected),
        "low_positive_fraction": float(low_positive),
        "background_sd_multiplier": float(background_sd),
        "negative_control_mad_multiplier": float(negative_mad),
        "min_fov_counted_fraction": float(min_fov),
        "binding_density_min": float(binding_min),
        "binding_density_max": float(binding_max),
        "fail_library_fraction": float(fail_library),
        "fail_detected_fraction": float(fail_detected),
        "filter_min_count": int(filter_min_count),
        "filter_min_samples": int(filter_min_samples),
        "strict_exclusion": bool(strict_exclusion),
    }
    st.session_state["qc_settings"] = qc_settings
    return qc_settings


def _render_qc_evaluation(results_dir: Path) -> None:
    import plotly.express as px
    import streamlit as st

    qc_path = results_dir / "qc_summary.csv"
    if not qc_path.exists():
        st.info("Run processing/QC first to view sample QC evaluation.")
        return

    qc = pd.read_csv(qc_path)
    st.subheader("Sample QC evaluation")
    status_counts = qc.get("qc_status", pd.Series(dtype=str)).value_counts().to_dict()
    cols = st.columns(4)
    cols[0].metric("Samples", len(qc))
    cols[1].metric("PASS", status_counts.get("PASS", 0))
    cols[2].metric("WARN", status_counts.get("WARN", 0))
    cols[3].metric("FAIL", status_counts.get("FAIL", 0))

    issues = qc[qc.get("qc_status", "").isin(["WARN", "FAIL"])] if "qc_status" in qc else pd.DataFrame()
    if issues.empty:
        st.success("No problematic samples under the current QC thresholds.")
    else:
        st.warning("Samples con problemi QC rilevati.")
        st.dataframe(issues, use_container_width=True)

    chart_cols = [
        col
        for col in ["library_size", "detected_endogenous_probes", "positive_control_sum", "negative_control_mean"]
        if col in qc.columns
    ]
    selected_sample = None
    if chart_cols:
        selected_metric = st.selectbox("Metric to display", chart_cols)
        hover_cols = _unique_columns(
            [
                "qc_warnings",
                "qc_fail_reasons",
                "background_threshold",
                "positive_control_cv",
                "positive_control_linearity",
                "negative_control_mean",
                "negative_control_sd",
                "hk_geomean",
                "fov_registration_rate",
                "binding_density",
                "endogenous_above_background",
            ],
            available=qc.columns,
            exclude={selected_metric},
        )
        plot_columns = _unique_columns(
            ["sample_id", "qc_status", selected_metric, *hover_cols],
            available=qc.columns,
        )
        plot_qc = qc[plot_columns].copy()
        plot_qc["qc_status"] = plot_qc["qc_status"].fillna("PASS")
        plot_qc["qc_reason"] = plot_qc.apply(_qc_hover_reason, axis=1)
        fig = px.bar(
            plot_qc,
            x="sample_id",
            y=selected_metric,
            color="qc_status",
            custom_data=["sample_id"],
            hover_name="sample_id",
            hover_data={
                "sample_id": False,
                "qc_status": True,
                selected_metric: ":,.3f",
                "qc_reason": True,
                **{col: True for col in hover_cols},
            },
            color_discrete_map={
                "PASS": "#4C78A8",
                "WARN": "#F2C94C",
                "FAIL": "#D62728",
            },
            category_orders={"qc_status": ["PASS", "WARN", "FAIL"]},
        )
        fig.update_layout(
            xaxis_title="sample_id",
            yaxis_title=selected_metric,
            legend_title_text="QC status",
            clickmode="event+select",
        )
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="qc_metric_plot",
            on_select="rerun",
            selection_mode="points",
        )
        selected_from_event = _selected_sample_from_plotly_event(event) or _selected_sample_from_plotly_event(
            st.session_state.get("qc_metric_plot")
        )
        if selected_from_event:
            st.session_state["selected_qc_sample"] = selected_from_event
        selected_sample = st.session_state.get("selected_qc_sample")
        if selected_sample:
            selected_rows = qc[qc["sample_id"].astype(str).eq(str(selected_sample))]
            st.subheader("Selected sample")
            if selected_rows.empty:
                st.info(f"Selected sample: {selected_sample}")
            else:
                st.dataframe(selected_rows, use_container_width=True, hide_index=True)

    st.subheader("QC table")
    st.dataframe(
        qc.style.apply(lambda row: _highlight_qc_table_row(row, selected_sample), axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _qc_hover_reason(row: pd.Series) -> str:
    warnings = str(row.get("qc_warnings", "") or "").strip()
    failures = str(row.get("qc_fail_reasons", "") or "").strip()
    parts = _qc_reason_details(row, failures, "FAIL") + _qc_reason_details(row, warnings, "WARN")
    return " | ".join(parts) if parts else "No QC warning/failure"


def _unique_columns(
    columns: list[str],
    available: pd.Index | list[str] | tuple[str, ...] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    available_set = set(available) if available is not None else None
    excluded = exclude or set()
    selected = []
    for column in columns:
        if column in excluded:
            continue
        if available_set is not None and column not in available_set:
            continue
        if column not in selected:
            selected.append(column)
    return selected


def _qc_reason_details(row: pd.Series, reason_text: str, status: str) -> list[str]:
    details = []
    for reason in [item.strip() for item in reason_text.split(";") if item.strip()]:
        values = _qc_reason_values(row, reason)
        if values:
            details.append(f"{status} {reason}: {values}")
        else:
            details.append(f"{status} {reason}")
    return details


def _qc_reason_values(row: pd.Series, reason: str) -> str:
    reason_columns = {
        "low_library_size": ["library_size"],
        "very_low_library_size": ["library_size"],
        "low_detected_endogenous_probes": ["detected_endogenous_probes"],
        "very_low_detected_endogenous_probes": ["detected_endogenous_probes"],
        "low_positive_controls": ["positive_control_sum", "positive_control_cv"],
        "high_negative_controls": ["negative_control_mean", "negative_control_sd", "background_threshold"],
        "low_fov_counted_fraction": ["fov_counted_fraction", "fov_registration_rate"],
        "low_fov_registration_rate": ["fov_registration_rate", "fov_counted_fraction"],
        "low_binding_density": ["binding_density", "bindingdensity"],
        "high_binding_density": ["binding_density", "bindingdensity"],
        "low_housekeeper_geomean": ["hk_geomean"],
        "low_positive_control_linearity": ["positive_control_linearity"],
    }
    values = []
    seen = set()
    for column in reason_columns.get(reason, []):
        if column in seen or column not in row.index:
            continue
        seen.add(column)
        value = row.get(column)
        if _is_missing_display_value(value):
            continue
        values.append(f"{column}={_format_qc_hover_value(value)}")
    return ", ".join(values)


def _is_missing_display_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_qc_hover_value(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    return str(value)


def _selected_sample_from_plotly_event(event) -> str | None:
    if event is None:
        return None
    if hasattr(event, "to_dict"):
        try:
            event = event.to_dict()
        except (TypeError, ValueError):
            pass
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if hasattr(selection, "to_dict"):
        try:
            selection = selection.to_dict()
        except (TypeError, ValueError):
            pass
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        if point.get("x") is not None:
            return point.get("x")
        customdata = point.get("customdata")
        if isinstance(customdata, np.ndarray):
            flattened = customdata.ravel()
            if len(flattened) > 0:
                return str(flattened[0])
        if isinstance(customdata, (list, tuple)) and customdata:
            return customdata[0]
        if customdata is not None:
            return customdata
    customdata = getattr(point, "customdata", None)
    if isinstance(customdata, np.ndarray):
        flattened = customdata.ravel()
        if len(flattened) > 0:
            return str(flattened[0])
    if isinstance(customdata, (list, tuple)) and customdata:
        return customdata[0]
    if customdata is not None:
        return str(customdata)
    return getattr(point, "x", None)


def _highlight_qc_table_row(row: pd.Series, selected_sample: str | None) -> list[str]:
    if selected_sample is not None and str(row.get("sample_id")) == str(selected_sample):
        color = "background-color: #cfe8ff; font-weight: 600"
    elif row.get("qc_status") == "FAIL":
        color = "background-color: #fdecec"
    elif row.get("qc_status") == "WARN":
        color = "background-color: #fff6cc"
    else:
        color = ""
    return [color] * len(row)


def _filter_metadata_for_downstream_pages(metadata: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        return metadata
    if "sample_id" not in _ensure_sample_id(metadata).columns:
        return metadata
    try:
        return _filter_metadata_by_included_samples(_ensure_sample_id(metadata))
    except ValueError:
        return pd.DataFrame(columns=metadata.columns)


def _option_index(options: list[str], value: object, default: int = 0) -> int:
    text = str(value) if value is not None else ""
    return options.index(text) if text in options else default


def _int_default(value: object, fallback: int, minimum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    if minimum is not None and number < minimum:
        return fallback
    return number


def _float_default(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _render_processing_steps_workflow(
    config_path: Path,
    run_dir: Path,
    results_dir: Path,
    reports_dir: Path,
    metadata: pd.DataFrame,
) -> None:
    import streamlit as st

    st.write(f"Analysis folder: `{run_dir}`")
    processing_settings = _render_processing_settings(metadata)
    qc_settings = _render_qc_settings()
    settings_ready = bool(processing_settings)

    st.divider()
    step_col, files_col = st.columns([2, 1])
    with step_col:
        st.subheader("Step 1 - Sample QC, background correction, gene filtering")
        accepted_qc = st.checkbox(
            "Ho revisionato e accetto i parametri QC/background/filtering.",
            key="accept_step_prepare",
        )
        if st.button(
            "Run Step 1",
            type="primary",
            disabled=not (settings_ready and accepted_qc),
            key="run_step_prepare",
        ):
            _materialize_current_run(run_dir, processing_settings, qc_settings)
            _record_analysis_savepoint(config_path, "configured")
            _execute_steps(config_path, ["prepare"])
    with files_col:
        _render_step_outputs(
            "Step 1 outputs",
            [
                results_dir / "counts_raw.csv",
                results_dir / "metadata.csv",
                results_dir / "probe_annotation_parsed.csv",
                results_dir / "qc_summary.csv",
                results_dir / "sample_qc.csv",
                results_dir / "counts_background_corrected.csv",
                results_dir / "counts_filtered.csv",
                results_dir / "gene_filter_decisions.csv",
            ],
            key_prefix="step-prepare",
        )

    st.divider()
    step_col, files_col = st.columns([2, 1])
    with step_col:
        st.subheader("Step 2 - Normalization")
        accepted_norm = st.checkbox(
            "Ho revisionato e accetto il metodo di normalizzazione.",
            key="accept_step_normalize",
        )
        if st.button(
            "Run Step 2",
            type="primary",
            disabled=not (settings_ready and accepted_norm and (results_dir / "counts_filtered.csv").exists()),
            key="run_step_normalize",
        ):
            _materialize_current_run(run_dir, processing_settings, qc_settings)
            _execute_steps(config_path, ["normalize"])
    with files_col:
        _render_step_outputs(
            "Step 2 outputs",
            [
                results_dir / "counts_normalized.csv",
                results_dir / "normalization_audit.csv",
                results_dir / "selected_housekeepers.csv",
            ],
            key_prefix="step-normalize",
        )

    st.divider()
    step_col, files_col = st.columns([2, 1])
    with step_col:
        st.subheader("Step 3 - QC/report plots")
        accepted_report = st.checkbox(
            "Ho revisionato gli step precedenti e voglio generare i report.",
            key="accept_step_report",
        )
        if st.button(
            "Run Step 3",
            type="primary",
            disabled=not (settings_ready and accepted_report and (results_dir / "counts_normalized.csv").exists()),
            key="run_step_report",
        ):
            _materialize_current_run(run_dir, processing_settings, qc_settings)
            _execute_steps(config_path, ["report"])
    with files_col:
        _render_step_outputs(
            "Step 3 outputs",
            [
                reports_dir / "qc_library_size.png",
                reports_dir / "housekeeper_geomean_qc.png",
                reports_dir / "count_distribution_raw.png",
                reports_dir / "count_distribution_background_corrected.png",
                reports_dir / "count_distribution_normalized.png",
                reports_dir / "pca.png",
                reports_dir / "mds_plot.png",
                reports_dir / "sample_correlation_heatmap.png",
                reports_dir / "heatmap_top_variable.png",
            ],
            key_prefix="step-report",
        )

    if config_path.exists():
        with st.expander("Generated configuration"):
            st.code(config_path.read_text(encoding="utf-8"), language="yaml")


def _render_step_outputs(title: str, paths: list[Path], key_prefix: str) -> None:
    import streamlit as st

    st.subheader(title)
    existing = [path for path in paths if path.exists()]
    if not existing:
        st.caption("No file yet.")
        return
    for path in existing:
        with path.open("rb") as handle:
            st.download_button(
                label=path.name,
                data=handle.read(),
                file_name=path.name,
                key=f"download-{key_prefix}-{path}",
                use_container_width=True,
            )


def _materialize_current_run(
    run_dir: Path,
    settings: dict[str, object],
    qc_settings: dict[str, object],
) -> None:
    if st_session_get("input_mode") == "Saved analysis":
        _materialize_saved_run(run_dir, settings, qc_settings)
        return

    _materialize_run(
        run_dir=run_dir,
        counts_file=st_session_get("counts_file"),
        metadata_file=st_session_get("metadata_file"),
        samplesheet_file=st_session_get("samplesheet_file"),
        rcc_files=st_session_get("rcc_files", []),
        rlf_file=st_session_get("rlf_file"),
        edited_samplesheet=st_session_get("edited_samplesheet"),
        edited_metadata=st_session_get("edited_metadata"),
        settings=settings,
        qc_settings=qc_settings,
        processed_counts=st_session_get("input_mode") == "Final processed counts",
    )


def _materialize_saved_run(
    run_dir: Path,
    settings: dict[str, object],
    qc_settings: dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    reports_dir = run_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = _read_saved_config(run_dir)
    metadata = st_session_get("edited_metadata")
    if not isinstance(metadata, pd.DataFrame):
        metadata = _read_saved_metadata(run_dir)
    metadata = _ensure_sample_id(metadata)
    metadata = _filter_metadata_by_included_samples(metadata)
    metadata = _add_composite_group_column(metadata, settings)
    metadata.to_csv(results_dir / "metadata.csv", index=False)

    config.setdefault("project", {"name": run_dir.name})
    config.setdefault("inputs", {})
    config["outputs"] = {
        "results_dir": str(results_dir),
        "reports_dir": str(reports_dir),
    }
    config["analysis"] = {
        "group_column": settings["group_column"],
        "group_columns": settings.get("group_columns", [settings["group_column"]]),
        "reference_group": settings["reference_group"],
        "case_group": settings["case_group"],
        "contrasts": settings.get("contrasts", []),
        "min_count": settings["min_count"],
        "min_samples": settings["min_samples"],
        "top_variable_genes": settings["top_variable_genes"],
        "skip_low_count_filter": not (results_dir / "counts_filtered.csv").exists(),
    }
    config["normalization"] = {
        **(config.get("normalization", {}) if isinstance(config.get("normalization"), dict) else {}),
        "method": settings["normalization_method"],
    }
    config["qc"] = qc_settings or config.get("qc", {}) or {}
    config.setdefault("r", {"executable": "Rscript"})

    _write_saved_config(run_dir / "pipeline.yaml", config)


def st_session_get(key: str, default=None):
    import streamlit as st

    return st.session_state.get(key, default)


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=sep)


def _normalize_counts_preview(table: pd.DataFrame) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        table.to_csv(temp_path, index=False)
        return read_count_table(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _metadata_template_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    return pd.DataFrame(
        {
            "sample_id": [_normalize_sample_id(sample) for sample in sample_cols],
            "condition": "",
        }
    )


def _build_samplesheet_from_rcc_uploads(rcc_files: list) -> pd.DataFrame:
    rows = []
    for uploaded_file in rcc_files:
        sample_id = _sample_id_from_rcc_upload(uploaded_file)
        rows.append(
            {
                "RCC_FILE": uploaded_file.name,
                "RCC_FILE_NAME": uploaded_file.name,
                "SAMPLE_ID": _normalize_sample_id(sample_id),
                "condition": "",
                "INCLUDE": 1,
            }
        )
    return pd.DataFrame(rows)


def _sample_id_from_rcc_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".RCC"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        _save_upload(uploaded_file, temp_path)
        sample_id, _ = read_rcc_file(temp_path)
        return sample_id
    except Exception:
        return Path(uploaded_file.name).stem
    finally:
        temp_path.unlink(missing_ok=True)


def _normalize_sample_id(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def _render_loaded_samples_summary(
    metadata: pd.DataFrame,
    title: str = "Campioni caricati",
    pre_qc: pd.DataFrame | None = None,
    editable_sample_id: bool = False,
    key: str | None = None,
) -> pd.DataFrame:
    import streamlit as st

    samples = _ensure_sample_id(metadata)
    if "sample_id" not in samples.columns:
        st.warning("Non riesco ancora a identificare una colonna `sample_id` nei campioni caricati.")
        return pd.DataFrame(columns=["sample_id"])

    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].map(_normalize_sample_id)
    samples["_row_id"] = samples.get("_row_id", samples["sample_id"])
    samples = samples.drop_duplicates(subset=["sample_id"], keep="first")
    if pre_qc is not None and not pre_qc.empty:
        qc = pre_qc.copy()
        qc["sample_id"] = qc["sample_id"].map(_normalize_sample_id)
        samples = samples.merge(qc, on="sample_id", how="left")
    if "pre_qc_status" not in samples.columns:
        samples["total_counts"] = pd.NA
        samples["pre_qc_status"] = "UNKNOWN"
        samples["pre_qc_reason"] = "counts_not_available"
    samples["total_counts"] = pd.to_numeric(samples["total_counts"], errors="coerce").round().astype("Int64")
    samples["pre_qc_status"] = samples["pre_qc_status"].fillna("FAILED")
    samples["pre_qc_reason"] = samples["pre_qc_reason"].fillna("sample_not_found_in_counts")
    excluded = _excluded_sample_ids()
    samples["include"] = ~samples["sample_id"].isin(excluded)

    st.subheader(title)
    status_counts = samples["pre_qc_status"].value_counts().to_dict()
    metric_cols = st.columns(3)
    metric_cols[0].metric("Campioni", len(samples))
    metric_cols[1].metric("Pre-QC OK", int(status_counts.get("OK", 0)))
    metric_cols[2].metric("Pre-QC failed", int(status_counts.get("FAILED", 0)))
    display_cols = [
        "_row_id",
        "file_name",
        "sample_id",
        "total_counts",
        "pre_qc_status",
        "pre_qc_reason",
        "include",
        *[
            col
            for col in samples.columns
            if col
            not in {
                "_row_id",
                "file_name",
                "sample_id",
                "include",
                "pre_qc_status",
                "total_counts",
                "pre_qc_reason",
                "condition",
                "INCLUDE",
            }
        ],
    ]
    display_cols = [col for col in display_cols if col in samples.columns]
    disabled = [col for col in display_cols if col not in {"include"}]
    if editable_sample_id:
        disabled = [col for col in disabled if col != "sample_id"]
    visible = samples[display_cols].copy()
    styled = visible.style.apply(_highlight_failed_sample_row, axis=1)
    column_order = [
        col
        for col in [
            "_row_id",
            "file_name",
            "sample_id",
            "total_counts",
            "pre_qc_status",
            "pre_qc_reason",
            "include",
        ]
        if col in visible.columns
    ]
    column_order.extend([col for col in visible.columns if col not in column_order])
    edited = st.data_editor(
        styled,
        use_container_width=True,
        hide_index=True,
        disabled=disabled,
        column_order=column_order,
        column_config={
            "_row_id": None,
            "total_counts": st.column_config.NumberColumn("total_counts", format="%d"),
            "pre_qc_status": st.column_config.TextColumn("pre_QC_status"),
            "include": st.column_config.CheckboxColumn(
                "include",
                help="Deseleziona per escludere il campione dai passaggi successivi.",
            ),
            "sample_id": st.column_config.TextColumn(
                "sample_id",
                help="Modifica l'identificativo del campione.",
            ),
        },
        key=key or f"loaded_samples_{_safe_name(title)}",
    )
    edited = pd.DataFrame(edited)
    if "sample_id" in edited.columns:
        edited["sample_id"] = edited["sample_id"].map(_normalize_sample_id)
    _update_excluded_samples_from_summary(edited)
    return edited


def _highlight_failed_sample_row(row: pd.Series) -> list[str]:
    color = "background-color: #fdecec" if row.get("pre_qc_status") == "FAILED" else ""
    return [color] * len(row)


def _samplesheet_loaded_samples_frame(samplesheet: pd.DataFrame) -> pd.DataFrame:
    if samplesheet is None or samplesheet.empty:
        return pd.DataFrame(columns=["sample_id", "file_name"])
    table = samplesheet.copy()
    if "SAMPLE_ID" not in table.columns:
        return pd.DataFrame(columns=["sample_id", "file_name"])
    table["sample_id"] = table["SAMPLE_ID"].map(_normalize_sample_id)
    if "RCC_FILE_NAME" in table.columns:
        table["file_name"] = table["RCC_FILE_NAME"].map(lambda value: Path(str(value)).name)
    elif "RCC_FILE" in table.columns:
        table["file_name"] = table["RCC_FILE"].map(lambda value: Path(str(value)).name)
    else:
        table["file_name"] = ""
    table["_row_id"] = table["file_name"].where(table["file_name"].astype(str).ne(""), table["sample_id"])
    return table[["_row_id", "file_name", "sample_id"]]


def _apply_loaded_sample_ids_to_samplesheet(
    samplesheet: pd.DataFrame,
    edited_samples: pd.DataFrame,
) -> pd.DataFrame:
    output = samplesheet.copy()
    if output.empty or edited_samples.empty or "SAMPLE_ID" not in output.columns:
        return output
    if "sample_id" not in edited_samples.columns:
        return output

    row_id_column = "_row_id" if "_row_id" in edited_samples.columns else "file_name"
    if row_id_column not in edited_samples.columns:
        return output

    sample_ids_by_row = {
        str(row[row_id_column]): _normalize_sample_id(row["sample_id"])
        for _, row in edited_samples.iterrows()
        if pd.notna(row.get(row_id_column)) and str(row.get(row_id_column)).strip()
    }

    for index, row in output.iterrows():
        row_id = ""
        if "RCC_FILE_NAME" in output.columns and pd.notna(row.get("RCC_FILE_NAME")):
            row_id = Path(str(row["RCC_FILE_NAME"])).name
        elif "RCC_FILE" in output.columns and pd.notna(row.get("RCC_FILE")):
            row_id = Path(str(row["RCC_FILE"])).name
        if row_id in sample_ids_by_row:
            output.at[index, "SAMPLE_ID"] = sample_ids_by_row[row_id]
    return output


def _excluded_sample_ids() -> set[str]:
    raw = st_session_get("excluded_sample_ids", [])
    if not isinstance(raw, list):
        return set()
    return {_normalize_sample_id(sample) for sample in raw}


def _update_excluded_samples_from_summary(summary: pd.DataFrame) -> None:
    import streamlit as st

    if "sample_id" not in summary.columns or "include" not in summary.columns:
        return
    current = _excluded_sample_ids()
    visible_samples = {_normalize_sample_id(sample) for sample in summary["sample_id"]}
    excluded_visible = {
        _normalize_sample_id(row["sample_id"])
        for _, row in summary.iterrows()
        if not bool(row.get("include", True))
    }
    updated = (current - visible_samples) | excluded_visible
    st.session_state["excluded_sample_ids"] = sorted(updated)


def _counts_matrix_pre_qc(counts: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    rows = []
    for sample in sample_cols:
        values = pd.to_numeric(counts[sample], errors="coerce").fillna(0)
        total = int(round(float(values.sum())))
        rows.append(
            {
                "sample_id": _normalize_sample_id(sample),
                "total_counts": total,
                "pre_qc_status": "OK" if total > 0 else "FAILED",
                "pre_qc_reason": "counts_present" if total > 0 else "all_counts_zero",
            }
        )
    return pd.DataFrame(rows)


def _rcc_upload_pre_qc(rcc_files: list, samplesheet: pd.DataFrame) -> pd.DataFrame:
    sample_lookup = _rcc_upload_sample_lookup(samplesheet)
    rows = []
    for uploaded_file in rcc_files:
        sample_id = sample_lookup.get(uploaded_file.name, _normalize_sample_id(Path(uploaded_file.name).stem))
        suffix = Path(uploaded_file.name).suffix or ".RCC"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            _save_upload(uploaded_file, temp_path)
            _, counts = read_rcc_file(temp_path)
            total = int(round(float(pd.to_numeric(counts["Count"], errors="coerce").fillna(0).sum())))
            status = "OK" if total > 0 else "FAILED"
            reason = "counts_present" if total > 0 else "all_counts_zero"
        except Exception:
            total = pd.NA
            status = "FAILED"
            reason = "rcc_counts_not_readable"
        finally:
            temp_path.unlink(missing_ok=True)
        rows.append(
            {
                "sample_id": sample_id,
                "total_counts": total,
                "pre_qc_status": status,
                "pre_qc_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _rcc_upload_sample_lookup(samplesheet: pd.DataFrame) -> dict[str, str]:
    if samplesheet is None or samplesheet.empty or "SAMPLE_ID" not in samplesheet.columns:
        return {}
    output = {}
    for _, row in samplesheet.iterrows():
        sample_id = _normalize_sample_id(row.get("SAMPLE_ID", ""))
        for column in ("RCC_FILE_NAME", "RCC_FILE"):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                output[Path(str(value)).name] = sample_id
    return output


def _saved_analysis_pre_qc(run_dir: Path) -> pd.DataFrame:
    for filename in ["counts_raw.csv", "counts_normalized.csv", "counts_filtered.csv"]:
        path = run_dir / "results" / filename
        if path.exists():
            return _counts_matrix_pre_qc(pd.read_csv(path))
    return pd.DataFrame(columns=["sample_id", "total_counts", "pre_qc_status", "pre_qc_reason"])


def _filter_metadata_by_included_samples(metadata: pd.DataFrame) -> pd.DataFrame:
    excluded = _excluded_sample_ids()
    if not excluded or "sample_id" not in metadata.columns:
        return metadata
    output = metadata.copy()
    output["sample_id"] = output["sample_id"].map(_normalize_sample_id)
    output = output[~output["sample_id"].isin(excluded)].copy()
    _require_included_samples(output["sample_id"].tolist())
    return output


def _filter_counts_by_included_samples(counts: pd.DataFrame) -> pd.DataFrame:
    excluded = _excluded_sample_ids()
    if not excluded:
        return counts
    output = counts.copy()
    sample_cols = [col for col in output.columns if col not in {"CodeClass", "Name"}]
    drop_cols = [col for col in sample_cols if _normalize_sample_id(col) in excluded]
    output = output.drop(columns=drop_cols)
    remaining = [col for col in output.columns if col not in {"CodeClass", "Name"}]
    _require_included_samples(remaining)
    return output


def _apply_recorded_sample_id_renames_to_counts(counts: pd.DataFrame) -> pd.DataFrame:
    rename_map = st_session_get("sample_id_renames", {})
    if not isinstance(rename_map, dict) or not rename_map:
        return counts
    normalized_map = {
        _normalize_sample_id(old): _normalize_sample_id(new)
        for old, new in rename_map.items()
        if str(new).strip()
    }
    return counts.rename(
        columns={
            col: normalized_map[_normalize_sample_id(col)]
            for col in counts.columns
            if col not in {"CodeClass", "Name"} and _normalize_sample_id(col) in normalized_map
        }
    )


def _filter_samplesheet_by_included_samples(samplesheet: pd.DataFrame) -> pd.DataFrame:
    excluded = _excluded_sample_ids()
    if not excluded or "SAMPLE_ID" not in samplesheet.columns:
        return samplesheet
    output = samplesheet.copy()
    output["SAMPLE_ID"] = output["SAMPLE_ID"].map(_normalize_sample_id)
    output = output[~output["SAMPLE_ID"].isin(excluded)].copy()
    _require_included_samples(output["SAMPLE_ID"].tolist())
    return output


def _require_included_samples(samples: list[object]) -> None:
    if len(samples) == 0:
        raise ValueError("At least one sample must remain included in the analysis.")


def _render_input_preview(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    import streamlit as st

    editable_metadata = _ensure_sample_id(metadata)
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    missing_metadata = sorted(set(sample_cols) - set(editable_metadata.get("sample_id", [])))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Probe", f"{len(counts):,}")
    metric_cols[1].metric("Samples counts", f"{len(sample_cols):,}")
    metric_cols[2].metric("Metadata rows", f"{len(editable_metadata):,}")
    metric_cols[3].metric("Missing metadata", f"{len(missing_metadata):,}")

    if "sample_id" not in editable_metadata.columns:
        st.error("Metadata must contain a `sample_id` column.")
    elif missing_metadata:
        st.warning("Missing metadata rows for: " + ", ".join(missing_metadata))

    left, right = st.columns(2)
    with left:
        st.subheader("Counts")
        st.dataframe(counts.head(20), use_container_width=True)
    with right:
        st.subheader("Metadata")
        edited = _render_metadata_editor(editable_metadata, key="metadata")

    return edited


def _render_metadata_only_preview(counts: pd.DataFrame, metadata: pd.DataFrame, key: str) -> pd.DataFrame:
    import streamlit as st

    editable_metadata = _ensure_sample_id(metadata)
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    missing_metadata = sorted(set(sample_cols) - set(editable_metadata.get("sample_id", [])))

    if "sample_id" not in editable_metadata.columns:
        st.error("Metadata must contain a `sample_id` column.")
    elif missing_metadata:
        st.warning("Missing metadata rows for: " + ", ".join(missing_metadata))

    st.subheader("Metadata")
    return _render_metadata_editor(editable_metadata, key=key)


def _render_metadata_template_tools(metadata: pd.DataFrame, key: str) -> pd.DataFrame:
    import streamlit as st

    table = _ensure_sample_id(metadata)
    if "sample_id" not in table.columns:
        return table

    template = _metadata_template_for_download(table)
    with st.expander("Modifica metadata via CSV", expanded=False):
        tool_cols = st.columns(2)
        tool_cols[0].download_button(
            "Scarica template samplesheet",
            template.to_csv(index=False).encode("utf-8"),
            file_name="samplesheet_template.csv",
            mime="text/csv",
            key=f"{key}_download",
            use_container_width=True,
        )
        uploaded_template = tool_cols[1].file_uploader(
            "Ricarica template modificato",
            type=["csv", "tsv", "txt"],
            key=f"{key}_upload",
        )
        if uploaded_template is not None:
            updated = _apply_uploaded_metadata_template(table, _read_uploaded_table(uploaded_template))
            st.session_state[f"{key}_table"] = updated
            st.success("Template caricato. Le modifiche sono state applicate alla tabella metadata.")
            return updated
    return table


def _metadata_template_for_download(metadata: pd.DataFrame) -> pd.DataFrame:
    table = _ensure_sample_id(metadata)
    output = table.copy()
    output["sample_id"] = output["sample_id"].map(_normalize_sample_id)
    output.insert(1, "new_sample_id", "")
    return output


def _apply_uploaded_metadata_template(metadata: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    table = _ensure_sample_id(metadata)
    incoming = _ensure_sample_id(uploaded)
    if "sample_id" not in incoming.columns:
        raise ValueError("Uploaded template must contain `sample_id`.")

    incoming["sample_id"] = incoming["sample_id"].map(_normalize_sample_id)
    table["sample_id"] = table["sample_id"].map(_normalize_sample_id)
    renamed_rows = incoming.copy()
    rename_map = {}
    if "new_sample_id" in renamed_rows.columns:
        for _, row in renamed_rows.iterrows():
            new_value = row.get("new_sample_id")
            if pd.notna(new_value) and str(new_value).strip():
                rename_map[_normalize_sample_id(row["sample_id"])] = _normalize_sample_id(new_value)
        renamed_rows["sample_id"] = renamed_rows.apply(
            lambda row: rename_map.get(row["sample_id"], row["sample_id"]),
            axis=1,
        )
        renamed_rows = renamed_rows.drop(columns=["new_sample_id"])

    metadata_columns = [col for col in renamed_rows.columns if col != "sample_id"]
    base = table.drop(columns=[col for col in metadata_columns if col in table.columns])
    base["sample_id"] = base["sample_id"].map(lambda value: rename_map.get(value, value))
    output = base.merge(renamed_rows[["sample_id", *metadata_columns]], on="sample_id", how="left")
    output = output.fillna("")
    _record_sample_id_renames(rename_map)
    _apply_sample_id_renames_to_current_samplesheet(rename_map)
    return output


def _record_sample_id_renames(rename_map: dict[str, str]) -> None:
    import streamlit as st

    if not rename_map:
        return
    current = st.session_state.get("sample_id_renames", {})
    if not isinstance(current, dict):
        current = {}
    current.update(rename_map)
    st.session_state["sample_id_renames"] = current


def _apply_sample_id_renames_to_current_samplesheet(rename_map: dict[str, str]) -> None:
    import streamlit as st

    samplesheet = st.session_state.get("edited_samplesheet")
    if not rename_map or not isinstance(samplesheet, pd.DataFrame) or "SAMPLE_ID" not in samplesheet.columns:
        return
    updated = samplesheet.copy()
    updated["SAMPLE_ID"] = updated["SAMPLE_ID"].map(
        lambda value: rename_map.get(_normalize_sample_id(value), _normalize_sample_id(value))
    )
    st.session_state["edited_samplesheet"] = updated


def _render_processed_counts_preview(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    import streamlit as st

    st.info("This input is treated as already processed: the UI will use the uploaded counts as the final matrix.")
    return _render_metadata_only_preview(counts, metadata, key="processed_counts_metadata")


def _render_samplesheet_upload_editor(samplesheet: pd.DataFrame, key: str = "samplesheet") -> pd.DataFrame:
    import streamlit as st

    normalized = samplesheet.copy()
    if "SAMPLE_ID" in normalized.columns:
        normalized["SAMPLE_ID"] = normalized["SAMPLE_ID"].map(_normalize_sample_id)
    table_key = f"{key}_table"
    editor_key = f"{key}_upload_editor"
    if not isinstance(st.session_state.get(table_key), pd.DataFrame):
        st.session_state[table_key] = normalized

    missing = {"RCC_FILE", "SAMPLE_ID"} - set(normalized.columns)
    if missing:
        st.error("The samplesheet must contain: " + ", ".join(sorted(missing)))

    st.subheader("Samplesheet")
    st.caption("At this stage you can edit only `SAMPLE_ID`.")
    table = st.session_state[table_key]
    disabled = [col for col in table.columns if col != "SAMPLE_ID"]
    st.data_editor(
        table,
        use_container_width=True,
        num_rows="fixed",
        disabled=disabled,
        key=editor_key,
        on_change=_apply_data_editor_changes,
        args=(editor_key, table_key, ["RCC_FILE", "RCC_FILE_NAME"], "edited_samplesheet"),
    )
    edited = st.session_state[table_key].copy()
    if "SAMPLE_ID" in edited.columns:
        edited["SAMPLE_ID"] = edited["SAMPLE_ID"].map(_normalize_sample_id)
    st.session_state[table_key] = edited
    st.session_state["edited_samplesheet"] = edited
    return edited


def _render_metadata_editor(metadata: pd.DataFrame, key: str) -> pd.DataFrame:
    import streamlit as st

    table_key = f"{key}_table"
    table = _sync_metadata_state(table_key, metadata)

    controls = st.columns([2, 1, 2, 1])
    new_column = controls[0].text_input("New column", placeholder="TIME", key=f"{key}_new_column")
    if controls[1].button("Add column", key=f"{key}_add_column"):
        column = _parse_single_column_name(new_column)
        if column and column != "sample_id" and column not in table.columns:
            table[column] = ""

    removable = [col for col in table.columns if col != "sample_id"]
    selected_column = controls[2].selectbox(
        "Column to delete",
        removable,
        key=f"{key}_drop_column",
        disabled=not removable,
    )
    if controls[3].button(
        "Delete selected column",
        key=f"{key}_delete_column",
        disabled=not removable,
    ):
        table = table.drop(columns=[selected_column])

    st.session_state[table_key] = table
    editor_key = f"{key}_editor"
    st.data_editor(
        table,
        use_container_width=True,
        num_rows="fixed",
        disabled=["sample_id"],
        key=editor_key,
        on_change=_apply_data_editor_changes,
        args=(editor_key, table_key, ["sample_id"], None),
    )
    edited = st.session_state[table_key].copy()
    edited["sample_id"] = table["sample_id"].map(_normalize_sample_id)
    st.session_state[table_key] = edited
    return edited


def _apply_data_editor_changes(
    editor_key: str,
    table_key: str,
    locked_columns: list[str],
    mirror_key: str | None,
) -> None:
    import streamlit as st

    table = st.session_state.get(table_key)
    state = st.session_state.get(editor_key)
    if not isinstance(table, pd.DataFrame) or not isinstance(state, dict):
        return

    updated = table.copy()
    edited_rows = state.get("edited_rows", {})
    for raw_index, changes in edited_rows.items():
        row_index = int(raw_index)
        if row_index >= len(updated):
            continue
        for column, value in changes.items():
            if column in locked_columns or column not in updated.columns:
                continue
            updated.iat[row_index, updated.columns.get_loc(column)] = value

    if "sample_id" in updated.columns:
        updated["sample_id"] = table["sample_id"].map(_normalize_sample_id)
    if "SAMPLE_ID" in updated.columns:
        updated["SAMPLE_ID"] = updated["SAMPLE_ID"].map(_normalize_sample_id)

    st.session_state[table_key] = updated
    if mirror_key:
        st.session_state[mirror_key] = updated


def _sync_metadata_state(table_key: str, metadata: pd.DataFrame) -> pd.DataFrame:
    import streamlit as st

    base = _ensure_sample_id(metadata)
    base["sample_id"] = base["sample_id"].map(_normalize_sample_id)
    current = st.session_state.get(table_key)
    if not isinstance(current, pd.DataFrame):
        st.session_state[table_key] = base
        return base.copy()

    current = _ensure_sample_id(current)
    current["sample_id"] = current["sample_id"].map(_normalize_sample_id)
    sample_ids = base["sample_id"].tolist()
    if current["sample_id"].tolist() == sample_ids:
        return current.copy()

    merged = base[["sample_id"]].merge(current, on="sample_id", how="left")
    for column in base.columns:
        if column != "sample_id" and column not in merged.columns:
            merged[column] = base[column]
    merged = merged.fillna("")
    st.session_state[table_key] = merged
    return merged


def _ensure_sample_id(metadata: pd.DataFrame) -> pd.DataFrame:
    table = metadata.copy()
    if "sample_id" not in table.columns and "SAMPLE_ID" in table.columns:
        table = table.rename(columns={"SAMPLE_ID": "sample_id"})
    if "sample_id" in table.columns:
        first = ["sample_id"]
        rest = [col for col in table.columns if col != "sample_id"]
        table = table[first + rest]
    return table


def _parse_column_names(value: str) -> list[str]:
    names = []
    for raw_name in value.replace(";", ",").split(","):
        name = raw_name.strip()
        if not name:
            continue
        safe = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
        safe = safe.strip("_")
        if safe and safe not in names:
            names.append(safe)
    return names


def _parse_single_column_name(value: str) -> str:
    names = _parse_column_names(value)
    return names[0] if names else ""


def _render_analysis_settings(metadata: pd.DataFrame) -> dict[str, object]:
    import streamlit as st

    st.subheader("Experiment design")
    candidate_group_cols = [col for col in metadata.columns if col != "sample_id"]
    default_group = candidate_group_cols[0] if candidate_group_cols else None
    saved_config = _current_saved_config()
    saved_analysis = saved_config.get("analysis", {}) if isinstance(saved_config.get("analysis"), dict) else {}
    saved_group_columns = saved_analysis.get("group_columns") or [saved_analysis.get("group_column")]
    if isinstance(saved_group_columns, str):
        saved_group_columns = [saved_group_columns]
    default_group_columns = [
        str(column) for column in saved_group_columns if column and str(column) in candidate_group_cols
    ]

    if default_group is None:
        st.error("Add at least one experimental column besides `sample_id`.")
        return {}

    with st.expander("Metadata grouping for QC/report coloring", expanded=False):
        group_cols = st.multiselect(
            "Group columns",
            candidate_group_cols,
            default=default_group_columns or [default_group],
            help=(
                "Optional. Contrasts are built with the filters below; this grouping is kept for "
                "QC/report coloring and legacy outputs."
            ),
        )
    if not group_cols:
        st.error("Select at least one metadata column.")
        return {}

    group_col = _composite_group_column_name(group_cols)
    grouped_metadata = _add_composite_group_column(
        metadata,
        {"group_column": group_col, "group_columns": group_cols},
    )
    groups = _metadata_group_values(metadata, group_cols)
    if not groups:
        st.error("The selected group columns do not contain usable values.")
        return {}
    if len(groups) < 2:
        st.warning("At least two groups are required for differential comparison.")
    with st.expander("Generated metadata groups", expanded=False):
        counts = _metadata_group_counts(metadata, group_cols)
        st.dataframe(counts, use_container_width=True, hide_index=True)

    contrast_mode = st.radio(
        "Contrast mode",
        ["Single contrast", "Multiple contrasts", "Custom sample groups"],
        horizontal=True,
        key="contrast_mode",
    )
    if contrast_mode == "Custom sample groups":
        contrasts = _render_custom_sample_contrast_settings(metadata, saved_analysis)
    else:
        contrasts = _render_contrast_settings(metadata, saved_analysis, contrast_mode)
    if not contrasts:
        st.warning("No valid contrast is configured.")
    inferred_group_cols = _contrast_filter_features(contrasts, metadata)
    if inferred_group_cols:
        group_cols = inferred_group_cols
        group_col = _composite_group_column_name(group_cols)
        grouped_metadata = _add_composite_group_column(
            metadata,
            {"group_column": group_col, "group_columns": group_cols},
        )
        with st.expander("Analysis groups used in all-sample plots", expanded=False):
            st.caption("These groups are built from the features used by the configured contrast filters.")
            counts = _metadata_group_counts(metadata, group_cols)
            st.dataframe(counts, use_container_width=True, hide_index=True)
    reference_group = contrasts[0]["reference_group"] if contrasts else ""
    case_group = contrasts[0]["case_group"] if contrasts else ""
    _render_contrast_sample_lists(grouped_metadata, group_col, contrasts)

    st.subheader("Analysis parameters")
    col_a, col_b, col_c = st.columns(3)
    min_count = col_a.number_input(
        "Min count",
        min_value=0,
        value=_saved_analysis_int(saved_analysis, "min_count", 10, minimum=0),
        step=1,
    )
    min_samples = col_b.number_input(
        "Min samples",
        min_value=1,
        value=_saved_analysis_int(saved_analysis, "min_samples", 2, minimum=1),
        step=1,
    )
    top_variable = col_c.number_input(
        "Top variable genes",
        min_value=5,
        value=_saved_analysis_int(saved_analysis, "top_variable_genes", 50, minimum=5),
        step=5,
    )

    normalization = saved_config.get("normalization", {})
    saved_method = normalization.get("method") if isinstance(normalization, dict) else None
    methods = ["nanostringnorm", "library_size", "hk_geomean_all", "hk_geomean_geNorm"]
    method_index = methods.index(str(saved_method)) if saved_method in methods else 0
    method = st.selectbox("Normalization", methods, index=method_index)

    settings = {
        "group_column": group_col,
        "group_columns": group_cols,
        "reference_group": reference_group,
        "case_group": case_group,
        "contrasts": contrasts,
        "min_count": int(min_count),
        "min_samples": int(min_samples),
        "top_variable_genes": int(top_variable),
        "normalization_method": method,
    }
    st.session_state["analysis_settings"] = settings
    return settings


def _saved_analysis_int(
    saved_analysis: dict[str, object],
    key: str,
    default: int,
    minimum: int,
) -> int:
    value = saved_analysis.get(key)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _metadata_group_counts(metadata: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if len(group_columns) == 1:
        group_series = metadata[group_columns[0]].dropna().astype(str)
    else:
        group_series = _build_composite_group_series(metadata, group_columns).dropna().astype(str)
    return (
        group_series.value_counts()
        .rename_axis("group")
        .reset_index(name="samples")
        .sort_values("group")
    )


def _render_contrast_settings(
    metadata: pd.DataFrame,
    saved_analysis: dict[str, object],
    contrast_mode: str,
) -> list[dict[str, object]]:
    import streamlit as st

    saved_contrasts = _saved_contrasts(saved_analysis)
    if contrast_mode == "Single contrast":
        contrast = _render_filter_contrast_editor(
            metadata,
            key_prefix="single_filter_contrast",
            default=saved_contrasts[0] if saved_contrasts else {},
            expanded=True,
        )
        return [contrast] if contrast else []

    if "filter_contrast_count" not in st.session_state:
        st.session_state["filter_contrast_count"] = max(1, len(saved_contrasts))
    if st.button("Add contrast", key="add_filter_contrast"):
        st.session_state["filter_contrast_count"] += 1
        st.rerun()

    contrasts = []
    for index in range(int(st.session_state["filter_contrast_count"])):
        default = saved_contrasts[index] if index < len(saved_contrasts) else {}
        with st.expander(f"Contrast {index + 1}", expanded=index == 0):
            contrast = _render_filter_contrast_editor(
                metadata,
                key_prefix=f"multi_filter_contrast_{index}",
                default=default,
                expanded=index == 0,
            )
            if contrast:
                contrasts.append(contrast)
    st.caption(f"Valid contrasts: {len(contrasts)}")
    return contrasts


def _render_filter_contrast_editor(
    metadata: pd.DataFrame,
    key_prefix: str,
    default: dict[str, object],
    expanded: bool,
) -> dict[str, object]:
    import streamlit as st

    default_filters = _default_filter_sets(metadata, default)
    comparison_id = st.text_input(
        "Comparison ID",
        value=str(default.get("comparison_id") or f"{key_prefix}_case_vs_reference"),
        key=f"{key_prefix}_comparison_id",
    )
    col_ref, col_case = st.columns(2)
    with col_ref:
        st.caption("Reference filters")
        reference_filters = _render_metadata_filter_builder(
            metadata,
            key_prefix=f"{key_prefix}_reference",
            default_filters=default_filters["reference"],
        )
    with col_case:
        st.caption("Case filters")
        case_filters = _render_metadata_filter_builder(
            metadata,
            key_prefix=f"{key_prefix}_case",
            default_filters=default_filters["case"],
        )

    reference_samples = _apply_metadata_filters(metadata, reference_filters)
    case_samples = _apply_metadata_filters(metadata, case_filters)
    overlap = sorted(set(reference_samples) & set(case_samples))
    if overlap:
        st.error("Samples selected in both reference and case: " + ", ".join(overlap))
        return {}
    if not reference_samples or not case_samples:
        st.warning("Both reference and case must contain at least one sample.")
        return {}

    if expanded:
        st.caption(f"Reference samples: {len(reference_samples)} | Case samples: {len(case_samples)}")
    reference_label = _metadata_filter_group_label(reference_filters, "Reference")
    case_label = _metadata_filter_group_label(case_filters, "Case")
    return {
        "comparison_id": _safe_name(comparison_id) or "case_vs_reference",
        "reference_group": reference_label,
        "case_group": case_label,
        "reference_samples": reference_samples,
        "case_samples": case_samples,
        "reference_filters": reference_filters,
        "case_filters": case_filters,
    }


def _render_metadata_filter_builder(
    metadata: pd.DataFrame,
    key_prefix: str,
    default_filters: list[dict[str, str]],
) -> list[dict[str, str]]:
    import streamlit as st

    features = _filterable_metadata_columns(metadata)
    if not features:
        return []
    count_key = f"{key_prefix}_filter_count"
    if count_key not in st.session_state:
        st.session_state[count_key] = max(1, len(default_filters))
    if st.button("Add feature", key=f"{key_prefix}_add_filter"):
        st.session_state[count_key] += 1
        st.rerun()

    filters = []
    running = metadata.copy()
    for index in range(int(st.session_state[count_key])):
        default = default_filters[index] if index < len(default_filters) else {}
        feature_default = str(default.get("feature") or features[0])
        feature_index = features.index(feature_default) if feature_default in features else 0
        col_feature, col_op, col_value, col_remove = st.columns([2, 1, 2, 0.8])
        feature = col_feature.selectbox(
            "Feature",
            features,
            index=feature_index,
            key=f"{key_prefix}_feature_{index}",
        )
        operator = col_op.selectbox(
            "Rule",
            ["is", "is not"],
            index=0 if str(default.get("operator") or "is") == "is" else 1,
            key=f"{key_prefix}_operator_{index}",
        )
        values = _metadata_values_for_feature(running, feature) or _metadata_values_for_feature(metadata, feature)
        value_default = str(default.get("value") or (values[0] if values else ""))
        value_index = values.index(value_default) if value_default in values else 0
        value = col_value.selectbox(
            "Value",
            values,
            index=value_index,
            key=f"{key_prefix}_value_{index}",
        )
        item = {"feature": feature, "operator": operator, "value": value}
        filters.append(item)
        running = _metadata_table_for_filters(running, [item])
        if col_remove.button(
            "Remove",
            key=f"{key_prefix}_remove_{index}",
            disabled=int(st.session_state[count_key]) <= 1,
        ):
            _remove_filter_row_state(key_prefix, index, int(st.session_state[count_key]))
            st.session_state[count_key] = max(1, int(st.session_state[count_key]) - 1)
            st.rerun()
    return filters


def _remove_filter_row_state(key_prefix: str, remove_index: int, count: int) -> None:
    import streamlit as st

    fields = ["feature", "operator", "value"]
    for index in range(remove_index, max(count - 1, remove_index)):
        for field in fields:
            current_key = f"{key_prefix}_{field}_{index}"
            next_key = f"{key_prefix}_{field}_{index + 1}"
            if next_key in st.session_state:
                st.session_state[current_key] = st.session_state[next_key]
    last_index = max(count - 1, 0)
    for field in fields:
        st.session_state.pop(f"{key_prefix}_{field}_{last_index}", None)


def _default_filter_sets(metadata: pd.DataFrame, contrast: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    reference_filters = _normalize_filter_rows(contrast.get("reference_filters"))
    case_filters = _normalize_filter_rows(contrast.get("case_filters"))
    if reference_filters or case_filters:
        return {"reference": reference_filters, "case": case_filters}
    columns = _filterable_metadata_columns(metadata)
    if not columns:
        return {"reference": [], "case": []}
    values = _metadata_values_for_feature(metadata, columns[0])
    if not values:
        return {"reference": [], "case": []}
    return {
        "reference": [{"feature": columns[0], "operator": "is", "value": values[0]}],
        "case": [{"feature": columns[0], "operator": "is not", "value": values[0]}],
    }


def _normalize_filter_rows(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    filters = []
    for row in value:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "")).strip()
        operator = str(row.get("operator", "is")).strip()
        filter_value = str(row.get("value", "")).strip()
        if feature and operator in {"is", "is not"} and filter_value:
            filters.append({"feature": feature, "operator": operator, "value": filter_value})
    return filters


def _contrast_filter_features(
    contrasts: list[dict[str, object]],
    metadata: pd.DataFrame | None = None,
) -> list[str]:
    available = set(metadata.columns) if isinstance(metadata, pd.DataFrame) else None
    features = []
    for contrast in contrasts:
        for key in ["reference_filters", "case_filters"]:
            for item in _normalize_filter_rows(contrast.get(key)):
                feature = item["feature"]
                if available is not None and feature not in available:
                    continue
                if feature not in features:
                    features.append(feature)
    return features


def _metadata_filter_group_label(filters: list[dict[str, str]], fallback: str) -> str:
    parts = []
    for item in filters:
        feature = str(item.get("feature", "")).strip()
        value = str(item.get("value", "")).strip()
        if not feature or not value:
            continue
        operator = "!=" if item.get("operator") == "is not" else "="
        parts.append(f"{feature}{operator}{value}")
    return " + ".join(parts) if parts else fallback


def _filterable_metadata_columns(metadata: pd.DataFrame) -> list[str]:
    return [
        column
        for column in metadata.columns
        if column != "sample_id" and len(_metadata_values_for_feature(metadata, column)) > 0
    ]


def _metadata_values_for_feature(metadata: pd.DataFrame, feature: str) -> list[str]:
    if feature not in metadata.columns:
        return []
    values = metadata[feature].dropna().astype(str)
    values = values[values.str.strip().ne("")]
    return sorted(values.drop_duplicates().tolist())


def _metadata_table_for_filters(metadata: pd.DataFrame, filters: list[dict[str, str]]) -> pd.DataFrame:
    output = metadata.copy()
    for item in filters:
        feature = item.get("feature")
        if feature not in output.columns:
            continue
        values = output[feature].astype(str)
        target = str(item.get("value", ""))
        if item.get("operator") == "is not":
            output = output[values.ne(target)].copy()
        else:
            output = output[values.eq(target)].copy()
    return output


def _apply_metadata_filters(metadata: pd.DataFrame, filters: list[dict[str, str]]) -> list[str]:
    if "sample_id" not in metadata.columns:
        return []
    filtered = _metadata_table_for_filters(metadata, filters)
    return filtered["sample_id"].dropna().astype(str).tolist()


def _render_custom_sample_contrast_settings(
    metadata: pd.DataFrame,
    saved_analysis: dict[str, object],
) -> list[dict[str, object]]:
    import streamlit as st

    sample_ids = _metadata_sample_ids(metadata)
    if len(sample_ids) < 2:
        st.error("At least two samples are required for a custom comparison.")
        return []

    saved_contrasts = _saved_contrasts(saved_analysis)
    saved_first = saved_contrasts[0] if saved_contrasts else {}
    default_reference = [
        sample
        for sample in _as_string_list(saved_first.get("reference_samples"))
        if sample in sample_ids
    ]
    default_case = [
        sample
        for sample in _as_string_list(saved_first.get("case_samples"))
        if sample in sample_ids
    ]
    if not default_reference:
        default_reference = sample_ids[:1]
    if not default_case:
        default_case = [sample for sample in sample_ids if sample not in default_reference][:1]

    comparison_id = st.text_input(
        "Comparison ID",
        value=str(saved_first.get("comparison_id") or "custom_case_vs_reference"),
        key="custom_sample_comparison_id",
    )
    col_ref, col_case = st.columns(2)
    reference_samples = col_ref.multiselect(
        "Reference samples",
        sample_ids,
        default=default_reference,
        key="custom_reference_samples",
    )
    case_samples = col_case.multiselect(
        "Case samples",
        sample_ids,
        default=default_case,
        key="custom_case_samples",
    )
    overlap = sorted(set(reference_samples) & set(case_samples))
    if overlap:
        st.error("A sample cannot be both reference and case: " + ", ".join(overlap))
        return []
    if not reference_samples or not case_samples:
        st.warning("Select at least one reference sample and one case sample.")
        return []

    return [
        {
            "comparison_id": _safe_name(comparison_id) or "custom_case_vs_reference",
            "reference_group": "reference",
            "case_group": "case",
            "reference_samples": reference_samples,
            "case_samples": case_samples,
        }
    ]


def _render_contrast_sample_lists(
    metadata: pd.DataFrame,
    group_col: str,
    contrasts: list[dict[str, object]],
) -> None:
    import streamlit as st

    if not contrasts:
        return
    st.subheader("Samples in current comparison")
    for index, contrast in enumerate(contrasts):
        title = str(contrast.get("comparison_id") or f"contrast_{index + 1}")
        container = st.expander(title, expanded=index == 0) if len(contrasts) > 1 else st.container()
        with container:
            reference_samples = _contrast_reference_samples(metadata, group_col, contrast)
            case_samples = _contrast_case_samples(metadata, group_col, contrast)
            col_ref, col_case = st.columns(2)
            with col_ref:
                st.caption(f"Reference ({len(reference_samples)})")
                st.dataframe(pd.DataFrame({"sample_id": reference_samples}), use_container_width=True, hide_index=True)
            with col_case:
                st.caption(f"Case ({len(case_samples)})")
                st.dataframe(pd.DataFrame({"sample_id": case_samples}), use_container_width=True, hide_index=True)


def _contrast_reference_samples(
    metadata: pd.DataFrame,
    group_col: str,
    contrast: dict[str, object],
) -> list[str]:
    explicit = _as_string_list(contrast.get("reference_samples"))
    if explicit:
        return explicit
    return _samples_for_group(metadata, group_col, str(contrast.get("reference_group", "")))


def _contrast_case_samples(
    metadata: pd.DataFrame,
    group_col: str,
    contrast: dict[str, object],
) -> list[str]:
    explicit = _as_string_list(contrast.get("case_samples"))
    if explicit:
        return explicit
    return _samples_for_group(metadata, group_col, str(contrast.get("case_group", "")))


def _samples_for_group(metadata: pd.DataFrame, group_col: str, group: str) -> list[str]:
    if group_col not in metadata.columns or "sample_id" not in metadata.columns:
        return []
    table = metadata.copy()
    table[group_col] = table[group_col].astype(str)
    return table.loc[table[group_col].eq(group), "sample_id"].astype(str).tolist()


def _metadata_sample_ids(metadata: pd.DataFrame) -> list[str]:
    if "sample_id" not in metadata.columns:
        return []
    return metadata["sample_id"].dropna().map(_normalize_sample_id).drop_duplicates().tolist()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _saved_contrasts(saved_analysis: dict[str, object]) -> list[dict[str, object]]:
    raw_contrasts = saved_analysis.get("contrasts")
    if isinstance(raw_contrasts, list):
        return _normalize_contrast_rows(raw_contrasts)

    reference_group = saved_analysis.get("reference_group")
    case_group = saved_analysis.get("case_group")
    if reference_group and case_group:
        return [
            {
                "comparison_id": _contrast_id(str(case_group), str(reference_group)),
                "reference_group": str(reference_group),
                "case_group": str(case_group),
            }
        ]
    return []


def _default_pairwise_contrasts(groups: list[str]) -> list[dict[str, str]]:
    if len(groups) < 2:
        return []
    reference_group = groups[0]
    return [
        {
            "comparison_id": _contrast_id(group, reference_group),
            "reference_group": reference_group,
            "case_group": group,
        }
        for group in groups[1:]
    ]


def _normalize_contrast_rows(rows: list[object]) -> list[dict[str, object]]:
    normalized = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        reference_group = str(row.get("reference_group", "")).strip()
        case_group = str(row.get("case_group", "")).strip()
        comparison_id = str(row.get("comparison_id", "")).strip()
        if not comparison_id and reference_group and case_group:
            comparison_id = _contrast_id(case_group, reference_group)
        if not comparison_id:
            comparison_id = f"comparison_{index}"
        normalized.append(
            {
                "comparison_id": comparison_id,
                "reference_group": reference_group,
                "case_group": case_group,
                "reference_samples": _as_string_list(row.get("reference_samples")),
                "case_samples": _as_string_list(row.get("case_samples")),
                "reference_filters": _normalize_filter_rows(row.get("reference_filters")),
                "case_filters": _normalize_filter_rows(row.get("case_filters")),
            }
        )
    return normalized


def _valid_contrasts(rows: list[dict[str, object]], groups: list[str]) -> list[dict[str, object]]:
    valid = []
    seen = set()
    seen_ids = set()
    group_set = set(groups)
    for row in _normalize_contrast_rows(rows):
        reference_group = row["reference_group"]
        case_group = row["case_group"]
        if reference_group == case_group or reference_group not in group_set or case_group not in group_set:
            continue
        comparison_id = _safe_name(row["comparison_id"]) or _contrast_id(case_group, reference_group)
        if comparison_id in seen_ids:
            continue
        key = (comparison_id, reference_group, case_group)
        if key in seen:
            continue
        seen.add(key)
        seen_ids.add(comparison_id)
        valid.append(
            {
                "comparison_id": comparison_id,
                "reference_group": reference_group,
                "case_group": case_group,
                "reference_samples": row.get("reference_samples", []),
                "case_samples": row.get("case_samples", []),
            }
        )
    return valid


def _contrast_id(case_group: str, reference_group: str) -> str:
    return f"{_safe_name(case_group)}_vs_{_safe_name(reference_group)}"


def _composite_group_column_name(group_columns: list[str]) -> str:
    if len(group_columns) == 1:
        return group_columns[0]
    safe_parts = [_safe_name(str(column)) for column in group_columns]
    return COMPOSITE_GROUP_PREFIX + "__".join(safe_parts)


def _metadata_group_values(metadata: pd.DataFrame, group_columns: list[str]) -> list[str]:
    if len(group_columns) == 1:
        return [str(value) for value in metadata[group_columns[0]].dropna().unique()]

    group_series = _build_composite_group_series(metadata, group_columns)
    return group_series.dropna().drop_duplicates().astype(str).tolist()


def _build_composite_group_series(metadata: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    def label_row(row: pd.Series) -> str | None:
        parts = []
        for column in group_columns:
            value = row.get(column)
            if pd.isna(value) or str(value).strip() == "":
                return None
            parts.append(f"{column}={value}")
        return COMPOSITE_GROUP_SEPARATOR.join(parts)

    return metadata.apply(label_row, axis=1)


def _add_composite_group_column(table: pd.DataFrame, settings: dict[str, object]) -> pd.DataFrame:
    group_columns = settings.get("group_columns")
    if not isinstance(group_columns, list) or len(group_columns) <= 1:
        return table
    group_columns = [str(column) for column in group_columns if str(column) in table.columns]
    if len(group_columns) <= 1:
        return table

    output = table.copy()
    output[_composite_group_column_name(group_columns)] = _build_composite_group_series(output, group_columns)
    return output


def _metadata_preview_from_samplesheet(uploaded_file, samplesheet: pd.DataFrame) -> pd.DataFrame:
    if {"RCC_FILE", "SAMPLE_ID"} - set(samplesheet.columns):
        return pd.DataFrame(columns=["sample_id"])

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        _save_upload(uploaded_file, temp_path)
        return metadata_from_samplesheet(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _metadata_preview_from_samplesheet_frame(samplesheet: pd.DataFrame) -> pd.DataFrame:
    required = {"RCC_FILE", "SAMPLE_ID"}
    if required - set(samplesheet.columns):
        return pd.DataFrame(columns=["sample_id"])

    metadata = samplesheet.drop(columns=["RCC_FILE", "RCC_FILE_NAME"], errors="ignore").copy()
    metadata["SAMPLE_ID"] = metadata["SAMPLE_ID"].map(_normalize_sample_id)
    metadata = metadata.rename(columns={"SAMPLE_ID": "sample_id"})
    return metadata.drop_duplicates(subset=["sample_id"], keep="first")


def _render_samplesheet_editor(samplesheet: pd.DataFrame) -> pd.DataFrame:
    return _render_samplesheet_upload_editor(samplesheet)


def _materialize_run(
    run_dir: Path,
    counts_file,
    metadata_file,
    samplesheet_file,
    rcc_files,
    rlf_file,
    edited_samplesheet: pd.DataFrame | None,
    edited_metadata: pd.DataFrame | None,
    settings: dict[str, object],
    qc_settings: dict[str, object] | None = None,
    processed_counts: bool = False,
) -> None:
    input_dir = run_dir / "input"
    results_dir = run_dir / "results"
    reports_dir = run_dir / "reports"
    input_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    counts_path = None
    metadata_path = None
    samplesheet_path = None
    rlf_path = None

    if processed_counts:
        counts_path = input_dir / counts_file.name
        metadata_path = input_dir / metadata_file.name
        counts = _normalize_counts_preview(_read_uploaded_table(counts_file))
        metadata_source = edited_metadata if edited_metadata is not None else _read_uploaded_table(metadata_file)
        metadata = _ensure_sample_id(metadata_source)
        if "sample_id" not in metadata.columns:
            raise ValueError("Metadata must contain a `sample_id` or `SAMPLE_ID` column.")
        metadata["sample_id"] = metadata["sample_id"].map(_normalize_sample_id)
        counts = _filter_counts_by_included_samples(counts)
        metadata = _filter_metadata_by_included_samples(metadata)
        metadata = _add_composite_group_column(metadata, settings)
        counts_sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
        rename_map = {col: _normalize_sample_id(col) for col in counts_sample_cols}
        counts = counts.rename(columns=rename_map)
        counts = _apply_recorded_sample_id_renames_to_counts(counts)
        sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
        missing_metadata = sorted(set(sample_cols) - set(metadata["sample_id"]))
        if missing_metadata:
            raise ValueError("Missing metadata rows for: " + ", ".join(missing_metadata))

        _save_upload(counts_file, counts_path)
        metadata.to_csv(metadata_path, index=False)
        _write_processed_counts_outputs(counts, metadata, results_dir)
    elif rcc_files:
        rcc_dir = input_dir / "rcc"
        rcc_dir.mkdir(exist_ok=True)
        name_to_path = {}
        for uploaded_file in rcc_files:
            rcc_path = rcc_dir / uploaded_file.name
            _save_upload(uploaded_file, rcc_path)
            name_to_path[uploaded_file.name] = rcc_path

        if rlf_file is not None:
            rlf_path = input_dir / rlf_file.name
            _save_upload(rlf_file, rlf_path)

        samplesheet_path = input_dir / "samplesheet.csv"
        generated = edited_samplesheet.copy() if edited_samplesheet is not None else pd.DataFrame()
        if generated.empty:
            generated = _build_samplesheet_from_rcc_uploads(rcc_files)
        generated = _combine_samplesheet_and_metadata(generated, edited_metadata)
        generated = _filter_samplesheet_by_included_samples(generated)
        generated = _add_composite_group_column(generated, settings)
        generated["SAMPLE_ID"] = generated["SAMPLE_ID"].map(_normalize_sample_id)
        generated["RCC_FILE"] = generated["RCC_FILE_NAME"].map(
            lambda name: str(Path("rcc") / str(name_to_path.get(str(name), rcc_dir / str(name)).name))
        )
        generated.to_csv(samplesheet_path, index=False)
    elif samplesheet_file is not None:
        samplesheet_path = input_dir / samplesheet_file.name
        if edited_samplesheet is not None:
            generated = _combine_samplesheet_and_metadata(edited_samplesheet, edited_metadata)
            generated = _filter_samplesheet_by_included_samples(generated)
            generated = _add_composite_group_column(generated, settings)
            generated.to_csv(samplesheet_path, index=False)
        else:
            generated = _read_uploaded_table(samplesheet_file)
            generated = _filter_samplesheet_by_included_samples(generated)
            generated = _add_composite_group_column(generated, settings)
            generated.to_csv(samplesheet_path, index=False)
    else:
        counts_path = input_dir / counts_file.name
        metadata_path = input_dir / metadata_file.name
        counts = _normalize_counts_preview(_read_uploaded_table(counts_file))
        counts = _apply_recorded_sample_id_renames_to_counts(counts)
        counts = _filter_counts_by_included_samples(counts)
        if edited_metadata is not None:
            metadata = edited_metadata.copy()
        else:
            metadata = _read_uploaded_table(metadata_file)
        metadata = _ensure_sample_id(metadata)
        metadata["sample_id"] = metadata["sample_id"].map(_normalize_sample_id)
        metadata = _filter_metadata_by_included_samples(metadata)
        metadata = _add_composite_group_column(metadata, settings)
        counts.to_csv(counts_path, index=False)
        metadata.to_csv(metadata_path, index=False)

    inputs = {"probe_annotation": str(rlf_path) if rlf_path else ""}
    if samplesheet_path is not None:
        inputs["samplesheet"] = str(samplesheet_path)
    else:
        inputs["counts"] = str(counts_path)
        inputs["metadata"] = str(metadata_path)

    config = {
        "project": {"name": run_dir.name},
        "inputs": inputs,
        "outputs": {
            "results_dir": str(results_dir),
            "reports_dir": str(reports_dir),
        },
        "analysis": {
            "group_column": settings["group_column"],
            "group_columns": settings.get("group_columns", [settings["group_column"]]),
            "reference_group": settings["reference_group"],
            "case_group": settings["case_group"],
            "contrasts": settings.get("contrasts", []),
            "min_count": settings["min_count"],
            "min_samples": settings["min_samples"],
            "top_variable_genes": settings["top_variable_genes"],
            "skip_low_count_filter": bool(processed_counts),
        },
        "normalization": {
            "method": settings["normalization_method"],
            "housekeeping_genes": [],
        },
        "qc": qc_settings or {},
        "r": {"executable": "Rscript"},
    }

    existing_config = _read_saved_config(run_dir)
    existing_ui = existing_config.get("ui") if isinstance(existing_config, dict) else None
    if isinstance(existing_ui, dict):
        config["ui"] = existing_ui

    _write_saved_config(run_dir / "pipeline.yaml", config)


def _write_processed_counts_outputs(counts: pd.DataFrame, metadata: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    numeric_counts = counts.copy()
    numeric_counts[sample_cols] = numeric_counts[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    qc_summary = pd.DataFrame(
        {
            "sample_id": sample_cols,
            "library_size": numeric_counts[sample_cols].sum(axis=0).to_numpy(dtype=float),
            "detected_endogenous_probes": (numeric_counts[sample_cols] > 0).sum(axis=0).to_numpy(dtype=int),
            "positive_control_sum": np.nan,
            "negative_control_mean": np.nan,
            "qc_status": "PREPROCESSED",
            "qc_mode": "preprocessed",
            "hk_geomean": np.nan,
            "hk_geomean_flag": "SKIPPED",
            "positive_control_linearity": np.nan,
            "positive_control_linearity_flag": "SKIPPED",
            "fov_registration_rate": np.nan,
            "fov_registration_flag": "SKIPPED",
            "binding_density": np.nan,
            "binding_density_flag": "SKIPPED",
            "exclusion_status": "INCLUDED",
            "exclusion_reason": "",
            "qc_warnings": "Final counts loaded: technical QC was not recalculated",
            "qc_fail_reasons": "",
        }
    )
    probe_annotation = numeric_counts[["CodeClass", "Name"]].drop_duplicates()
    parsed_annotation = parse_probe_annotation(numeric_counts)
    filter_decisions = numeric_counts[["CodeClass", "Name"]].copy()
    filter_decisions["filter_status"] = "PREPROCESSED"
    filter_decisions["filter_reason"] = "Final counts loaded"

    numeric_counts.to_csv(results_dir / "counts_raw.csv", index=False)
    numeric_counts.to_csv(results_dir / "counts_background_corrected.csv", index=False)
    numeric_counts.to_csv(results_dir / "counts_filtered.csv", index=False)
    numeric_counts.to_csv(results_dir / "counts_normalized.csv", index=False)
    metadata.to_csv(results_dir / "metadata.csv", index=False)
    probe_annotation.to_csv(results_dir / "probe_annotation.csv", index=False)
    parsed_annotation.to_csv(results_dir / "probe_annotation_parsed.csv", index=False)
    qc_summary.to_csv(results_dir / "qc_summary.csv", index=False)
    qc_summary[
        [
            "sample_id",
            "qc_mode",
            "qc_status",
            "hk_geomean",
            "hk_geomean_flag",
            "positive_control_linearity",
            "positive_control_linearity_flag",
            "fov_registration_rate",
            "fov_registration_flag",
            "binding_density",
            "binding_density_flag",
            "exclusion_status",
            "exclusion_reason",
        ]
    ].to_csv(results_dir / "sample_qc.csv", index=False)
    qc_summary[["sample_id", "qc_status", "qc_warnings", "qc_fail_reasons"]].to_csv(
        results_dir / "sample_qc_decisions.csv",
        index=False,
    )
    filter_decisions.to_csv(results_dir / "gene_filter_decisions.csv", index=False)


def _combine_samplesheet_and_metadata(
    samplesheet: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    output = samplesheet.copy()
    if metadata is None or metadata.empty or "sample_id" not in metadata.columns:
        return output

    output["SAMPLE_ID"] = output["SAMPLE_ID"].map(_normalize_sample_id)
    metadata_for_merge = metadata.copy()
    metadata_for_merge["sample_id"] = metadata_for_merge["sample_id"].map(_normalize_sample_id)
    metadata_for_merge = metadata_for_merge.rename(columns={"sample_id": "SAMPLE_ID"})

    metadata_columns = [
        col
        for col in metadata_for_merge.columns
        if col != "SAMPLE_ID" and col not in {"RCC_FILE", "RCC_FILE_NAME"}
    ]
    output = output.drop(columns=[col for col in metadata_columns if col in output.columns])
    return output.merge(metadata_for_merge[["SAMPLE_ID", *metadata_columns]], on="SAMPLE_ID", how="left")


def _save_upload(uploaded_file, path: Path) -> None:
    uploaded_file.seek(0)
    path.write_bytes(uploaded_file.read())


def _execute_steps(config_path: Path, steps: list[str]) -> None:
    import streamlit as st

    config = load_config(config_path)
    for step in steps:
        with st.status(f"Running {step}...", expanded=True) as status:
            try:
                if step == "prepare":
                    prepare_pipeline(config)
                elif step == "normalize":
                    run_r_script(config, "normalize.R")
                elif step == "differential":
                    run_r_script(config, "differential_expression.R")
                elif step == "report":
                    run_r_script(config, "report.R")
            except subprocess.CalledProcessError as exc:
                status.update(label=f"{step} failed", state="error")
                st.error(f"Error in step `{step}`: {exc}")
                break
            except Exception as exc:  # UI boundary: show actionable feedback.
                status.update(label=f"{step} failed", state="error")
                st.error(f"Error in step `{step}`: {exc}")
                break
            else:
                _record_analysis_savepoint(config_path, step)
                status.update(label=f"{step} completed", state="complete")


def _render_generated_files(results_dir: Path, reports_dir: Path, key_prefix: str) -> None:
    import streamlit as st

    st.subheader("Generated files")
    files = sorted(results_dir.glob("*.csv")) + sorted(reports_dir.glob("*.png"))
    if not files:
        st.caption("No output generated for this analysis.")
        return

    for path in files:
        with path.open("rb") as handle:
            st.download_button(
                label=f"Download {path.name}",
                data=handle.read(),
                file_name=path.name,
                key=f"download-{key_prefix}-{path}",
            )


def _render_results_explorer(results_dir: Path, reports_dir: Path) -> None:
    import streamlit as st

    norm_path = results_dir / "counts_normalized.csv"
    de_path, comparison_label = _select_differential_result(results_dir)

    if de_path is None or not de_path.exists():
        _render_qc_explorer(results_dir)
        st.info("Run differential analysis to explore results.")
        return

    _render_qc_explorer(results_dir)

    de = pd.read_csv(de_path)
    st.subheader("Differential expression")
    if comparison_label:
        st.caption(f"Selected contrast: {comparison_label}")
    de = _prepare_de_table(de)

    threshold_cols = st.columns([1, 1, 1, 1])
    p_value_metric = threshold_cols[0].radio(
        "P-value metric",
        ["adj.P.Val", "P.Value"],
        horizontal=True,
        key="selected_p_value_metric",
    )
    max_p_value = threshold_cols[1].slider(
        f"{p_value_metric} threshold",
        0.0,
        1.0,
        0.05,
        0.01,
        key="selected_p_value_threshold",
    )
    min_abs_logfc = threshold_cols[2].slider(
        "|logFC| threshold",
        0.0,
        5.0,
        1.0,
        0.1,
        key="selected_logfc_threshold",
    )
    max_heatmap_genes = threshold_cols[3].number_input(
        "Max heatmap genes",
        min_value=5,
        max_value=500,
        value=50,
        step=5,
    )

    selected = _filter_de_table(
        de,
        p_value_metric=p_value_metric,
        max_p_value=max_p_value,
        min_abs_logfc=min_abs_logfc,
    )
    st.caption(f"Analyzed genes: {len(de):,} | Genes above current thresholds: {len(selected):,}")
    _render_de_result_table(de, selected, norm_path)
    _render_interactive_volcano(de, norm_path, results_dir, p_value_metric, max_p_value, min_abs_logfc)
    _render_gene_search_panel(de, selected, norm_path, results_dir)
    _render_selected_gene_plots(selected, norm_path, results_dir, max_heatmap_genes)

    _render_report_images(reports_dir)


def _render_de_result_table(de: pd.DataFrame, selected: pd.DataFrame, norm_path: Path) -> None:
    import streamlit as st

    st.subheader("Genes above threshold")
    table_scope_options = ["Genes above current thresholds", "All analyzed genes"]
    if st.session_state.get("de_table_scope") not in table_scope_options:
        st.session_state["de_table_scope"] = table_scope_options[0]
    controls = st.columns([1.3, 1, 1.2])
    table_scope = controls[0].radio(
        "Result list",
        table_scope_options,
        horizontal=True,
        key="de_table_scope",
    )
    sort_options = [col for col in ["adj.P.Val", "P.Value", "logFC"] if col in de]
    sort_col = controls[1].selectbox("Sort by", sort_options) if sort_options else None
    gene_filter = controls[2].text_input("Search table", key="de_table_gene_filter")

    filtered = selected.copy() if table_scope == "Genes above current thresholds" else de.copy()
    if gene_filter and "Name" in filtered:
        filtered = filtered[filtered["Name"].astype(str).str.contains(gene_filter, case=False, na=False)]
    if sort_col:
        filtered = filtered.sort_values(sort_col)

    st.metric("Selected features", f"{len(filtered):,}")
    st.dataframe(filtered, use_container_width=True, height=260)

    if not filtered.empty:
        st.download_button(
            "Download filtered results",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="differential_expression_filtered.csv",
        )

    if norm_path.exists() and "Name" in filtered and not filtered.empty:
        gene_options = filtered["Name"].astype(str).tolist()
        selected_gene = st.selectbox("Show expression from this table", gene_options)
        if st.button("Show gene from table", key="show_gene_from_de_table"):
            st.session_state["active_gene"] = selected_gene
            st.rerun()


def _select_differential_result(results_dir: Path) -> tuple[Path | None, str]:
    import streamlit as st

    summary_path = results_dir / "comparison_summary.csv"
    legacy_path = results_dir / "differential_expression.csv"
    if not summary_path.exists():
        return legacy_path, ""

    summary = pd.read_csv(summary_path)
    if summary.empty or "result_file" not in summary.columns:
        return legacy_path, ""

    summary = summary.copy()
    summary["result_path"] = summary["result_file"].map(lambda value: results_dir / str(value))
    summary = summary[summary["result_path"].map(lambda path: Path(path).exists())]
    if summary.empty:
        return legacy_path, ""

    st.subheader("Comparison summary")
    visible_cols = [
        col
        for col in [
            "comparison_id",
            "reference_group",
            "case_group",
            "n_reference",
            "n_case",
            "n_significant",
            "n_up",
            "n_down",
            "top_gene",
        ]
        if col in summary.columns
    ]
    st.dataframe(summary[visible_cols], use_container_width=True, hide_index=True)

    labels = [
        f"{row.comparison_id}: {row.case_group} vs {row.reference_group}"
        for row in summary.itertuples(index=False)
    ]
    selected = st.selectbox("Contrast to explore", labels, key="selected_comparison_result")
    row = summary.iloc[labels.index(selected)]
    _render_selected_comparison_samples(results_dir, row)
    return Path(row["result_path"]), selected


def _render_selected_comparison_samples(results_dir: Path, summary_row: pd.Series) -> None:
    import streamlit as st

    metadata = _read_results_metadata(results_dir)
    analysis_samples = _read_analysis_sample_columns(results_dir)
    settings = _read_analysis_settings(results_dir)
    group_col = settings.get("group_column")
    if metadata.empty or "sample_id" not in metadata.columns:
        return

    reference_group = str(summary_row.get("reference_group", "reference"))
    case_group = str(summary_row.get("case_group", "case"))
    explicit_reference = _summary_sample_list(summary_row.get("reference_samples"))
    explicit_case = _summary_sample_list(summary_row.get("case_samples"))
    recap = metadata.copy()
    recap["sample_id"] = recap["sample_id"].astype(str)

    if explicit_reference or explicit_case:
        sample_to_group = {
            **{sample: f"Group 1: {reference_group}" for sample in explicit_reference},
            **{sample: f"Group 2: {case_group}" for sample in explicit_case},
        }
        recap = recap[recap["sample_id"].isin(sample_to_group)].copy()
        recap["comparison_group"] = recap["sample_id"].map(sample_to_group)
    elif isinstance(group_col, str) and group_col in metadata.columns:
        recap[group_col] = recap[group_col].astype(str)
        recap = recap[recap[group_col].isin([reference_group, case_group])].copy()
        recap["comparison_group"] = recap[group_col].map(
            {
                reference_group: f"Group 1: {reference_group}",
                case_group: f"Group 2: {case_group}",
            }
        )
    else:
        return

    if analysis_samples and not (explicit_reference or explicit_case):
        recap = recap[recap["sample_id"].isin(analysis_samples)].copy()
    if recap.empty:
        return

    group_columns = settings.get("group_columns") or [group_col]
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    display_cols = [
        col
        for col in ["sample_id", "comparison_group", *group_columns]
        if col in recap.columns
    ]
    display_cols = list(dict.fromkeys(display_cols))
    recap = recap[display_cols].sort_values(["comparison_group", "sample_id"])

    with st.expander("Samples in selected comparison", expanded=True):
        st.markdown(_table_html(recap), unsafe_allow_html=True)


def _table_html(table: pd.DataFrame) -> str:
    styles = """
<style>
.compact-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.compact-table th,
.compact-table td {
  border-bottom: 1px solid rgba(49, 51, 63, 0.18);
  padding: 0.35rem 0.5rem;
  text-align: left;
  vertical-align: top;
}
.compact-table th {
  font-weight: 600;
}
</style>
"""
    return styles + table.to_html(index=False, escape=True, border=0, classes="compact-table")


def _prepare_de_table(de: pd.DataFrame) -> pd.DataFrame:
    output = de.copy()
    output["adj.P.Val"] = pd.to_numeric(output["adj.P.Val"], errors="coerce").fillna(1.0)
    output["P.Value"] = pd.to_numeric(output.get("P.Value", 1.0), errors="coerce").fillna(1.0)
    output["logFC"] = pd.to_numeric(output["logFC"], errors="coerce").fillna(0.0)
    return output


def _filter_de_table(
    de: pd.DataFrame,
    p_value_metric: str,
    max_p_value: float,
    min_abs_logfc: float,
) -> pd.DataFrame:
    metric = p_value_metric if p_value_metric in de else "adj.P.Val"
    return de[(de[metric] <= max_p_value) & (de["logFC"].abs() >= min_abs_logfc)].copy()


def _render_gene_search_panel(
    de: pd.DataFrame,
    selected: pd.DataFrame,
    norm_path: Path,
    results_dir: Path,
) -> None:
    import streamlit as st

    if not norm_path.exists():
        return

    normalized = pd.read_csv(norm_path)
    if "Name" not in normalized:
        return

    all_genes = normalized["Name"].astype(str).dropna().drop_duplicates().sort_values().tolist()
    selected_genes = selected["Name"].astype(str).dropna().drop_duplicates().tolist() if "Name" in selected else []
    selected_genes = sorted([gene for gene in selected_genes if gene in set(all_genes)])
    de_genes = set(de["Name"].astype(str)) if "Name" in de else set()
    all_genes = [gene for gene in all_genes if gene in de_genes or not de_genes]

    with st.expander("Gene and expression profile", expanded=True):
        scope_col, search_col, button_col = st.columns([0.9, 1.7, 0.7])
        search_scope_options = ["All analyzed genes", "Genes above current thresholds"]
        if st.session_state.get("gene_search_scope") not in search_scope_options:
            st.session_state["gene_search_scope"] = search_scope_options[0]
        scope = scope_col.radio(
            "Source",
            search_scope_options,
            horizontal=True,
            key="gene_search_scope",
        )
        gene_options = selected_genes if scope == "Genes above current thresholds" else all_genes
        if not gene_options:
            st.caption("No gene passes the current thresholds. Switch Source to 'All analyzed genes'.")
            return

        current_gene = st.session_state.get("active_gene")
        index = gene_options.index(current_gene) if current_gene in gene_options else 0
        searched_gene = search_col.selectbox(
            "Gene",
            gene_options,
            index=index,
            key=f"gene_search_selectbox_{scope}",
        )
        if button_col.button("Show", key="show_gene_from_search"):
            st.session_state["active_gene"] = searched_gene

        active_gene = st.session_state.get("active_gene")
        if active_gene:
            _render_gene_expression_profile(active_gene, norm_path, results_dir, de=de)


def _render_interactive_volcano(
    de: pd.DataFrame,
    norm_path: Path,
    results_dir: Path,
    p_value_metric: str,
    max_p_value: float,
    min_abs_logfc: float,
) -> None:
    import plotly.express as px
    import streamlit as st

    required = {"Name", "logFC", "adj.P.Val", "P.Value"}
    if not required.issubset(de.columns):
        return

    metric = p_value_metric if p_value_metric in de else "adj.P.Val"
    metric_label = "adjusted p-value" if metric == "adj.P.Val" else "nominal p-value"
    volcano = de.copy()
    volcano["neg_log10_p_metric"] = -volcano[metric].clip(lower=1e-300).map(math.log10)
    volcano["adj_p_label"] = volcano["adj.P.Val"].map(lambda value: f"{value:.3g}")
    volcano["p_label"] = volcano["P.Value"].map(lambda value: f"{value:.3g}")
    volcano["regulation"] = "not_significant"
    volcano.loc[
        (volcano[metric] <= max_p_value) & (volcano["logFC"] >= min_abs_logfc),
        "regulation",
    ] = "up"
    volcano.loc[
        (volcano[metric] <= max_p_value) & (volcano["logFC"] <= -min_abs_logfc),
        "regulation",
    ] = "down"
    active_gene = st.session_state.get("active_gene")

    fig = px.scatter(
        volcano,
        x="logFC",
        y="neg_log10_p_metric",
        color="regulation",
        color_discrete_map={
            "up": "#B23A48",
            "down": "#2F6690",
            "not_significant": "#8A8F98",
        },
        custom_data=["Name", "adj_p_label", "p_label"],
        hover_data={
            "Name": True,
            "logFC": ":.3f",
            "adj_p_label": True,
            "p_label": True,
            "adj.P.Val": False,
            "P.Value": False,
            "neg_log10_p_metric": False,
            "regulation": False,
        },
    )
    fig.update_traces(
        marker={"size": 7, "opacity": 0.78},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "logFC: %{x:.3f}<br>"
            f"-log10 {metric}: %{{y:.3f}}<br>"
            "adj.P.Val: %{customdata[1]}<br>"
            "P.Value: %{customdata[2]}<extra></extra>"
        ),
    )
    fig.add_vline(x=-min_abs_logfc, line_width=1, line_dash="dash", line_color="#666666")
    fig.add_vline(x=min_abs_logfc, line_width=1, line_dash="dash", line_color="#666666")
    fig.add_hline(y=-math.log10(max(max_p_value, 1e-300)), line_width=1, line_dash="dash", line_color="#666666")
    if active_gene:
        active_row = volcano[volcano["Name"].astype(str).eq(str(active_gene))]
        if not active_row.empty:
            fig.add_scatter(
                x=active_row["logFC"],
                y=active_row["neg_log10_p_metric"],
                mode="markers",
                marker={
                    "size": 16,
                    "symbol": "circle-open",
                    "color": "#111111",
                    "line": {"width": 3},
                },
                name="selected gene",
                customdata=active_row[["Name", "adj_p_label", "p_label"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "logFC: %{x:.3f}<br>"
                    f"-log10 {metric}: %{{y:.3f}}<br>"
                    "adj.P.Val: %{customdata[1]}<br>"
                    "P.Value: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
    fig.update_layout(
        height=500,
        margin={"l": 45, "r": 20, "t": 18, "b": 45},
        legend_title_text="",
        xaxis_title="log2 fold change",
        yaxis_title=f"-log10 {metric_label}",
    )

    st.subheader("Interactive plots")
    _, volcano_col, _ = st.columns([0.12, 1, 0.12])
    with volcano_col:
        state = st.plotly_chart(
            fig,
            use_container_width=True,
            key="interactive_volcano",
            on_select="rerun",
            selection_mode="points",
        )
    clicked_gene = _selected_gene_from_plotly_state(state)
    if clicked_gene and clicked_gene != st.session_state.get("active_gene"):
        st.session_state["active_gene"] = clicked_gene
        st.rerun()
    if not st.session_state.get("active_gene"):
        st.caption("Click a volcano point or search for a gene below to view expression.")


def _selected_gene_from_plotly_state(state) -> str | None:
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    if not selection:
        return None

    points = selection.get("points") if isinstance(selection, dict) else getattr(selection, "points", None)
    if isinstance(points, list) and points:
        point = points[0]
        customdata = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
        if isinstance(customdata, np.ndarray):
            flattened = customdata.ravel()
            if len(flattened):
                return str(flattened[0])
        if isinstance(customdata, (list, tuple)) and customdata:
            return str(customdata[0])
        if customdata is not None and str(customdata):
            return str(customdata)
        point_y = point.get("y") if isinstance(point, dict) else getattr(point, "y", None)
        if point_y:
            return str(point_y)
        point_text = point.get("text") if isinstance(point, dict) else getattr(point, "text", None)
        if point_text:
            return str(point_text)
    return None


def _render_selected_gene_plots(
    selected: pd.DataFrame,
    norm_path: Path,
    results_dir: Path,
    max_heatmap_genes: int,
) -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    if selected.empty:
        st.info("No gene passes the current thresholds: dynamic PCA and heatmap are not drawn.")
        return
    if not norm_path.exists():
        st.warning("Normalized matrix is not available for dynamic PCA and heatmap.")
        return

    normalized = pd.read_csv(norm_path)
    selected_genes = selected["Name"].astype(str).tolist()
    matrix = normalized[normalized["Name"].astype(str).isin(selected_genes)].copy()
    if matrix.empty:
        st.warning("Selected genes are not present in the normalized matrix.")
        return
    ordered_selection = selected.assign(abs_logFC=selected["logFC"].abs()).sort_values(
        "abs_logFC",
        ascending=False,
    )
    ordered_genes = ordered_selection["Name"].astype(str).tolist()
    matrix = _order_matrix_by_genes(matrix, ordered_genes)
    metadata = _read_results_metadata(results_dir)
    matrix, metadata, group_col, comparison_groups = _filter_matrix_to_selected_contrast(
        matrix,
        metadata,
        selected,
        results_dir,
    )
    if matrix.empty or len([col for col in matrix.columns if col not in {"CodeClass", "Name"}]) < 2:
        st.info("The selected contrast has fewer than two matching samples for dynamic PCA and heatmap.")
        return

    with st.expander("Dynamic PCA and heatmap", expanded=False):
        if group_col and len(comparison_groups) == 2:
            st.caption(f"Samples shown: {comparison_groups[1]} vs {comparison_groups[0]}")
        pca_col, heatmap_col = st.columns(2)

        with pca_col:
            _render_selected_gene_pca(matrix, metadata, px, group_col, comparison_groups)

        with heatmap_col:
            heatmap_selection = ordered_selection.head(int(max_heatmap_genes))
            heatmap_selection = heatmap_selection.sort_values("logFC", ascending=False)
            heatmap_genes = heatmap_selection["Name"].astype(str).tolist()
            heatmap_matrix = _order_matrix_by_genes(matrix, heatmap_genes)
            _render_selected_gene_heatmap(heatmap_matrix, go)

    with st.expander("Expression profiles for genes above threshold", expanded=False):
        show_all = st.checkbox(
            "Show profiles for all selected genes",
            value=False,
            key="show_all_selected_gene_profiles",
        )
        if show_all:
            max_profile_genes = st.number_input(
                "Max displayed genes",
                min_value=1,
                max_value=max(1, len(matrix)),
                value=min(24, len(matrix)),
                step=1,
                key="max_selected_gene_profiles",
            )
            _render_all_selected_gene_profiles(matrix.head(int(max_profile_genes)), metadata, px, results_dir)


def _order_matrix_by_genes(matrix: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    indexed = matrix.copy()
    indexed["Name"] = indexed["Name"].astype(str)
    indexed = indexed.drop_duplicates(subset=["Name"], keep="first").set_index("Name")
    present_genes = [gene for gene in genes if gene in indexed.index]
    return indexed.loc[present_genes].reset_index()


def _filter_matrix_to_selected_contrast(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    de: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None, list[str]]:
    sample_cols = [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    if "sample_id" not in metadata.columns or not sample_cols:
        return matrix, metadata, None, []

    reference_samples, case_samples, sample_groups = _comparison_sample_lists_from_summary(results_dir, de)
    if reference_samples or case_samples:
        ordered_samples = [sample for sample in [*reference_samples, *case_samples] if sample in sample_cols]
        if len(ordered_samples) < 2:
            return matrix.iloc[:, :0].copy(), metadata, "comparison_group", []
        filtered_metadata = metadata.copy()
        filtered_metadata["sample_id"] = filtered_metadata["sample_id"].astype(str)
        filtered_metadata = filtered_metadata[filtered_metadata["sample_id"].isin(ordered_samples)].copy()
        filtered_metadata["comparison_group"] = filtered_metadata["sample_id"].map(sample_groups)
        filtered_metadata = filtered_metadata.set_index("sample_id").loc[ordered_samples].reset_index()
        comparison_groups = list(dict.fromkeys(sample_groups[sample] for sample in ordered_samples if sample in sample_groups))
        return (
            matrix[["CodeClass", "Name", *ordered_samples]].copy(),
            filtered_metadata,
            "comparison_group",
            comparison_groups,
        )

    group_col, comparison_groups = _selected_de_group_info(de, metadata, results_dir)
    if not group_col or len(comparison_groups) < 2:
        return matrix, metadata, group_col, comparison_groups

    filtered_metadata = metadata.copy()
    filtered_metadata["sample_id"] = filtered_metadata["sample_id"].astype(str)
    filtered_metadata[group_col] = filtered_metadata[group_col].astype(str)
    filtered_metadata = filtered_metadata[filtered_metadata[group_col].isin(comparison_groups)]
    ordered_samples = [
        sample
        for group in comparison_groups
        for sample in filtered_metadata.loc[filtered_metadata[group_col].eq(group), "sample_id"].tolist()
        if sample in sample_cols
    ]
    if len(ordered_samples) < 2:
        return matrix.iloc[:, :0].copy(), filtered_metadata, group_col, comparison_groups

    filtered_metadata = filtered_metadata[filtered_metadata["sample_id"].isin(ordered_samples)].copy()
    filtered_metadata = _add_comparison_group_labels(filtered_metadata, group_col, comparison_groups)
    return matrix[["CodeClass", "Name", *ordered_samples]].copy(), filtered_metadata, group_col, comparison_groups


def _comparison_sample_lists_from_summary(
    results_dir: Path,
    de: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, str]]:
    if de.empty or "comparison_id" not in de.columns:
        return [], [], {}
    summary_path = results_dir / "comparison_summary.csv"
    if not summary_path.exists():
        return [], [], {}
    summary = pd.read_csv(summary_path)
    if "comparison_id" not in summary.columns:
        return [], [], {}
    comparison_id = str(de["comparison_id"].dropna().astype(str).iloc[0])
    rows = summary[summary["comparison_id"].astype(str).eq(comparison_id)]
    if rows.empty:
        return [], [], {}
    row = rows.iloc[0]
    reference_samples = _summary_sample_list(row.get("reference_samples"))
    case_samples = _summary_sample_list(row.get("case_samples"))
    labels = _comparison_group_labels(
        [
            str(row.get("reference_group") or "Reference"),
            str(row.get("case_group") or "Case"),
        ]
    )
    sample_groups = {
        **{sample: labels[0] for sample in reference_samples},
        **{sample: labels[1] for sample in case_samples},
    }
    return reference_samples, case_samples, sample_groups


def _summary_sample_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(sample).strip() for sample in value if str(sample).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [sample.strip() for sample in str(value).split(";") if sample.strip()]


def _add_comparison_group_labels(
    table: pd.DataFrame,
    group_col: str,
    comparison_groups: list[str],
) -> pd.DataFrame:
    output = table.copy()
    if len(comparison_groups) < 2 or group_col not in output.columns:
        output["comparison_group"] = np.nan
        return output
    output[group_col] = output[group_col].astype(str)
    labels = _comparison_group_labels(comparison_groups)
    output["comparison_group"] = output[group_col].map(
        {
            comparison_groups[0]: labels[0],
            comparison_groups[1]: labels[1],
        }
    )
    return output


def _comparison_group_labels(comparison_groups: list[str]) -> list[str]:
    if len(comparison_groups) >= 2 and [group.lower() for group in comparison_groups[:2]] == [
        "reference",
        "case",
    ]:
        return ["Reference", "Case"]
    if len(comparison_groups) >= 2:
        return [str(comparison_groups[0]), str(comparison_groups[1])]
    return comparison_groups


def _selected_de_group_info(
    de: pd.DataFrame,
    metadata: pd.DataFrame,
    results_dir: Path,
) -> tuple[str | None, list[str]]:
    settings = _read_analysis_settings(results_dir)
    group_col = settings.get("group_column")
    if not isinstance(group_col, str) or group_col not in metadata.columns:
        group_col = None

    required = {"reference_group", "case_group"}
    if group_col and required.issubset(de.columns) and not de.empty:
        reference_values = de["reference_group"].dropna().astype(str)
        case_values = de["case_group"].dropna().astype(str)
        if reference_values.empty or case_values.empty:
            return _comparison_group_info(metadata, results_dir)
        reference_group = str(reference_values.iloc[0])
        case_group = str(case_values.iloc[0])
        observed = set(metadata[group_col].dropna().astype(str))
        groups = [group for group in [reference_group, case_group] if group in observed]
        if len(groups) == 2:
            return group_col, groups

    return _comparison_group_info(metadata, results_dir)


def _read_results_metadata(results_dir: Path) -> pd.DataFrame:
    metadata_path = results_dir / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        settings = _read_analysis_settings(results_dir)
        return _add_composite_group_column(metadata, settings)
    return pd.DataFrame(columns=["sample_id"])


def _read_analysis_sample_columns(results_dir: Path) -> list[str]:
    for file_name in ["counts_normalized.csv", "counts_filtered.csv"]:
        matrix_path = results_dir / file_name
        if matrix_path.exists():
            matrix = pd.read_csv(matrix_path, nrows=0)
            return [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    return []


def _read_analysis_settings(results_dir: Path) -> dict[str, object]:
    config_path = results_dir.parent / "pipeline.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("analysis", {}) or {}


def _comparison_group_info(data: pd.DataFrame, results_dir: Path) -> tuple[str | None, list[str]]:
    settings = _read_analysis_settings(results_dir)
    group_col = settings.get("group_column")
    if not isinstance(group_col, str) or group_col not in data.columns:
        candidates = [col for col in data.columns if col not in {"sample_id", "normalized_count", "Name"}]
        group_col = candidates[0] if candidates else None
    if group_col is None:
        return None, []

    reference_group = settings.get("reference_group")
    case_group = settings.get("case_group")
    configured_groups = [
        str(group)
        for group in [reference_group, case_group]
        if group is not None and str(group) in set(data[group_col].dropna().astype(str))
    ]
    if len(configured_groups) >= 2:
        return group_col, configured_groups[:2]

    observed_groups = data[group_col].dropna().astype(str).drop_duplicates().tolist()
    return group_col, observed_groups[:2]


def _render_selected_gene_pca(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    px,
    group_col: str | None,
    comparison_groups: list[str],
) -> None:
    import streamlit as st

    sample_cols = [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    if len(sample_cols) < 2 or len(matrix) < 2:
        st.caption("At least two samples and two selected genes are required for PCA.")
        return

    values = matrix[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float).T
    values = np.log2(values + 1)
    values = values - values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    values = values / np.where(sd == 0, 1, sd)

    _, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    scores = values @ vt.T[:, :2]
    variance = singular_values**2
    explained = variance / variance.sum() if variance.sum() else np.zeros_like(variance)

    pca_df = pd.DataFrame(
        {
            "sample_id": sample_cols,
            "PC1": scores[:, 0],
            "PC2": scores[:, 1] if scores.shape[1] > 1 else 0,
        }
    )
    if "sample_id" in metadata.columns:
        pca_df = pca_df.merge(metadata, on="sample_id", how="left")
    color_col = "comparison_group" if "comparison_group" in pca_df.columns else None
    if color_col is None and group_col in pca_df.columns:
        color_col = group_col
    category_orders = {}
    if color_col == "comparison_group" and len(comparison_groups) == 2:
        category_orders[color_col] = _comparison_group_labels(comparison_groups)
    color_map = _contrast_color_map(category_orders.get(color_col, [])) if color_col else {}

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_col,
        color_discrete_map=color_map,
        category_orders=category_orders,
        hover_name="sample_id",
        title=f"PCA of selected contrast samples ({len(matrix)} geni)",
    )
    fig.update_layout(
        height=460,
        margin={"l": 45, "r": 18, "t": 45, "b": 45},
        xaxis_title=f"PC1 ({explained[0] * 100:.1f}%)" if len(explained) else "PC1",
        yaxis_title=f"PC2 ({explained[1] * 100:.1f}%)" if len(explained) > 1 else "PC2",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_selected_gene_heatmap(matrix: pd.DataFrame, go) -> None:
    import streamlit as st

    sample_cols = [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    if matrix.empty or not sample_cols:
        st.caption("No gene available for the heatmap.")
        return

    values = matrix[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    values = np.log2(values.to_numpy(dtype=float) + 1)
    row_mean = values.mean(axis=1, keepdims=True)
    row_sd = values.std(axis=1, keepdims=True)
    z_values = (values - row_mean) / np.where(row_sd == 0, 1, row_sd)
    gene_names = matrix["Name"].astype(str).tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=sample_cols,
            y=gene_names,
            customdata=np.array([[gene] * len(sample_cols) for gene in gene_names]),
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "row z"},
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Sample: %{x}<br>"
                "row z: %{z:.2f}<extra></extra>"
            ),
        )
    )
    overlay_x = []
    overlay_y = []
    overlay_gene = []
    for gene in gene_names:
        for sample in sample_cols:
            overlay_x.append(sample)
            overlay_y.append(gene)
            overlay_gene.append(gene)
    fig.add_trace(
        go.Scatter(
            x=overlay_x,
            y=overlay_y,
            mode="markers",
            marker={"size": 13, "opacity": 0.01, "color": "#111111"},
            customdata=overlay_gene,
            hoverinfo="skip",
            showlegend=False,
            name="gene selection",
        )
    )
    fig.update_layout(
        title=f"Selected-contrast heatmap sorted by logFC ({len(matrix)} geni)",
        height=460,
        margin={"l": 95, "r": 18, "t": 45, "b": 85},
    )
    state = st.plotly_chart(
        fig,
        use_container_width=True,
        key="selected_gene_heatmap",
        on_select="rerun",
        selection_mode="points",
    )
    selected_gene = _selected_gene_from_plotly_state(state)
    if selected_gene and st.session_state.get("active_gene") != selected_gene:
        st.session_state["active_gene"] = selected_gene
        st.rerun()


def _render_all_selected_gene_profiles(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    px,
    results_dir: Path,
) -> None:
    import streamlit as st

    sample_cols = [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    if matrix.empty or not sample_cols:
        st.caption("No profile available.")
        return

    long_profiles = matrix[["Name", *sample_cols]].melt(
        id_vars="Name",
        var_name="sample_id",
        value_name="normalized_count",
    )
    long_profiles["normalized_count"] = pd.to_numeric(
        long_profiles["normalized_count"],
        errors="coerce",
    ).fillna(0)
    if "sample_id" in metadata.columns:
        long_profiles = long_profiles.merge(metadata, on="sample_id", how="left")

    if "comparison_group" in long_profiles.columns:
        group_col = "comparison_group"
        comparison_groups = long_profiles[group_col].dropna().astype(str).drop_duplicates().tolist()[:2]
    else:
        group_col, comparison_groups = _comparison_group_info(long_profiles, results_dir)
    if group_col:
        long_profiles[group_col] = long_profiles[group_col].astype(str)
    color_col = group_col
    color_map = {}
    category_orders = {"Name": matrix["Name"].astype(str).tolist(), "sample_id": sample_cols}
    if color_col:
        if color_col == "comparison_group":
            color_map = _contrast_color_map(comparison_groups)
        else:
            color_map = _discrete_color_map(comparison_groups)
    genes = matrix["Name"].astype(str).tolist()
    long_profiles["Name"] = pd.Categorical(long_profiles["Name"].astype(str), categories=genes, ordered=True)

    rows = max(1, math.ceil(len(genes) / 4))
    bar_tab, box_tab = st.tabs(["Samples", "Group boxplot"])
    with bar_tab:
        fig = px.bar(
            long_profiles,
            x="sample_id",
            y="normalized_count",
            color=color_col,
            facet_col="Name",
            facet_col_wrap=4,
            color_discrete_map=color_map,
            category_orders=category_orders,
            hover_data=["sample_id", "normalized_count"],
            title=f"Expression per sample ({len(genes)} geni visualizzati)",
        )
        fig.update_layout(
            height=min(1600, max(420, rows * 260)),
            margin={"l": 45, "r": 18, "t": 55, "b": 42},
            showlegend=bool(color_col),
        )
        fig.update_xaxes(showticklabels=False, title="")
        fig.update_yaxes(title="Normalized count")
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("Name=", "")))
        _, chart_col, _ = st.columns([0.08, 1, 0.08])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True)

    with box_tab:
        if not group_col:
            st.caption("No group column available for the boxplot.")
            return
        box_data = long_profiles.copy()
        box_data[group_col] = box_data[group_col].astype(str)
        if comparison_groups:
            box_data = box_data[box_data[group_col].isin(comparison_groups)]
        if box_data.empty:
            st.caption("No sample available for the two comparison groups.")
            return
        fig = px.box(
            box_data,
            x=group_col,
            y="normalized_count",
            color=group_col,
            color_discrete_map=color_map,
            points="all",
            facet_col="Name",
            facet_col_wrap=4,
            category_orders={"Name": genes, group_col: comparison_groups},
            hover_name="sample_id",
            title=f"Comparison group boxplot ({len(genes)} geni visualizzati)",
        )
        fig.update_layout(
            height=min(1600, max(420, rows * 260)),
            margin={"l": 45, "r": 18, "t": 55, "b": 42},
            showlegend=False,
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Normalized count")
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("Name=", "")))
        _, chart_col, _ = st.columns([0.08, 1, 0.08])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True)


def _render_gene_expression_profile(
    gene: str,
    norm_path: Path,
    results_dir: Path,
    compact: bool = False,
    de: pd.DataFrame | None = None,
) -> None:
    import plotly.express as px
    import streamlit as st

    if not norm_path.exists():
        st.warning("Normalized matrix is not available to draw the expression profile.")
        return

    normalized = pd.read_csv(norm_path)
    profile = normalized[normalized["Name"].astype(str).eq(str(gene))]
    if profile.empty:
        st.warning(f"Gene `{gene}` not found in the normalized matrix.")
        return

    sample_cols = [col for col in profile.columns if col not in {"CodeClass", "Name"}]
    long_profile = profile[sample_cols].T.reset_index()
    long_profile.columns = ["sample_id", "normalized_count"]
    long_profile["normalized_count"] = pd.to_numeric(long_profile["normalized_count"], errors="coerce").fillna(0)
    metadata = _read_results_metadata(results_dir)
    if "sample_id" in metadata.columns:
        long_profile = long_profile.merge(metadata, on="sample_id", how="left")

    explicit_reference_samples = []
    explicit_case_samples = []
    explicit_sample_groups = {}
    if de is not None:
        explicit_reference_samples, explicit_case_samples, explicit_sample_groups = _comparison_sample_lists_from_summary(
            results_dir,
            de,
        )
        group_col, comparison_groups = _selected_de_group_info(de, long_profile, results_dir)
    else:
        group_col, comparison_groups = _comparison_group_info(long_profile, results_dir)
    if group_col:
        long_profile[group_col] = long_profile[group_col].astype(str)

    st.markdown(f"**Expression: `{gene}`**")
    profile_view = st.radio(
        "Expression profile view",
        ["Selected contrast samples", "All samples by analysis groups"],
        horizontal=True,
        key=f"gene_profile_view_{gene}",
    )
    plot_profile = long_profile.copy()
    x_col = "sample_id"
    color_col = group_col
    color_map = {}
    category_orders = {"sample_id": sample_cols}
    if group_col:
        category_orders[group_col] = long_profile[group_col].dropna().astype(str).drop_duplicates().tolist()

    if profile_view == "Selected contrast samples" and explicit_sample_groups:
        ordered_samples = [
            sample
            for sample in [*explicit_reference_samples, *explicit_case_samples]
            if sample in set(plot_profile["sample_id"].astype(str))
        ]
        plot_profile = plot_profile[plot_profile["sample_id"].astype(str).isin(ordered_samples)].copy()
        plot_profile["comparison_group"] = plot_profile["sample_id"].astype(str).map(explicit_sample_groups)
        group_order = list(
            dict.fromkeys(
                explicit_sample_groups[sample] for sample in ordered_samples if sample in explicit_sample_groups
            )
        )
        plot_profile["comparison_group"] = pd.Categorical(
            plot_profile["comparison_group"],
            categories=group_order,
            ordered=True,
        )
        plot_profile = plot_profile.sort_values(["comparison_group", "sample_id"])
        color_col = "comparison_group"
        category_orders = {
            "sample_id": ordered_samples,
            "comparison_group": group_order,
        }
    elif profile_view == "Selected contrast samples" and group_col and len(comparison_groups) == 2:
        plot_profile = _add_comparison_group_labels(plot_profile, group_col, comparison_groups)
        plot_profile = plot_profile[plot_profile["comparison_group"].notna()].copy()
        group_order = _comparison_group_labels(comparison_groups)
        plot_profile["comparison_group"] = pd.Categorical(
            plot_profile["comparison_group"],
            categories=group_order,
            ordered=True,
        )
        plot_profile = plot_profile.sort_values(["comparison_group", "sample_id"])
        color_col = "comparison_group"
        category_orders = {
            "sample_id": plot_profile["sample_id"].astype(str).tolist(),
            "comparison_group": group_order,
        }
    elif profile_view == "All samples by analysis groups" and group_col:
        group_order = long_profile[group_col].dropna().astype(str).drop_duplicates().tolist()
        plot_profile[group_col] = pd.Categorical(plot_profile[group_col], categories=group_order, ordered=True)
        plot_profile = plot_profile.sort_values([group_col, "sample_id"])
        category_orders = {
            "sample_id": plot_profile["sample_id"].astype(str).tolist(),
            group_col: group_order,
        }
    if color_col:
        if color_col == "comparison_group":
            color_map = _contrast_color_map(category_orders.get(color_col, []))
        else:
            color_map = _discrete_color_map(category_orders.get(color_col, []))

    bar_fig = px.bar(
        plot_profile,
        x=x_col,
        y="normalized_count",
        color=color_col,
        color_discrete_map=color_map,
        category_orders=category_orders,
        hover_data=["sample_id", "normalized_count"],
        title="Samples" if profile_view == "All samples by analysis groups" else "Selected contrast samples",
    )
    bar_fig.update_layout(
        height=420,
        margin={"l": 45, "r": 18, "t": 45, "b": 60},
        showlegend=bool(group_col),
        xaxis_title="Sample",
        yaxis_title="Normalized count",
    )

    box_fig = None
    if group_col and profile_view == "All samples by analysis groups":
        box_data = long_profile.copy()
        box_data[group_col] = box_data[group_col].astype(str)
        if not box_data.empty:
            box_fig = px.box(
                box_data,
                x=group_col,
                y="normalized_count",
                color=group_col,
                color_discrete_map=color_map,
                points="all",
                hover_name="sample_id",
                category_orders={group_col: category_orders.get(group_col, [])},
                title="Group boxplot",
            )
            box_fig.update_layout(
                height=420,
                margin={"l": 45, "r": 18, "t": 45, "b": 60},
                showlegend=False,
                xaxis_title=group_col,
                yaxis_title="Normalized count",
            )
    elif color_col == "comparison_group" and not plot_profile.empty:
        box_fig = px.box(
            plot_profile,
            x="comparison_group",
            y="normalized_count",
            color="comparison_group",
            color_discrete_map=color_map,
            points="all",
            hover_name="sample_id",
            category_orders={"comparison_group": category_orders["comparison_group"]},
            title="Selected contrast distribution",
        )
        box_fig.update_layout(
            height=420,
            margin={"l": 45, "r": 18, "t": 45, "b": 60},
            showlegend=False,
            xaxis_title="Comparison group",
            yaxis_title="Normalized count",
        )

    _, bar_col, box_col, _ = st.columns([0.08, 1, 1, 0.08])
    with bar_col:
        st.plotly_chart(bar_fig, use_container_width=True)
    with box_col:
        if box_fig is not None:
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.caption("No group available for the boxplot.")


def _discrete_color_map(groups: list[str]) -> dict[str, str]:
    palette = [
        "#2F6690",
        "#B23A48",
        "#5B8E7D",
        "#E09F3E",
        "#6D597A",
        "#4D908E",
        "#9D4EDD",
        "#577590",
        "#BC6C25",
        "#386641",
    ]
    return {str(group): palette[index % len(palette)] for index, group in enumerate(groups)}


def _contrast_color_map(groups: list[str]) -> dict[str, str]:
    contrast_palette = ["#1F77B4", "#D62728"]
    output = {}
    for index, group in enumerate(groups):
        text = str(group)
        lower = text.lower()
        if "reference" in lower or "group 1" in lower:
            output[text] = contrast_palette[0]
        elif "case" in lower or "group 2" in lower:
            output[text] = contrast_palette[1]
        else:
            output[text] = contrast_palette[index % len(contrast_palette)]
    return output


def _render_qc_explorer(results_dir: Path) -> None:
    import streamlit as st

    qc_path = results_dir / "qc_summary.csv"
    if not qc_path.exists():
        return

    qc = pd.read_csv(qc_path)
    st.subheader("Quality control")
    if "qc_status" in qc:
        status_counts = qc["qc_status"].value_counts().to_dict()
        cols = st.columns(2)
        cols[0].metric("PASS", status_counts.get("PASS", 0))
        cols[1].metric("WARN", status_counts.get("WARN", 0))
    st.dataframe(qc, use_container_width=True)


def _render_report_images(reports_dir: Path) -> None:
    import streamlit as st

    images = [
        reports_dir / "volcano_plot.png",
        reports_dir / "qc_library_size.png",
        reports_dir / "pca.png",
        reports_dir / "heatmap_top_variable.png",
    ]
    existing = [path for path in images if path.exists()]
    if not existing:
        return

    st.subheader("Figures")
    for index in range(0, len(existing), 2):
        columns = st.columns(2)
        for column, path in zip(columns, existing[index : index + 2]):
            with column:
                with st.expander(path.stem.replace("_", " "), expanded=False):
                    st.image(str(path), caption=path.name, use_column_width=True)


if __name__ == "__main__":
    _run_app()
