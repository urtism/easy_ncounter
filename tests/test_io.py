from pathlib import Path

from easy_ncounter.cli import _add_analysis_group_column
from easy_ncounter.io import (
    metadata_from_samplesheet,
    parse_probe_annotation,
    read_count_table,
    read_counts_from_samplesheet,
    read_rcc_file,
    read_rcc_metrics,
)
from easy_ncounter.qc import background_correct_counts, compute_qc_summary, filter_endogenous_counts
from easy_ncounter.ui import _valid_contrasts


def test_read_count_table_adds_code_class(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("Name,sample_01,sample_02\nGENE1,10,20\n", encoding="utf-8")

    table = read_count_table(path)

    assert list(table.columns) == ["CodeClass", "Name", "sample_01", "sample_02"]
    assert table.loc[0, "CodeClass"] == "Endogenous"


def test_parse_probe_annotation_standardizes_optional_columns(tmp_path: Path) -> None:
    counts = read_count_table(Path("data/raw/counts.csv"))
    annotation = tmp_path / "probes.csv"
    annotation.write_text(
        "\n".join(
            [
                "Name,CodeClass,Pathway,cell type,Related Probe",
                "ACTB,Housekeeping,Core,Stromal,ACTB_2",
                "GENE1,Endogenous,Inflammation,T cell,",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_probe_annotation(counts, annotation)

    actb = parsed.loc[parsed["probe_id"] == "ACTB"].iloc[0]
    assert bool(actb["is_housekeeper"])
    assert actb["pathway"] == "Core"
    assert actb["cell_type"] == "Stromal"
    assert "raw_annotation_columns" in parsed.columns


def test_add_analysis_group_column_combines_multiple_metadata_features() -> None:
    import pandas as pd

    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_01", "sample_02"],
            "DISEASE": ["control", "disease"],
            "TIME": ["T0", "T1"],
        }
    )

    output = _add_analysis_group_column(
        metadata,
        {
            "group_column": "__group__DISEASE__TIME",
            "group_columns": ["DISEASE", "TIME"],
        },
    )

    assert output["__group__DISEASE__TIME"].tolist() == [
        "DISEASE=control | TIME=T0",
        "DISEASE=disease | TIME=T1",
    ]


def test_valid_contrasts_filters_invalid_and_duplicates() -> None:
    contrasts = _valid_contrasts(
        [
            {"comparison_id": "treated vs control", "reference_group": "control", "case_group": "treated"},
            {"comparison_id": "treated vs control", "reference_group": "control", "case_group": "treated"},
            {"comparison_id": "bad", "reference_group": "control", "case_group": "control"},
            {"comparison_id": "missing", "reference_group": "control", "case_group": "other"},
        ],
        ["control", "treated"],
    )

    assert contrasts == [
        {
            "comparison_id": "treated_vs_control",
            "reference_group": "control",
            "case_group": "treated",
        }
    ]


def test_read_counts_from_nf_core_style_samplesheet(tmp_path: Path) -> None:
    rcc_path = tmp_path / "sample 1.RCC"
    rcc_path.write_text(
        """
<root>
  <Sample_ID>instrument_name</Sample_ID>
  <Lane_Attributes><CodeClass>Positive</CodeClass><Name>POS_A</Name><Count>100</Count></Lane_Attributes>
  <Lane_Attributes><CodeClass>Negative</CodeClass><Name>NEG_A</Name><Count>5</Count></Lane_Attributes>
  <Lane_Attributes><CodeClass>Endogenous</CodeClass><Name>GENE1</Name><Count>42</Count></Lane_Attributes>
</root>
""".strip(),
        encoding="utf-8",
    )
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        f"RCC_FILE,RCC_FILE_NAME,SAMPLE_ID,condition\n{rcc_path.name},{rcc_path.name},sample 1,treated\n",
        encoding="utf-8",
    )

    counts = read_counts_from_samplesheet(samplesheet)
    metadata = metadata_from_samplesheet(samplesheet)

    assert list(counts.columns) == ["CodeClass", "Name", "sample_1"]
    assert counts.loc[counts["Name"] == "GENE1", "sample_1"].item() == 42
    assert metadata.to_dict("records") == [{"sample_id": "sample_1", "condition": "treated"}]


def test_read_counts_from_samplesheet_resolves_paths_relative_to_samplesheet(tmp_path: Path) -> None:
    rcc_dir = tmp_path / "rcc"
    rcc_dir.mkdir()
    rcc_path = rcc_dir / "sample.RCC"
    rcc_path.write_text(
        """
<root>
  <Sample_ID>sample</Sample_ID>
  <Lane_Attributes><CodeClass>Endogenous</CodeClass><Name>GENE1</Name><Count>7</Count></Lane_Attributes>
</root>
""".strip(),
        encoding="utf-8",
    )
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        "RCC_FILE,RCC_FILE_NAME,SAMPLE_ID,condition\nrcc/sample.RCC,sample.RCC,sample,treated\n",
        encoding="utf-8",
    )

    counts = read_counts_from_samplesheet(samplesheet)

    assert counts.loc[counts["Name"] == "GENE1", "sample"].item() == 7


def test_read_rcc_file_supports_text_section_format(tmp_path: Path) -> None:
    rcc_path = tmp_path / "sample.RCC"
    rcc_path.write_text(
        """
[Header]
FileVersion,1.0
[Sample_Attributes]
Sample_ID,text_sample
[Code_Summary]
CodeClass,Name,Accession,Count
Positive,POS_A,,120
Negative,NEG_A,,5
Endogenous,GENE1,,42
""".strip(),
        encoding="utf-8",
    )

    sample_id, counts = read_rcc_file(rcc_path)

    assert sample_id == "text_sample"
    assert counts.loc[counts["Name"] == "GENE1", "Count"].item() == 42


def test_read_rcc_file_supports_angle_bracket_sections(tmp_path: Path) -> None:
    rcc_path = tmp_path / "sample.RCC"
    rcc_path.write_text(
        """
<Header>
FileVersion,2.0
</Header>
<Sample_Attributes>
ID,24012023-P1
GeneRLF,NS_Hs_Transplant_v1.0
</Sample_Attributes>
<Code_Summary>
CodeClass,Name,Accession,Count
Endogenous,AICDA,NM_020661.2,18
Positive,POS_A,,120
</Code_Summary>
""".strip(),
        encoding="utf-8",
    )

    sample_id, counts = read_rcc_file(rcc_path)

    assert sample_id == "24012023-P1"
    assert counts.loc[counts["Name"] == "AICDA", "Count"].item() == 18


def test_read_rcc_metrics_extracts_lane_metrics(tmp_path: Path) -> None:
    rcc_path = tmp_path / "sample.RCC"
    rcc_path.write_text(
        """
<Sample_Attributes>
ID,24012023-P1
</Sample_Attributes>
<Lane_Attributes>
FovCount,100
FovCounted,80
BindingDensity,1.2
</Lane_Attributes>
<Code_Summary>
CodeClass,Name,Accession,Count
Endogenous,AICDA,NM_020661.2,18
</Code_Summary>
""".strip(),
        encoding="utf-8",
    )

    metrics = read_rcc_metrics(rcc_path)

    assert metrics["sample_id_in_rcc"] == "24012023-P1"
    assert metrics["fov_counted_fraction"] == 0.8
    assert metrics["bindingdensity"] == 1.2


def test_qc_summary_adds_status_and_warnings() -> None:
    table = read_count_table(
        Path("data/raw/counts.csv")
    )

    summary = compute_qc_summary(table)

    assert "qc_status" in summary.columns
    assert "background_threshold" in summary.columns
    assert set(summary["qc_status"]).issubset({"PASS", "WARN"})


def test_bruker_like_qc_summary_adds_expected_columns() -> None:
    table = read_count_table(Path("data/raw/counts.csv"))

    summary = compute_qc_summary(table, {"qc_mode": "bruker_like"})

    for column in [
        "qc_mode",
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
    ]:
        assert column in summary.columns
    assert set(summary["qc_mode"]) == {"bruker_like"}


def test_background_correction_and_gene_filtering() -> None:
    table = read_count_table(Path("data/raw/counts.csv"))
    metadata = Path("data/metadata/samples.csv")
    metadata_table = __import__("pandas").read_csv(metadata)
    qc = compute_qc_summary(table)

    corrected = background_correct_counts(table, qc)
    filtered, decisions = filter_endogenous_counts(
        corrected,
        qc,
        metadata_table,
        "condition",
        {"filter_min_count": 10, "filter_min_samples": 2},
    )

    assert set(filtered["CodeClass"]) == {"Endogenous"}
    assert "kept" in decisions.columns
