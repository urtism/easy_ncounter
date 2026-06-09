from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def read_counts(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if path.is_dir():
        return read_rcc_directory(path)
    return read_count_table(path)


def read_count_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    table = read_delimited_table(path)

    if "Name" not in table.columns:
        first = table.columns[0]
        table = table.rename(columns={first: "Name"})

    if "CodeClass" not in table.columns:
        table.insert(0, "CodeClass", "Endogenous")

    return table


def read_delimited_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def read_samplesheet(path: str | Path) -> pd.DataFrame:
    table = read_delimited_table(path)
    missing = {"RCC_FILE", "SAMPLE_ID"} - set(table.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Samplesheet is missing required columns: {joined}")

    table = table.copy()
    table["SAMPLE_ID"] = table["SAMPLE_ID"].astype(str).map(_normalize_sample_id)
    if "RCC_FILE_NAME" not in table.columns:
        table["RCC_FILE_NAME"] = table["RCC_FILE"].map(lambda value: Path(str(value)).name)
    return table


def read_counts_from_samplesheet(path: str | Path) -> pd.DataFrame:
    samplesheet_path = Path(path)
    samplesheet = read_samplesheet(samplesheet_path)
    rows = []

    for _, sample in samplesheet.iterrows():
        rcc_path = _resolve_rcc_path(samplesheet_path, sample["RCC_FILE"])
        _, counts = read_rcc_file(rcc_path)
        counts = counts.assign(sample_id=sample["SAMPLE_ID"])
        rows.append(counts)

    long_counts = pd.concat(rows, ignore_index=True)
    matrix = long_counts.pivot_table(
        index=["CodeClass", "Name"],
        columns="sample_id",
        values="Count",
        aggfunc="mean",
    )
    matrix = matrix.reset_index().rename_axis(columns=None)
    sample_cols = [col for col in matrix.columns if col not in {"CodeClass", "Name"}]
    matrix[sample_cols] = matrix[sample_cols].round().astype("Int64")
    return matrix


def read_rcc_metrics_from_samplesheet(path: str | Path) -> pd.DataFrame:
    samplesheet_path = Path(path)
    samplesheet = read_samplesheet(samplesheet_path)
    rows = []

    for _, sample in samplesheet.iterrows():
        rcc_path = _resolve_rcc_path(samplesheet_path, sample["RCC_FILE"])
        metrics = read_rcc_metrics(rcc_path)
        metrics["sample_id"] = sample["SAMPLE_ID"]
        metrics["rcc_file"] = str(rcc_path)
        rows.append(metrics)

    return pd.DataFrame(rows)


def metadata_from_samplesheet(path: str | Path) -> pd.DataFrame:
    samplesheet = read_samplesheet(path)
    metadata = samplesheet.drop(columns=["RCC_FILE", "RCC_FILE_NAME"], errors="ignore")
    metadata = metadata.rename(columns={"SAMPLE_ID": "sample_id"})
    return metadata.drop_duplicates(subset=["sample_id"], keep="first")


def read_rcc_directory(path: Path) -> pd.DataFrame:
    rcc_files = sorted(path.glob("*.RCC")) + sorted(path.glob("*.rcc"))
    if not rcc_files:
        raise FileNotFoundError(f"No RCC files found in {path}")

    sample_tables = []
    for rcc_file in rcc_files:
        sample_id, counts = read_rcc_file(rcc_file)
        counts = counts.rename(columns={"Count": sample_id})
        sample_tables.append(counts)

    merged = sample_tables[0]
    for table in sample_tables[1:]:
        merged = merged.merge(table, on=["CodeClass", "Name"], how="outer")

    return merged


def read_rcc_file(path: Path) -> tuple[str, pd.DataFrame]:
    try:
        return _read_rcc_xml(path)
    except ET.ParseError:
        return _read_rcc_text(path)


def read_rcc_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sections = _parse_rcc_sections(text)
    metrics: dict[str, object] = {"sample_id_in_rcc": _sample_id_from_sections(sections) or path.stem}

    for section_name in ("Header", "Sample_Attributes", "Sample Attributes", "Lane_Attributes", "Lane Attributes"):
        for row in sections.get(section_name, []):
            if len(row) < 2:
                continue
            key = _normalize_rcc_key(row[0])
            value = row[1].strip()
            if key in {
                "id",
                "fileversion",
                "softwareversion",
                "systemtype",
                "generlf",
                "assaytype",
                "fovcount",
                "fovcounted",
                "bindingdensity",
                "scannerid",
                "cartridgeid",
                "cartridgebarcode",
            }:
                metrics[key] = _coerce_number(value)

    if "fovcount" in metrics and "fovcounted" in metrics and float(metrics["fovcount"]) > 0:
        metrics["fov_counted_fraction"] = float(metrics["fovcounted"]) / float(metrics["fovcount"])
    return metrics


def _read_rcc_xml(path: Path) -> tuple[str, pd.DataFrame]:
    tree = ET.parse(path)
    root = tree.getroot()

    sample_id = _find_text(root, ".//Sample_ID") or path.stem
    rows: list[dict[str, str | int]] = []

    for lane in root.findall(".//Lane_Attributes"):
        code_class = _find_text(lane, "CodeClass") or "Endogenous"
        name = _find_text(lane, "Name")
        count = _find_text(lane, "Count")
        if name is None or count is None:
            continue
        rows.append({"CodeClass": code_class, "Name": name, "Count": int(float(count))})

    if not rows:
        raise ValueError(f"No probe counts found in {path}")

    return sample_id, pd.DataFrame(rows)


def _read_rcc_text(path: Path) -> tuple[str, pd.DataFrame]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sections = _parse_rcc_sections(text)

    sample_id = _sample_id_from_sections(sections) or path.stem
    count_rows = _count_rows_from_sections(sections)
    if not count_rows:
        raise ValueError(f"No probe counts found in {path}")

    return sample_id, pd.DataFrame(count_rows)


def _parse_rcc_sections(text: str) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip()
            sections.setdefault(current_section, [])
            continue
        if line.startswith("</") and line.endswith(">"):
            current_section = None
            continue
        if line.startswith("<") and line.endswith(">"):
            current_section = line.strip("<>").strip()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            continue
        sections[current_section].append(next(csv.reader([raw_line])))

    return sections


def _sample_id_from_sections(sections: dict[str, list[list[str]]]) -> str | None:
    for section_name in ("Sample_Attributes", "Sample Attributes", "Header"):
        rows = sections.get(section_name, [])
        for row in rows:
            if len(row) < 2:
                continue
            key = _normalize_rcc_key(row[0])
            if key in {"id", "sample_id", "sampleid", "sample_name", "samplename"}:
                return row[1].strip()
    return None


def _count_rows_from_sections(sections: dict[str, list[list[str]]]) -> list[dict[str, str | int]]:
    for section_name in ("Code_Summary", "Code Summary", "Lane_Attributes", "Lane Attributes"):
        rows = sections.get(section_name, [])
        parsed_rows = _parse_count_section(rows)
        if parsed_rows:
            return parsed_rows
    return []


def _parse_count_section(rows: list[list[str]]) -> list[dict[str, str | int]]:
    if not rows:
        return []

    header_index = _find_count_header_index(rows)
    if header_index is None:
        return []

    header = [_normalize_rcc_key(value) for value in rows[header_index]]
    indexes = _count_column_indexes(header)
    if indexes is None:
        return []

    count_rows = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(indexes.values()):
            continue
        name = row[indexes["name"]].strip()
        count = row[indexes["count"]].strip()
        if not name or not count:
            continue
        code_class = row[indexes["code_class"]].strip() if "code_class" in indexes else "Endogenous"
        count_rows.append({"CodeClass": code_class or "Endogenous", "Name": name, "Count": int(float(count))})

    return count_rows


def _find_count_header_index(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows):
        normalized = {_normalize_rcc_key(value) for value in row}
        if {"name", "count"}.issubset(normalized):
            return index
    return None


def _count_column_indexes(header: list[str]) -> dict[str, int] | None:
    aliases = {
        "code_class": ("codeclass", "code_class", "codeclassname", "class"),
        "name": ("name", "gene", "gene_name", "genename", "target"),
        "count": ("count", "counts"),
    }
    indexes: dict[str, int] = {}
    for output_name, names in aliases.items():
        for candidate in names:
            if candidate in header:
                indexes[output_name] = header.index(candidate)
                break
    if "name" not in indexes or "count" not in indexes:
        return None
    return indexes


def _normalize_rcc_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace(".", "_").replace("-", "_")


def _coerce_number(value: str) -> object:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _find_text(root: ET.Element, pattern: str) -> str | None:
    element = root.find(pattern)
    if element is None or element.text is None:
        return None
    return element.text.strip()


def _normalize_sample_id(value: str) -> str:
    return value.strip().replace(" ", "_")


def _resolve_rcc_path(samplesheet_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return samplesheet_path.parent / path
