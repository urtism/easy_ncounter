from __future__ import annotations

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
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    table = pd.read_csv(path, sep=sep)

    if "Name" not in table.columns:
        first = table.columns[0]
        table = table.rename(columns={first: "Name"})

    if "CodeClass" not in table.columns:
        table.insert(0, "CodeClass", "Endogenous")

    return table


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


def _find_text(root: ET.Element, pattern: str) -> str | None:
    element = root.find(pattern)
    if element is None or element.text is None:
        return None
    return element.text.strip()

