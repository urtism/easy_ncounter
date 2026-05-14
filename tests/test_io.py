from pathlib import Path

from easy_ncounter.io import read_count_table


def test_read_count_table_adds_code_class(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"
    path.write_text("Name,sample_01,sample_02\nGENE1,10,20\n", encoding="utf-8")

    table = read_count_table(path)

    assert list(table.columns) == ["CodeClass", "Name", "sample_01", "sample_02"]
    assert table.loc[0, "CodeClass"] == "Endogenous"

