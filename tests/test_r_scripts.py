from pathlib import Path


def test_differential_expression_uses_explicit_bh_adjustment() -> None:
    script = Path("scripts/differential_expression.R").read_text(encoding="utf-8")

    assert 'adjust.method = "BH"' in script
