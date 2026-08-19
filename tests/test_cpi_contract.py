from pathlib import Path

from mdrlab.core import load, normalize, to_csv

ROOT = Path(__file__).parents[1]


def test_cpi_raw_matches_golden() -> None:
    actual = to_csv(normalize(load(ROOT / "fixtures/raw/bls_cuur0000sa0.json")))
    expected = (ROOT / "fixtures/golden/bls_cuur0000sa0.csv").read_text(encoding="utf-8")
    assert actual == expected


def test_cpi_series_identity() -> None:
    rows = normalize(load(ROOT / "fixtures/raw/bls_cuur0000sa0.json"))
    assert rows
    assert {row["series_id"] for row in rows} == {"CUUR0000SA0"}
