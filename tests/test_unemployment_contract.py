from pathlib import Path

from mdrlab.core import load, normalize, to_csv

ROOT = Path(__file__).parents[1]


def test_unemployment_raw_matches_golden() -> None:
    actual = to_csv(normalize(load(ROOT / "fixtures/raw/bls_lns14000000.json")))
    expected = (ROOT / "fixtures/golden/bls_lns14000000.csv").read_text(encoding="utf-8")
    assert actual == expected


def test_unemployment_series_identity() -> None:
    rows = normalize(load(ROOT / "fixtures/raw/bls_lns14000000.json"))
    assert rows
    assert {row["series_id"] for row in rows} == {"LNS14000000"}
