from pathlib import Path
import pytest
from mdrlab.core import load, normalize, to_csv
from mdrlab.errors import DataContractError

ROOT = Path(__file__).parents[1]

def test_raw_matches_golden():
    actual = to_csv(normalize(load(ROOT / "fixtures/raw/bls_lns11300000.json")))
    expected = (ROOT / "fixtures/golden/bls_lns11300000.csv").read_text()
    assert actual == expected

@pytest.mark.parametrize("name", ["duplicate_key.json", "invalid_period.json", "numeric_as_text_noise.json", "schema_drift.json"])
def test_corruption_is_rejected(name):
    with pytest.raises(DataContractError):
        normalize(load(ROOT / "fixtures/cases" / name))

def test_order_is_canonicalized():
    assert normalize(load(ROOT / "fixtures/cases/reverse_order.json")) == normalize(load(ROOT / "fixtures/raw/bls_lns11300000.json"))
