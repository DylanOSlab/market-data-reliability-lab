from pathlib import Path

import pytest

from mdrlab.core import load, normalize
from mdrlab.errors import DataContractError


def test_missing_value_field_raises() -> None:
    """Ensure that an observation missing the `value` key raises a DataContractError."""
    fixture_path = Path(__file__).parents[1] / "fixtures" / "cases" / "missing_value.json"
    payload = load(fixture_path)
    with pytest.raises(DataContractError, match="missing value field"):
        normalize(payload)
