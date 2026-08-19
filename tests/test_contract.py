from pathlib import Path

import pytest

from mdrlab.core import load, normalize, to_csv
from mdrlab.errors import DataContractError

ROOT = Path(__file__).parents[1]


def test_raw_matches_golden() -> None:
    actual = to_csv(normalize(load(ROOT / "fixtures/raw/bls_lns11300000.json")))

    expected = (ROOT / "fixtures/golden/bls_lns11300000.csv").read_text(encoding="utf-8")

    assert actual == expected


@pytest.mark.parametrize(
    "name",
    [
        "duplicate_key.json",
        "invalid_period.json",
        "numeric_as_text_noise.json",
        "schema_drift.json",
    ],
)
def test_corruption_is_rejected(name: str) -> None:
    fixture_path = ROOT / "fixtures/cases" / name

    with pytest.raises(DataContractError):
        normalize(load(fixture_path))


def test_order_is_canonicalized() -> None:
    reversed_rows = normalize(load(ROOT / "fixtures/cases/reverse_order.json"))

    expected_rows = normalize(load(ROOT / "fixtures/raw/bls_lns11300000.json"))

    assert reversed_rows == expected_rows


@pytest.mark.parametrize(
    "missing_marker",
    [".", "-"],
)
def test_bls_missing_value_markers_are_skipped(
    missing_marker: str,
) -> None:
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "LNS11300000",
                    "data": [
                        {
                            "year": "2025",
                            "period": "M12",
                            "value": missing_marker,
                        },
                        {
                            "year": "2025",
                            "period": "M11",
                            "value": "62.5",
                        },
                    ],
                }
            ]
        },
    }

    rows = normalize(payload)

    assert rows == [
        {
            "series_id": "LNS11300000",
            "date": "2025-11-01",
            "value": "62.5",
        }
    ]


def test_unknown_non_numeric_value_is_rejected() -> None:
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "LNS11300000",
                    "data": [
                        {
                            "year": "2025",
                            "period": "M12",
                            "value": "not-available",
                        }
                    ],
                }
            ]
        },
    }

    with pytest.raises(
        DataContractError,
        match="value is not numeric",
    ):
        normalize(payload)


def test_non_finite_value_is_rejected() -> None:
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "LNS11300000",
                    "data": [
                        {
                            "year": "2025",
                            "period": "M12",
                            "value": "NaN",
                        }
                    ],
                }
            ]
        },
    }

    with pytest.raises(
        DataContractError,
        match="value must be finite",
    ):
        normalize(payload)
