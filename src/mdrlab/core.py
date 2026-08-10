from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

from .errors import DataContractError


PERIOD = re.compile(r"^M(0[1-9]|1[0-2])$")
MISSING_VALUES = {".", ""}


def load(path: str | Path) -> dict[str, Any]:
    """Load a JSON document from a file."""

    file_path = Path(path)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataContractError(
            f"could not read input file: {file_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataContractError(
            f"input file is not valid JSON: {file_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise DataContractError("JSON root must be an object")

    return payload


def normalize(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Validate and normalize one BLS monthly time series."""

    if not isinstance(payload, dict):
        raise DataContractError("payload must be an object")

    status = payload.get("status")

    if status is not None and status != "REQUEST_SUCCEEDED":
        raise DataContractError("BLS request did not succeed")

    results = payload.get("Results")

    if not isinstance(results, dict):
        raise DataContractError("missing or invalid Results object")

    series_collection = results.get("series")

    if not isinstance(series_collection, list):
        raise DataContractError("missing or invalid Results.series")

    if len(series_collection) != 1:
        raise DataContractError("expected exactly one series")

    series = series_collection[0]

    if not isinstance(series, dict):
        raise DataContractError("series entry must be an object")

    series_id = series.get("seriesID")

    if not isinstance(series_id, str) or not series_id.strip():
        raise DataContractError("missing or invalid seriesID")

    observations = series.get("data")

    if not isinstance(observations, list):
        raise DataContractError("missing or invalid series data")

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise DataContractError(
                f"observation at index {index} must be an object"
            )

        year = observation.get("year")
        period = observation.get("period")
        raw_value = observation.get("value")

        if (
            not isinstance(year, str)
            or not year.isdigit()
            or len(year) != 4
        ):
            raise DataContractError(
                f"invalid year at observation index {index}"
            )

        if not isinstance(period, str) or not PERIOD.fullmatch(period):
            raise DataContractError(
                f"invalid monthly period at observation index {index}"
            )

        if raw_value is None:
            raise DataContractError(
                f"missing value at observation index {index}"
            )

        if isinstance(raw_value, bool):
            raise DataContractError(
                f"value is not numeric at observation index {index}"
            )

        if isinstance(raw_value, str):
            value_text = raw_value.strip()

            if value_text in MISSING_VALUES:
                continue
        elif isinstance(raw_value, (int, float)):
            value_text = str(raw_value)
        else:
            raise DataContractError(
                f"value is not numeric at observation index {index}"
            )

        try:
            numeric_value = float(value_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataContractError(
                f"value is not numeric at observation index {index}"
            ) from exc

        if not math.isfinite(numeric_value):
            raise DataContractError(
                f"value must be finite at observation index {index}"
            )

        date = f"{year}-{period[1:]}-01"
        key = (series_id, date)

        if key in seen:
            raise DataContractError(
                f"duplicate series/date key: {series_id}, {date}"
            )

        seen.add(key)

        rows.append(
            {
                "series_id": series_id,
                "date": date,
                "value": format(numeric_value, "g"),
            }
        )

    return sorted(rows, key=lambda row: row["date"])


def to_csv(rows: list[dict[str, str]]) -> str:
    """Convert normalized rows to deterministic CSV output."""

    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=["series_id", "date", "value"],
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()
