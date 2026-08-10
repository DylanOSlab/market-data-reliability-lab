from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

from .errors import DataContractError


MONTHLY_PERIOD_PATTERN = re.compile(r"^M(0[1-9]|1[0-2])$")

# Missing-value markers observed in real BLS API responses.
BLS_MISSING_VALUES = {".", "-"}


def load(path: str | Path) -> dict[str, Any]:
    """
    Load and parse a JSON document from disk.

    Raises:
        DataContractError:
            If the file cannot be read, contains invalid JSON, or the
            root JSON value is not an object.
    """
    file_path = Path(path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataContractError(
            f"could not read input file: {file_path}"
        ) from exc

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DataContractError(
            f"input file is not valid JSON: {file_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise DataContractError("JSON root must be an object")

    return payload


def normalize(payload: dict[str, Any]) -> list[dict[str, str]]:
    """
    Validate and normalize one BLS monthly time series.

    The normalized output contains:

    - series_id
    - date in YYYY-MM-01 format
    - value as a canonical numeric string

    Observations containing a recognized BLS missing-value marker are
    omitted from the normalized result.
    """
    if not isinstance(payload, dict):
        raise DataContractError("payload must be an object")

    _validate_response_status(payload)

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

    series_id = series_id.strip()

    observations = series.get("data")

    if not isinstance(observations, list):
        raise DataContractError("missing or invalid series data")

    normalized_rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for observation_index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise DataContractError(
                f"observation at index {observation_index} "
                "must be an object"
            )

        year = observation.get("year")
        period = observation.get("period")
        raw_value = observation.get("value")

        normalized_date = _normalize_monthly_date(
            year=year,
            period=period,
            observation_index=observation_index,
        )

        normalized_value = _normalize_value(
            raw_value=raw_value,
            observation_index=observation_index,
        )

        # A None result represents a recognized BLS missing value.
        if normalized_value is None:
            continue

        key = (series_id, normalized_date)

        if key in seen_keys:
            raise DataContractError(
                "duplicate series/date key: "
                f"{series_id}, {normalized_date}"
            )

        seen_keys.add(key)

        normalized_rows.append(
            {
                "series_id": series_id,
                "date": normalized_date,
                "value": normalized_value,
            }
        )

    return sorted(
        normalized_rows,
        key=lambda row: (row["series_id"], row["date"]),
    )


def to_csv(rows: list[dict[str, str]]) -> str:
    """
    Convert normalized rows into deterministic CSV text.
    """
    if not isinstance(rows, list):
        raise DataContractError("rows must be a list")

    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=["series_id", "date", "value"],
        lineterminator="\n",
        extrasaction="raise",
    )

    writer.writeheader()

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DataContractError(
                f"row at index {row_index} must be an object"
            )

        required_fields = {
            "series_id",
            "date",
            "value",
        }

        missing_fields = required_fields - set(row)

        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))

            raise DataContractError(
                f"row at index {row_index} is missing fields: "
                f"{missing_text}"
            )

        try:
            writer.writerow(row)
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                f"could not write row at index {row_index}"
            ) from exc

    return output.getvalue()


def _validate_response_status(payload: dict[str, Any]) -> None:
    """
    Validate the optional BLS response status.

    Some stored fixtures may omit the status. If a status is present,
    the status must indicate a successful BLS request.
    """
    status = payload.get("status")

    if status is None:
        return

    if status == "REQUEST_SUCCEEDED":
        return

    messages = payload.get("message")
    message_text = ""

    if isinstance(messages, list):
        message_text = "; ".join(
            str(message)
            for message in messages
        )
    elif messages is not None:
        message_text = str(messages)

    if message_text:
        raise DataContractError(
            f"BLS request did not succeed: {message_text}"
        )

    raise DataContractError("BLS request did not succeed")


def _normalize_monthly_date(
    *,
    year: Any,
    period: Any,
    observation_index: int,
) -> str:
    """
    Validate a BLS monthly period and convert it to YYYY-MM-01.
    """
    if not isinstance(year, str):
        raise DataContractError(
            f"invalid year at observation index {observation_index}"
        )

    if len(year) != 4 or not year.isdigit():
        raise DataContractError(
            f"invalid year at observation index "
            f"{observation_index}: {year!r}"
        )

    if not isinstance(period, str):
        raise DataContractError(
            f"invalid period at observation index {observation_index}"
        )

    if MONTHLY_PERIOD_PATTERN.fullmatch(period) is None:
        raise DataContractError(
            f"invalid monthly period at observation index "
            f"{observation_index}: {period!r}"
        )

    month = period[1:]

    return f"{year}-{month}-01"


def _normalize_value(
    *,
    raw_value: Any,
    observation_index: int,
) -> str | None:
    """
    Normalize a BLS observation value.

    Returns:
        A canonical numeric string for a valid numeric value.
        None for a recognized BLS missing-value marker.

    Raises:
        DataContractError:
            If the value is absent, empty, non-numeric, or non-finite.
    """
    if raw_value is None:
        raise DataContractError(
            f"missing value field at observation index "
            f"{observation_index}"
        )

    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(raw_value, bool):
        raise DataContractError(
            f"value is not numeric at observation index "
            f"{observation_index}: {raw_value!r}"
        )

    if isinstance(raw_value, str):
        value_text = raw_value.strip()

        if value_text in BLS_MISSING_VALUES:
            return None

        if value_text == "":
            raise DataContractError(
                f"value is empty at observation index "
                f"{observation_index}"
            )

        value_for_conversion: str | int | float = value_text

    elif isinstance(raw_value, (int, float)):
        value_for_conversion = raw_value

    else:
        raise DataContractError(
            f"value is not numeric at observation index "
            f"{observation_index}: {raw_value!r}"
        )

    try:
        numeric_value = float(value_for_conversion)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataContractError(
            f"value is not numeric at observation index "
            f"{observation_index}: {raw_value!r}"
        ) from exc

    if not math.isfinite(numeric_value):
        raise DataContractError(
            f"value must be finite at observation index "
            f"{observation_index}: {raw_value!r}"
        )

    return format(numeric_value, "g")
