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
BLS_MISSING_VALUES = {".", ""}


def load(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON document from disk.

    Parameters
    ----------
    path:
        Path to the JSON fixture or downloaded API response.

    Returns
    -------
    dict
        Parsed JSON document.

    Raises
    ------
    DataContractError
        If the file cannot be read, contains invalid JSON, or the JSON
        document is not an object.
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

    - ``series_id``
    - ``date`` as an ISO month-start date
    - ``value`
