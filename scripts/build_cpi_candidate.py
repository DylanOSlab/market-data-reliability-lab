from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SERIES_ID = "CUUR0000SA0"
SOURCE_URL = f"https://api.bls.gov/publicAPI/v1/timeseries/data/{SERIES_ID}"
RAW_PATH = ROOT / "fixtures" / "raw" / "bls_cuur0000sa0.json"
GOLDEN_PATH = ROOT / "fixtures" / "golden" / "bls_cuur0000sa0.csv"
PROVENANCE_PATH = ROOT / "provenance" / "bls_cuur0000sa0.json"
TEST_PATH = ROOT / "tests" / "test_cpi_contract.py"


def fetch() -> bytes:
    try:
        with urlopen(SOURCE_URL, timeout=30) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download official BLS CPI data: {exc}") from exc


def validate_envelope(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("BLS response root must be an object")
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError(f"BLS request failed: {payload.get('message')}")
    results = payload.get("Results")
    if not isinstance(results, dict):
        raise TypeError("Missing Results object")
    series = results.get("series")
    if not isinstance(series, list) or len(series) != 1:
        raise ValueError("Expected exactly one BLS series")
    if series[0].get("seriesID") != SERIES_ID:
        raise ValueError("Unexpected BLS series ID")
    return payload


def normalized_csv(payload: dict) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from mdrlab.core import normalize

    rows = normalize(payload)
    if not rows:
        raise ValueError("CPI normalization produced no rows")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["series_id", "date", "value"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def test_source() -> str:
    return """from pathlib import Path

from mdrlab.core import load, normalize, to_csv

ROOT = Path(__file__).parents[1]


def test_cpi_raw_matches_golden() -> None:
    actual = to_csv(normalize(load(ROOT / "fixtures/raw/bls_cuur0000sa0.json")))
    expected = (ROOT / "fixtures/golden/bls_cuur0000sa0.csv").read_text(
        encoding="utf-8"
    )
    assert actual == expected


def test_cpi_series_identity() -> None:
    rows = normalize(load(ROOT / "fixtures/raw/bls_cuur0000sa0.json"))
    assert rows
    assert {row["series_id"] for row in rows} == {"CUUR0000SA0"}
"""


def main() -> None:
    raw_bytes = fetch()
    try:
        payload = validate_envelope(json.loads(raw_bytes))
    except json.JSONDecodeError as exc:
        raise ValueError("BLS response is not valid JSON") from exc

    golden = normalized_csv(payload)
    canonical_raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(canonical_raw).hexdigest()

    for path in (RAW_PATH, GOLDEN_PATH, PROVENANCE_PATH, TEST_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    RAW_PATH.write_bytes(canonical_raw)
    GOLDEN_PATH.write_text(golden, encoding="utf-8")
    PROVENANCE_PATH.write_text(
        json.dumps(
            {
                "series_id": SERIES_ID,
                "source": "U.S. Bureau of Labor Statistics Public Data API v1",
                "source_url": SOURCE_URL,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
                "generator": "scripts/build_cpi_candidate.py",
                "review_required": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    TEST_PATH.write_text(test_source(), encoding="utf-8")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("# CPI candidate generated\n\n")
            handle.write(f"- Series: `{SERIES_ID}`\n")
            handle.write(f"- Raw SHA-256: `{digest}`\n")
            handle.write("- Golden review required: `true`\n")


if __name__ == "__main__":
    main()
