from __future__ import annotations
import csv, io, json, re
from pathlib import Path
from .errors import DataContractError

PERIOD = re.compile(r"^M(0[1-9]|1[0-2])$")

def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def normalize(payload: dict) -> list[dict[str, str]]:
    if "Results" not in payload or "series" not in payload["Results"]:
        raise DataContractError("missing Results.series")
    series = payload["Results"]["series"]
    if len(series) != 1 or "seriesID" not in series[0] or "data" not in series[0]:
        raise DataContractError("expected exactly one well-formed series")
    sid = series[0]["seriesID"]
    rows, seen = [], set()
    for item in series[0]["data"]:
        year, period, value = item.get("year"), item.get("period"), item.get("value")
        if not isinstance(year, str) or not year.isdigit() or not PERIOD.match(str(period)):
            raise DataContractError("invalid year or monthly period")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise DataContractError("value is not numeric") from exc
        date = f"{year}-{period[1:]}-01"
        key = (sid, date)
        if key in seen:
            raise DataContractError("duplicate series/date key")
        seen.add(key)
        rows.append({"series_id": sid, "date": date, "value": format(number, "g")})
    return sorted(rows, key=lambda r: r["date"])

def to_csv(rows: list[dict[str, str]]) -> str:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["series_id", "date", "value"], lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()
