# Market Data Reliability Lab

A small, reproducible benchmark for testing economic and market-data ingestion systems against failures derived from real public data.

## MVP scope

The first dataset is the U.S. Bureau of Labor Statistics (BLS) labor-force participation series `LNS11300000`. The checked-in raw fixture preserves the BLS JSON shape and a documented observation (`2025 M12 = 62.4`). Tests run offline and never depend on the availability of an external API.

## Reliability contract

A conforming pipeline must:

1. Validate the envelope and series identity.
2. Reject invalid periods and non-numeric values.
3. Reject duplicate `(series_id, date)` keys.
4. Normalize `M01..M12` to ISO month-start dates.
5. Sort ascending by date.
6. Produce output identical to the golden CSV.

## Layout

```text
src/mdrlab/          adapters, schema, normalization, CLI
fixtures/raw/        immutable source snapshots
fixtures/cases/      deterministic corrupted variants
fixtures/golden/     canonical answers
provenance/          source URL, retrieval time, checksum
scripts/             refresh and case-generation commands
tests/               unit and contract tests
.github/workflows/   CI and scheduled upstream smoke test
```

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest
python -m mdrlab.cli validate fixtures/raw/bls_lns11300000.json
python -m mdrlab.cli normalize fixtures/raw/bls_lns11300000.json
```

## Generate failure cases

```bash
python scripts/generate_cases.py
pytest
```

## Refresh from the official BLS API

```bash
python scripts/fetch_bls_snapshot.py
```

Refresh is intentionally separate from CI's deterministic offline tests. The scheduled smoke job checks the upstream contract without silently changing golden files.

## Initial failure catalog

- `duplicate_key.json`: duplicate observation
- `invalid_period.json`: period `M13`
- `numeric_as_text_noise.json`: value contains a thousands separator/noise
- `schema_drift.json`: `Results` renamed to `results`
- `reverse_order.json`: valid records in descending order, expected to normalize successfully

## Next increments

- Add CPI (`CUUR0000SA0`) and unemployment (`LNS14000000`).
- Add FRED/ALFRED vintage tests for revision-aware processing.
- Add exchange-rate or securities-market sources only after licensing and redistribution checks.
- Publish benchmark results as JUnit and JSON artifacts.
