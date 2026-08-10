from pathlib import Path
from urllib.request import urlopen
URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/LNS11300000"
with urlopen(URL, timeout=30) as response:
    body = response.read()
Path("fixtures/raw/bls_lns11300000.latest.json").write_bytes(body)
print("Wrote fixtures/raw/bls_lns11300000.latest.json")
