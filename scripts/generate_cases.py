import copy
import json
from pathlib import Path

src = json.loads(Path("fixtures/raw/bls_lns11300000.json").read_text())
out = Path("fixtures/cases")
out.mkdir(parents=True, exist_ok=True)


def write(name, doc):
    (out / name).write_text(json.dumps(doc, indent=2) + "\n")


base = src["Results"]["series"][0]["data"]
d = copy.deepcopy(src)
d["Results"]["series"][0]["data"].append(copy.deepcopy(base[0]))
write("duplicate_key.json", d)
d = copy.deepcopy(src)
d["Results"]["series"][0]["data"][0]["period"] = "M13"
write("invalid_period.json", d)
d = copy.deepcopy(src)
d["Results"]["series"][0]["data"][0]["value"] = "62.4%"
write("numeric_as_text_noise.json", d)
d = copy.deepcopy(src)
d["results"] = d.pop("Results")
write("schema_drift.json", d)
d = copy.deepcopy(src)
d["Results"]["series"][0]["data"].reverse()
write("reverse_order.json", d)
print("Generated deterministic cases")
