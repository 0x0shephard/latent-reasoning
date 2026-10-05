"""Build the Kaggle notebook for the directed counterfactual (ledger §112)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_directed_counterfactual_long.ipynb"
RUN_COMMIT = "ce97cfd8fdf118a93395900c62bc0c7ac8e415b8"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# On long chains, does the thought that decodes a value carry it? (ledger §112)

## tl;dr

§111 found that on typical GSM8k-Aug questions no single thought transports a value:
swapping a thought's state between counterfactual twins leaves the own answer in place
86–92% of the time, and the counterfactual is carried by the whole latent-tail K/V
together with the question. Transport through a single thought appeared only in chains of
three or more steps, in small cells. This notebook preregisters that reading: twins are
built only from **chains with ≥ 3 equations**, the number entering the chain **latest** is
changed so earlier intermediates stay fixed, and a pair is **uniquely located** when exactly
one odd slot decodes the changed value. **No training**, about 35–50 minutes of GPU.

| check | rule |
|---|---|
| E1 long-chain tail | whole-tail K/V swap moves ≥ 50% of first tokens to the twin's answer |
| E2 single-thought transport | pooled, the better of slot 1 / slot 3 state swaps ≥ 0.25 |
| E3 unique-location specificity (primary) | at the uniquely located thought: its state ≥ 0.30 and ≥ 2× the other feeding thought's; the following position's K/V ≥ 2× the other even position's |

Attach only the official reproduction dataset.
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "ce97cfd8fdf118a93395900c62bc0c7ac8e415b8"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""   # "" = discover under /kaggle/input
OUTPUT_DIR = "/kaggle/working/codi_directed_counterfactual_long"
RUN_SMOKE = True

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
assert len(RUN_COMMIT) == 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
''')

md("### Environment and focused tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub>=0.34,<1.0", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\ntry:\n print(is_torchao_available())\nexcept ImportError:\n print('incompatible')"],
    capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_counterfactual_chain.py",
                "tests/test_directed_counterfactual.py",
                "tests/test_directed_counterfactual_runner.py",
                "tests/test_cache_carrier.py"], check=True)

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

REPRODUCTION_SUMMARY = discover_file(REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
print("reproduction", REPRODUCTION_SUMMARY)
''')

md("## Run: smoke (12 pairs), then up to 512 long-chain pairs")
code(r'''
def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s ({attempt + 1}/{attempts})")
        time.sleep(wait)

def command(output_dir, *extra):
    return [sys.executable, "-u", "scripts/run_codi_directed_counterfactual.py",
            "--config", "configs/official_codi_gpt2.yaml", "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", "32", "--precision", "float32", "--device", "cuda", "--profile", "long", *extra]

if RUN_SMOKE:
    run_with_retry(command(OUTPUT_DIR + "_smoke", "--smoke"))
run_with_retry(command(OUTPUT_DIR))
summary = json.loads(open(pathlib.Path(OUTPUT_DIR) / "summary.json").read())
print("gate:", summary["gate"])
print("baseline:", summary["baseline"])
print(summary["decision"])
''')

md("## Results")
code(r'''
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

conds = summary["report"]["conditions"]
rows = []
for name, v in conds.items():
    a = v["all"]
    rows.append({"condition": name, "target (twin's answer)": round(a["target"], 3), "retain (own)": round(a["retain"], 3),
                 "other": round(a["other"], 3)})
print("first token after the swap, both directions pooled, n =", conds["state_1"]["all"]["n"])
display(pd.DataFrame(rows).set_index("condition"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
xs = list(range(6))
for kind, color in (("state", "#b65f24"), ("v89", "#2a6f97"), ("kv", "#444444")):
    axes[0].plot(xs, [conds[f"{kind}_{s}"]["all"]["target"] for s in xs], marker="o", ms=4, color=color, label=kind)
axes[0].plot([0, 2, 4], [conds[f"v1011_{s}"]["all"]["target"] for s in (0, 2, 4)], marker="s", ms=4, color="#a9d6e5", label="v1011 (control)")
axes[0].set(xlabel="slot / position", ylabel="target rate (answer becomes the twin's)", title="single-site swaps, pooled")
axes[0].legend(fontsize=8)
strata = sorted(summary["strata_counts"])
for name, color in (("state_1", "#b65f24"), ("state_3", "#e0a070"), ("v89_2", "#2a6f97"), ("v89_4", "#61a5c2")):
    axes[1].plot(strata, [conds[name]["by_stratum"][k]["target"] or 0 for k in strata], marker="o", ms=4, color=color, label=name)
axes[1].set(xlabel="stratum (chain length n, perturbed step k)", ylabel="target rate", title="by where the number enters the chain")
axes[1].tick_params(axis="x", rotation=45); axes[1].legend(fontsize=8)
plt.show()

print("strata counts:", summary["strata_counts"])
loc = summary["report"]["located"]; spec = summary["report"]["located_specificity"]
print("located fraction:", round(loc["fraction_located"], 3), "rows by slot:", loc["rows_by_slot"], "multi-slot rows:", loc["rows_multi_slot"])
spec_rows = []
for s, v in spec.items():
    spec_rows.append({"located at slot": s, "n": v["n"],
                      "state at site": v["state_at_site"]["target"],
                      **{f"state at slot {o}": w["target"] for o, w in v["state_elsewhere"].items()},
                      f"store at position {int(s) + 1}": v["store_at_site"]["target"],
                      **{f"store at position {o}": w["target"] for o, w in v["store_elsewhere"].items()}})
print("D3: located specificity (target rates among located pairs)")
display(pd.DataFrame(spec_rows))
print("located at the terminal slot:", summary["report"]["located_terminal"])
uniq = summary["report"]["unique_location"]; uspec = summary["report"]["unique_specificity"]
print("uniquely located fraction:", round(uniq["fraction_unique"], 3), "rows by slot:", uniq["rows_by_slot"])
urows = []
for s, v in uspec.items():
    urows.append({"uniquely located at slot": s, "n": v["n"],
                  "state at site": v["state_at_site"]["target"],
                  **{f"state at slot {o}": w["target"] for o, w in v["state_elsewhere"].items()},
                  f"K/V at position {int(s) + 1}": v["kv_store_at_site"]["target"],
                  **{f"K/V at position {o}": w["target"] for o, w in v["kv_store_elsewhere"].items()},
                  f"values L8-9 at position {int(s) + 1}": v["store_at_site"]["target"]})
print("E3: unique-location specificity (target rates among uniquely located rows)")
display(pd.DataFrame(urows))
print("uniquely located at the terminal slot:", summary["report"]["unique_terminal"])
print("native decoding (baseline exact match", round(summary["baseline"]["native_exact_match"], 3), ")")
display(pd.DataFrame(summary["report"]["native"]).T)
display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary["checks"].items()}]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print()
for w in summary["warnings"]:
    print("WARNING:", w)
print("seconds:", round(summary["seconds"]))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
