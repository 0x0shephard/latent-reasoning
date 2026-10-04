"""Build the Kaggle notebook for the cache-carrier audit (ledger §108)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_cache_carrier_audit.ipynb"
RUN_COMMIT = "f8fa7281a0bbc60a158460c412a098fc13c6a0da"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Which route carries a CODI thought's effect: its state or its K/V? (ledger §108 audit)

## tl;dr

Every thought slot acts on the answer two ways: its output **state** is projected into
the next thought's input (that is what a linear monitor reads), and the **K/V** it writes
are attended to by every later position. §106 swapped states only: slots 1 and 3 changed
about a third of answers, slot 5 changed none because its projected state is discarded.
SCIT (EMNLP 2026) swapped cache segments only and found the carrier in the layers 8–9
value cache. This notebook swaps either route, or both, between paired questions, with
**no training** and about 30–40 minutes of GPU.

| check | rule |
|---|---|
| A1 replication | state swap ≥ 0.25 change at slots 1 and 3; 0 at slot 5 |
| A2 terminal slot is a memory | K/V swap at slot 5 changes ≥ 10% of first tokens |
| A3 route split | slots 1 and 3: cache-dominant / state-dominant / shared at ratio 1.5 |
| A4 values over keys | values ≥ keys at a majority of slot 1, slot 3, slot 5, tail |
| A5 layer localisation | values at layers 8–9 ≥ layers 10–11 at the tail |
| A6 tail transplant | complete latent-tail transplant sends changed answers to the donor's (≥ 0.5, ≥ null + 0.3) |

Reported as a profile, not a single go/no-go. Attach only the official reproduction dataset.
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "f8fa7281a0bbc60a158460c412a098fc13c6a0da"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""   # "" = discover under /kaggle/input
OUTPUT_DIR = "/kaggle/working/codi_cache_carrier_audit"
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
                "tests/test_cache_carrier.py",
                "tests/test_cache_carrier_audit_runner.py"], check=True)

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

REPRODUCTION_SUMMARY = discover_file(REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
print("reproduction", REPRODUCTION_SUMMARY)
''')

md("## Run: smoke on 24 questions, then the full 512")
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
    return [sys.executable, "-u", "scripts/run_codi_cache_carrier_audit.py",
            "--config", "configs/official_codi_gpt2.yaml", "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", "32", "--precision", "float32", "--device", "cuda", *extra]

if RUN_SMOKE:
    run_with_retry(command(OUTPUT_DIR + "_smoke", "--smoke"))
run_with_retry(command(OUTPUT_DIR))
summary = json.loads(open(pathlib.Path(OUTPUT_DIR) / "summary.json").read())
print("baseline:", summary["baseline"])
print(summary["decision"])
''')

md("## Results")
code(r'''
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

slots = summary["report"]["slots"]; tail = summary["report"]["tail"]
conditions = list(next(iter(slots.values())).keys())
table = pd.DataFrame({int(k): {c: round(v[c]["change_rate"], 3) for c in conditions} for k, v in slots.items()}).T
table.index.name = "slot"
print("first-token change rate per slot and route (donor = paired question)")
display(table)

tail_table = pd.DataFrame({name: {"change rate": round(v["change_rate"], 3), "accuracy": round(v["accuracy"], 3),
                                  "toward donor (among changed)": v["toward_donor_among_changed"], "donor null": v["donor_null"]}
                           for name, v in tail.items()}).T
print("all six slots at once")
display(tail_table)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
xs = [int(k) for k in slots]
for c, color in (("hidden", "#b65f24"), ("kv", "#2a6f97"), ("v", "#61a5c2"), ("k", "#a9d6e5"), ("hidden_kv", "#444444")):
    axes[0].plot(xs, [slots[str(x)][c]["change_rate"] for x in xs], marker="o", ms=4, color=color, label=c)
axes[0].set(xlabel="slot", ylabel="first-token change rate", title="state route (orange) vs cache routes (blue)")
axes[0].legend(fontsize=8)
for c, color in (("v_layers_0_7", "#c8c8c8"), ("v_layers_8_9", "#2a6f97"), ("v_layers_10_11", "#8a8a8a")):
    axes[1].plot(xs, [slots[str(x)][c]["change_rate"] for x in xs], marker="o", ms=4, color=color, label=c)
axes[1].set(xlabel="slot", ylabel="first-token change rate", title="values by layer group (SCIT: 8-9 should carry)")
axes[1].legend(fontsize=8)
plt.show()

native = summary["report"]["native"]
rows = [{"where": f"slot {s}", "condition": c, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}}
        for s, conds in native["slots"].items() for c, out in conds.items()]
rows += [{"where": "tail", "condition": c, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}}
         for c, out in native["tail"].items()]
print("native decoding (baseline exact match", round(summary["baseline"]["native_exact_match"], 3), ")")
display(pd.DataFrame(rows))
display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary["checks"].items()}]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Profile:", summary["decision"]["claim"])
print()
c = summary["checks"]
if c["a2_terminal_slot_is_a_memory"]:
    print("Slot 5, inert as a thought in section 106, is consumed as a memory: a monitor reading slot states reads the wrong object there.")
else:
    print("Slot 5 is inert through both routes: the terminal thought contributes nothing after it is written.")
print("Feeding slots:", c["a3_route_split"])
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
