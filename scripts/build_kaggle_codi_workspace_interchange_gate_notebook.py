"""Build the Kaggle notebook for the workspace interchange go/no-go (ledger §105)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_workspace_interchange_gate.ipynb"
RUN_COMMIT = "c94d7f043c44c78915b437898f583a1bce1e677c"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does CODI's latent workspace mediate the answer under interchange? (ledger §105 go/no-go)

## tl;dr

§55 showed the odd thought slots hold the solution's intermediate values. Before any
training on the trajectory, this notebook asks the official model three questions, with
**no training** and about 25 minutes of GPU:

1. **Mediation.** If we swap a slot's state between two questions and let the model carry
   on, does the answer change? Does it change through a *low-rank* subspace of the slot?
2. **Specificity.** Is the effect larger at the odd (value) slots than at the even ones?
3. **Divergence.** Do gradient scores over the slot's principal directions rank them the
   way the interchange effects do? If yes, an intervention-selected trajectory target
   cannot differ from a gradient-selected one and the method route closes.

| gate | threshold |
|---|---|
| M1 slots matter | full-slot swap at each feeding odd slot (1, 3) changes the first answer token for ≥ 30% of pairs |
| M2 low-rank mediation | some rank ≤ 64 reaches ≥ 50% of the full-slot change rate at each odd slot |
| M3 specificity | mean odd-slot change rate ≥ 1.5× mean even-slot change rate, at rank 64 and at full rank |
| M4 divergence | Spearman(gradient score, interchange effect) over the first 64 PCs ≤ 0.7 at each odd slot |

**GO = all four.** Each failing gate names what it closes. Slot 5 is the terminal thought:
the released path projects its state and then feeds the end-of-thought token, so its output
state cannot affect the answer. It is swept as a built-in negative control (its change rate
must be zero) and excluded from the gates. Attach only the official reproduction dataset.
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "c94d7f043c44c78915b437898f583a1bce1e677c"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""   # "" = discover under /kaggle/input
OUTPUT_DIR = "/kaggle/working/codi_workspace_interchange_gate"
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
                "tests/test_workspace_interchange.py",
                "tests/test_workspace_interchange_gate_runner.py"], check=True)

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
    return [sys.executable, "-u", "scripts/run_codi_workspace_interchange_gate.py",
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
import numpy as np
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

slots = summary["report"]["slots"]
ranks = [r for r in slots["1"]["first_token"] if r != "full"]
rows = []
for k, s in slots.items():
    row = {"slot": int(k), "kind": "odd (values)" if int(k) % 2 else "even"}
    for r, v in s["first_token"].items():
        row[f"change @ {r}"] = round(v["change_rate"], 3)
    row["spearman grad vs intervention"] = None if s["spearman_gradient_vs_intervention"] is None else round(s["spearman_gradient_vs_intervention"], 2)
    rows.append(row)
table = pd.DataFrame(rows).set_index("slot")
display(table)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
xs = [int(r) for r in ranks] + [768]
for k, s in slots.items():
    ys = [s["first_token"][r]["change_rate"] for r in ranks] + [s["first_token"]["full"]["change_rate"]]
    axes[0].plot(xs, ys, marker="o", ms=3, color="#b65f24" if int(k) % 2 else "#8a8a8a", label=f"slot {k}")
axes[0].set(xscale="log", xlabel="interchanged rank (PCs of the slot state)", ylabel="first-token change rate",
            title="odd slots (orange) vs even slots (grey)")
axes[0].legend(fontsize=7, ncol=2)
for k in ("1", "3"):
    s = slots[k]
    if s.get("per_direction_change_rate"):
        axes[1].scatter(s["per_direction_gradient_score"], s["per_direction_change_rate"], s=14, alpha=0.7, label=f"slot {k}  rho={s['spearman_gradient_vs_intervention']:.2f}")
axes[1].set(xscale="log", xlabel="gradient score per PC (relevance)", ylabel="interchange change rate per PC",
            title="do gradients rank the mediating directions?")
axes[1].legend(fontsize=8)
plt.show()

native_rows = []
for k in ("1", "3"):
    for label, v in slots[k].get("native", {}).items():
        native_rows.append({"slot": int(k), "swap": label, "native change rate": round(v["change_rate"], 3),
                            "exact match under swap": round(v["exact_match"], 3),
                            "equals donor's answer (among changed)": round(v["equals_donor_answer_among_changed"], 3)})
print("terminal slot 5 change rate (must be 0):", slots["5"]["first_token"]["full"]["change_rate"])
print("native decoding, odd slots (baseline exact match", round(summary["baseline"]["native_exact_match"], 3), ")")
display(pd.DataFrame(native_rows))
for k in ("1", "3"):
    print(f"slot {k}: top-8 PCs by intervention {slots[k]['top8_by_intervention']} | by gradient {slots[k]['top8_by_gradient']}")
display(pd.DataFrame([summary["gate"]]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print()
g = summary["gate"]
if g["go"]:
    print("GO. The next stage is interchange-intervention training on the odd slots (to be preregistered as its own section).")
else:
    print("STOP. The named gate closes the corresponding part of the trajectory route; see the ledger section 105 for what each means.")
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
