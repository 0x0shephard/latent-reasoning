"""Build the Kaggle notebook for the templated-task latent-path diagnostic (ledger §91)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_templated_latent_diagnostic.ipynb"
RUN_COMMIT = "1a876c2090b98286cf8ae19092c2582062c5e231"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can the official CODI latent path do the templated task?

## tl;dr

Ledger §90: fresh CODI students trained for 3,000 steps on generated two- and
three-step arithmetic word problems never left the exact-match floor (≈1%), so the
§89 comparison of distillation targets was floor against floor. This notebook asks
the one question that decides how to read that: **can the unmodified official
checkpoint's latent path solve these problems at all?**

It rebuilds the same 2,000 test problems (same generator, seed and split order) and
scores the official checkpoint two ways, with no training:

| path | input | what it tells us |
|---|---|---|
| latent | question + BOT + 6 latent thoughts + forced answer decode (released path) | can a *converged* latent model do this task |
| explicit CoT | question alone, greedy | the teacher side (≈83% on the §89 check split) |

**Reading (preregistered):** latent ≥ 50% → **BUDGET** (the §90 floor is a training-budget
failure); latent ≤ 20% → **TASK** (the latent path itself finds these problems hard);
between → **MIXED**. Runtime ≈ 3–5 minutes on a T4/P100.
""")

md("## Setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = (
    "/kaggle/input/datasets/jonraza15/"
    "corrected-official-codi-answer-cue-endpoint-tsv-c/"
    "latent-reasoning/outputs/official_codi_gpt2/eval/"
    "revision_fd641b3d/full_gsm8k/summary.json"
)
OUTPUT_DIR = "/kaggle/working/codi_templated_latent_diagnostic"
SMOKE_LIMIT = 64          # first pass scores 64 problems per split as a pipeline check
RUN_SMOKE = True

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
assert len(RUN_COMMIT) == 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip())
''')

md("### Environment and focused tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2",
    "accelerate==1.7.0", "huggingface-hub>=0.34,<1.0",
    "safetensors==0.5.3", "pytest>=8,<10",
], check=True)
probe = subprocess.run([
    sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\n"
    "try:\n print(is_torchao_available())\n"
    "except ImportError:\n print('incompatible')",
], capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_templated_arithmetic.py",
    "tests/test_templated_latent_diagnostic.py",
], check=True)

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
print("reproduction", REPRODUCTION_SUMMARY)
''')

md("## Run: smoke on 64 problems, then the full 2,000")
code(r'''
def command(output_dir, *extra):
    return [sys.executable, "-u", "scripts/run_codi_templated_latent_diagnostic.py",
            "--config", "configs/official_codi_gpt2.yaml",
            "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", "64",
            "--precision", "float32", "--device", "cuda", *extra]

if RUN_SMOKE:
    subprocess.run(command(OUTPUT_DIR + "_smoke", "--limit", str(SMOKE_LIMIT)), check=True)
subprocess.run(command(OUTPUT_DIR), check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("## Results")
code(r'''
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
test = summary["results"]["test"]
sel = summary["results"]["selection"]
print(f"official latent path,   test (n={test['examples']}):      {test['latent']['accuracy']:.3f}")
print(f"official explicit CoT,  test (n={test['examples']}):      {test['explicit_cot']['accuracy']:.3f}")
print(f"official latent path,   selection (n={sel['examples']}): {sel['latent']['accuracy']:.3f}")
print("agreement on test:", test["agreement"])
print("for scale: §90 from-scratch students after 3,000 steps ≈ 0.010–0.013 on the same test problems;"
      " official CODI GPT-2 on GSM8K ≈ 0.43")

rows = []
for key in test["latent"]["breakdown"]:
    rows.append({"group": key, "latent": test["latent"]["breakdown"][key]["accuracy"],
                 "explicit CoT": test["explicit_cot"]["breakdown"][key]["accuracy"],
                 "n": test["latent"]["breakdown"][key]["examples"]})
table = pd.DataFrame(rows).set_index("group")
display(table)
ax = table[["latent", "explicit CoT"]].plot.barh(figsize=(8, 5), color=["#b65f24", "#315f8c"])
ax.set(xlabel="exact match on the 2,000 templated test problems", xlim=(0, 1),
       title="Official checkpoint: latent path vs explicit CoT, by template and step count")
plt.tight_layout(); plt.show()
display(pd.DataFrame(test["samples"]))
''')

md("## Takeaways")
code(r'''
print("Reading:", summary["decision"]["reading"])
print()
if summary["decision"]["reading"].startswith("BUDGET"):
    print("The task is learnable by the latent path. The §90 floor was training budget, so the")
    print("only remaining informative design is a warm-start from the official weights (no reinit),")
    print("fine-tuned on the templated task under the same none/variance/causal arms.")
elif summary["decision"]["reading"].startswith("TASK"):
    print("The converged latent path cannot do these problems either. No from-scratch design at this")
    print("budget was going to reach a non-trivial exact match; the §90 STOP is a task property, and")
    print("the write-up should say so. Look at the per-template table: step-3 and multiplication-heavy")
    print("templates are the usual failure modes for a 6-thought latent GPT-2.")
else:
    print("Partial ability. Check which templates the latent path solves; a warm-start experiment")
    print("restricted to those templates would be the honest follow-up.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
print("\nartefacts:", sorted(p.name for p in pathlib.Path(OUTPUT_DIR).glob("*")))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
