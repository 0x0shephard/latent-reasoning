"""Build the Kaggle notebook for patch-selected two-term distillation (ledger §101)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_patch_selected_distillation.ipynb"
RUN_COMMIT = "__RUN_COMMIT__"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Patch-selected two-term distillation: can interventions on the *student* beat the full state? (ledger §101)

## tl;dr

The §100 primary (five seeds) found that copying the teacher's full decision state
repairs a damaged CODI student best, that a 12-direction answer-directed target repairs
**as many** questions as the full state but **breaks more** borderline ones, and that
teacher-side intervention and gradient scores pick the same directions. This notebook
builds the target from interventions on the **student**, which no teacher-side score
can see:

| set | how it is chosen | rank | pressure |
|---|---|---|---|
| **teaching** | *transfer patching*: patch the teacher's coordinates into the damaged student's decision state; keep the directions that most raise the student's answers | 12 | norm-matched ×1.0 |
| **anchor** | *breakage patching*: patch the student's drift into the official model's own latent state; keep the directions in which the drift breaks the most answers (excluding the teaching set) | 24 | norm-matched ×0.3 |

| arm | teaching | anchor | re-selected every 250 steps |
|---|---|---|---|
| `patch` | transfer-patch | — | no |
| `patch_anchor` | transfer-patch | breakage-patch | no |
| `patch_anchor_adaptive` | transfer-patch | breakage-patch | yes |
| `causal_anchor` (control) | §100 causal set | breakage-patch | no |

Seeds 1–3 reproduce the §100 damage and data order exactly, so every arm is **paired
with the §100 `none`, `full`, `causal` and `relevance` runs** of the same seed, read
from the published primary output. Nothing is retrained. ≈26 min per run, 12 runs.

**Predictions written before the run** (from the §100 per-question addendum): per seed
against `none`, `patch_anchor` repairs ≥ 59 questions and breaks < 36.6 (full's figure).

### What to attach

1. The official CODI reproduction dataset (as always).
2. The **published §100 primary output** (the pooled one, e.g. `dannyism11/seed-1-2-3`):
   it supplies the teacher cache, selectors, the damage calibration and the baseline
   runs' per-question predictions.
""")

md(r"""
## Go/no-go and gates

- **Anchor meaningful:** breakage of the selected anchor set on the official latent
  state ≥ 2× the mean breakage of 20 random 24-direction sets; else STOP.
- **Reported:** Jaccard of the transfer-patch teaching set with the §100 causal set.
  If ≥ 0.75, the `patch` arm is expected to tie `causal` and the test is about the anchor.
- **P1** `patch_anchor − full`, **P2** `− causal`, **P3** `− relevance`; **P4** `patch −
  causal`; **P5** `adaptive − fixed`; **P6** `causal_anchor − causal`. Paired bootstrap over
  the 1,319 GSM8K test questions on seed-mean correctness, with per-seed signs.
- **CONFIRMED** = P1 ∧ P2 ∧ P3 with 3/3 seeds each; **ANCHOR** = P6 ∧ ¬P1; **TIE** within 3
  points; otherwise **PARTIAL**.

This stage was designed after the §100 result and is labelled as such in the ledger.
""")

md("## Data and setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "__RUN_COMMIT__"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""   # "" = discover the official reproduction summary under /kaggle/input
PRIMARY_OUTPUT_INPUT = ""         # "" = discover the published §100 primary output (codi_recovery_primary)
PRIMARY_SMOKE_INPUT = ""          # "" = discover its codi_recovery_smoke sibling (used by the smoke pass)

# ---- per-session controls ----
SEEDS = "1,2,3"
ARMS = "patch,patch_anchor,patch_anchor_adaptive,causal_anchor"
RUN_PRELIMINARY_ONLY = False      # True = selection + go/no-go only (a few minutes)
RUN_SMOKE = True
MAX_SECONDS = 30600
PREVIOUS_OUTPUT_INPUT = ""        # attached copy of this notebook's earlier output; "" = discover

OUTPUT_DIR = "/kaggle/working/codi_patch_selected"
SMOKE_OUTPUT_DIR = "/kaggle/working/codi_patch_selected_smoke"

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
                "tests/test_causal_subspace_distillation.py",
                "tests/test_patch_selected_distillation_runner.py"], check=True)
''')

md("### Resolve inputs: reproduction summary, the §100 primary output, a previous session")
code(r'''
def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

def discover_dir(explicit, contract, *, want_test=False):
    if explicit and pathlib.Path(explicit).is_dir():
        return str(explicit)
    best = None
    for path in glob.glob("/kaggle/input/**/summary.json", recursive=True):
        try:
            record = json.loads(open(path).read())
        except Exception:
            continue
        if record.get("contract") != contract:
            continue
        if want_test and not record.get("test"):
            continue
        seeds = len(record.get("test", {}).get("arms", {}).get("none", {}).get("seeds", [])) if record.get("test") else 0
        if best is None or seeds > best[0]:
            best = (seeds, str(pathlib.Path(path).parent))
    return best[1] if best else None

REPRODUCTION_SUMMARY = discover_file(REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
PRIMARY = discover_dir(PRIMARY_OUTPUT_INPUT, "official_codi_recovery_subspace_distillation_v1", want_test=True)
assert PRIMARY, "Attach the published section-100 primary output (codi_recovery_primary with predictions.jsonl)"
PRIMARY_SMOKE = discover_dir(PRIMARY_SMOKE_INPUT, "official_codi_recovery_subspace_distillation_v1_smoke")
CONTRACT = "official_codi_patch_selected_distillation_v1"
PREVIOUS = discover_dir(PREVIOUS_OUTPUT_INPUT, CONTRACT)
print("reproduction", REPRODUCTION_SUMMARY)
print("primary output", PRIMARY)
print("primary smoke output", PRIMARY_SMOKE or "(none: smoke pass will be skipped)")
print("resume from", PREVIOUS or "(fresh start)")
primary_summary = json.loads(open(pathlib.Path(PRIMARY) / "summary.json").read())
print("primary seeds per arm:", {a: r["seeds"] for a, r in primary_summary["test"]["arms"].items()})
''')

md("## Smoke pass, then the run")
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

def runner_command(output_dir, shared, *extra):
    return [sys.executable, "-u", "scripts/run_codi_patch_selected_distillation.py",
            "--config", "configs/official_codi_gpt2.yaml",
            "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--shared-from", shared,
            "--arms", ARMS, "--batch-size", "16", "--micro-batch-size", "8", "--eval-batch-size", "32",
            "--bootstrap-samples", "10000", "--precision", "float32", "--device", "cuda", *extra]

if RUN_SMOKE and PRIMARY_SMOKE:
    run_with_retry(runner_command(SMOKE_OUTPUT_DIR, PRIMARY_SMOKE, "--smoke", "--seeds", "1"))
    smoke = json.loads(open(pathlib.Path(SMOKE_OUTPUT_DIR) / "summary.json").read())
    print("smoke status:", smoke["status"], "|", smoke["decision"])

extra = ["--seeds", SEEDS, "--max-seconds", str(MAX_SECONDS)]
if RUN_PRELIMINARY_ONLY:
    extra.append("--preliminary-only")
if PREVIOUS:
    extra += ["--resume-from", PREVIOUS]
run_with_retry(runner_command(OUTPUT_DIR, PRIMARY, *extra))
summary = json.loads(open(pathlib.Path(OUTPUT_DIR) / "summary.json").read())
print("status:", summary["status"])
print(summary["decision"])
''')

md("### Selection: which directions did the patches choose, and how do they compare with the causal set?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

g = summary["preliminary"]["gate_patch_selected"]
print("checks:", g["checks"])
print("teaching set (transfer patch):", g["teaching"])
print("section-100 causal set:        ", summary["selectors"]["causal_set"], "| jaccard", round(g["jaccard_teaching_causal"], 2))
print("anchor set (breakage patch):   ", g["anchor"])
print(f"student accuracy on the pool: unpatched {g['unpatched_accuracy']:.3f} -> teaching set patched in {g['patched_accuracy']:.3f}")
print(f"anchor breakage {g['anchor_breakage']:.3f} vs random sets {g['random_breakage_mean']:.3f} (ratio {g['anchor_over_random']:.1f})")
for name, run in summary.get("runs", {}).items():
    hist = run["selection_history"]
    if len(hist) > 1:
        print(name, "re-selections:", [(h["step"], round(h.get("jaccard_with_previous_teaching", 1.0), 2)) for h in hist[1:]])
''')

md("### GSM8K test exact match, paired with the §100 runs of the same seeds")
code(r'''
test = summary.get("test")
if not test:
    print("No aggregate yet:", summary["decision"])
else:
    arms = test["arms"]
    order = ["none", "variance", "random", "relevance", "causal", "full", "patch", "causal_anchor", "patch_anchor", "patch_anchor_adaptive"]
    rows = []
    for arm in order:
        if arm in arms:
            r = arms[arm]
            rows.append({"arm": arm, "source": r["source"], **{f"seed {s}": round(100 * a, 2) for s, a in zip(r["seeds"], r["test_accuracy_by_seed"])},
                         "mean": round(100 * r["test_accuracy_mean"], 2)})
    display(pd.DataFrame(rows).set_index("arm"))
    comps = pd.DataFrame([{"comparison": k, "mean (pts)": 100 * v["mean_difference"],
                           "lower": 100 * v["bootstrap_95ci"][0], "upper": 100 * v["bootstrap_95ci"][1],
                           "per-seed (pts)": [round(x, 2) for x in test["per_seed_differences"][k]]}
                          for k, v in test["comparisons"].items()])
    display(comps)
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(comps) + 1.5), constrained_layout=True)
    y = np.arange(len(comps))
    for i, (_, r) in enumerate(comps.iterrows()):
        ax.errorbar(r["mean (pts)"], y[i], xerr=[[r["mean (pts)"] - r["lower"]], [r["upper"] - r["mean (pts)"]]],
                    fmt="o", capsize=4, color="#315f8c")
    ax.axvline(0, color="#222", linewidth=1)
    ax.set(yticks=y, yticklabels=comps["comparison"], xlabel="paired GSM8K exact-match difference, points (95% bootstrap over questions)")
    plt.show()
    dec = test["decomposition_vs_none"]
    display(pd.DataFrame([{"arm": a, "repairs / seed": round(d["repairs_mean"], 1), "breaks / seed": round(d["breaks_mean"], 1),
                           "net": round(d["repairs_mean"] - d["breaks_mean"], 1)} for a, d in dec.items()]).set_index("arm")
            .loc[[a for a in order if a in dec]])
    print("predictions: patch_anchor repairs >= 59 and breaks < 36.6")
    display(pd.DataFrame([test["gate"]]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Status:", summary["status"])
print("Decision:", summary["decision"]["claim"])
if summary["status"] == "paused":
    print("\nNEXT: publish", OUTPUT_DIR, "as a Kaggle dataset, attach it, and Run All again.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
output = pathlib.Path(OUTPUT_DIR)
print("artefacts:", sorted(p.name for p in output.glob("*")))
print("runs:", sorted(p.name for p in (output / "runs").glob("*")))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
