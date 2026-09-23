"""Build the Kaggle notebook for the templated-task subspace-distillation test (ledger §89)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_templated_subspace_distillation.ipynb"
RUN_COMMIT = "2bc825c48157e39d27df51f5bf7a046b1414707d"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Is the variance-distillation tax permanent? A templated-task test

## tl;dr

On GSM8k-Aug (ledger §86–§88, two seeds) distilling a fresh CODI student toward the
teacher's **top-variance** decision-state directions made it *worse* than no
distillation, while the **intervention-selected (causal)** directions cost less. Those
students never solved GSM8K, so the finding was about early loss. Here the same GPT-2
teacher, selectors and norm-matched distillation are applied to generated two- and
three-step arithmetic word problems in GSM8k-Aug format, a task the student can
finish in 3,000 steps. Exact match at the end is the primary outcome.

| arm | selector over the teacher's decision-state PCs |
|---|---|
| `none` | answer cross-entropy only |
| `variance` | first r PCs (LoRi-style) |
| `causal` | greedy **retain-only** intervention on the teacher (§36 analytic tier) |

### Go/no-go first (about 10 minutes)

Set `RUN_PRELIMINARY_ONLY = True` and Run All. The runner checks that the teacher
solves ≥80% of held-out templated problems by explicit CoT generation, that the rank
rule selects a rank, and that dense colon first-token accuracy is ≥80%. Only if
`preliminary.json` says GO should you set `RUN_PRELIMINARY_ONLY = False` and spend
the quota.

### Two accounts, several sessions

- `SEEDS = "1,2,3"` on this account and `SEEDS = "4,5"` on the other (five seeds total).
- Each session stops cleanly after `MAX_SECONDS`; **publish the output directory as a
  dataset**, attach it to the next session, and the run resumes.
- Attach the other account's finished output as `OTHER_ACCOUNT_OUTPUT_INPUT` (or leave
  discovery on) and the Results cell pools all seeds.

Budget: roughly 45 minutes per (arm, seed) at 3,000 steps, so ≈7 h for three seeds
and ≈4.5 h for two.
""")

md(r"""
## Context & Methods

Teacher decision states \(h\) (last normalised state at the answer colon) with mean
\(\mu\) and PC basis \(V\). The retain-only edit keeps the selected coordinates,
\(h' = \mu + V_S V_S^\top (h-\mu)\), and the teacher's own readout scores \(h'\). The
**causal** set is built greedily by adding the PC that most raises retain-only
first-token accuracy (margin as tie-break) on a disjoint split. The **variance** set
is the first r PCs. Rank r is chosen on the teacher only.

Student loss: answer cross-entropy plus
\(\mathrm{smoothL1}\big(V_S^\top(h_s-\mu),\,V_S^\top(h_t-\mu)\big)/\sigma_S\), with the
distillation gradient rescaled to the cross-entropy gradient norm every step
(**norm-matched**), as in §86.

### Gates (paired bootstrap over questions, seed-mean exact match on 2,000 test problems)

- **T1** `none − variance`: lower bound > 0 → the variance tax **persists**; upper
  bound < 0 → **reversed**; interval covers 0 → **transient**.
- **H1** `causal − variance` lower bound > 0.
- **H1b** `causal − none`, two-sided.
- Secondary: steps to 30% selection-split exact match (reported, not gated).

Claims: **PERMANENT** = T1 ∧ H1; **PARTIAL** = one of them; **REVERSED**; **TRANSIENT**.

### Boundaries

- Generated problems say nothing about GSM8K difficulty; this decides whether the
  §86 effect is a permanent property of the selectors or an early-training artefact.
- Students learn from fresh adapters on the checkpoint's embeddings and readout.
- The test split is read once per trained student.
""")

md("## Data and setup")
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
OUTPUT_DIR = "/kaggle/working/codi_templated_subspace_distillation"
SMOKE_OUTPUT_DIR = "/kaggle/working/codi_templated_subspace_distillation_smoke"

# ---- per-account and per-session controls ----
RUN_PRELIMINARY_ONLY = True      # go/no-go only (about 10 min); set False to train
SEEDS = "1,2,3"                  # "1,2,3" on this account, "4,5" on the other
MAX_SECONDS = 30600              # stop cleanly after 8.5 h with a resumable checkpoint
RUN_SMOKE = True                 # two-minute pipeline check before the real run
PREVIOUS_OUTPUT_INPUT = ""       # attached copy of this account's earlier output; "" = discover
OTHER_ACCOUNT_OUTPUT_INPUT = ""  # attached copy of the other account's finished output; "" = discover

import glob, json, os, pathlib, subprocess, sys
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
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip())
''')

md("### Install the checkpoint-compatible environment and run focused tests")
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
    "tests/test_templated_subspace_distillation_runner.py",
    "tests/test_causal_subspace_distillation.py",
], check=True)
''')

md("### Resolve inputs: reproduction summary, previous session, other account")
code(r'''
def all_candidates(suffix):
    values = []
    for root in ("/kaggle/working", "/kaggle/input"):
        values.extend(glob.glob(f"{root}/**/{suffix}", recursive=True))
    return sorted(set(values), key=lambda v: (len(pathlib.Path(v).parts), v))

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = all_candidates(suffix)
    return found[0] if found else None

def discover_output_dirs(explicit, contract, *, want_seeds):
    """Attached output directories of this experiment, split by the seeds they hold."""
    if explicit and pathlib.Path(explicit).is_dir():
        candidates = [pathlib.Path(explicit)]
    else:
        candidates = [pathlib.Path(p).parent for p in glob.glob("/kaggle/input/**/summary.json", recursive=True)]
    mine, others = [], []
    for directory in candidates:
        try:
            record = json.loads((directory / "summary.json").read_text())
        except Exception:
            continue
        if record.get("contract") != contract:
            continue
        seeds = set(record.get("preregistration", {}).get("seeds_this_run", []))
        (mine if seeds & want_seeds else others).append(str(directory))
    return mine, others

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
CONTRACT = "official_codi_templated_subspace_distillation_v1"
want = {int(s) for s in SEEDS.split(",") if s.strip()}
mine, others = discover_output_dirs(PREVIOUS_OUTPUT_INPUT, CONTRACT, want_seeds=want)
if OTHER_ACCOUNT_OUTPUT_INPUT and pathlib.Path(OTHER_ACCOUNT_OUTPUT_INPUT).is_dir():
    others = [OTHER_ACCOUNT_OUTPUT_INPUT]
RESUME_FROM = ",".join(mine)
AGGREGATE_FROM = ",".join(others)
print("reproduction", REPRODUCTION_SUMMARY)
print("resume from", RESUME_FROM or "(fresh start)")
print("aggregate with", AGGREGATE_FROM or "(this account only)")
''')

md("## Smoke pass: generator, teacher check, selectors, three tiny students")
code(r'''
def runner_command(output_dir, *extra):
    return [
        sys.executable, "-u", "scripts/run_codi_templated_subspace_distillation.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--output-dir", output_dir,
        "--batch-size", "16", "--micro-batch-size", "8", "--eval-batch-size", "32",
        "--bootstrap-samples", "10000",
        "--precision", "float32", "--device", "cuda",
        *extra,
    ]

if RUN_SMOKE:
    subprocess.run(runner_command(SMOKE_OUTPUT_DIR, "--smoke"), check=True)
    smoke = json.loads((pathlib.Path(SMOKE_OUTPUT_DIR) / "summary.json").read_text())
    assert smoke["contract"].endswith("_smoke")
    print("smoke status:", smoke["status"], "|", smoke["decision"])
''')

md("## Results — go/no-go on the teacher, then this account's seeds")
code(r'''
extra = ["--seeds", SEEDS, "--max-seconds", str(MAX_SECONDS)]
if RUN_PRELIMINARY_ONLY:
    extra.append("--preliminary-only")
if RESUME_FROM:
    extra += ["--resume-from", RESUME_FROM]
if AGGREGATE_FROM:
    extra += ["--aggregate-from", AGGREGATE_FROM]
subprocess.run(runner_command(OUTPUT_DIR, *extra), check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print("status:", summary["status"])
print(summary["decision"])
''')

md("### Go/no-go: can the teacher do the templated task, and are the selectors distinguishable?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
ARM_COLORS = {"none": "#8a8a8a", "variance": "#315f8c", "causal": "#b65f24"}

preliminary = json.loads((pathlib.Path(OUTPUT_DIR) / "preliminary.json").read_text())
print("teacher CoT generation accuracy on templated problems:",
      round(preliminary["teacher_generation_accuracy"], 3), f"(n={preliminary['teacher_check_examples']})")
print("dense colon first-token accuracy:", round(preliminary["dense_first_token_accuracy"], 3))
print("gate:", preliminary["gate"])
display(pd.DataFrame(preliminary["teacher_samples"]))
selectors = json.loads((pathlib.Path(OUTPUT_DIR) / "selectors.json").read_text())
print("top-4 eigenvalue share:", round(selectors["eigenvalue_share_top4"], 3))
display(pd.DataFrame(selectors["audit"]).T)
rows = []
for rank, report in selectors["reports_by_rank"].items():
    for name in ("variance", "relevance", "causal", "random"):
        rows.append({"rank": int(rank), "set": name,
                     "first-token accuracy": report[name]["first_token_accuracy"],
                     "retention of dense": report[name]["retention_of_dense"],
                     "variance share": report[name]["variance_share"]})
    rows.append({"rank": int(rank), "set": "dense", "first-token accuracy": report["dense"]["first_token_accuracy"]})
display(pd.DataFrame(rows))
print("chosen rank:", selectors["rank"])
if selectors["rank"] is not None:
    for name, idx in selectors["sets"][str(selectors["rank"])].items():
        print(f"  {name:9s} PCs: {idx}")
''')

md("### Learning curves: selection-split exact match and NLL per arm")
code(r'''
if summary.get("runs"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    first_seed = min(r["seed"] for r in summary["runs"].values())
    for name, run in summary["runs"].items():
        curve = pd.DataFrame(run["curve"])
        if curve.empty:
            continue
        kw = dict(color=ARM_COLORS.get(run["arm"], "#000"), alpha=0.75, linewidth=1.3,
                  label=run["arm"] if run["seed"] == first_seed else None)
        axes[0].plot(curve["step"], curve["accuracy"], **kw)
        axes[1].plot(curve["step"], curve["answer_nll"], **kw)
        if curve["distillation_scale"].notna().any():
            axes[2].plot(curve["step"], curve["distillation_scale"], **kw)
    axes[0].axhline(0.30, color="#999", linestyle="--", linewidth=0.8)
    axes[0].set(title="selection-split exact match (primary signal)", xlabel="step")
    axes[1].set(title="selection-split answer NLL (teacher-forced)", xlabel="step")
    axes[2].set(title="distillation gradient scale (norm-matched)", xlabel="step", yscale="log")
    axes[0].legend()
    plt.show()
    display(pd.DataFrame([{"run": k, "arm": v["arm"], "seed": v["seed"],
                           "steps to 30% selection EM": v["steps_to_threshold"],
                           "final selection EM": v["curve"][-1]["accuracy"] if v["curve"] else None}
                          for k, v in summary["runs"].items()]))
else:
    print("No completed runs yet in this output.")
''')

md("### Test exact match by arm and seed, comparisons, and gates")
code(r'''
test = summary.get("test")
if not test:
    print("No test aggregate yet:", summary["decision"])
else:
    arms = test["arms"]
    rows = []
    for arm, rec in arms.items():
        for seed, acc in zip(rec["seeds"], rec["test_accuracy_by_seed"]):
            rows.append({"arm": arm, "seed": seed, "test exact match": acc})
    table = pd.DataFrame(rows).pivot(index="arm", columns="seed", values="test exact match")
    table["mean"] = table.mean(axis=1)
    table["test NLL"] = [arms[a]["test_nll_mean"] for a in table.index]
    display(table.loc[[a for a in ARM_COLORS if a in table.index]])

    comps = pd.DataFrame([{"comparison": k, "mean difference": v["mean_difference"],
                           "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
                          for k, v in test["comparisons"].items()])
    display(comps)
    fig, ax = plt.subplots(figsize=(9.5, 4), constrained_layout=True)
    y = np.arange(len(comps))
    for i, (_, r) in enumerate(comps.iterrows()):
        ax.errorbar(r["mean difference"], y[i],
                    xerr=[[r["mean difference"] - r["lower"]], [r["upper"] - r["mean difference"]]],
                    fmt="o", capsize=4, color="#315f8c")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set(yticks=y, yticklabels=comps["comparison"],
           xlabel="paired exact-match difference (95% bootstrap over questions)",
           title="Arm comparisons on the 2,000 generated test problems")
    plt.show()
    display(pd.DataFrame([{"comparison": k, "mean NLL difference": v["mean_difference"],
                           "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
                          for k, v in test["nll_comparisons"].items()]))
    if test["gate"]:
        display(pd.DataFrame([test["gate"]]))
''')

md("## Takeaways")
code(r'''
print("Status:", summary["status"])
print("Decision:", summary["decision"]["claim"])
if summary["status"] == "preliminary_only":
    print("\nGO. Set RUN_PRELIMINARY_ONLY = False, publish this output (the teacher cache and selectors"
          " are reused), and Run All to train.")
elif summary["status"] == "stopped":
    print("\nSTOP. The teacher or the selectors failed the go/no-go on the templated task; nothing was trained.")
elif summary["status"] == "paused":
    print("\nNEXT: publish", OUTPUT_DIR, "as a Kaggle dataset, attach it to a new session, and Run All again.")
elif summary.get("test") is None:
    print("\nNEXT: the other account must finish its seeds; then attach both outputs and rerun to aggregate.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
for name in ("summary.json", "problems.json", "preliminary.json", "teacher_cache.pt", "selectors.json"):
    print(name, "present" if (output / name).is_file() else "MISSING")
print("runs:", sorted(p.name for p in (output / "runs").glob("*")))
if summary.get("test"):
    assert (output / "predictions.jsonl").is_file()
print("Publish this directory directly; no same-named archive is created.")
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
