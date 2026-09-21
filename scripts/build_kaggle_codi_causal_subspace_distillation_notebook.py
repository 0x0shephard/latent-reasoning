"""Build the Kaggle notebook for causal-subspace distillation (ledger §86)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_causal_subspace_distillation.ipynb"
RUN_COMMIT = "30a7731a4f9e1c72c2da1788c7e1aa8768afa812"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does a causally selected distillation subspace beat variance selection?

## tl;dr

Subspace-distillation methods pick the teacher subspace by variance or gradient
relevance. Ledger §40 showed the decision state's top-variance directions are inert.
This notebook distils a fresh latent student toward rank-r subspaces of the frozen
CODI teacher's decision state, selected four ways, and asks which transfers accuracy.

| arm | selector over the teacher's PCs |
|---|---|
| `variance` | first r PCs (LoRi-style) |
| `relevance` | top-r by Fisher score of the gold-token gradient (Flex-KD-style) |
| `causal` | greedy **retain-only** intervention on the teacher (§36 analytic tier) |
| `random` | r seeded PCs |
| `full` | all 768 dims |
| `none` | answer cross-entropy only |

**Student:** the checkpoint's embeddings and readout, fresh LoRA and projector. The
distillation gradient is **norm-matched** to the answer cross-entropy every step.
**Rank** is chosen on the teacher only: among r ∈ {8, 12, 16} where the causal set
beats the variance set by ≥5 points and retains ≥50% of dense first-token accuracy,
the rank maximising gap × retention. If none qualifies, nothing is trained.

### Running across two accounts and several sessions

- Set `SEEDS = "1"` on one account and `SEEDS = "2"` on the other.
- Each session stops cleanly after `MAX_SECONDS` with a resumable checkpoint.
  **Publish the output directory as a dataset**, attach it to the next session, and
  the run continues where it stopped (set `PREVIOUS_OUTPUT_INPUT` or leave discovery on).
- When both accounts have finished, attach the other account's output as
  `OTHER_ACCOUNT_OUTPUT_INPUT` and the final cell aggregates both seeds.

Budget: roughly 12–13 GPU hours per account for six arms at 10,000 steps.
""")

md(r"""
## Context & Methods

For teacher decision states \(h\) with mean \(\mu\) and PC basis \(V\), the retain-only
edit keeps only the selected coordinates:

\[
h' = \mu + V_S V_S^\top (h-\mu),
\]

and the teacher's own readout scores \(h'\). The **causal** set is built greedily by
adding the PC that most raises the gold first-token log-probability of \(h'\) on a
disjoint split; it is an intervention on the teacher, not a statistic of it.

The student loss is answer cross-entropy plus
\(\mathrm{smoothL1}\big(V_S^\top(h_s-\mu),\,V_S^\top(h_t-\mu)\big)/\sigma_S\), with the
distillation gradient rescaled to the cross-entropy gradient norm.

### Gates (paired bootstrap over questions, seed-mean correctness)

- **S1** `full − none` > 0: distillation is detectable at this budget.
- **H1** `causal − variance` > 0. **H2** `causal − relevance` > 0. **H3** `causal − random` > 0.
- **Headline** = S1 ∧ H1 ∧ H3. Teacher-forced NLL is the secondary outcome.

### Key assumptions

- Students learn the latent task from scratch (0.4 epoch); absolute accuracy may be
  low, and the selection-split NLL curve is the sensitive secondary signal.
- Only the LoRA adapters and projector are reset; embeddings and readout are the
  checkpoint's, so the causal subspace is defined through the student's own readout.
- The test set is read once per trained model.
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
OUTPUT_DIR = "/kaggle/working/codi_causal_subspace_distillation"
SMOKE_OUTPUT_DIR = "/kaggle/working/codi_causal_subspace_distillation_smoke"

# ---- per-account and per-session controls ----
SEEDS = "1"                      # "1" on this account, "2" on the other
MAX_SECONDS = 30600              # stop cleanly after 8.5 h with a resumable checkpoint
RUN_SMOKE = True                 # three-minute pipeline check before the real run
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
    "tests/test_causal_subspace_distillation.py",
    "tests/test_causal_subspace_distillation_runner.py",
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
CONTRACT = "official_codi_causal_subspace_distillation_v1"
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

md("## Smoke pass: teacher cache, selectors, and three tiny students")
code(r'''
def runner_command(output_dir, *extra):
    return [
        sys.executable, "-u", "scripts/run_codi_causal_subspace_distillation.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--output-dir", output_dir,
        "--batch-size", "16", "--micro-batch-size", "8", "--eval-batch-size", "32",
        "--bootstrap-samples", "10000",
        "--precision", "float32", "--device", "cuda",
        *extra,
    ]

if RUN_SMOKE:
    subprocess.run(runner_command(SMOKE_OUTPUT_DIR, "--smoke", "--seeds", "1"), check=True)
    smoke = json.loads((pathlib.Path(SMOKE_OUTPUT_DIR) / "summary.json").read_text())
    assert smoke["contract"].endswith("_smoke")
    print("smoke status:", smoke["status"], "|", smoke["decision"])
''')

md("## Results — teacher-side selection, then this account's seeds")
code(r'''
extra = ["--seeds", SEEDS, "--max-seconds", str(MAX_SECONDS)]
if RESUME_FROM:
    extra += ["--resume-from", RESUME_FROM]
if AGGREGATE_FROM:
    extra += ["--aggregate-from", AGGREGATE_FROM]
subprocess.run(runner_command(OUTPUT_DIR, *extra), check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print("status:", summary["status"])
print(summary["decision"])
''')

md("### Were the selectors distinguishable on the teacher?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
ARM_COLORS = {"none": "#8a8a8a", "full": "#222222", "variance": "#315f8c",
              "relevance": "#c9a227", "causal": "#b65f24", "random": "#5c8a3a"}

selectors = json.loads((pathlib.Path(OUTPUT_DIR) / "selectors.json").read_text())
print("top-4 eigenvalue share:", round(selectors["eigenvalue_share_top4"], 3))
display(pd.DataFrame(selectors["audit"]).T)
rows = []
for rank, report in selectors["reports_by_rank"].items():
    for name in ("variance", "relevance", "causal", "random"):
        rows.append({"rank": int(rank), "set": name,
                     "first-token accuracy": report[name]["first_token_accuracy"],
                     "retention of dense": report[name]["retention_of_dense"],
                     "variance share": report[name]["variance_share"],
                     "indices": report[name]["indices"]})
    rows.append({"rank": int(rank), "set": "dense", "first-token accuracy": report["dense"]["first_token_accuracy"]})
teacher_table = pd.DataFrame(rows)
display(teacher_table.drop(columns=["indices"]))
chosen = selectors["rank"]
print("chosen rank:", chosen)
if chosen is not None:
    sets = selectors["sets"][str(chosen)]
    for name, idx in sets.items():
        print(f"  {name:9s} PCs: {idx}")
    print("  jaccard:", selectors["reports_by_rank"][str(chosen)]["jaccard"])
''')

md("### Learning curves: selection-split NLL and exact match per arm")
code(r'''
if summary.get("runs"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for name, run in summary["runs"].items():
        curve = pd.DataFrame(run["curve"])
        if curve.empty:
            continue
        kw = dict(color=ARM_COLORS.get(run["arm"], "#000"), alpha=0.75, linewidth=1.3,
                  label=run["arm"] if run["seed"] == min(r["seed"] for r in summary["runs"].values()) else None)
        axes[0].plot(curve["step"], curve["answer_nll"], **kw)
        axes[1].plot(curve["step"], curve["accuracy"], **kw)
        if curve["distillation_scale"].notna().any():
            axes[2].plot(curve["step"], curve["distillation_scale"], **kw)
    axes[0].set(title="selection-split answer NLL (teacher-forced)", xlabel="step")
    axes[1].set(title="selection-split exact match", xlabel="step")
    axes[2].set(title="distillation gradient scale (norm-matched)", xlabel="step", yscale="log")
    axes[0].legend()
    plt.show()
else:
    print("No completed runs yet in this output.")
''')

md("### Test accuracy by arm and seed, comparisons, and gates")
code(r'''
test = summary.get("test")
if not test:
    print("No test aggregate yet:", summary["decision"])
else:
    arms = test["arms"]
    rows = []
    for arm, rec in arms.items():
        for seed, acc in zip(rec["seeds"], rec["test_accuracy_by_seed"]):
            rows.append({"arm": arm, "seed": seed, "test accuracy": acc})
    table = pd.DataFrame(rows).pivot(index="arm", columns="seed", values="test accuracy")
    table["mean"] = table.mean(axis=1)
    table["test NLL"] = [arms[a]["test_nll_mean"] for a in table.index]
    display(table.loc[[a for a in ARM_COLORS if a in table.index]])

    comps = pd.DataFrame([{"comparison": k, "mean difference": v["mean_difference"],
                           "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
                          for k, v in test["comparisons"].items()])
    display(comps)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ordered = comps.sort_values("mean difference"); y = np.arange(len(ordered))
    for i, (_, r) in enumerate(ordered.iterrows()):
        ax.errorbar(r["mean difference"], y[i],
                    xerr=[[r["mean difference"] - r["lower"]], [r["upper"] - r["mean difference"]]],
                    fmt="o", capsize=4, color="#315f8c")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set(yticks=y, yticklabels=ordered["comparison"],
           xlabel="paired exact-match difference (95% bootstrap over questions)",
           title="Arm comparisons on the test read")
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
if summary["status"] == "paused":
    print("\nNEXT: publish", OUTPUT_DIR, "as a Kaggle dataset, attach it to a new session, and Run All again.")
elif summary["status"] == "stopped":
    print("\nThe teacher-side rank rule stopped the run before training; see the selector table above.")
elif summary.get("test") is None:
    print("\nNEXT: the other account must finish its seed; then attach both outputs and rerun to aggregate.")
print("\nTIP: publishing this output lets the other account reuse teacher_cache.pt and selectors.json"
      " (about 35 minutes of GPU time) by attaching it; discovery picks it up automatically.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
for name in ("summary.json", "sampling.json", "teacher_cache.pt", "selectors.json"):
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
