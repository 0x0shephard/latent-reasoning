"""Build the Kaggle notebook for the recovery test of distillation targets (ledger §94)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_recovery_subspace_distillation.ipynb"
RUN_COMMIT = "9e78741751cd7c62f04b9c30bd62fcec584bc45c"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Which distillation target recovers a damaged latent reasoner? (ledger §94)

## tl;dr

Three earlier runs (§86–§90) could not decide whether an **intervention-selected**
subspace of the teacher's decision state transfers better than a **variance-selected**
one, because from-scratch students never reached measurable accuracy. This experiment
puts the student in the measurable regime from step 0: the official CODI checkpoint's
**projector** (the module that turns each latent thought into the next input) is
damaged with calibrated Gaussian noise (σ chosen in the go/no-go so that ≥ 20 points of
selection-split accuracy are lost; a full reset proved bimodal, ledger §96), and the
student fine-tunes on GSM8k-Aug under one of six targets. **Primary outcome: GSM8K test exact match on all 1,319
questions at the final step**, five seeds, paired by seed.

| arm | target for the student's latent decision state |
|---|---|
| `none` | answer cross-entropy only |
| `full` | all 768 dims of the teacher's explicit-CoT decision state |
| `variance` | first r teacher PCs (LoRi-style) |
| `relevance` | top-r PCs by Fisher score of the gold-token gradient (Flex-KD-style) |
| `causal` | greedy **retain-only** intervention on the teacher (§36 tier) |
| `random` | r seeded PCs |

Distillation gradients are **norm-matched** to the cross-entropy gradient. Stage 2 rescales
that match (×0.3, ×3) for `variance` and `causal`; stage 3 repeats four arms under a second
damage mode (every LoRA B matrix × 0.5).

### Stages and accounts (≈30 min per run; 60 GPU-hours total, ≈27 needed)

| `STAGE` | damage | arms | `SEEDS` | account |
|---|---|---|---|---|
| `"primary"` | projector_noise | six | `"1,2,3"` on A, `"4,5"` on B | both |
| `"weights"` | projector_noise | variance_x0.3, variance_x3, causal_x0.3, causal_x3 | `"1,2,3"` | A |
| `"damage"` | lora_half | none, variance, causal, random | `"1,2,3"` | B |

**Order of operations**

1. Account A: `STAGE = "primary"`, `RUN_PRELIMINARY_ONLY = True`, Run All (≈2 h). This runs
   the go/no-go (including the σ sweep) and **two** recovery pilots and fixes the step budget.
   **Publish the output.**
2. Both accounts: attach A's published output (shared artefacts are reused), set
   `RUN_PRELIMINARY_ONLY = False`, `STAGE = "primary"`, their `SEEDS`, Run All. Each session
   pauses cleanly after `MAX_SECONDS`; publish, attach, rerun to resume.
3. A: `STAGE = "weights"`. B: `STAGE = "damage"`. Attach A's primary output for shared artefacts.
4. Final aggregate: attach every output and rerun the Results cell on either account.
""")

md(r"""
## Context & Methods

Teacher decision states \(h\) (last normalised state at the answer colon, explicit-CoT
path) with mean \(\mu\) and PC basis \(V\). The retain-only edit \(h' = \mu + V_S V_S^\top (h-\mu)\)
is scored by the teacher's own readout; the **causal** set adds greedily the PC that most
raises retain-only first-token accuracy. Rank r is chosen on the teacher (§86 rule).

Student loss: answer cross-entropy plus
\(\mathrm{smoothL1}\big(V_S^\top(h_s-\mu),\,V_S^\top(h_t-\mu)\big)/\sigma_S\), where \(h_s\) is the
student's **latent-path** decision state; the distillation gradient is rescaled to the CE
gradient norm each step. Each step also records the cosine between the two gradients.

### Go/no-go (before any counted seed)

1. Headroom: the damage removes ≥ 20 points of selection-split exact match. For `projector_noise`
   σ is the smallest of {0.25, 0.5, 1, 2} that does so (sweep recorded).
2. Causal and variance sets share ≤ 8 of 12 PCs (the templated teacher failed this, §90).
3. Variance/causal distillation loss at the damaged weights ≥ 1.5× its value at the official weights (§97).
4. Two `none` pilots (seeds 0 and 100, 2,000 steps): the recovery threshold is `damaged + ½ × gap`
   on the selection split; budget = 1,000 if **both** cross by step 1,000, 2,000 if both cross by
   2,000, else STOP. A disagreeing pair is a STOP (that is what killed the full-reset mode, §96).

### Gates (paired bootstrap over 1,319 test questions, seed-mean correctness)

R0 `none` reaches the recovery threshold. S1 `full − none`. **H1 `causal − variance`**, H2 `causal − relevance`,
**H3 `causal − random`**. `variance − none`, `causal − none` two-sided.
CONFIRMED = H1 ∧ H3; REVERSED; NULL if H1 covers 0 within 3 points; PARTIAL; INCONCLUSIVE.

### Boundaries

This is the continued-training (recovery) regime, not learning from scratch (§93). The
official checkpoint's weights are restored before every damage; nothing is written back.
""")

md("## Data and setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "9e78741751cd7c62f04b9c30bd62fcec584bc45c"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = (
    "/kaggle/input/datasets/jonraza15/"
    "corrected-official-codi-answer-cue-endpoint-tsv-c/"
    "latent-reasoning/outputs/official_codi_gpt2/eval/"
    "revision_fd641b3d/full_gsm8k/summary.json"
)

# ---- per-account and per-session controls ----
STAGE = "primary"                # "primary" | "weights" | "damage"
SEEDS = "1,2,3"                  # primary: "1,2,3" on account A, "4,5" on B; weights/damage: "1,2,3"
RUN_PRELIMINARY_ONLY = True      # go/no-go + pilot only (about 1 h); set False to train
MAX_SECONDS = 30600              # stop cleanly after 8.5 h with a resumable checkpoint
RUN_SMOKE = True                 # three-minute pipeline check before the real run
PREVIOUS_OUTPUT_INPUT = ""       # attached copy of THIS stage's earlier output on this account; "" = discover
OTHER_OUTPUT_INPUTS = ""         # comma-separated attached outputs of this experiment (any stage/account); "" = discover

STAGES = {
    "primary": ("projector_noise", "none,full,variance,relevance,causal,random"),
    "weights": ("projector_noise", "variance_x0.3,variance_x3,causal_x0.3,causal_x3"),
    "damage":  ("lora_half", "none,variance,causal,random"),
}
DAMAGE, ARMS = STAGES[STAGE]
OUTPUT_DIR = f"/kaggle/working/codi_recovery_{STAGE}"
SMOKE_OUTPUT_DIR = "/kaggle/working/codi_recovery_smoke"

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
print("stage", STAGE, "| damage", DAMAGE, "| arms", ARMS, "| seeds", SEEDS)
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
    "tests/test_recovery_subspace_distillation_runner.py",
], check=True)
''')

md("### Resolve inputs: reproduction summary, this stage's previous session, other outputs")
code(r'''
def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

CONTRACT = "official_codi_recovery_subspace_distillation_v1"
want = {int(s) for s in SEEDS.split(",") if s.strip()}

def discover_outputs():
    """Attached outputs of this experiment: (same stage+seeds on this account) vs (everything else)."""
    candidates = [pathlib.Path(p).parent for p in glob.glob("/kaggle/input/**/summary.json", recursive=True)]
    mine, others = [], []
    for directory in candidates:
        try:
            record = json.loads((directory / "summary.json").read_text())
        except Exception:
            continue
        if record.get("contract") != CONTRACT:
            continue
        pre = record.get("preregistration", {})
        same_stage = record.get("damage") == DAMAGE and set(pre.get("arms", [])) == set(ARMS.split(","))
        if same_stage and set(pre.get("seeds_this_run", [])) & want:
            mine.append(str(directory))
        else:
            others.append(str(directory))
    return mine, others

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
mine, others = discover_outputs()
if PREVIOUS_OUTPUT_INPUT and pathlib.Path(PREVIOUS_OUTPUT_INPUT).is_dir():
    mine = [PREVIOUS_OUTPUT_INPUT]
if OTHER_OUTPUT_INPUTS:
    others = [p for p in OTHER_OUTPUT_INPUTS.split(",") if pathlib.Path(p).is_dir()]
RESUME_FROM = ",".join(mine)
AGGREGATE_FROM = ",".join(others)
print("reproduction", REPRODUCTION_SUMMARY)
print("resume from", RESUME_FROM or "(fresh start)")
print("shared artefacts / pooled runs from", AGGREGATE_FROM or "(none attached)")
''')

md("## Smoke pass: splits, teacher cache, selectors, damage, a tiny pilot and tiny students")
code(r'''
def runner_command(output_dir, *extra):
    return [
        sys.executable, "-u", "scripts/run_codi_recovery_subspace_distillation.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--output-dir", output_dir,
        "--damage", DAMAGE, "--arms", ARMS,
        "--batch-size", "16", "--micro-batch-size", "8", "--eval-batch-size", "32",
        "--bootstrap-samples", "10000",
        "--precision", "float32", "--device", "cuda",
        *extra,
    ]

import time

def run_with_retry(command, attempts=3, wait=60):
    """Hugging Face downloads occasionally drop mid-file; the hub resumes partial files, so retry."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s (attempt {attempt + 1}/{attempts})")
        time.sleep(wait)

if RUN_SMOKE:
    run_with_retry(runner_command(SMOKE_OUTPUT_DIR, "--smoke", "--seeds", "1"))
    smoke = json.loads((pathlib.Path(SMOKE_OUTPUT_DIR) / "summary.json").read_text())
    assert smoke["contract"].endswith("_smoke")
    print("smoke status:", smoke["status"], "|", smoke["decision"])
''')

md("## Results — go/no-go and pilot, then this account's seeds for this stage")
code(r'''
extra = ["--seeds", SEEDS, "--max-seconds", str(MAX_SECONDS)]
if RUN_PRELIMINARY_ONLY:
    extra.append("--preliminary-only")
if RESUME_FROM:
    extra += ["--resume-from", RESUME_FROM]
if AGGREGATE_FROM:
    extra += ["--aggregate-from", AGGREGATE_FROM]
run_with_retry(runner_command(OUTPUT_DIR, *extra))
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print("status:", summary["status"])
print(summary["decision"])
''')

md("### Go/no-go: headroom, selector divergence, term ratio, and the recovery pilot")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
ARM_COLORS = {"none": "#8a8a8a", "full": "#222222", "variance": "#315f8c", "relevance": "#c9a227",
              "causal": "#b65f24", "random": "#5c8a3a",
              "variance_x0.3": "#7fa6cc", "variance_x3": "#1d3a57", "causal_x0.3": "#e0a37a", "causal_x3": "#6e3812"}

pre = summary["preliminary"]
gate0 = pre[f"gate_{DAMAGE}"]
print("official selection EM:", round(gate0["official_selection"]["accuracy"], 3),
      "| damaged selection EM:", round(gate0["damaged_selection"]["accuracy"], 3))
print("rank:", gate0["rank"], "| variance/causal shared PCs:", gate0["shared_variance_causal_pcs"])
print("term loss official -> damaged:", {k: (round(gate0["official_term_loss"][k], 3), round(gate0["damaged_term_loss"][k], 3))
                                         for k in gate0["damaged_term_loss"]})
print("gap:", round(gate0["gap"], 3), "| recovery threshold:", round(gate0["recovery_threshold"], 3))
if gate0.get("sigma_sweep"):
    print("sigma:", gate0["sigma"])
    display(pd.DataFrame(gate0["sigma_sweep"]))
print("checks:", gate0["checks"])
sel = summary["selectors"]
if sel["sets"]:
    for name, idx in sel["sets"].items():
        print(f"  {name:9s} PCs: {idx}")
    display(pd.DataFrame(sel["audit"]).T)
pilot = pre.get(f"pilot_{DAMAGE}")
if pilot:
    pilots = pilot.get("pilots") or [{"seed": 0, **{k: pilot[k] for k in ("curve", "test", "steps_to_threshold")}}]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for p_, colour in zip(pilots, ("#8a8a8a", "#b65f24", "#315f8c")):
        curve = pd.DataFrame(p_["curve"])
        ax.plot(curve["step"], curve["accuracy"], color=colour, marker="o", ms=3, label=f"none, seed {p_['seed']}")
    ax.axhline(gate0["recovery_threshold"], color="#999", linestyle="--", linewidth=0.8)
    ax.axhline(gate0["official_selection"]["accuracy"], color="#222", linestyle=":", linewidth=0.8)
    ax.set(title=f"recovery pilots: selection-split exact match | budget = {pilot['chosen_steps']}",
           xlabel="step", ylim=(0, 1))
    ax.legend(fontsize=8)
    plt.show()
    for p_ in pilots:
        print(f"pilot seed {p_['seed']}: test {p_['test']} | steps to recovery threshold: {p_['steps_to_threshold']}")
''')

md("### Learning curves: selection-split exact match, NLL, and CE-vs-distillation gradient cosine")
code(r'''
if summary.get("runs"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    seen = set()
    for name, run in summary["runs"].items():
        curve = pd.DataFrame(run["curve"])
        if curve.empty:
            continue
        label = run["arm"] if run["arm"] not in seen else None; seen.add(run["arm"])
        kw = dict(color=ARM_COLORS.get(run["arm"], "#000"), alpha=0.7, linewidth=1.2, label=label)
        axes[0].plot(curve["step"], curve["accuracy"], **kw)
        axes[1].plot(curve["step"], curve["answer_nll"], **kw)
        if curve["gradient_cosine"].notna().any():
            axes[2].plot(curve["step"], curve["gradient_cosine"], **kw)
    axes[0].axhline(gate0["recovery_threshold"], color="#999", linestyle="--", linewidth=0.8)
    axes[0].axhline(gate0["official_selection"]["accuracy"], color="#222", linestyle=":", linewidth=0.8)
    axes[0].set(title="selection-split exact match (dashed: recovery threshold, dotted: official)", xlabel="step")
    axes[1].set(title="selection-split answer NLL (teacher-forced)", xlabel="step")
    axes[2].axhline(0, color="#222", linewidth=0.8)
    axes[2].set(title="cosine(CE gradient, distillation gradient)", xlabel="step")
    axes[0].legend(fontsize=8)
    plt.show()
    display(pd.DataFrame([{"run": k, "arm": v["arm"], "seed": v["seed"], "steps to recovery threshold": v["steps_to_threshold"],
                           "mean gradient cosine": v["mean_gradient_cosine"],
                           "final selection EM": v["curve"][-1]["accuracy"] if v["curve"] else None,
                           "test EM": v["test"]["accuracy"]} for k, v in summary["runs"].items()]))
else:
    print("No completed runs yet in this output.")
''')

md("### GSM8K test exact match by arm and seed, comparisons, and gates")
code(r'''
test = summary.get("test")
if not test:
    print("No test aggregate yet:", summary["decision"])
else:
    arms = test["arms"]
    rows = []
    for arm, rec in arms.items():
        for seed, acc in zip(rec["seeds"], rec["test_accuracy_by_seed"]):
            rows.append({"arm": arm, "seed": seed, "test EM": acc})
    table = pd.DataFrame(rows).pivot(index="arm", columns="seed", values="test EM")
    table["mean"] = table.mean(axis=1)
    table["test NLL"] = [arms[a]["test_nll_mean"] for a in table.index]
    table["mean grad cosine"] = [np.mean([c for c in arms[a]["mean_gradient_cosine"] if c is not None])
                                 if any(c is not None for c in arms[a]["mean_gradient_cosine"]) else np.nan
                                 for a in table.index]
    display(table.loc[[a for a in ARM_COLORS if a in table.index]])

    comps = pd.DataFrame([{"comparison": k, "mean difference": v["mean_difference"],
                           "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
                          for k, v in test["comparisons"].items()])
    display(comps)
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(comps) + 1.5), constrained_layout=True)
    y = np.arange(len(comps))
    for i, (_, r) in enumerate(comps.iterrows()):
        ax.errorbar(r["mean difference"], y[i],
                    xerr=[[r["mean difference"] - r["lower"]], [r["upper"] - r["mean difference"]]],
                    fmt="o", capsize=4, color="#315f8c")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set(yticks=y, yticklabels=comps["comparison"],
           xlabel="paired GSM8K exact-match difference (95% bootstrap over 1,319 questions)",
           title=f"Arm comparisons, damage = {DAMAGE}")
    plt.show()
    display(pd.DataFrame([{"comparison": k, "mean NLL difference": v["mean_difference"],
                           "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
                          for k, v in test["nll_comparisons"].items()]))
    display(pd.DataFrame([test["gate"]]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Stage:", STAGE, "| damage:", DAMAGE)
print("Status:", summary["status"])
print("Decision:", summary["decision"]["claim"])
if summary["status"] == "preliminary_only":
    print("\nGO. Publish this output; on both accounts set RUN_PRELIMINARY_ONLY = False, attach it, and Run All.")
elif summary["status"] == "stopped":
    print("\nSTOP. See the go/no-go cell; nothing counted was trained.")
elif summary["status"] == "paused":
    print("\nNEXT: publish", OUTPUT_DIR, "as a Kaggle dataset, attach it to a new session, and Run All again.")
elif summary.get("test") and summary["decision"]["claim"].startswith("partial"):
    print("\nNEXT: attach the other account's output for this stage and rerun to aggregate all seeds.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
for name in ("summary.json", "sampling.json", "teacher_cache.pt", "selectors.json", "preliminary.json"):
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
