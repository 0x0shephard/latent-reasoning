"""Build the Kaggle notebook for trajectory-level supervision (ledger §84)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_trajectory_supervision.ipynb"
RUN_COMMIT = "12f5eacb438d25ff83db29fe51caa3ef15ad7d3b"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does supervising CODI's value slots beat KaVa's redundancy-selected targets?

## tl;dr

CODI supervises only the answer-cue endpoint. KaVa adds key/value targets for every
latent slot at teacher positions chosen by R-KV, a redundancy score. Neither knows
what the slots are for. Ledger §55 does: the odd slots hold the solution's
intermediate values, as an unordered set.

This notebook warm-starts the frozen official checkpoint six ways, identical in
data, order, optimizer, 1,024 steps, and base objective, differing only in what the
auxiliary term supervises:

| arm | targets | positions | slots |
|---|---|---|---|
| `codi` | none | — | — |
| `kava` | K/V | R-KV (λ=0.1) | all six |
| `value_odd` | K/V | first token of each `<<…=v>>` result | 1, 3, 5 |
| `random_odd` | K/V | random trace positions, count-matched | 1, 3, 5 |
| `value_even` | K/V | value tokens | 0, 2, 4 |
| `recon_odd` | slot readout cross-entropy → value token | value tokens | 1, 3, 5 |

Every auxiliary gradient is rescaled to the endpoint term's **gradient norm** each
step, so arms differ in *what* they supervise, not how hard. Three seeds per arm.
A drift **screen** on `codi` seed 1 runs before the test set is read.

The notebook runs a three-minute **smoke pass** of the whole pipeline first, then the
frozen protocol. Expect roughly three to four hours on a T4.
""")

md(r"""
## Context & Methods

### Base objective and matched pressure

Base loss: student gold-answer NLL plus the official endpoint distillation,
smooth-L1 over all 13 states scaled by the teacher standard deviation. For each
auxiliary arm the gradient of the auxiliary loss is scaled to the gradient norm of the
endpoint term before being added, so no arm can win merely by pushing harder.

### Assignment

§55 showed the value store is unordered, so slot-to-value assignment is a
per-example minimum-cost injective matching on the detached loss. Examples with
fewer values than slots leave surplus slots unsupervised.

### Screen and gates

- **Screen:** `codi` seed 1 selection-split accuracy ≥ frozen − 3 points, else STOP
  and the test set is not read.
- **H1** `kava − codi` lower bound > 0.
- **H2** `value_odd − kava` > 0 **and** `value_odd − random_odd` > 0.
- **H3** `value_odd − value_even` > 0.
- **H4** `recon_odd − value_odd`, two-sided, descriptive.
- **Headline:** H2 and `value_odd − codi` > 0.

Paired bootstrap over questions on per-question seed-mean correctness.

### Key assumptions

- GSM8k-Aug rows need at least two equations so the truncated trace carries a value;
  the filter applies to every arm.
- The teacher path is frozen (no teacher CE) in every arm.
- Decoding follows the official native protocol, no forced cue.
- A 1,024-step warm start is a cheap held-out gate, not a retraining; a null bounds
  only this budget.
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
OUTPUT_DIR = "/kaggle/working/codi_trajectory_supervision"
SMOKE_OUTPUT_DIR = "/kaggle/working/codi_trajectory_supervision_smoke"

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
print("code commit", subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip())
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
    "tests/test_trajectory_supervision.py",
    "tests/test_trajectory_supervision_runner.py",
    "tests/test_official_codi_kv.py",
], check=True)
''')

md("### Resolve the reproduction summary")
code(r'''
def all_candidates(suffix):
    values = []
    for root in ("/kaggle/working", "/kaggle/input"):
        values.extend(glob.glob(f"{root}/**/{suffix}", recursive=True))
    return sorted(set(values), key=lambda value: (len(pathlib.Path(value).parts), value))

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    candidates = all_candidates(suffix)
    return candidates[0] if candidates else None

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
print("reproduction", REPRODUCTION_SUMMARY)
print("No artifact from any earlier experiment is required.")
''')

md("## Smoke pass: the entire pipeline on sixteen rows")
code(r'''
def runner_command(output_dir, *extra):
    return [
        sys.executable, "-u", "scripts/run_codi_trajectory_supervision.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--output-dir", output_dir,
        "--batch-size", "8", "--eval-batch-size", "32",
        "--bootstrap-samples", "10000",
        "--precision", "float32", "--device", "cuda",
        *extra,
    ]

subprocess.run(runner_command(SMOKE_OUTPUT_DIR, "--smoke"), check=True)
smoke = json.loads((pathlib.Path(SMOKE_OUTPUT_DIR) / "summary.json").read_text())
assert smoke["contract"].endswith("_smoke")
print("smoke decision:", smoke["decision"])
print("smoke arms:", list(smoke["runs"]))
''')

md("## Results — the frozen protocol: screen, eighteen training runs, one test read")
code(r'''
subprocess.run(runner_command(OUTPUT_DIR), check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
print("screen:", summary["screen"])
''')

md("### Did the warm start keep the model intact?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
ARM_COLORS = {"codi": "#222222", "kava": "#315f8c", "value_odd": "#b65f24",
              "random_odd": "#8a8a8a", "value_even": "#c9a227", "recon_odd": "#5c8a3a"}

display(pd.DataFrame([summary["screen"]]))
print("frozen selection accuracy", summary["frozen"]["selection"]["accuracy"])
if summary["test"] is None:
    print("The screen failed; the test set was not read. Nothing below will run.")
''')

md("### Training curves: answer loss, endpoint loss, and how hard each auxiliary pushed")
code(r'''
if summary["test"] is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for name, run in summary["runs"].items():
        arm = run["arm"]
        curve = pd.DataFrame(run["training"]["curve"])
        if curve.empty:
            continue
        kw = dict(color=ARM_COLORS.get(arm, "#000"), alpha=0.7, linewidth=1.2)
        axes[0].plot(curve["step"], curve["answer_loss"], label=arm if run["seed"] == 1 else None, **kw)
        axes[1].plot(curve["step"], curve["endpoint_loss"], **kw)
        if curve["auxiliary_scale"].notna().any():
            axes[2].plot(curve["step"], curve["auxiliary_scale"], **kw)
    axes[0].set(title="student answer NLL", xlabel="step")
    axes[1].set(title="endpoint distillation loss", xlabel="step")
    axes[2].set(title="auxiliary gradient scale (norm-matched)", xlabel="step", yscale="log")
    axes[0].legend()
    plt.show()
''')

md("### Do KaVa's picks already land on the value tokens?")
code(r'''
if summary["test"] is not None:
    arms = summary["test"]["arms"]
    diag = pd.DataFrame({
        "arm": list(arms),
        "supervised slot fraction": [arms[a]["mean_supervised_slot_fraction"] for a in arms],
        "R-KV picks on value tokens": [arms[a]["mean_rkv_value_overlap"] for a in arms],
        "mean auxiliary scale": [arms[a]["mean_auxiliary_scale"] for a in arms],
    })
    display(diag)
    print("values per training row:", summary["sampling"]["values_per_row"])
''')

md("### Test accuracy by arm and seed")
code(r'''
if summary["test"] is not None:
    arms = summary["test"]["arms"]
    rows = []
    for arm, record in arms.items():
        for seed, acc in zip(summary["preregistration"]["training_seeds"], record["test_accuracy_by_seed"]):
            rows.append({"arm": arm, "seed": seed, "test accuracy": acc,
                         "selection accuracy": record["selection_accuracy_by_seed"][seed - 1]})
    table = pd.DataFrame(rows)
    display(table.pivot(index="arm", columns="seed", values="test accuracy").assign(
        mean=lambda d: d.mean(axis=1)).loc[list(arms)])
    print("frozen checkpoint test accuracy", summary["test"]["frozen_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    order = list(arms)
    for i, arm in enumerate(order):
        accs = arms[arm]["test_accuracy_by_seed"]
        ax.scatter([i] * len(accs), accs, color=ARM_COLORS[arm], s=40, zorder=3)
        ax.hlines(np.mean(accs), i - 0.3, i + 0.3, color=ARM_COLORS[arm], linewidth=2)
    ax.axhline(summary["test"]["frozen_accuracy"], color="#222222", linestyle="--",
               linewidth=1, label="frozen checkpoint")
    ax.set(xticks=range(len(order)), xticklabels=order, ylabel="GSM8K exact match",
           title="Test accuracy per seed (bars = seed mean)")
    ax.legend()
    plt.show()
''')

md("### Paired comparisons and the preregistered gates")
code(r'''
if summary["test"] is not None:
    comps = summary["test"]["comparisons"]
    comp_table = pd.DataFrame([
        {"comparison": k, "mean difference": v["mean_difference"],
         "lower": v["bootstrap_95ci"][0], "upper": v["bootstrap_95ci"][1]}
        for k, v in comps.items()
    ])
    display(comp_table)

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    ordered = comp_table.sort_values("mean difference")
    y = np.arange(len(ordered))
    for i, (_, r) in enumerate(ordered.iterrows()):
        ax.errorbar(r["mean difference"], y[i],
                    xerr=[[r["mean difference"] - r["lower"]], [r["upper"] - r["mean difference"]]],
                    fmt="o", capsize=4, color="#315f8c")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set(yticks=y, yticklabels=ordered["comparison"],
           xlabel="paired accuracy difference (95% bootstrap over questions)",
           title="Arm comparisons on the locked test read")
    plt.show()

    display(pd.DataFrame([summary["test"]["gate"]]))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print("Screen passed:", summary["decision"]["screen_passed"])
print("Headline gate:", summary["decision"].get("final_passed"))
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
assert (output / "summary.json").is_file()
if summary["test"] is not None:
    assert (output / "trajectory_supervision.pt").is_file()
    assert (output / "predictions.jsonl").is_file()
print("Kaggle output directory", output)
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
