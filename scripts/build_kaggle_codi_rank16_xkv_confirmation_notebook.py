"""Build the Kaggle notebook for rank-16 xKV mechanism confirmation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_rank16_xkv_mechanism_confirmation.ipynb"
RUN_COMMIT = "2e9b78b01b82e618405a825535173e134612fd3d"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does the rank-16 task-aware xKV result replicate, and what causes it?

## tl;dr

This is a preregistered confirmation notebook, not another rank search.

- GSM8K rows `0:256` are the completed discovery set and are never reused.
- Rows `256:768` form a new 512-question confirmation set.
- Rows `768:1319` remain locked until the complete primary gate passes.
- Rank 16, the layer groups, Fisher weights, group utilities and layer-11 basis
  are imported unchanged from the completed predecessor artifact.
- If confirmation succeeds, factorial ablations and 100 energy-matched random
  protections isolate allocation, Fisher weighting and layer-11 protection.
- Transfer generation on SVAMP, MultiArith and GSM-Hard runs only after the
  locked final holdout passes.

This reference implementation reconstructs dense KV tensors. It measures quality
and modeled storage, not production latency.
""")

md(r"""
## Context & Methods

For question (i), the primary effect is

\[
d_i=\operatorname{NLL}_{\text{ordinary xKV},i}
-\operatorname{NLL}_{\text{full method},i}.
\]

Positive values favor the full method. The primary confirmation gate requires:

1. the paired-bootstrap 95% lower bound for mean `d` to exceed zero;
2. full/ordinary modeled cache bits to match within 0.1%;
3. at least 95% first-token agreement with dense CODI; and
4. a paired generation-accuracy lower bound above the preregistered −2 point
   non-inferiority margin.

Only after that gate passes do we test three mechanisms with one-sided paired
sign-flip tests and Holm correction: allocation/Fisher versus ordinary xKV,
Fisher's increment over allocation, and layer-11 protection's increment. A
causal-protection claim additionally requires beating at least 95 of 100
energy-matched random protected bases with a positive paired interval.

### Key assumptions

- The supplied predecessor artifact is exactly the completed experiment that
  discovered the rank-16 result.
- The direct-cache artifact supplies the same frozen layer-11 basis and covariance.
- Public GSM8K rationales reconstruct answer boundaries only; student inputs still
  contain the question alone.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
SOURCE_ARTIFACT_INPUT = ""
PREVIOUS_EXPERIMENT_ARTIFACT_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_rank16_xkv_confirmation"

CONFIRMATION_START = 256
CONFIRMATION_EXAMPLES = 512
FINAL_START = 768
RANK = 16
RANDOM_CONTROLS = 100
RUN_TRANSFER_IF_FINAL_PASSES = True
TRANSFER_EXAMPLES = 256

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
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2",
    "accelerate==1.7.0", "huggingface-hub==0.32.4",
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
    "tests/test_rank16_xkv_confirmation.py",
    "tests/test_rank16_xkv_runner.py",
    "tests/test_task_aware_xkv.py",
    "tests/test_official_codi_kv.py",
    "tests/test_preanswer_kv_subspace.py",
], check=True)
''')

md("### Resolve the three frozen inputs")
code(r'''
import torch

def discover_file(explicit, suffix):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    candidates = []
    for root in ("/kaggle/working", "/kaggle/input"):
        candidates.extend(glob.glob(f"{root}/**/{suffix}", recursive=True))
    return sorted(set(candidates), key=lambda value: (len(pathlib.Path(value).parts), value))[0] if candidates else None

def discover_torch_contract(explicit, suffix, contract):
    candidates = [discover_file(explicit, suffix)] if explicit else []
    if not explicit:
        for root in ("/kaggle/working", "/kaggle/input"):
            candidates.extend(glob.glob(f"{root}/**/{suffix}", recursive=True))
    for candidate in filter(None, candidates):
        try:
            artifact = torch.load(candidate, map_location="cpu", weights_only=False)
            if artifact.get("contract") == contract:
                return candidate
        except Exception:
            continue
    return None

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
SOURCE_ARTIFACT = discover_torch_contract(
    SOURCE_ARTIFACT_INPUT,
    "direct_cache_task_subspaces.pt",
    "official_codi_direct_cache_task_sensitive_independent_kv_v1",
)
PREVIOUS_EXPERIMENT_ARTIFACT = discover_torch_contract(
    PREVIOUS_EXPERIMENT_ARTIFACT_INPUT,
    "task_aware_protected_xkv.pt",
    "official_codi_task_aware_protected_residual_xkv_v1",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
assert SOURCE_ARTIFACT, "Attach direct_cache_task_subspaces.pt from the source experiment"
assert PREVIOUS_EXPERIMENT_ARTIFACT, "Attach task_aware_protected_xkv.pt from the completed predecessor"
print("reproduction", REPRODUCTION_SUMMARY)
print("source artifact", SOURCE_ARTIFACT)
print("previous experiment", PREVIOUS_EXPERIMENT_ARTIFACT)
''')

md("## Results — confirmation, mechanism isolation and locked replication")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_rank16_xkv_mechanism_confirmation.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--source-artifact", SOURCE_ARTIFACT,
    "--previous-experiment-artifact", PREVIOUS_EXPERIMENT_ARTIFACT,
    "--output-dir", OUTPUT_DIR,
    "--rank", str(RANK),
    "--confirmation-start", str(CONFIRMATION_START),
    "--confirmation-examples", str(CONFIRMATION_EXAMPLES),
    "--final-start", str(FINAL_START),
    "--random-controls", str(RANDOM_CONTROLS),
    "--random-candidates", "512",
    "--bootstrap-samples", "10000", "--signflip-samples", "10000",
    "--batch-size", "16", "--generation-batch-size", "8",
    "--precision", "float32", "--device", "cuda",
]
if RUN_TRANSFER_IF_FINAL_PASSES:
    command.extend(["--run-transfer", "--transfer-examples", str(TRANSFER_EXAMPLES)])
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("### Primary confirmation gate")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"full": "#315f8c", "ordinary": "#8a8a8a", "fail": "#b65f24"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 11})

primary = summary["confirmation"]["primary"]
primary_table = pd.DataFrame([{
    "mean NLL advantage": primary["mean_nll_advantage"],
    "NLL lower": primary["nll_bootstrap_95ci"][0],
    "NLL upper": primary["nll_bootstrap_95ci"][1],
    "full / ordinary bits": primary["full_to_ordinary_cache_bits_ratio"],
    "ordinary accuracy": primary["ordinary_accuracy"],
    "full accuracy": primary["full_accuracy"],
    "accuracy lower": primary["accuracy_bootstrap_95ci"][0],
    "passed": primary["gate"]["passed"],
}])
display(primary_table)
display(pd.DataFrame([primary["gate"]]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for axis, mean, interval, title, ylabel in (
    (axes[0], primary["mean_nll_advantage"], primary["nll_bootstrap_95ci"],
     "Rank-16 answer-NLL advantage", "ordinary NLL − full-method NLL"),
    (axes[1], primary["mean_accuracy_advantage"], primary["accuracy_bootstrap_95ci"],
     "Paired generation-accuracy difference", "full accuracy − ordinary accuracy"),
):
    axis.errorbar([0], [mean], yerr=[[mean-interval[0]], [interval[1]-mean]],
                  fmt="o", capsize=7, color=COLORS["full"])
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set(title=title, ylabel=ylabel, xticks=[])
axes[1].axhline(-0.02, color=COLORS["fail"], linestyle="--", label="−2 point margin")
axes[1].legend()
plt.show()
''')

md("### Factorial mechanism tests and matched-random protection")
code(r'''
if summary["mechanism"] is None:
    print("Mechanism stage correctly stopped because the primary confirmation gate failed.")
else:
    effects = summary["mechanism"]["effects"]
    rows = []
    for name, record in effects.items():
        rows.append({
            "effect": name,
            "mean": record["mean"],
            "lower": record["bootstrap_95ci"][0],
            "upper": record["bootstrap_95ci"][1],
            "p-value": record["one_sided_sign_flip_pvalue"],
            "Holm threshold": record["holm"]["holm_threshold"],
            "Holm passed": record["holm"]["rejected"],
        })
    effect_table = pd.DataFrame(rows)
    display(effect_table)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    y = np.arange(len(effect_table))
    means = effect_table["mean"].to_numpy()
    lower = effect_table["lower"].to_numpy(); upper = effect_table["upper"].to_numpy()
    ax.errorbar(means, y, xerr=np.vstack((means-lower, upper-means)),
                fmt="o", capsize=6, color=COLORS["full"])
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set(title="Which component explains the rank-16 effect?",
           xlabel="paired answer-NLL advantage", yticks=y,
           yticklabels=effect_table["effect"])
    plt.show()

    random_record = summary["mechanism"]["random_protection"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.hist(random_record["random_mean_nll_deltas"], bins=15,
            color="#d0d0d0", edgecolor="#555555")
    full_delta = summary["confirmation"]["teacher_forced"]["full_method"][
        "mean_full_answer_nll_delta"
    ]
    ax.axvline(full_delta, color=COLORS["full"], linewidth=2, label="frozen layer-11 protection")
    ax.set(title="Frozen layer-11 protection versus 100 matched-random bases",
           xlabel="mean answer-NLL change", ylabel="random controls")
    ax.legend(); plt.show()
    print({key: random_record[key] for key in (
        "mean_full_advantage_over_random", "bootstrap_95ci", "full_beats_random_fraction"
    )})
''')

md("### Locked final replication")
code(r'''
if summary["final_replication"] is None:
    print("The 551-question final slice remains untouched because confirmation failed.")
else:
    final_primary = summary["final_replication"]["primary"]
    display(pd.DataFrame([{
        "split": "confirmation",
        "examples": CONFIRMATION_EXAMPLES,
        "NLL advantage": primary["mean_nll_advantage"],
        "lower": primary["nll_bootstrap_95ci"][0],
        "upper": primary["nll_bootstrap_95ci"][1],
        "passed": primary["gate"]["passed"],
    }, {
        "split": "locked final",
        "examples": summary["preregistration"]["final_slice"][1] - FINAL_START,
        "NLL advantage": final_primary["mean_nll_advantage"],
        "lower": final_primary["nll_bootstrap_95ci"][0],
        "upper": final_primary["nll_bootstrap_95ci"][1],
        "passed": final_primary["gate"]["passed"],
    }]))
''')

md("### Greedy-generation quality at rank 16")
code(r'''
generation_rows = []
for split_name in ("confirmation", "final_replication"):
    record = summary.get(split_name)
    if record is None:
        continue
    for arm, values in record["generation"].items():
        cache = values.get("cache", {})
        generation_rows.append({
            "split": "confirmation" if split_name == "confirmation" else "locked final",
            "arm": arm,
            "accuracy": values["accuracy"],
            "correct": values["correct"],
            "examples": values["examples"],
            "sequence agreement": values["exact_sequence_agreement"],
            "compression ratio": cache.get("modelled_compression_ratio", 1.0),
        })
generation_table = pd.DataFrame(generation_rows)
display(generation_table.sort_values(["split", "compression ratio"]))
if len(generation_table):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for split_name, group in generation_table.groupby("split"):
        ax.scatter(group["compression ratio"], group["accuracy"], s=65, label=split_name)
        for _, row in group.iterrows():
            ax.annotate(row["arm"], (row["compression ratio"], row["accuracy"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(title="Rank-16 generation quality at actual modeled storage",
           xlabel="modeled cache compression ratio", ylabel="GSM8K accuracy")
    ax.legend(); plt.show()
''')

md("### Frozen cross-dataset transfer, only after final replication")
code(r'''
if summary["transfer"] is None:
    print("Transfer stage did not run because the locked final gate did not pass or transfer was disabled.")
else:
    rows = []
    for dataset, record in summary["transfer"].items():
        for arm, values in record["generation"].items():
            rows.append({"dataset": dataset, "arm": arm, "accuracy": values["accuracy"],
                         "correct": values["correct"], "examples": values["examples"]})
    display(pd.DataFrame(rows).sort_values(["dataset", "accuracy"], ascending=[True, False]))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print("Confirmation gate:", summary["decision"]["confirmation_passed"])
print("Locked-final gate:", summary["decision"]["final_passed"])
if summary["mechanism"] is not None:
    print("Allocation/Fisher mechanism:", summary["decision"]["allocation_fisher_passed"])
    print("Specific causal protection:", summary["decision"]["causal_protection_passed"])
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
import shutil
archive = shutil.make_archive(
    "/kaggle/working/codi_rank16_xkv_confirmation",
    "zip",
    root_dir="/kaggle/working",
    base_dir=pathlib.Path(OUTPUT_DIR).name,
)
print(archive)
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
