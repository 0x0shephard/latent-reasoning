"""Build the native-cache task-sensitive CODI Kaggle notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_direct_cache_task_subspaces.ipynb"
RUN_COMMIT = "18a55c056b3b43d8f2df09c73f3d61fc001b5583"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# CODI: learn answer-sensitive directions directly in every K and V cache

## What this notebook decides

The earlier task-sensitive experiment learned a subspace in the 768-dimensional
hidden state and then mapped it through the key and value projections. That mapping
and separate orthonormalization changed the projector that the causal intervention
actually tested. This experiment removes that mismatch.

For every one of CODI's 12 transformer blocks, it learns one subspace directly in
the cached **key** vectors and another directly in the cached **value** vectors.
Keys and values may receive different ranks. A final causal split tests whether
removing those native-cache directions damages the complete gold answer more than
equal-rank, energy-matched random directions while retaining them preserves the
dense model. Compression remains blocked unless at least two adjacent layers pass.
""")

md(r"""
## Context & Methods

For one cache kind, let (x=X-mu) be a centered 768-value key or value vector and
let (g=\partial L/\partial X) be the exact derivative of full-answer loss with
respect to that same cache tensor. Removing an orthogonal projector (P=UU^T)
has the first-order effect

\[
\Delta L \approx -g^T P x.
\]

The leading positive eigenvectors of

\[
M=-\tfrac12\,\mathbb E[xg^T+gx^T]
\]

maximize that predicted damage. We fit this matrix independently for K and V in
every layer. No hidden-to-cache mapping is used.

### Key assumptions and safeguards

- Four question-disjoint GSM8K-train splits: cache covariance fit (1,024), task
  discovery (1,024), rank selection (256), and causal confirmation (128).
- Official GSM8K test questions are excluded from all splits.
- Every native K/V tensor must have 100% autograd connectivity to answer loss.
- K and V ranks are selected independently from `1,2,4,8,16,28,32,48,64`.
- The smallest stable rank retaining 95% of the best held-out excess predicted
  effect is selected; this replaces the previous diffuse-spectrum rule.
- Causal specificity uses a paired bootstrap interval corrected across all 12
  layers. The primary gate is frozen before viewing causal results.
- Direct top-covariance, 50/50 task–covariance hybrid, key-only, value-only, and
  energy-matched random controls use the same K/V ranks.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_direct_cache_task_subspaces"
COMPRESSION_OUTPUT = "/kaggle/working/codi_direct_cache_task_xkv"

CACHE_FIT_EXAMPLES = 1024
TASK_DISCOVERY_EXAMPLES = 1024
RANK_SELECTION_EXAMPLES = 256
CAUSAL_CONFIRMATION_EXAMPLES = 128
RANK_GRID = "1,2,4,8,16,28,32,48,64"
RANDOM_CONTROLS = 4
RUN_COMPRESSION_IF_GATE_PASSES = True
ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION = False

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip()
print("code commit", CODE_COMMIT)
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
    "tests/test_direct_cache_task_subspace.py",
    "tests/test_preanswer_kv_subspace.py",
    "tests/test_direct_layerwise_kv.py",
    "tests/test_causal_xkv.py",
], check=True)
''')

md("### Resolve the completed official CODI reproduction")
code(r'''
def discover(explicit, suffix):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    assert matches, "Attach the completed official CODI reproduction dataset"
    return sorted(matches, key=lambda value: (len(pathlib.Path(value).parts), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
print(REPRODUCTION_SUMMARY)
''')

md("## Results — direct K/V discovery and causal confirmation")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_direct_cache_task_subspaces.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--fit-examples", str(CACHE_FIT_EXAMPLES),
    "--discovery-examples", str(TASK_DISCOVERY_EXAMPLES),
    "--rank-examples", str(RANK_SELECTION_EXAMPLES),
    "--causal-examples", str(CAUSAL_CONFIRMATION_EXAMPLES),
    "--rank-grid", RANK_GRID, "--random-controls", str(RANDOM_CONTROLS),
    "--fit-batch-size", "16", "--gradient-batch-size", "4",
    "--causal-batch-size", "8", "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(json.dumps({
    "key_ranks": summary["key_ranks_by_layer"],
    "value_ranks": summary["value_ranks_by_layer"],
    "gate": summary["gate"],
}, indent=2))
''')

md("### Verify exact native-cache gradient connectivity")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 11})
connectivity = np.asarray(summary["gradient_connectivity_fraction"])
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
image = ax.imshow(connectivity, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set(title="Exact pre-answer native-cache gradient connectivity",
       xlabel="cache tensor", ylabel="transformer block")
ax.set_xticks([0, 1], ["Key", "Value"]); ax.set_yticks(range(12))
for layer in range(12):
    for kind in range(2):
        ax.text(kind, layer, f"{connectivity[layer, kind]:.2f}",
                ha="center", va="center", color="white")
fig.colorbar(image, ax=ax, label="fraction of connected batches")
plt.show()
assert np.allclose(connectivity, 1), "Do not interpret results without 1.00 connectivity"
''')

md("### How many native K and V directions survive held-out rank selection?")
code(r'''
layer_table = pd.DataFrame(summary["layer_direction_summary"])
display(layer_table)

layers = layer_table["layer"].to_numpy()
width = 0.36
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
axes[0].bar(layers - width/2, layer_table["key_rank"], width,
            color="#315f8c", label="key rank")
axes[0].bar(layers + width/2, layer_table["value_rank"], width,
            color="#b65f24", label="value rank")
axes[0].axhline(28, color="#222222", linestyle="--", linewidth=1,
                label="original endpoint U28 reference")
axes[0].set(title="Key and value ranks selected independently in native cache space",
            ylabel="selected directions")
axes[0].legend(ncols=3)
axes[1].plot(layers, layer_table["key_half_split_overlap"].fillna(0),
             marker="o", color="#315f8c", label="key overlap")
axes[1].plot(layers, layer_table["value_half_split_overlap"].fillna(0),
             marker="s", color="#b65f24", label="value overlap")
axes[1].axhline(0.25, color="#222222", linestyle=":", linewidth=1,
                label="minimum overlap")
axes[1].set(xlabel="transformer block", ylabel="mean squared canonical correlation",
            xticks=layers)
axes[1].legend(ncols=3)
plt.show()
''')

md("### Does task geometry beat covariance and matched random directions?")
code(r'''
effect_rows = []
for layer in range(12):
    record = summary["causal_screen"][f"layer_{layer:02d}"]
    if "task_remove" not in record:
        continue
    random_mean = np.mean([
        arm["mean_full_answer_nll_delta"] for arm in record["random_remove"]
    ])
    effect_rows.append({
        "layer": layer,
        "native task": record["task_remove"]["mean_full_answer_nll_delta"],
        "native covariance": record["covariance_remove"]["mean_full_answer_nll_delta"],
        "task-covariance hybrid": record["hybrid_remove"]["mean_full_answer_nll_delta"],
        "energy-matched random": random_mean,
    })
effect_table = pd.DataFrame(effect_rows)
if len(effect_table):
    display(effect_table)
    effect_table.set_index("layer").plot.bar(
        figsize=(14, 6), color=["#315f8c", "#b65f24", "#8b6f47", "#888888"]
    )
    plt.axhline(0, color="#222222", linewidth=1)
    plt.title("Full-answer NLL damage at equal native K/V rank budgets")
    plt.xlabel("transformer block"); plt.ylabel("mean gold-answer NLL change")
    plt.legend(ncols=2); plt.tight_layout(); plt.show()
else:
    print("No native-cache K or V rank passed held-out selection.")
''')

md("### Separate the key and value causal contributions")
code(r'''
component_rows = []
for layer in range(12):
    record = summary["causal_screen"][f"layer_{layer:02d}"]
    if "key_task_remove" not in record:
        continue
    component_rows.append({
        "layer": layer,
        "key only": record["key_task_remove"]["mean_full_answer_nll_delta"],
        "value only": record["value_task_remove"]["mean_full_answer_nll_delta"],
        "joint independent K+V": record["task_remove"]["mean_full_answer_nll_delta"],
    })
component_table = pd.DataFrame(component_rows)
if len(component_table):
    display(component_table)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for name, color, marker in (
        ("key only", "#315f8c", "o"), ("value only", "#b65f24", "s"),
        ("joint independent K+V", "#555555", "^"),
    ):
        ax.plot(component_table["layer"], component_table[name], marker=marker,
                color=color, label=name)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set(title="Which native cache carries answer-sensitive information?",
           xlabel="transformer block", ylabel="mean gold-answer NLL change",
           xticks=range(12))
    ax.legend(); plt.show()
''')

md("### Inspect the preregistered causal gate")
code(r'''
gate_rows = []
for layer in range(12):
    record = summary["causal_screen"][f"layer_{layer:02d}"]
    gate_rows.append({
        "layer": layer,
        "K rank": record["key_rank"], "V rank": record["value_rank"],
        "specificity": record.get("specificity_mean_full_answer_nll_delta"),
        "family-wise lower": record.get("familywise_95ci", [None, None])[0],
        "retain top-1 agreement": record.get("task_retain", {}).get(
            "dense_first_token_top1_agreement"
        ),
        "retain NLL delta": record.get("task_retain", {}).get(
            "mean_full_answer_nll_delta"
        ),
        "passed": record["passed"],
    })
display(pd.DataFrame(gate_rows))
print("Overall causal gate:", "PASS" if summary["gate"]["passed"] else "FAIL")
print("Passing layers:", summary["gate"]["passing_layers"])
print("Selected adjacent group:", summary["gate"]["selected_contiguous_layers"])
''')

md("## Results — protected xKV quality proxy, only if the causal gate passes")
code(r'''
gate = summary["gate"]
should_run = RUN_COMPRESSION_IF_GATE_PASSES and (
    gate["passed"] or ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION
)
if should_run:
    selected_group = gate["selected_contiguous_layers"] or [8, 9, 10, 11]
    protected_rank = sum(
        summary["key_ranks_by_layer"][layer]
        + summary["value_ranks_by_layer"][layer]
        for layer in selected_group
    )
    compression_ranks = sorted({
        max(32, protected_rank), max(48, protected_rank), max(64, protected_rank)
    })
    command = [
        sys.executable, "-u", "scripts/run_codi_causal_xkv.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--layerwise-artifact", str(
            pathlib.Path(OUTPUT_DIR) / "direct_cache_task_subspaces.pt"
        ),
        "--output-dir", COMPRESSION_OUTPUT,
        "--examples", "256", "--max-new-tokens", "64",
        "--ranks", ",".join(map(str, compression_ranks)),
        "--batch-size", "8", "--precision", "float32", "--device", "cuda",
    ]
    if not gate["passed"]:
        command.append("--allow-unconfirmed")
    subprocess.run(command, check=True)
    compression = json.loads((pathlib.Path(COMPRESSION_OUTPUT) / "summary.json").read_text())
    display(pd.DataFrame(compression["results"]).T)
else:
    print("Compression correctly stopped: no adjacent causal native-cache core passed.")
''')

md("## Takeaways")
code(r'''
print("Native key ranks:", summary["key_ranks_by_layer"])
print("Native value ranks:", summary["value_ranks_by_layer"])
print("Causal gate:", "PASS" if gate["passed"] else "FAIL")
print("Artifact:", pathlib.Path(OUTPUT_DIR) / "direct_cache_task_subspaces.pt")
if not any(summary["key_ranks_by_layer"]) and not any(summary["value_ranks_by_layer"]):
    print("No direct K/V combination passed held-out rank selection.")
elif not gate["passed"]:
    print("Some direct K/V ranks were selected, but no adjacent causal core was confirmed.")
else:
    print("An adjacent native-cache causal core passed; inspect the gated xKV proxy.")
''')

md("### Save the full audit trail")
code(r'''
import shutil
archive = shutil.make_archive(
    "/kaggle/working/codi_direct_cache_task_subspaces", "zip",
    root_dir="/kaggle/working", base_dir=pathlib.Path(OUTPUT_DIR).name,
)
print(archive)
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
