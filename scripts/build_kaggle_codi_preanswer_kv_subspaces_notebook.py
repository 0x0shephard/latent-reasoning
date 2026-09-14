"""Build the exact-pre-answer-gradient CODI layerwise KV notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_preanswer_kv_subspaces.ipynb"
RUN_COMMIT = "6054944c911cccc13057cf7cff633a55bb986019"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# CODI: how many answer-relevant KV directions exist in every layer?

## tl;dr

This is the corrected experiment. It computes answer gradients with respect to the
**exact pre-answer key and value cache tensors** supplied to CODI's answer decoder.
It independently fits a covariance eigensystem at the `ln_1` input of every one of
GPT-2's 12 blocks, maps every candidate direction through that block's effective
LoRA-aware `W_K` and `W_V`, and reports a variable validated rank for each layer.

The previous notebook forced 28 directions and then discovered that its hooked-state
gradients were disconnected. Those 28-index lists are not evidence. This notebook
has a hard 100% cache-gradient-connectivity gate and cannot convert disconnected
gradients into a scientific zero.
""")

md(r"""
## Context & Methods

For layer (l), latent position (p), and eigenvector (e_{lj}), the notebook
measures the state coefficient

\[
a_{ilpj}=(h_{ilp}-\mu_{lp})^T e_{lj},
\]

maps the direction into cache feature space,

\[
r^K_{lj}=W_K^{(l)}e_{lj},\qquad r^V_{lj}=W_V^{(l)}e_{lj},
\]

and estimates the loss change caused by removing it:

\[
I_{ilpj}=-a_{ilpj}\left[\langle \nabla_K L,r^K_{lj}\rangle+
\langle \nabla_V L,r^V_{lj}\rangle\right].
\]

Directions must be positive in two deterministic halves, exceed a shuffled-example
null, and survive Benjamini–Hochberg correction across all 9,216 layer-direction
hypotheses. The operational rank is the smallest prefix retaining 95% of the
validated first-order effect, capped at 64. It is not forced to equal 28.

### Key assumptions

- Fit, direction-selection, and causal-confirmation questions are disjoint.
- The official GSM8K test questions are excluded from all three discovery splits.
- The answer objective is full teacher-forced gold-answer NLL, not only a binary
  first-token result.
- Retain/remove interventions edit all six latent K/V entries in the real cache.
- Four rank- and covariance-energy-matched random controls are used by default.
- The optional xKV stage is a quality/storage proxy; a fused kernel is required to
  claim production memory or latency gains.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
DISCOVERY_OUTPUT = "/kaggle/working/codi_preanswer_kv_subspaces"
COMPRESSION_OUTPUT = "/kaggle/working/codi_preanswer_kv_compression"

FIT_EXAMPLES = 1024
SELECTION_EXAMPLES = 1024
RANK_SELECTION_EXAMPLES = 256
CAUSAL_EXAMPLES = 128
MAXIMUM_RANK = 64
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
    "tests/test_preanswer_kv_subspace.py", "tests/test_direct_layerwise_kv.py",
    "tests/test_causal_xkv.py", "tests/test_official_codi_target_utility.py",
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
    assert matches, (
        "Attach the completed official CODI reproduction dataset containing " + suffix
    )
    return sorted(matches, key=lambda value: (len(pathlib.Path(value).parts), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
print(REPRODUCTION_SUMMARY)
''')

md("## Results — exact cache-gradient discovery and causal confirmation")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_preanswer_kv_subspace_discovery.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--output-dir", DISCOVERY_OUTPUT,
    "--fit-examples", str(FIT_EXAMPLES),
    "--select-examples", str(SELECTION_EXAMPLES),
    "--rank-examples", str(RANK_SELECTION_EXAMPLES),
    "--causal-examples", str(CAUSAL_EXAMPLES),
    "--maximum-rank", str(MAXIMUM_RANK),
    "--random-controls", str(RANDOM_CONTROLS),
    "--fit-batch-size", "16", "--selection-batch-size", "4",
    "--causal-batch-size", "8", "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(DISCOVERY_OUTPUT) / "summary.json").read_text())
print(json.dumps({
    "ranks_by_layer": summary["ranks_by_layer"],
    "gate": summary["gate"],
}, indent=2))
''')

md("### Connectivity is an implementation gate, not a scientific outcome")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

connectivity = np.asarray(summary["gradient_connectivity_fraction"])
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
image = ax.imshow(connectivity, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set(title="Exact pre-answer cache-gradient connectivity",
       xlabel="cache tensor", ylabel="transformer block")
ax.set_xticks([0, 1], ["Key", "Value"])
ax.set_yticks(range(12))
for layer in range(12):
    for kind in range(2):
        ax.text(kind, layer, f"{connectivity[layer, kind]:.2f}",
                ha="center", va="center",
                color="white" if connectivity[layer, kind] > 0.55 else "#222222")
fig.colorbar(image, ax=ax, label="fraction of connected batches")
plt.show()
assert np.allclose(connectivity, 1), "Do not interpret ranks unless connectivity is 1.00 everywhere"
''')

md("### How many directions were supported in each layer?")
code(r'''
layer_table = pd.DataFrame(summary["layer_direction_summary"])
display(layer_table[[
    "layer", "statistically_validated_direction_count", "operational_rank",
    "selected_variance_fraction", "causal_status",
]])

layers = layer_table["layer"].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
axes[0].bar(layers - 0.18, layer_table["statistically_validated_direction_count"],
            width=0.36, color="#315f8c", label="statistically validated")
axes[0].bar(layers + 0.18, layer_table["operational_rank"],
            width=0.36, color="#b65f24", label="95% effect rank")
axes[0].axhline(28, color="#222222", linestyle="--", linewidth=1, label="original U28 reference")
axes[0].set(title="Variable answer-relevant rank by transformer block",
            xlabel="transformer block", ylabel="number of directions", xticks=layers)
axes[0].legend()
axes[1].bar(layers, 100 * layer_table["selected_variance_fraction"], color="#315f8c")
axes[1].set(title="Covariance energy carried by selected directions",
            xlabel="transformer block", ylabel="selected variance (%)", xticks=layers)
plt.show()
''')

md("### Does removing the learned subspace hurt more than removing random space?")
code(r'''
means, lower, upper, plotted_layers = [], [], [], []
for layer in range(12):
    record = summary["causal_screen"][f"layer_{layer:02d}"]
    if "specificity_bootstrap_95ci" not in record:
        continue
    plotted_layers.append(layer)
    means.append(record["specificity_mean_full_answer_nll_delta"])
    lower.append(record["specificity_bootstrap_95ci"][0])
    upper.append(record["specificity_bootstrap_95ci"][1])

fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
if plotted_layers:
    ax.errorbar(
        plotted_layers, means,
        yerr=[[mean-lo for mean, lo in zip(means, lower)],
              [hi-mean for mean, hi in zip(means, upper)]],
        marker="o", capsize=4, color="#b65f24",
    )
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Learned removal damage minus matched-random removal damage",
       xlabel="transformer block", ylabel="full-answer NLL difference (paired 95% CI)",
       xticks=range(12))
plt.show()
''')

md("## Results — protected-core xKV quality proxy, only after the causal gate")
code(r'''
gate = summary["gate"]
should_run = RUN_COMPRESSION_IF_GATE_PASSES and (
    gate["passed"] or ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION
)
if should_run:
    selected_group = gate["selected_contiguous_layers"] or [8, 9, 10, 11]
    protected_rank = sum(summary["ranks_by_layer"][layer] for layer in selected_group)
    compression_ranks = sorted({
        max(32, protected_rank), max(48, protected_rank), max(64, protected_rank)
    })
    command = [
        sys.executable, "-u", "scripts/run_codi_causal_xkv.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--layerwise-artifact", str(pathlib.Path(DISCOVERY_OUTPUT) / "preanswer_kv_subspaces.pt"),
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
    print("Compression correctly stopped: no adjacent causal layer group passed.")
    print("Do not enable exploratory compression for a confirmatory claim.")
''')

md("## Takeaways")
code(r'''
print("Validated direction counts:", [
    row["statistically_validated_direction_count"]
    for row in summary["layer_direction_summary"]
])
print("Operational ranks:", summary["ranks_by_layer"])
print("Causal gate:", "PASS" if gate["passed"] else "FAIL")
print("Passing layers:", gate["passing_layers"])
print("Selected adjacent group:", gate["selected_contiguous_layers"])
print("Artifact:", pathlib.Path(DISCOVERY_OUTPUT) / "preanswer_kv_subspaces.pt")

if not any(summary["ranks_by_layer"]):
    print("No layer-local direction survived split stability and global FDR correction.")
elif not gate["passed"]:
    print("Candidate directions exist, but no adjacent causal KV core was confirmed.")
else:
    print("An adjacent causal KV core was confirmed; interpret the equal-budget compression stage.")
''')

md("### Save the complete Kaggle output")
code(r'''
import shutil
archive = shutil.make_archive(
    "/kaggle/working/codi_preanswer_kv_experiment", "zip",
    root_dir="/kaggle/working",
    base_dir=pathlib.Path(DISCOVERY_OUTPUT).name,
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
