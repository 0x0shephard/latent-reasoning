"""Build the task-sensitive CODI KV subspace Kaggle notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_task_sensitive_kv_subspaces.ipynb"
RUN_COMMIT = "52f8ba7e462c03feb94b87e6373783022c0e7006"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# CODI: learn task-sensitive KV subspaces from combinations of directions

## tl;dr

The previous corrected run proved that every pre-answer K/V tensor was connected to
answer loss, but no **individual covariance eigenvector** passed its 9,216-way
multiple-testing gate. This experiment tests the broader hypothesis that answer
relevance is carried by a linear combination of covariance directions.

For each of CODI's 12 transformer blocks, it builds a task matrix from the six latent
`ln_1` states and the exact pre-answer K/V gradients. Its leading positive
eigenvectors maximize the predicted answer-loss increase caused by removing the
whole subspace. Rank is selected on disjoint examples and causally confirmed against
top-covariance, key-only, value-only, and rank/energy-matched random controls.
""")

md(r"""
## Context & Methods

The cache gradients are mapped back into the hidden space:

\[
g^h_{lp}=(W_K^{(l)})^T g^K_{lp}+(W_V^{(l)})^T g^V_{lp}.
\]

For centred hidden state (x), removing projector (P=UU^T) has first-order
answer-loss change

\[
\Delta L\approx -(g^h)^T P x.
\]

Therefore the leading positive eigenvectors of

\[
M_l=-\frac12\mathbb E[x(g^h)^T+g^h x^T]
\]

identify the rank-(r) subspace with the greatest expected first-order removal
damage. Unlike the previous test, each learned direction can combine many ordinary
covariance eigenvectors.

### Key assumptions and safeguards

- Covariance fit, task discovery, rank selection, and causal confirmation are four
  question-disjoint GSM8K-train partitions.
- Official GSM8K test questions are excluded from all four partitions.
- Every K and V cache tensor must show 100% autograd connectivity.
- Candidate ranks are `1,2,4,8,16,28,32,48,64`.
- A rank must retain 95% of the positive spectrum available through rank 64, have
  half-split subspace overlap of at least 0.25, and beat shuffled gradients with a
  positive 95% bootstrap interval on the rank-selection split.
- Causal confirmation evaluates full gold-answer NLL and first-token distribution
  fidelity. Compression runs only after two adjacent layers pass.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_task_sensitive_kv"
COMPRESSION_OUTPUT = "/kaggle/working/codi_task_sensitive_kv_compression"

COVARIANCE_FIT_EXAMPLES = 1024
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
    "tests/test_task_sensitive_kv_subspace.py",
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

md("## Results — task-sensitive discovery, rank selection and causal confirmation")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_task_sensitive_kv_subspaces.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--fit-examples", str(COVARIANCE_FIT_EXAMPLES),
    "--discovery-examples", str(TASK_DISCOVERY_EXAMPLES),
    "--rank-examples", str(RANK_SELECTION_EXAMPLES),
    "--causal-examples", str(CAUSAL_CONFIRMATION_EXAMPLES),
    "--rank-grid", RANK_GRID,
    "--random-controls", str(RANDOM_CONTROLS),
    "--fit-batch-size", "16", "--gradient-batch-size", "4",
    "--causal-batch-size", "8", "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(json.dumps({
    "ranks_by_layer": summary["ranks_by_layer"],
    "gate": summary["gate"],
}, indent=2))
''')

md("### Verify that the exact cache gradients were connected")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

connectivity = np.asarray(summary["gradient_connectivity_fraction"])
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
image = ax.imshow(connectivity, vmin=0, vmax=1, cmap="Blues", aspect="auto")
ax.set(title="Exact pre-answer cache-gradient connectivity",
       xlabel="cache tensor", ylabel="transformer block")
ax.set_xticks([0, 1], ["Key", "Value"]); ax.set_yticks(range(12))
for layer in range(12):
    for kind in range(2):
        ax.text(kind, layer, f"{connectivity[layer, kind]:.2f}",
                ha="center", va="center", color="white")
fig.colorbar(image, ax=ax, label="fraction of connected batches")
plt.show()
assert np.allclose(connectivity, 1), "Do not interpret results unless connectivity is 1.00"
''')

md("### Selected rank and split stability")
code(r'''
layer_table = pd.DataFrame(summary["layer_direction_summary"])
display(layer_table[[
    "layer", "positive_joint_task_eigenvalues", "selected_rank",
    "half_split_subspace_overlap", "rank_selection_excess_damage",
    "selected_variance_fraction", "causal_status",
]])

layers = layer_table["layer"].to_numpy()
overlap = layer_table["half_split_subspace_overlap"].fillna(0).to_numpy()
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
bars = ax.bar(layers, layer_table["selected_rank"], color="#315f8c", label="selected rank")
ax.axhline(28, color="#222222", linestyle="--", linewidth=1, label="original U28 reference")
ax.set(title="Task-sensitive rank selected independently for each block",
       xlabel="transformer block", ylabel="selected rank", xticks=layers)
second = ax.twinx()
second.plot(layers, overlap, color="#b65f24", marker="o", label="half-split overlap")
second.axhline(0.25, color="#b65f24", linestyle=":", linewidth=1)
second.set_ylabel("mean squared canonical correlation", color="#b65f24")
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = second.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, loc="upper left")
plt.show()
''')

md("### Task spectrum: how concentrated is predicted removal damage?")
code(r'''
spectra = np.asarray([
    row["top_joint_task_eigenvalues"] for row in summary["layer_direction_summary"]
])
positive = np.maximum(spectra, 0)
denominator = positive.sum(axis=1, keepdims=True)
normalized = np.divide(
    positive, denominator, out=np.zeros_like(positive), where=denominator > 0
)
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
image = ax.imshow(normalized, aspect="auto", cmap="Blues", vmin=0)
ax.set(title="Share of positive task spectrum by direction",
       xlabel="task-sensitive direction rank", ylabel="transformer block")
ax.set_yticks(range(12)); ax.set_xticks([0, 7, 15, 27, 31, 47, 63], [1, 8, 16, 28, 32, 48, 64])
fig.colorbar(image, ax=ax, label="share of positive spectrum through rank 64")
plt.show()
''')

md("### Causal damage versus random and covariance baselines")
code(r'''
plot_rows = []
for layer in range(12):
    record = summary["causal_screen"][f"layer_{layer:02d}"]
    if "joint_remove" not in record:
        continue
    random_mean = np.mean([
        arm["mean_full_answer_nll_delta"] for arm in record["random_remove"]
    ])
    plot_rows.append({
        "layer": layer,
        "task-sensitive": record["joint_remove"]["mean_full_answer_nll_delta"],
        "top covariance": record["covariance_remove"]["mean_full_answer_nll_delta"],
        "matched random": random_mean,
    })

plot_table = pd.DataFrame(plot_rows)
if len(plot_table):
    display(plot_table)
    x = np.arange(len(plot_table)); width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - width, plot_table["task-sensitive"], width, color="#315f8c", label="task-sensitive")
    ax.bar(x, plot_table["top covariance"], width, color="#b65f24", label="top covariance")
    ax.bar(x + width, plot_table["matched random"], width, color="#888888", label="matched random")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set(title="Full-answer loss damage caused by subspace removal",
           xlabel="transformer block", ylabel="mean gold-answer NLL change",
           xticks=x, xticklabels=plot_table["layer"])
    ax.legend()
    plt.show()
else:
    print("No layer passed rank selection, so causal interventions were not run.")
''')

md("### Separate key and value contributions")
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
        "joint": record["joint_remove"]["mean_full_answer_nll_delta"],
    })
component_table = pd.DataFrame(component_rows)
if len(component_table):
    display(component_table)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(component_table["layer"], component_table["key only"], marker="o",
            color="#315f8c", label="key-only learned subspace")
    ax.plot(component_table["layer"], component_table["value only"], marker="s",
            color="#b65f24", label="value-only learned subspace")
    ax.plot(component_table["layer"], component_table["joint"], marker="^",
            color="#555555", label="joint learned subspace")
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set(title="Which part of the latent cache carries the causal signal?",
           xlabel="transformer block", ylabel="mean gold-answer NLL change",
           xticks=range(12))
    ax.legend()
    plt.show()
''')

md("## Results — protected-core xKV proxy, only after the causal gate")
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
        "--layerwise-artifact", str(pathlib.Path(OUTPUT_DIR) / "task_sensitive_kv_subspaces.pt"),
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
''')

md("## Takeaways")
code(r'''
print("Task-sensitive ranks:", summary["ranks_by_layer"])
print("Causal gate:", "PASS" if gate["passed"] else "FAIL")
print("Passing layers:", gate["passing_layers"])
print("Selected adjacent group:", gate["selected_contiguous_layers"])
print("Artifact:", pathlib.Path(OUTPUT_DIR) / "task_sensitive_kv_subspaces.pt")
if not any(summary["ranks_by_layer"]):
    print("No combination passed held-out rank selection; inspect overlap and rank-grid audit.")
elif not gate["passed"]:
    print("Low-rank combinations were selected, but an adjacent causal KV core was not confirmed.")
else:
    print("A task-sensitive adjacent KV core passed; interpret the equal-budget xKV comparison.")
''')

md("### Save the discovery artifacts")
code(r'''
import shutil
archive = shutil.make_archive(
    "/kaggle/working/codi_task_sensitive_kv", "zip",
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
