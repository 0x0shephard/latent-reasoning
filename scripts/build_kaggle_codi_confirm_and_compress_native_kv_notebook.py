"""Build one notebook for native-KV confirmation and gated residual compression."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_confirm_and_compress_native_kv.ipynb"
RUN_COMMIT = "__RUN_COMMIT__"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# CODI native KV: confirm the finding, then test protected compression

## What this notebook decides

This single notebook performs both required follow-ups without using the new
confirmation data to rediscover the answer:

1. **Confirm the frozen finding.** Layer 11/value-rank-1 is the preregistered
   primary hypothesis. Frozen layer 2, 3 and 5 bases are larger-sample secondary
   confirmations. All use 512 questions never used to fit, select, or causally
   screen the source artifact.
2. **Test protected residual compression.** Only if layer 11 replicates, preserve
   its frozen value direction and compare ordinary SVD/xKV, random protection, and
   causal protection on the untouched GSM8K test set.

The compression stage is a quality and modeled-storage proxy. It reconstructs a
dense cache for stock GPT-2, so its wall-clock seconds are not evidence of a fused
runtime speedup.
""")

md(r"""
## Context & Methods

The source experiment found candidate native-cache projectors (P=UU^T). For a
centered cache vector (x) and exact answer-loss gradient (g), the predicted
removal damage is (-g^TPx). Here the bases and ranks are frozen before the new
questions are evaluated.

### Primary confirmation gate

Layer 11/value-rank-1 passes only when:

- its paired removal damage relative to four covariance-energy-matched random
  controls has a positive 95% bootstrap lower bound;
- at least three of four disjoint 128-question folds have positive specificity;
- retaining the direction gives at least 95% first-token agreement with dense
  CODI and increases full-answer NLL by no more than 0.10.

Layers 2, 3 and 5 use the same rules plus a Bonferroni interval across all four
candidate layers. Ranks 2, 4 and 8 at layer 11 are diagnostics only and cannot
rescue a failed rank-1 primary result.

### Key assumptions

- The source artifact must have the native-cache discovery contract.
- Its original four split hashes are reconstructed and verified before the new
  confirmation tail is accepted.
- The GSM8K test set remains untouched until the gated compression comparison.
- A passing layer-11 result establishes a single-layer value-cache core, not a
  two-layer adjacent core and not a globally low-rank KV cache.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
SOURCE_ARTIFACT_INPUT = ""
DISCOVERY_OUTPUT = "/kaggle/working/codi_direct_cache_source"
CONFIRM_OUTPUT = "/kaggle/working/codi_native_kv_confirmation"
COMPRESSION_OUTPUT = "/kaggle/working/codi_native_kv_protected_xkv"

RERUN_DISCOVERY_IF_SOURCE_IS_MISSING = True
CONFIRM_EXAMPLES = 512
CONFIRM_FOLDS = 4
CANDIDATE_LAYERS = "2,3,5,11"
PRIMARY_LAYER = 11
RUN_COMPRESSION_IF_PRIMARY_PASSES = True

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
    "tests/test_confirm_direct_cache_subspace.py",
    "tests/test_direct_cache_task_subspace.py",
    "tests/test_preanswer_kv_subspace.py",
    "tests/test_direct_layerwise_kv.py",
    "tests/test_causal_xkv.py",
], check=True)
''')

md("### Resolve the official reproduction and frozen discovery artifact")
code(r'''
def discover(explicit, suffix, message):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    if not matches:
        return None
    return sorted(matches, key=lambda value: (len(pathlib.Path(value).parts), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
    "Attach the completed official CODI reproduction dataset",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"

SOURCE_ARTIFACT = discover(
    SOURCE_ARTIFACT_INPUT,
    "direct_cache_task_subspaces.pt",
    "Attach the preceding native-cache experiment output",
)
print("reproduction", REPRODUCTION_SUMMARY)
print("source artifact", SOURCE_ARTIFACT or "will be reproduced deterministically")
''')

md("### Reproduce the frozen source only when it was not attached")
code(r'''
if SOURCE_ARTIFACT is None:
    assert RERUN_DISCOVERY_IF_SOURCE_IS_MISSING, (
        "Attach direct_cache_task_subspaces.pt or enable deterministic rerun"
    )
    subprocess.run([
        sys.executable, "-u", "scripts/run_codi_direct_cache_task_subspaces.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--output-dir", DISCOVERY_OUTPUT,
        "--fit-examples", "1024", "--discovery-examples", "1024",
        "--rank-examples", "256", "--causal-examples", "128",
        "--rank-grid", "1,2,4,8,16,28,32,48,64",
        "--fit-batch-size", "16", "--gradient-batch-size", "4",
        "--causal-batch-size", "8", "--precision", "float32", "--device", "cuda",
    ], check=True)
    SOURCE_ARTIFACT = str(pathlib.Path(DISCOVERY_OUTPUT) / "direct_cache_task_subspaces.pt")
print("using frozen source", SOURCE_ARTIFACT)
''')

md("## Results — independent confirmation")
code(r'''
subprocess.run([
    sys.executable, "-u", "scripts/run_codi_confirm_and_compress_native_kv.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--source-artifact", SOURCE_ARTIFACT,
    "--output-dir", CONFIRM_OUTPUT,
    "--confirm-examples", str(CONFIRM_EXAMPLES),
    "--confirm-folds", str(CONFIRM_FOLDS),
    "--candidate-layers", CANDIDATE_LAYERS,
    "--primary-layer", str(PRIMARY_LAYER),
    "--batch-size", "16", "--gradient-batch-size", "4",
    "--precision", "float32", "--device", "cuda",
], check=True)
summary = json.loads((pathlib.Path(CONFIRM_OUTPUT) / "summary.json").read_text())
print(json.dumps({
    "primary_layer": summary["primary_layer"],
    "compression_gate": summary["compression_gate"],
}, indent=2))
''')

md("### Compare frozen candidates with multiplicity-corrected intervals")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 11})
rows = []
for name, record in summary["confirmations"].items():
    rows.append({
        "layer": int(name.split("_")[-1]),
        "K rank": record["key_rank"], "V rank": record["value_rank"],
        "specificity": record["specificity_mean_full_answer_nll_delta"],
        "familywise lower": record["candidate_familywise_95ci"][0],
        "familywise upper": record["candidate_familywise_95ci"][1],
        "positive folds": record["positive_folds"],
        "retain agreement": record["task_retain"]["dense_first_token_top1_agreement"],
        "candidate passed": record["candidate_passed"],
        "primary passed": record["primary_passed"],
    })
candidate_table = pd.DataFrame(rows).sort_values("layer")
display(candidate_table)

x = np.arange(len(candidate_table))
y = candidate_table["specificity"].to_numpy()
lower = candidate_table["familywise lower"].to_numpy()
upper = candidate_table["familywise upper"].to_numpy()
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.errorbar(x, y, yerr=np.vstack((y-lower, upper-y)), fmt="o", capsize=5,
            color="#315f8c", ecolor="#777777")
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Frozen native-cache effects on 512 new questions",
       xlabel="transformer block", ylabel="task removal minus matched-random NLL change",
       xticks=x, xticklabels=candidate_table["layer"])
plt.show()
''')

md("### Check replication across four disjoint confirmation folds")
code(r'''
fold_rows = []
for name, record in summary["confirmations"].items():
    layer = int(name.split("_")[-1])
    for fold, effect in enumerate(record["fold_specificity_means"]):
        fold_rows.append({"layer": layer, "fold": fold + 1, "specificity": effect})
fold_table = pd.DataFrame(fold_rows)
display(fold_table.pivot(index="layer", columns="fold", values="specificity"))
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
for layer, group in fold_table.groupby("layer"):
    ax.plot(group["fold"], group["specificity"], marker="o", label=f"layer {layer}")
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Replication across four disjoint 128-question folds",
       xlabel="confirmation fold", ylabel="specificity NLL change", xticks=range(1, 5))
ax.legend(ncols=2); plt.show()
''')

md("### Diagnose layer 11 ranks without changing the preregistered decision")
code(r'''
rank_rows = []
for name, record in summary["rank_sweep"].items():
    rank = int(name.split("_")[-1])
    rank_rows.append({
        "rank": rank,
        "removal NLL damage": record["remove"]["mean_full_answer_nll_delta"],
        "retain top-1 agreement": record["retain"]["dense_first_token_top1_agreement"],
        "retain NLL delta": record["retain"]["mean_full_answer_nll_delta"],
        "preregistered": record["preregistered"],
    })
rank_table = pd.DataFrame(rank_rows).sort_values("rank")
display(rank_table)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
axes[0].plot(rank_table["rank"], rank_table["removal NLL damage"], marker="o", color="#315f8c")
axes[0].set(title="Necessity by layer-11 value rank", xlabel="rank", ylabel="removal NLL change",
            xticks=rank_table["rank"])
axes[1].plot(rank_table["rank"], rank_table["retain top-1 agreement"], marker="s", color="#b65f24")
axes[1].axhline(0.95, color="#222222", linestyle="--", linewidth=1)
axes[1].set(title="Sufficiency by layer-11 value rank", xlabel="rank",
            ylabel="dense first-token agreement", xticks=rank_table["rank"])
plt.show()
print("Only rank 1 controls the primary pass/fail decision.")
''')

md("### Locate the rank-1 signal across CODI's six latent positions")
code(r'''
position_table = pd.DataFrame(summary["position_summary"])
display(position_table)
x = position_table["latent_position"].to_numpy() + 1
y = position_table["mean_predicted_removal_damage"].to_numpy()
lower = np.array([value[0] for value in position_table["bootstrap_95ci"]])
upper = np.array([value[1] for value in position_table["bootstrap_95ci"]])
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.errorbar(x, y, yerr=np.vstack((y-lower, upper-y)), fmt="o-", capsize=5,
            color="#315f8c", ecolor="#777777")
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Layer-11 value-rank-1 first-order effect by latent position",
       xlabel="continuous latent step", ylabel="predicted removal NLL damage",
       xticks=range(1, 7))
plt.show()
''')

md("## Results — gated protected residual-xKV comparison")
code(r'''
gate = summary["compression_gate"]
if RUN_COMPRESSION_IF_PRIMARY_PASSES and gate["passed"]:
    subprocess.run([
        sys.executable, "-u", "scripts/run_codi_causal_xkv.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--layerwise-artifact", str(
            pathlib.Path(CONFIRM_OUTPUT) / "confirmed_native_kv_artifact.pt"
        ),
        "--output-dir", COMPRESSION_OUTPUT,
        "--examples", "256", "--max-new-tokens", "64",
        "--ranks", "8,16,32,48,64",
        "--batch-size", "8", "--precision", "float32", "--device", "cuda",
    ], check=True)
    compression = json.loads((pathlib.Path(COMPRESSION_OUTPUT) / "summary.json").read_text())
    compression_table = pd.DataFrame(compression["results"]).T
    display(compression_table[[
        column for column in ("accuracy", "accuracy_retained_fraction",
                              "exact_sequence_agreement", "cache")
        if column in compression_table.columns
    ]])
else:
    compression = None
    print("Protected compression correctly stopped: layer-11 rank-1 did not replicate.")
''')

md("### Visualize quality against modeled cache storage")
code(r'''
if compression is not None:
    plot_rows = []
    for name, record in compression["results"].items():
        if name == "dense" or "cache" not in record:
            continue
        plot_rows.append({
            "arm": name,
            "modeled compression ratio": record["cache"]["modelled_compression_ratio"],
            "accuracy retained": record["accuracy_retained_fraction"],
            "exact sequence agreement": record["exact_sequence_agreement"],
        })
    quality_table = pd.DataFrame(plot_rows)
    display(quality_table)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for _, row in quality_table.iterrows():
        marker = "s" if "independent_kv_protected" in row["arm"] else "o"
        ax.scatter(row["modeled compression ratio"], row["accuracy retained"],
                   marker=marker, s=70, color="#315f8c" if marker == "s" else "#888888")
        ax.annotate(row["arm"], (row["modeled compression ratio"], row["accuracy retained"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(title="GSM8K quality versus modeled KV storage",
           xlabel="modeled cache compression ratio (higher is smaller)",
           ylabel="accuracy retained relative to dense CODI")
    plt.show()
    print(compression["timing_warning"])
''')

md("## Takeaways")
code(r'''
primary = summary["confirmations"][f"layer_{PRIMARY_LAYER:02d}"]
print("Layer-11 rank-1 primary:", "PASS" if primary["primary_passed"] else "FAIL")
print("Primary 95% interval:", primary["primary_bootstrap_95ci"])
print("Positive folds:", primary["positive_folds"], "of", CONFIRM_FOLDS)
print("Candidate family-wise passes:", {
    name: value["candidate_passed"] for name, value in summary["confirmations"].items()
})
print("Compression executed:", compression is not None)
print("Confirmation artifact:", pathlib.Path(CONFIRM_OUTPUT) / "confirmed_native_kv_artifact.pt")
''')

md("### Save the complete combined experiment")
code(r'''
import shutil
bundle_root = pathlib.Path("/kaggle/working/codi_native_kv_combined_bundle")
bundle_root.mkdir(exist_ok=True)
shutil.copytree(CONFIRM_OUTPUT, bundle_root / "confirmation", dirs_exist_ok=True)
if compression is not None:
    shutil.copytree(COMPRESSION_OUTPUT, bundle_root / "compression", dirs_exist_ok=True)
archive = shutil.make_archive(str(bundle_root), "zip", root_dir=bundle_root)
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
