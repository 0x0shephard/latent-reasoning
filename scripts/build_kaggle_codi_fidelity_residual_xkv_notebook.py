"""Build the Kaggle notebook for the fidelity-residual xKV experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_fidelity_residual_xkv.ipynb"
RUN_COMMIT = "2cf904dc0d65ddf5e207f76ae4e134fe55c91b82"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can a tiny fidelity-sensitive residual make xKV behave like dense CODI?

## tl;dr

The previous task/variance frontier stopped correctly: even rank 48 reached only
about 85% dense first-token agreement, below the frozen 95% requirement. This
experiment changes the mechanism rather than weakening the gate.

- It reconstructs every GSM8K-train question used by predecessor experiments.
- It then takes the next 512 untouched train questions to fit a label-free
  dense-top-1 margin sensitivity basis.
- A second fresh 512-question train slice screens 12 fixed candidates:
  ordinary per-layer xKV ranks `48, 64, 80, 96` plus residual ranks `1, 2, 4`.
- The residual stores only the reconstruction-error coordinates aligned with
  dense top-1-versus-runner-up margin gradients.
- The 551-question GSM8K final slice remains locked unless one fresh-screen
  candidate passes every preregistered check.

This is a dense-reconstruction quality and modeled-storage study, not a native
compressed-attention latency benchmark.
""")

md(r"""
## Context & Methods

Ordinary xKV approximates each per-layer cache matrix with an SVD. If
(\widehat X) is that approximation, this experiment stores a small correction

\[
\widehat X_{\text{corrected}}
= \widehat X + (X-\widehat X)BB^\top,
\]

only for the six latent cache rows. Columns of (B) are the leading feature
directions of exact gradients of the dense model's own top-1-versus-runner-up
logit margin. Gold labels are not used to fit (B).

A fresh-screen candidate must satisfy all five checks:

1. paired-bootstrap 95% lower bound for ordinary-xKV NLL minus corrected-xKV NLL > 0;
2. corrected/ordinary modeled cache bits match within 0.1%;
3. dense first-token top-1 agreement ≥ 95%;
4. generation accuracy versus dense is non-inferior within 2 points; and
5. generation accuracy versus ordinary xKV is non-inferior within 2 points.

Selection is frozen: maximum compression, then maximum fidelity, minimum KL
from dense, maximum NLL lower bound, and finally minimum residual rank. If no
candidate passes, the final test slice remains unopened.

### Key Assumptions

- The predecessor artifact contains the complete deterministic train-sampling
  audit from the earlier experiments.
- The failed frontier summary confirms that the final GSM8K slice is untouched.
- Static residual bases are model metadata; their per-request coordinates are
  included in modeled KV storage.
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
SOURCE_ARTIFACT_INPUT = (
    "/kaggle/input/datasets/jonraza15/"
    "codi-xkv-rank16-predecessor-artifacts-v2/"
    "direct_cache_task_subspaces/direct_cache_task_subspaces.pt"
)
PREVIOUS_EXPERIMENT_ARTIFACT_INPUT = (
    "/kaggle/input/datasets/jonraza15/"
    "codi-xkv-rank16-predecessor-artifacts-v2/"
    "task_aware_protected_xkv/task_aware_protected_xkv.pt"
)
# Attach the output dataset from kaggle_codi_xkv_fidelity_frontier.ipynb.
# Leave blank for contract-based discovery or paste the exact summary.json path.
PREVIOUS_FRONTIER_SUMMARY_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_fidelity_residual_xkv"

FRESH_FIT_EXAMPLES = 512
FRESH_SCREEN_EXAMPLES = 512
FINAL_START = 768
RANKS = "48,64,80,96"
RESIDUAL_RANKS = "1,2,4"

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
# TPOT is preinstalled in some Kaggle images but unused here.  Removing it
# avoids a resolver warning when datasets installs its compatible dill version.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "tpot"],
    check=False,
)
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
    "tests/test_fidelity_residual_xkv.py",
    "tests/test_fidelity_residual_xkv_runner.py",
    "tests/test_preanswer_kv_subspace.py",
    "tests/test_task_aware_xkv.py",
    "tests/test_xkv_fidelity_frontier.py",
    "tests/test_official_codi_kv.py",
], check=True)
''')

md("### Resolve the four frozen inputs")
code(r'''
import torch
import zipfile

def repack_kaggle_torch_archive(candidate):
    """Restore a torch.save ZIP when Kaggle exposes it as a .pt directory."""
    path = pathlib.Path(candidate)
    if path.is_file():
        return str(path)
    if not path.is_dir() or not (path / "data.pkl").is_file():
        return None
    restored_root = pathlib.Path("/kaggle/working/restored_torch_artifacts")
    restored_root.mkdir(parents=True, exist_ok=True)
    restored = restored_root / path.name
    temporary = restored.with_suffix(restored.suffix + ".tmp")
    archive_prefix = path.stem
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as handle:
        for source in sorted(path.rglob("*")):
            if source.is_file():
                relative = source.relative_to(path).as_posix()
                handle.write(source, arcname=f"{archive_prefix}/{relative}")
    temporary.replace(restored)
    print("restored Kaggle-expanded torch artifact", path, "->", restored)
    return str(restored)

def all_candidates(suffix):
    values = []
    for root in ("/kaggle/working", "/kaggle/input"):
        values.extend(glob.glob(f"{root}/**/{suffix}", recursive=True))
    return sorted(set(values), key=lambda value: (len(pathlib.Path(value).parts), value))

def discover_file(explicit, suffix):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    candidates = all_candidates(suffix)
    return candidates[0] if candidates else None

def discover_json_contract(explicit, contract):
    candidates = [explicit] if explicit else all_candidates("summary.json")
    for candidate in filter(None, candidates):
        try:
            record = json.loads(pathlib.Path(candidate).read_text())
            if record.get("contract") == contract:
                return str(candidate)
        except Exception:
            continue
    return None

def discover_torch_contract(explicit, suffix, contract):
    candidates = [explicit] if explicit else all_candidates(suffix)
    for candidate in filter(None, candidates):
        try:
            loadable = repack_kaggle_torch_archive(candidate)
            if loadable is None:
                continue
            artifact = torch.load(loadable, map_location="cpu", weights_only=False)
            if artifact.get("contract") == contract:
                return loadable
            print("ignored artifact with contract", artifact.get("contract"), loadable)
        except Exception as error:
            print("could not load artifact", candidate, repr(error))
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
PREVIOUS_FRONTIER_SUMMARY = discover_json_contract(
    PREVIOUS_FRONTIER_SUMMARY_INPUT,
    "official_codi_xkv_fidelity_frontier_holdout_v1",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
assert SOURCE_ARTIFACT, "Attach direct_cache_task_subspaces.pt"
assert PREVIOUS_EXPERIMENT_ARTIFACT, "Attach task_aware_protected_xkv.pt"
assert PREVIOUS_FRONTIER_SUMMARY, "Attach summary.json from the failed fidelity frontier"
print("reproduction", REPRODUCTION_SUMMARY)
print("source artifact", SOURCE_ARTIFACT)
print("task-aware predecessor", PREVIOUS_EXPERIMENT_ARTIFACT)
print("failed frontier", PREVIOUS_FRONTIER_SUMMARY)
''')

md("## Results — fresh screen and one locked final replication")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_fidelity_residual_xkv.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--source-artifact", SOURCE_ARTIFACT,
    "--previous-experiment-artifact", PREVIOUS_EXPERIMENT_ARTIFACT,
    "--previous-frontier-summary", PREVIOUS_FRONTIER_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--ranks", RANKS, "--residual-ranks", RESIDUAL_RANKS,
    "--fresh-fit-examples", str(FRESH_FIT_EXAMPLES),
    "--fresh-screen-examples", str(FRESH_SCREEN_EXAMPLES),
    "--final-start", str(FINAL_START),
    "--bootstrap-samples", "10000",
    "--minimum-first-token-fidelity", "0.95",
    "--storage-tolerance", "0.001",
    "--accuracy-noninferiority-margin", "0.02",
    "--gradient-batch-size", "4", "--batch-size", "16",
    "--generation-batch-size", "8",
    "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("### What did the margin-gradient basis capture?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"key": "#315f8c", "value": "#b65f24"}
MARKERS = {1: "o", 2: "s", 4: "D"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

basis_rows = []
for component, record in summary["basis_fit"]["audit"].items():
    layer = int(component.split("_")[1])
    kind = component.split("_")[2]
    basis_rows.append({
        "layer": layer,
        "cache tensor": kind,
        "rank-1 captured gradient energy": record[
            "cumulative_gradient_energy_fraction"
        ][0],
        "rank-4 captured gradient energy": record[
            "cumulative_gradient_energy_fraction"
        ][3],
        "gradient norm": record["gradient_frobenius_norm"],
    })
basis_table = pd.DataFrame(basis_rows)
display(basis_table)

fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
width = 0.38
layers = np.arange(12)
for offset, kind in ((-width/2, "key"), (width/2, "value")):
    rows = basis_table[basis_table["cache tensor"] == kind].sort_values("layer")
    ax.bar(
        layers + offset,
        rows["rank-4 captured gradient energy"],
        width=width,
        color=COLORS[kind],
        label=kind,
    )
ax.set(
    title="Rank-4 coverage of dense top-1 margin-gradient energy",
    xlabel="transformer block",
    ylabel="captured gradient-energy fraction",
    xticks=layers,
    ylim=(0, 1),
)
ax.legend()
plt.show()
''')

md("### Does the tiny residual cross the 95% fidelity threshold?")
code(r'''
rows = []
for record in summary["fresh_screen"]["candidates"]:
    rows.append({
        "candidate": record["name"],
        "xKV rank": record["rank"],
        "residual rank": record["residual_rank"],
        "compression ratio": record["modelled_compression_ratio"],
        "first-token fidelity": record["first_token_fidelity"],
        "KL from dense": record["mean_first_token_kl_from_dense"],
        "mean NLL advantage": record["mean_nll_advantage"],
        "NLL lower": record["nll_bootstrap_95ci"][0],
        "NLL upper": record["nll_bootstrap_95ci"][1],
        "bits ratio": record["full_to_ordinary_cache_bits_ratio"],
        "teacher preeligible": record["teacher_preeligible"],
        "screen passed": record["screen_passed"],
    })
screen_table = pd.DataFrame(rows).sort_values(["xKV rank", "residual rank"])
display(screen_table)

rank_colors = {48: "#8a8a8a", 64: "#b65f24", 80: "#315f8c", 96: "#757b48"}
fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
for _, row in screen_table.iterrows():
    ax.scatter(
        row["compression ratio"], row["first-token fidelity"],
        color=rank_colors[row["xKV rank"]],
        marker=MARKERS[row["residual rank"]],
        s=100 if row["screen passed"] else 65,
        facecolors=(rank_colors[row["xKV rank"]]
                    if row["screen passed"] else "none"),
        linewidths=1.5,
    )
    ax.annotate(
        f"r{int(row['xKV rank'])}/res{int(row['residual rank'])}",
        (row["compression ratio"], row["first-token fidelity"]),
        xytext=(4, 4), textcoords="offset points", fontsize=8,
    )
ax.axhline(0.95, color="#222222", linestyle="--", linewidth=1,
           label="95% fidelity threshold")
ax.set(
    title="Fresh-screen fidelity versus modeled cache compression",
    xlabel="modeled cache compression ratio",
    ylabel="dense first-token top-1 agreement",
)
ax.legend(loc="best")
plt.show()
''')

md("### Paired answer-NLL evidence versus ordinary xKV at the same rank")
code(r'''
ordered = screen_table.sort_values("NLL lower")
fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
y = np.arange(len(ordered))
means = ordered["mean NLL advantage"].to_numpy()
lower = ordered["NLL lower"].to_numpy()
upper = ordered["NLL upper"].to_numpy()
for index, (_, row) in enumerate(ordered.iterrows()):
    ax.errorbar(
        means[index], y[index],
        xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
        fmt=MARKERS[row["residual rank"]], capsize=4,
        color=rank_colors[row["xKV rank"]],
    )
ax.axvline(0, color="#222222", linewidth=1)
ax.set(
    title="Residual xKV advantage with paired 95% bootstrap intervals",
    xlabel="ordinary xKV NLL − residual xKV NLL (positive favors residual)",
    yticks=y,
    yticklabels=ordered["candidate"],
)
plt.show()
''')

md("### Deterministic selection and the locked final decision")
code(r'''
selected = summary["selected_candidate"]
if selected is None:
    print("No fresh-screen candidate passed. The 551-question final slice remains untouched.")
else:
    display(pd.DataFrame([selected]))
    print("Selected once from the fresh screen:", selected["name"])

final = summary["final_replication"]
if final is None:
    print("Locked final evaluation did not run.")
else:
    display(pd.DataFrame([{
        "candidate": final["selected_candidate"]["name"],
        "mean NLL advantage": final["mean_nll_advantage"],
        "NLL lower": final["nll_bootstrap_95ci"][0],
        "NLL upper": final["nll_bootstrap_95ci"][1],
        "bits ratio": final["full_to_ordinary_cache_bits_ratio"],
        "first-token fidelity": final["teacher_forced"]["selected"][
            "dense_first_token_top1_agreement"
        ],
        "dense accuracy": final["generation"]["dense"]["accuracy"],
        "ordinary accuracy": final["generation"]["ordinary_xkv"]["accuracy"],
        "selected accuracy": final["generation"]["selected"]["accuracy"],
        "passed": final["gate"]["passed"],
    }]))
    display(pd.DataFrame([final["gate"]]))

    estimates = [
        final["mean_nll_advantage"],
        final["generation"]["selected"]["accuracy"]
        - final["generation"]["dense"]["accuracy"],
        final["generation"]["selected"]["accuracy"]
        - final["generation"]["ordinary_xkv"]["accuracy"],
    ]
    intervals = [
        final["nll_bootstrap_95ci"],
        final["accuracy_difference_vs_dense_95ci"],
        final["accuracy_difference_vs_ordinary_95ci"],
    ]
    labels = ["NLL advantage", "accuracy vs dense", "accuracy vs ordinary"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, estimate, interval, label in zip(axes, estimates, intervals, labels):
        axis.errorbar(
            [0], [estimate],
            yerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt="o", capsize=7, color="#315f8c",
        )
        axis.axhline(0, color="#222222", linewidth=1)
        axis.set(title=label, xticks=[])
    for axis in axes[1:]:
        axis.axhline(-0.02, color="#b65f24", linestyle="--", linewidth=1)
    fig.suptitle("Locked final estimates with paired 95% bootstrap intervals")
    plt.show()
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print("Fresh-screen gate:", summary["decision"]["screen_passed"])
print("Locked-final gate:", summary["decision"]["final_passed"])
if summary["selected_candidate"] is not None:
    print("Frozen selected candidate:", summary["selected_candidate"])
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
assert (output / "summary.json").is_file()
assert (output / "fidelity_residual_xkv.pt").is_file()
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
