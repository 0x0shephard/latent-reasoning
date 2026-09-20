"""Build the Kaggle notebook for the locked xKV fidelity-recovery frontier."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_xkv_fidelity_frontier.ipynb"
RUN_COMMIT = "181ea6c166bbe5f086ec9a048dc28793f0126173"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can xKV recover dense fidelity without giving up its matched-storage advantage?

## tl;dr

The previous rank-16 result beat ordinary xKV in answer NLL and GSM8K accuracy,
but it failed the frozen 95% first-token-fidelity requirement. This notebook does
not relabel that outcome as a success. It runs one preregistered recovery study:

- GSM8K rows `0:256` remain discovery-only.
- The already-open rows `256:768` are used only to select among 15 fixed
  rank × task-weight candidates.
- Rows `768:1319` remain locked unless one candidate passes every development
  screen, then exactly one selected candidate is evaluated once.
- Candidate ranks are `16, 24, 32, 40, 48`; answer-Fisher task weights are
  `0, 0.5, 1`. The complement is cache-variance weight.
- Every candidate retains the frozen layer-11 protected basis and is compared
  with ordinary xKV at the same rank and modeled storage.

The result is a dense-reconstruction quality and modeled-storage experiment,
not a native xKV kernel or latency benchmark.
""")

md(r"""
## Context & Methods

The recovery frontier tests whether the old failure came from over-specializing
the compression geometry to answer gradients. For candidate (c), group utility
is the frozen convex blend

\[
u_c = w\,u_{\text{answer-Fisher}} + (1-w)\,u_{\text{cache variance}},
\]

after separately normalizing both utility families. Feature-level Fisher weights
are interpolated toward all-ones with the same (w). Nothing is refit on the
locked final slice.

A development candidate must satisfy all five checks:

1. paired-bootstrap 95% lower bound for ordinary-xKV NLL minus candidate NLL > 0;
2. candidate/ordinary modeled cache bits within 0.1%;
3. dense first-token top-1 agreement ≥ 95%;
4. paired generation accuracy versus dense has lower bound > −2 points; and
5. paired generation accuracy versus ordinary xKV has lower bound > −2 points.

Among passing candidates, selection is deterministic: maximum modeled
compression, then maximum NLL lower bound, then maximum task weight. If none
passes, execution stops and the final 551 questions remain untouched. The final
gate repeats the same five checks without reselection.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"

# These three paths match the predecessor datasets already attached in the
# preceding experiment. Change them only if your Kaggle dataset slug differs.
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
# Leave blank to find the rank-16 output by its embedded contract.
RANK16_SUMMARY_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_xkv_fidelity_frontier"

DEVELOPMENT_START = 256
DEVELOPMENT_EXAMPLES = 512
FINAL_START = 768
RANKS = "16,24,32,40,48"
TASK_WEIGHTS = "0.0,0.5,1.0"

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
    "tests/test_xkv_fidelity_frontier.py",
    "tests/test_xkv_fidelity_frontier_runner.py",
    "tests/test_rank16_xkv_confirmation.py",
    "tests/test_task_aware_xkv.py",
    "tests/test_official_codi_kv.py",
    "tests/test_preanswer_kv_subspace.py",
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
RANK16_SUMMARY = discover_json_contract(
    RANK16_SUMMARY_INPUT,
    "official_codi_rank16_xkv_mechanism_holdout_v1",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
assert SOURCE_ARTIFACT, "Attach direct_cache_task_subspaces.pt"
assert PREVIOUS_EXPERIMENT_ARTIFACT, "Attach task_aware_protected_xkv.pt"
assert RANK16_SUMMARY, "Attach summary.json from the completed rank-16 confirmation"
print("reproduction", REPRODUCTION_SUMMARY)
print("source artifact", SOURCE_ARTIFACT)
print("task-aware predecessor", PREVIOUS_EXPERIMENT_ARTIFACT)
print("rank-16 result", RANK16_SUMMARY)
''')

md("## Results — development frontier and one locked final replication")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_xkv_fidelity_frontier.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--source-artifact", SOURCE_ARTIFACT,
    "--previous-experiment-artifact", PREVIOUS_EXPERIMENT_ARTIFACT,
    "--rank16-summary", RANK16_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--ranks", RANKS, "--task-weights", TASK_WEIGHTS,
    "--development-start", str(DEVELOPMENT_START),
    "--development-examples", str(DEVELOPMENT_EXAMPLES),
    "--final-start", str(FINAL_START),
    "--bootstrap-samples", "10000",
    "--minimum-first-token-fidelity", "0.95",
    "--storage-tolerance", "0.001",
    "--accuracy-noninferiority-margin", "0.02",
    "--batch-size", "16", "--generation-batch-size", "8",
    "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("### Which blend restores fidelity while preserving the NLL advantage?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {0.0: "#8a8a8a", 0.5: "#b65f24", 1.0: "#315f8c"}
MARKERS = {16: "o", 24: "s", 32: "^", 40: "D", 48: "P"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

rows = []
for record in summary["development_frontier"]["candidates"]:
    rows.append({
        "candidate": record["name"],
        "rank": record["rank"],
        "task weight": record["task_weight"],
        "compression ratio": record["modelled_compression_ratio"],
        "first-token fidelity": record["first_token_fidelity"],
        "mean NLL advantage": record["mean_nll_advantage"],
        "NLL lower": record["nll_bootstrap_95ci"][0],
        "NLL upper": record["nll_bootstrap_95ci"][1],
        "teacher preeligible": record["teacher_preeligible"],
        "screen passed": record["screen_passed"],
    })
frontier = pd.DataFrame(rows).sort_values(["rank", "task weight"])
display(frontier)

fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
for _, row in frontier.iterrows():
    ax.scatter(
        row["compression ratio"], row["first-token fidelity"],
        color=COLORS[row["task weight"]], marker=MARKERS[row["rank"]],
        s=95 if row["screen passed"] else 60,
        facecolors=COLORS[row["task weight"]] if row["screen passed"] else "none",
        linewidths=1.5,
    )
    ax.annotate(row["candidate"],
                (row["compression ratio"], row["first-token fidelity"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.axhline(0.95, color="#222222", linestyle="--", linewidth=1,
           label="95% fidelity threshold")
ax.set(
    title="Development fidelity frontier (open markers failed the full screen)",
    xlabel="modeled cache compression ratio",
    ylabel="dense first-token top-1 agreement",
)
ax.legend(loc="best")
plt.show()
''')

md("### Paired answer-NLL evidence with 95% bootstrap intervals")
code(r'''
ordered = frontier.sort_values("NLL lower")
fig, ax = plt.subplots(figsize=(10.5, 7), constrained_layout=True)
y = np.arange(len(ordered))
means = ordered["mean NLL advantage"].to_numpy()
lower = ordered["NLL lower"].to_numpy()
upper = ordered["NLL upper"].to_numpy()
colors = [COLORS[value] for value in ordered["task weight"]]
for index in range(len(ordered)):
    ax.errorbar(
        means[index], y[index],
        xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
        fmt="o", capsize=4, color=colors[index],
    )
ax.axvline(0, color="#222222", linewidth=1)
ax.set(
    title="Candidate advantage over ordinary xKV at the same rank",
    xlabel="ordinary xKV NLL − candidate NLL (positive favors candidate)",
    yticks=y,
    yticklabels=ordered["candidate"],
)
plt.show()
''')

md("### Deterministic selection and the locked final decision")
code(r'''
selected = summary["selected_candidate"]
if selected is None:
    print("No development candidate passed. The 551-question final slice remains untouched.")
else:
    display(pd.DataFrame([selected]))
    print("Selected once from development:", selected["name"])

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
print("Development screen:", summary["decision"]["development_passed"])
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
assert (output / "xkv_fidelity_frontier.pt").is_file()
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
