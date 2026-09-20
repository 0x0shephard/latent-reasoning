"""Build the Kaggle notebook for storage-matched adaptive K/V allocation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_adaptive_kv_allocation.ipynb"
RUN_COMMIT = "0198274e165b7434c52b4e581d1fd82e134f41a4"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can adaptive K/V rank allocation beat uniform per-layer xKV?

## tl;dr

The previous experiment showed that a tiny global gradient residual did not
improve ordinary xKV. This notebook tests a different hypothesis: uniform rank
is the wrong way to spend a fixed cache budget.

- Reconstruct every GSM8K-train question used by the predecessor chain.
- Fit rank-utility curves on the next 512 untouched train questions using the
  exact answer-NLL gradient of every pre-answer cache row.
- Allocate rank independently to all 24 layer×K/V components.
- Screen nine frozen candidates on another 512 untouched train questions:
  ordinary-xKV budgets `48, 64, 80` × reconstruction/Fisher blends `0, .5, 1`.
- Count explicit padding so each adaptive arm uses exactly the same modeled
  cache bits as ordinary grouped per-layer xKV at its comparison rank.
- Open the untouched 551-question final test slice only if a fresh candidate
  passes the complete gate.

This is a dense-reconstruction quality and modeled-storage experiment, not a
native compressed-attention latency benchmark.
""")

md(r"""
## Context & Methods

For each calibration request and each layer key/value matrix, the experiment
computes its SVD. The utility of retaining singular mode `j` combines:

1. its reconstruction energy, `sigma_j²`; and
2. its squared first-order answer-loss effect,
   `(gradient · singular_mode_j)²`.

Both utility families are globally normalized before blending. The allocator
then chooses independent prefix ranks under the ordinary-xKV factor-bit budget.
Any rank-unit remainder is recorded as padding, preventing the adaptive method
from receiving hidden free storage.

A candidate must satisfy all five preregistered checks:

1. paired-bootstrap 95% lower bound for ordinary NLL minus adaptive NLL > 0;
2. adaptive/ordinary modeled bits match within 0.1%;
3. dense first-token top-1 agreement ≥ 95%;
4. generation accuracy versus dense is non-inferior within 2 points; and
5. generation accuracy versus ordinary xKV is non-inferior within 2 points.

### Key Assumptions

- The attached predecessor summary is the failed fidelity-residual run and its
  final test field is still null.
- Train sampling is deterministic, so the predecessor prefix and two new slices
  can be reconstructed exactly.
- Static utility curves are model metadata; every request-specific factor and
  all budget padding are included in modeled cache storage.
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
# Attach the published output directory from
# kaggle_codi_fidelity_residual_xkv.ipynb. Leave blank for contract discovery.
PREVIOUS_SUMMARY_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_adaptive_kv_allocation"

FRESH_CALIBRATION_EXAMPLES = 512
FRESH_SCREEN_EXAMPLES = 512
FINAL_START = 768
BASELINE_RANKS = "48,64,80"
ANSWER_WEIGHTS = "0,0.5,1"
MAXIMUM_COMPONENT_RANK = 96

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
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False
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
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True
    )
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_adaptive_kv_allocation.py",
    "tests/test_adaptive_kv_allocation_runner.py",
    "tests/test_preanswer_kv_subspace.py",
    "tests/test_task_aware_xkv.py",
    "tests/test_xkv_fidelity_frontier.py",
    "tests/test_official_codi_kv.py",
], check=True)
''')

md("### Resolve the reproduction and failed-predecessor summaries")
code(r'''
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

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
PREVIOUS_SUMMARY = discover_json_contract(
    PREVIOUS_SUMMARY_INPUT,
    "official_codi_fidelity_residual_xkv_holdout_v1",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
assert PREVIOUS_SUMMARY, (
    "Attach the published output directory from the fidelity-residual experiment"
)
previous = json.loads(pathlib.Path(PREVIOUS_SUMMARY).read_text())
assert previous["decision"]["screen_passed"] is False
assert previous["final_replication"] is None
print("reproduction", REPRODUCTION_SUMMARY)
print("failed fidelity-residual predecessor", PREVIOUS_SUMMARY)
print("predecessor final slice remains locked", previous["split_hashes"]["final"])
''')

md("## Results — fresh allocation screen and one locked final replication")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_adaptive_kv_allocation.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--previous-summary", PREVIOUS_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--baseline-ranks", BASELINE_RANKS,
    "--answer-weights", ANSWER_WEIGHTS,
    "--maximum-component-rank", str(MAXIMUM_COMPONENT_RANK),
    "--fresh-calibration-examples", str(FRESH_CALIBRATION_EXAMPLES),
    "--fresh-screen-examples", str(FRESH_SCREEN_EXAMPLES),
    "--final-start", str(FINAL_START),
    "--bootstrap-samples", "10000",
    "--minimum-first-token-fidelity", "0.95",
    "--storage-tolerance", "0.001",
    "--accuracy-noninferiority-margin", "0.02",
    "--gradient-batch-size", "2", "--batch-size", "16",
    "--generation-batch-size", "8",
    "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("### Where did the calibration find useful rank?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"key": "#315f8c", "value": "#b65f24"}
WEIGHT_LABELS = {0.0: "reconstruction", 0.5: "hybrid", 1.0: "answer-Fisher"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

audit_rows = []
for name, record in summary["allocation_fit"]["component_audit"].items():
    parts = name.split("_")
    audit_rows.append({
        "layer": int(parts[1]), "cache tensor": parts[2],
        "reconstruction utility": record["reconstruction_utility_total"],
        "answer-Fisher utility": record["answer_fisher_utility_total"],
        "rank-16 Fisher fraction": record["rank_16_answer_fisher_fraction"],
    })
audit_table = pd.DataFrame(audit_rows).sort_values(["layer", "cache tensor"])
display(audit_table)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
width = 0.38
layers = np.arange(12)
for axis, metric, title in (
    (axes[0], "reconstruction utility", "Per-component reconstruction utility"),
    (axes[1], "answer-Fisher utility", "Per-component answer-NLL utility"),
):
    for offset, kind in ((-width / 2, "key"), (width / 2, "value")):
        rows = audit_table[audit_table["cache tensor"] == kind].sort_values("layer")
        axis.bar(layers + offset, rows[metric], width=width,
                 color=COLORS[kind], label=kind)
    axis.set(title=title, xlabel="transformer block", ylabel="mean utility",
             xticks=layers)
axes[0].legend()
plt.show()
''')

md("### How did the frozen allocator split rank between keys and values?")
code(r'''
allocation_rows = []
for candidate in summary["fresh_screen"]["candidates"]:
    ranks = candidate["teacher_forced"]["cache"]["effective_rank_by_component"]
    for component, values in ranks.items():
        layer, kind = component.split(":")
        allocation_rows.append({
            "candidate": candidate["name"], "baseline rank": candidate["rank"],
            "answer weight": candidate["answer_weight"], "layer": int(layer),
            "cache tensor": kind, "mean allocated rank": values["mean"],
        })
allocation_table = pd.DataFrame(allocation_rows)
display(allocation_table)

hybrid = allocation_table[allocation_table["answer weight"] == 0.5]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
for axis, rank in zip(axes, (48, 64, 80)):
    current = hybrid[hybrid["baseline rank"] == rank]
    for offset, kind in ((-width / 2, "key"), (width / 2, "value")):
        rows = current[current["cache tensor"] == kind].sort_values("layer")
        axis.bar(layers + offset, rows["mean allocated rank"], width=width,
                 color=COLORS[kind], label=kind)
    axis.set(title=f"ordinary-xKV rank-{rank} bit budget",
             xlabel="transformer block", xticks=layers)
axes[0].set_ylabel("mean allocated component rank")
axes[0].legend()
plt.show()
''')

md("### Does adaptive allocation improve fidelity and answer NLL?")
code(r'''
screen_rows = []
for record in summary["fresh_screen"]["candidates"]:
    screen_rows.append({
        "candidate": record["name"], "baseline rank": record["rank"],
        "utility blend": WEIGHT_LABELS[record["answer_weight"]],
        "compression ratio": record["modelled_compression_ratio"],
        "first-token fidelity": record["first_token_fidelity"],
        "KL from dense": record["mean_first_token_kl_from_dense"],
        "mean NLL advantage": record["mean_nll_advantage"],
        "NLL lower": record["nll_bootstrap_95ci"][0],
        "NLL upper": record["nll_bootstrap_95ci"][1],
        "bits ratio": record["adaptive_to_ordinary_cache_bits_ratio"],
        "teacher preeligible": record["teacher_preeligible"],
        "screen passed": record["screen_passed"],
    })
screen_table = pd.DataFrame(screen_rows).sort_values(
    ["baseline rank", "utility blend"]
)
display(screen_table)

rank_colors = {48: "#8a8a8a", 64: "#b65f24", 80: "#315f8c"}
blend_markers = {"reconstruction": "o", "hybrid": "s", "answer-Fisher": "D"}
fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
for _, row in screen_table.iterrows():
    color = rank_colors[row["baseline rank"]]
    ax.scatter(row["compression ratio"], row["first-token fidelity"],
               color=color, marker=blend_markers[row["utility blend"]], s=75,
               facecolors=color if row["screen passed"] else "none", linewidths=1.5)
    ax.annotate(row["candidate"].replace("adaptive_kv_", ""),
                (row["compression ratio"], row["first-token fidelity"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.axhline(0.95, color="#222222", linestyle="--", linewidth=1,
           label="95% fidelity threshold")
ax.set(title="Fresh-screen fidelity at exactly matched modeled storage",
       xlabel="modeled cache compression ratio",
       ylabel="dense first-token top-1 agreement")
ax.legend(loc="best")
plt.show()
''')

md("### Paired answer-NLL evidence versus ordinary xKV")
code(r'''
ordered = screen_table.sort_values("NLL lower")
fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
y = np.arange(len(ordered))
for index, (_, row) in enumerate(ordered.iterrows()):
    mean, lower, upper = row["mean NLL advantage"], row["NLL lower"], row["NLL upper"]
    ax.errorbar(mean, y[index], xerr=[[mean - lower], [upper - mean]],
                fmt=blend_markers[row["utility blend"]], capsize=4,
                color=rank_colors[row["baseline rank"]])
ax.axvline(0, color="#222222", linewidth=1)
ax.set(title="Adaptive-allocation advantage with paired 95% bootstrap intervals",
       xlabel="ordinary xKV NLL − adaptive NLL (positive favors adaptive)",
       yticks=y, yticklabels=ordered["candidate"])
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
        "bits ratio": final["adaptive_to_ordinary_cache_bits_ratio"],
        "first-token fidelity": final["teacher_forced"]["selected"][
            "dense_first_token_top1_agreement"
        ],
        "dense accuracy": final["generation"]["dense"]["accuracy"],
        "ordinary accuracy": final["generation"]["ordinary_xkv"]["accuracy"],
        "selected accuracy": final["generation"]["selected"]["accuracy"],
        "passed": final["gate"]["passed"],
    }]))
    display(pd.DataFrame([final["gate"]]))
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
assert (output / "adaptive_kv_allocation.pt").is_file()
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
