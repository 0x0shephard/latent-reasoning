"""Build the Kaggle notebook for confirmed task-aware protected residual xKV."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_task_aware_protected_xkv.ipynb"
RUN_COMMIT = "__RUN_COMMIT__"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can causal, attention-aware residuals improve xKV for CODI?

## tl;dr

This notebook runs the next confirmatory experiment, then measures the resulting
quality/storage trade-off. It does **not** assume that the earlier exploratory
layer-2 and layer-3 findings are true.

1. Reconstruct every earlier train split and the previous layer-11 confirmation.
2. Confirm frozen layer-2 `(K8,V48)` and layer-3 `(K16,V64)` hypotheses on the
   next 512 unseen GSM8K training questions.
3. Fit a fixed answer-Fisher weighting on a separate 512-question calibration
   split. Exact K/V gradients include the query, softmax-attention and value paths.
4. On untouched GSM8K test questions, compare per-layer SVD, ordinary grouped
   xKV, answer-Fisher xKV, sparse causal-residual xKV, the full adaptive method,
   20 matched-random protected residuals, and INT8/INT4 controls.

The reference path reconstructs dense caches. It can establish quality and
modeled storage, but not production latency.
""")

md(r"""
## Context & Methods

### Why the protected residual is sparse

The confirmed early-layer candidates contain many feature directions. Inserting
all of them into the xKV token factor would consume 136 rank units before xKV had
room for anything else. Instead, the notebook applies ordinary xKV to the bulk
cache and stores exact correction coordinates only for CODI's six continuous
latent rows:

\[
\widehat X_{\text{final}}
=\widehat X_{\text{xKV}}
+M_{\text{latent}}(X-\widehat X_{\text{xKV}})UU^T.
\]

The fixed basis `U` is model metadata. Each request stores only the small latent
coordinate residual `(X-X_hat)U`.

### Answer-Fisher weighting

For each K/V feature, calibration estimates

\[
s_j=\sqrt{\mathbb{E}[(\partial\mathcal L/\partial X_j)^2]}.
\]

Weighted SVD minimizes `||(X-X_hat)diag(s)||_F`. This is a deployable
query/attention-aware proxy because the fixed gradients pass through QK scores,
softmax and value aggregation. It is not presented as an exact implementation of
the KQ-SVD paper.

### Confirmatory gates

Layers 2 and 3 are frozen co-primary hypotheses. Each must have:

- a positive Bonferroni-corrected 95% interval against energy-matched random bases;
- positive specificity in at least three of four disjoint folds;
- at least 95% retained first-token agreement and no more than +0.10 answer NLL.

Layer 11 is imported only from the previous disjoint confirmation. If an early
layer fails, it is excluded from the protected method and the strict combined
claim is marked failed.
""")

md("## Data and setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
SOURCE_ARTIFACT_INPUT = ""
PREVIOUS_CONFIRMATION_SUMMARY_INPUT = ""
PREVIOUS_CONFIRMATION_ARTIFACT_INPUT = ""

DISCOVERY_OUTPUT = "/kaggle/working/codi_direct_cache_source"
PREVIOUS_CONFIRM_OUTPUT = "/kaggle/working/codi_native_kv_confirmation"
OUTPUT_DIR = "/kaggle/working/codi_task_aware_protected_xkv"

RERUN_MISSING_PREDECESSORS = True
CONFIRM_EXAMPLES = 512
CALIBRATION_EXAMPLES = 512
TEST_EXAMPLES = 256
GENERATION_EXAMPLES = 128
RANKS = "16,32,48"
GENERATION_RANKS = "32,48"
FOCAL_RANK = 32
RANDOM_PROTECTION_CONTROLS = 20

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
assert RUN_COMMIT != "__RUN_COMMIT__", "Use the pinned commit printed with this notebook"
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
    "tests/test_task_aware_xkv.py",
    "tests/test_confirm_direct_cache_subspace.py",
    "tests/test_direct_cache_task_subspace.py",
    "tests/test_preanswer_kv_subspace.py",
    "tests/test_causal_xkv.py",
], check=True)
''')

md("### Resolve the official and predecessor artifacts")
code(r'''
def discover_file(explicit, suffix):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    return sorted(matches, key=lambda value: (len(pathlib.Path(value).parts), value))[0] if matches else None

def discover_json_contract(explicit, contract):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        assert json.loads(path.read_text()).get("contract") == contract
        return str(path)
    for candidate in sorted(glob.glob("/kaggle/input/**/summary.json", recursive=True)):
        try:
            if json.loads(pathlib.Path(candidate).read_text()).get("contract") == contract:
                return candidate
        except Exception:
            continue
    return None

REPRODUCTION_SUMMARY = discover_file(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
SOURCE_ARTIFACT = discover_file(SOURCE_ARTIFACT_INPUT, "direct_cache_task_subspaces.pt")
PREVIOUS_CONFIRMATION_SUMMARY = discover_json_contract(
    PREVIOUS_CONFIRMATION_SUMMARY_INPUT,
    "official_codi_frozen_native_kv_confirmation_and_compression_gate_v1",
)
PREVIOUS_CONFIRMATION_ARTIFACT = discover_file(
    PREVIOUS_CONFIRMATION_ARTIFACT_INPUT, "confirmed_native_kv_artifact.pt"
)
print("reproduction", REPRODUCTION_SUMMARY)
print("source", SOURCE_ARTIFACT or "will rerun")
print("previous summary", PREVIOUS_CONFIRMATION_SUMMARY or "will rerun")
print("previous artifact", PREVIOUS_CONFIRMATION_ARTIFACT or "will rerun")
''')

md("### Reproduce missing frozen predecessors without changing their specifications")
code(r'''
SOURCE_WAS_RERUN = SOURCE_ARTIFACT is None
if SOURCE_WAS_RERUN:
    assert RERUN_MISSING_PREDECESSORS
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

if (SOURCE_WAS_RERUN or PREVIOUS_CONFIRMATION_SUMMARY is None
        or PREVIOUS_CONFIRMATION_ARTIFACT is None):
    assert RERUN_MISSING_PREDECESSORS
    subprocess.run([
        sys.executable, "-u", "scripts/run_codi_confirm_and_compress_native_kv.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--source-artifact", SOURCE_ARTIFACT,
        "--output-dir", PREVIOUS_CONFIRM_OUTPUT,
        "--confirm-examples", "512", "--confirm-folds", "4",
        "--candidate-layers", "2,3,5,11", "--primary-layer", "11",
        "--batch-size", "16", "--gradient-batch-size", "4",
        "--precision", "float32", "--device", "cuda",
    ], check=True)
    PREVIOUS_CONFIRMATION_SUMMARY = str(pathlib.Path(PREVIOUS_CONFIRM_OUTPUT) / "summary.json")
    PREVIOUS_CONFIRMATION_ARTIFACT = str(
        pathlib.Path(PREVIOUS_CONFIRM_OUTPUT) / "confirmed_native_kv_artifact.pt"
    )
print("using source", SOURCE_ARTIFACT)
print("using previous confirmation", PREVIOUS_CONFIRMATION_SUMMARY)
''')

md("## Results — independent early-layer confirmation and compression benchmark")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_task_aware_protected_xkv.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--source-artifact", SOURCE_ARTIFACT,
    "--previous-confirmation-summary", PREVIOUS_CONFIRMATION_SUMMARY,
    "--previous-confirmation-artifact", PREVIOUS_CONFIRMATION_ARTIFACT,
    "--output-dir", OUTPUT_DIR,
    "--confirm-examples", str(CONFIRM_EXAMPLES),
    "--calibration-examples", str(CALIBRATION_EXAMPLES),
    "--test-examples", str(TEST_EXAMPLES),
    "--generation-examples", str(GENERATION_EXAMPLES),
    "--ranks", RANKS, "--generation-ranks", GENERATION_RANKS,
    "--focal-rank", str(FOCAL_RANK),
    "--random-protection-controls", str(RANDOM_PROTECTION_CONTROLS),
    "--batch-size", "16", "--gradient-batch-size", "4",
    "--generation-batch-size", "8", "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print("strict combined gate", summary["confirmation"]["strict_layers_2_3_11_gate_passed"])
print("included layers", summary["confirmation"]["included_layers"])
''')

md("### Did frozen layers 2 and 3 replicate?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"task": "#315f8c", "baseline": "#8a8a8a", "quant": "#b65f24"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 11})

confirmation_rows = []
for name, record in summary["confirmation"]["new_early_layers"].items():
    confirmation_rows.append({
        "layer": int(name.split("_")[-1]),
        "K rank": record["key_rank"], "V rank": record["value_rank"],
        "specificity": record["specificity_mean_full_answer_nll_delta"],
        "lower": record["familywise_95ci"][0], "upper": record["familywise_95ci"][1],
        "positive folds": record["positive_folds"],
        "retain agreement": record["task_retain"]["dense_first_token_top1_agreement"],
        "passed": record["passed"],
    })
confirmation_table = pd.DataFrame(confirmation_rows).sort_values("layer")
display(confirmation_table)
x = np.arange(len(confirmation_table)); y = confirmation_table["specificity"].to_numpy()
lower = confirmation_table["lower"].to_numpy(); upper = confirmation_table["upper"].to_numpy()
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.errorbar(x, y, yerr=np.vstack((y-lower, upper-y)), fmt="o", capsize=6,
            color=COLORS["task"], ecolor=COLORS["baseline"])
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Frozen early-layer effects on the next 512 unseen questions",
       xlabel="transformer block", ylabel="NLL damage beyond matched random",
       xticks=x, xticklabels=confirmation_table["layer"])
plt.show()
''')

md("### Make the paired same-rank decision against ordinary xKV")
code(r'''
paired_rows = []
for name, record in summary["paired_ordinary_xkv_comparisons"].items():
    paired_rows.append({
        "rank": record["rank"],
        "full NLL advantage": record["mean_full_method_nll_advantage"],
        "lower": record["paired_bootstrap_95ci"][0],
        "upper": record["paired_bootstrap_95ci"][1],
        "full / ordinary cache bits": record["full_to_ordinary_cache_bits_ratio"],
        "strict quality win": record["strict_quality_win"],
    })
paired_table = pd.DataFrame(paired_rows).sort_values("rank")
display(paired_table)
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
y = paired_table["full NLL advantage"].to_numpy()
lower = paired_table["lower"].to_numpy(); upper = paired_table["upper"].to_numpy()
ax.errorbar(paired_table["rank"], y, yerr=np.vstack((y-lower, upper-y)),
            fmt="s", capsize=6, color=COLORS["task"], ecolor=COLORS["baseline"])
ax.axhline(0, color="#222222", linewidth=1)
ax.set(title="Full method versus ordinary xKV on the same test questions",
       xlabel="shared xKV rank", ylabel="ordinary xKV NLL minus full-method NLL")
plt.show()
''')

md("### Inspect the fixed calibration utility used for adaptive group ranks")
code(r'''
utility_table = pd.DataFrame([
    {"layer group": group, "answer-Fisher utility": value}
    for group, value in summary["compression_calibration"]["group_utilities"].items()
]).sort_values("layer group")
display(utility_table)
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
ax.bar(utility_table["layer group"], utility_table["answer-Fisher utility"],
       color=COLORS["task"])
ax.set(title="Calibration-fitted K/V gradient utility by xKV layer group",
       xlabel="grouped transformer blocks", ylabel="mean squared gradient-RMS")
plt.show()

adaptive_ranks = summary["results"][f"full_task_aware_xkv_r{FOCAL_RANK}"][
    "teacher_forced"
]["cache"]["effective_rank_by_group"]
display(pd.DataFrame([
    {"layer group": group, **record} for group, record in adaptive_ranks.items()
]).sort_values("layer group"))
''')

md("### Compare teacher-forced fidelity at actual modeled storage")
code(r'''
quality_rows = []
for name, record in summary["results"].items():
    teacher = record["teacher_forced"]
    cache = teacher.get("cache")
    if cache is None:
        continue
    quality_rows.append({
        "arm": name,
        "compression ratio": cache["modelled_compression_ratio"],
        "answer NLL delta": teacher["mean_full_answer_nll_delta"],
        "first-token agreement": teacher["dense_first_token_top1_agreement"],
        "top-5 overlap": teacher["first_token_top5_overlap"],
        "KL from dense": teacher["mean_first_token_kl_from_dense"],
    })
quality_table = pd.DataFrame(quality_rows).sort_values(
    ["compression ratio", "answer NLL delta"], ascending=[True, True]
)
display(quality_table)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for _, row in quality_table.iterrows():
    is_full = row["arm"].startswith("full_task_aware")
    color = COLORS["task"] if is_full else COLORS["quant"] if "int" in row["arm"] else COLORS["baseline"]
    marker = "s" if is_full else "o"
    axes[0].scatter(row["compression ratio"], row["answer NLL delta"],
                    color=color, marker=marker, s=60)
    axes[1].scatter(row["compression ratio"], row["first-token agreement"],
                    color=color, marker=marker, s=60)
axes[0].axhline(0, color="#222222", linewidth=1)
axes[0].set(title="Answer loss versus modeled cache storage",
            xlabel="compression ratio (higher is smaller)", ylabel="answer NLL change")
axes[1].set(title="First-token fidelity versus modeled cache storage",
            xlabel="compression ratio (higher is smaller)", ylabel="agreement with dense CODI")
plt.show()
''')

md("### Does causal protection beat 20 matched-random protected residuals?")
code(r'''
random_comparison = summary["random_protected_comparison"]
random_table = pd.DataFrame(random_comparison["results"]).T
display(random_table[["mean_full_answer_nll_delta", "dense_first_token_top1_agreement",
                      "mean_first_token_kl_from_dense"]])
task_value = summary["results"][f"full_task_aware_xkv_r{FOCAL_RANK}"][
    "teacher_forced"
]["mean_full_answer_nll_delta"]
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.hist(random_table["mean_full_answer_nll_delta"], bins=10,
        color="#d0d0d0", edgecolor="#555555", label="matched-random protected residuals")
ax.axvline(task_value, color=COLORS["task"], linewidth=2,
           label="causally protected residual")
ax.set(title=f"Rank-{FOCAL_RANK}: causal residual against 20 equal-rank random controls",
       xlabel="mean answer NLL change", ylabel="number of random controls")
ax.legend(); plt.show()
print({key: random_comparison[key] for key in (
    "mean_task_nll_advantage_over_random", "bootstrap_95ci", "task_beats_random_fraction"
)})
''')

md("### Check full greedy-generation quality")
code(r'''
generation_rows = []
for name, record in summary["results"].items():
    generation = record.get("generation")
    if generation is None:
        continue
    cache = generation.get("cache", {})
    generation_rows.append({
        "arm": name,
        "compression ratio": cache.get("modelled_compression_ratio", 1.0),
        "accuracy": generation["accuracy"],
        "accuracy retained": generation.get("accuracy_retained_fraction", 1.0),
        "exact sequence agreement": generation.get("exact_sequence_agreement", 1.0),
    })
generation_table = pd.DataFrame(generation_rows).sort_values("compression ratio")
display(generation_table)
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
for _, row in generation_table.iterrows():
    is_full = row["arm"].startswith("full_task_aware")
    ax.scatter(row["compression ratio"], row["accuracy retained"],
               color=COLORS["task"] if is_full else COLORS["baseline"],
               marker="s" if is_full else "o", s=65)
    ax.annotate(row["arm"], (row["compression ratio"], row["accuracy retained"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.axhline(1, color="#222222", linewidth=1, linestyle="--")
ax.set(title=f"Greedy GSM8K generation on {GENERATION_EXAMPLES} untouched test questions",
       xlabel="modeled pre-answer cache compression ratio",
       ylabel="accuracy retained relative to dense CODI")
plt.show()
''')

md("## Takeaways")
code(r'''
strict = summary["confirmation"]["strict_layers_2_3_11_gate_passed"]
comparison = summary["random_protected_comparison"]
print("Strict layers 2+3+11 confirmation:", "PASS" if strict else "FAIL")
print("Protected layers actually used:", summary["confirmation"]["included_layers"])
print("Causal residual NLL advantage over random:",
      comparison["mean_task_nll_advantage_over_random"])
print("Advantage 95% interval:", comparison["bootstrap_95ci"])
print("Fraction of random controls beaten:", comparison["task_beats_random_fraction"])
print("Paired same-rank xKV decisions:")
for name, record in summary["paired_ordinary_xkv_comparisons"].items():
    print(name, record)
print("\nDo not claim xKV improvement unless the causal method beats ordinary xKV and the")
print("20-control interval is positive at a comparable modeled storage ratio.")
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
import shutil
archive = shutil.make_archive(
    "/kaggle/working/codi_task_aware_protected_xkv",
    "zip",
    root_dir=OUTPUT_DIR,
)
print(archive)
print("artifact", pathlib.Path(OUTPUT_DIR) / "task_aware_protected_xkv.pt")
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
