"""Build the Kaggle notebook for the fixed per-head K/V subspace experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_fixed_basis_kv.ipynb"
RUN_COMMIT = "9d44081921157702457c54843fea94b27842ae64"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does CODI's KV cache have a fixed per-head subspace?

## tl;dr

The completed endpoint work found a **fixed** low-dimensional answer subspace at
CODI's final hidden state: one basis, fitted once on training questions, reused by
every question. This notebook asks the same question inside the KV cache.

- Fit one orthonormal basis per (layer, kind, head) from the second moments of the
  vectors that head actually writes, on 1,024 GSM8K-train questions.
- Project every key and value onto the leading `r` directions of its own basis.
- Sweep `r` in `8, 16, 24, 32, 40, 48` against a head width of 64, on a disjoint
  256-question selection split.
- Freeze one operating point there, then read the full 1,319-question test once.

This is **not** the xKV question. xKV factorizes each request's cache matrix, so
every request stores a token factor *and its own decoder* — which is why the
previous chain measured only 1.48x at CODI's ~100-token cache length. A fixed
basis is model metadata, so a rank-`r` component stores `r` coordinates per token
instead of 64, a ratio of `head_dim / rank` that does not decay at short context:
rank 32 is 2x, rank 16 is 4x, rank 8 is 8x.

Because GPT-2 has no rotary embedding on its keys, the projection folds into
attention exactly: `q·(UUᵀk) = (Uᵀq)·(Uᵀk)`. The measured arm **is** an
`r`-dimensional per-head attention, executed unfused. Wall clock and allocated
memory are **not measured** here.
""")

md(r"""
## Context & Methods

For each (layer, kind, head) the calibration pass accumulates

\[
M=\sum_t x_t x_t^\top,\qquad x_t\in\mathbb R^{64},
\]

over every cached vector that head writes during the released forced-cue
generation path. Row types are recorded separately — `question` (prompt plus
`<bot>`), `latent` (the six continuous thoughts), `cue` (the forced answer cue),
and `answer` (decoded tokens to EOS) — so the diagnostic can ask whether the
latent workspace occupies a different subspace than question tokens.

### Arms

- `uniform_r{r}` — every component gets rank `r`.
- `energy_r{r}` — same total budget, allocated greedily by eigenvalue. Eigenvalues
  are non-increasing, so greedy allocation is exactly optimal for retained energy.
- `random_s{seed}_r{r}` — seeded random orthonormal bases at identical rank and
  identical storage. **This is the control that matters.**
- `key_only_r{r}` / `value_only_r{r}` — compress only keys or only values.

### Frozen decision rules

The operating point is chosen on the **selection split only**: smallest rank,
preferring `uniform` over `energy` at equal rank, retaining ≥98% of dense exact
match and ≥95% dense first-token agreement. If no rank passes, the run stops and
the test set is not read.

The locked final gate requires all four:

1. paired exact-match difference versus dense has a 95% lower bound above −2 points;
2. the selected arm beats **every** random basis at equal rank with a positive
   paired lower bound;
3. dense first-token top-1 agreement ≥95%;
4. point retention ≥98% of dense exact match.

The primary outcome is **paired exact match**, not answer NLL. The previous chain
gated on NLL differences of a few thousandths, which at n=551 needed roughly 1,300
paired questions to detect.

### Key assumptions

- Bases are per head. A 768-wide basis would mix heads and could not fold into
  per-head dot products.
- No test question enters fitting or rank selection; the test set is read once,
  after the operating point is frozen.
- Compression is `head_dim / rank` per component. Fixed bases are model metadata
  and are reported separately as `basis_parameters`.
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
OUTPUT_DIR = "/kaggle/working/codi_fixed_basis_kv"

BASIS_FIT_EXAMPLES = 1024
SELECTION_EXAMPLES = 256
SAMPLING_SEED = 20260921
RANK_GRID = "8,16,24,32,40,48"
RANDOM_BASIS_SEEDS = "20260921,20260922"

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
    "tests/test_fixed_basis_kv.py",
    "tests/test_fixed_basis_kv_runner.py",
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
print("This experiment needs no predecessor artifact from the xKV chain.")
''')

md("## Results — calibration, selection sweep, and one locked test read")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_fixed_basis_kv.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--output-dir", OUTPUT_DIR,
    "--basis-fit-examples", str(BASIS_FIT_EXAMPLES),
    "--selection-examples", str(SELECTION_EXAMPLES),
    "--sampling-seed", str(SAMPLING_SEED),
    "--rank-grid", RANK_GRID,
    "--random-basis-seeds", RANDOM_BASIS_SEEDS,
    "--minimum-retention", "0.98",
    "--minimum-first-token-fidelity", "0.95",
    "--accuracy-noninferiority-margin", "0.02",
    "--bootstrap-samples", "10000",
    "--batch-size", "16", "--generation-batch-size", "16",
    "--max-new-tokens", "64",
    "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("### How low-dimensional is each head's cache subspace?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"key": "#315f8c", "value": "#b65f24"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

head_dim = summary["geometry"]["head_dim"]
layers = np.arange(summary["geometry"]["layers"])
energy = summary["calibration"]["energy_ranks"]

rows = []
for label, table in energy.items():
    target = label.replace("rank_for_", "").replace("_energy", "")
    for component, value in table.items():
        parts = component.split("_")
        rows.append({
            "energy target": target, "layer": int(parts[1]),
            "cache tensor": parts[2], "mean rank": value,
        })
energy_table = pd.DataFrame(rows)
display(energy_table.pivot_table(
    index=["layer", "cache tensor"], columns="energy target", values="mean rank"
))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
width = 0.38
for axis, target in zip(axes, sorted(energy_table["energy target"].unique())):
    current = energy_table[energy_table["energy target"] == target]
    for offset, kind in ((-width / 2, "key"), (width / 2, "value")):
        values = current[current["cache tensor"] == kind].sort_values("layer")
        axis.bar(layers + offset, values["mean rank"], width=width,
                 color=COLORS[kind], label=kind)
    axis.axhline(head_dim, color="#222222", linestyle="--", linewidth=1)
    axis.set(title=f"rank needed for {target} energy", xlabel="transformer block",
             xticks=layers)
axes[0].set_ylabel(f"mean rank (head width {head_dim})")
axes[0].legend()
plt.show()
''')

md("### Do the latent workspace rows use the same subspace as question tokens?")
code(r'''
capture = summary["calibration"]["row_type_capture"]
if not capture:
    print("Row-type capture was not computed (a row type had no observations).")
else:
    rows = []
    for comparison, by_rank in capture.items():
        for rank, table in by_rank.items():
            expectation = table["isotropic_expectation"]
            for component, value in table.items():
                if component == "isotropic_expectation":
                    continue
                parts = component.split("_")
                rows.append({
                    "comparison": comparison, "rank": int(rank), "layer": int(parts[1]),
                    "cache tensor": parts[2], "capture": value,
                    "isotropic expectation": expectation,
                })
    capture_table = pd.DataFrame(rows)
    display(capture_table.pivot_table(
        index=["comparison", "rank"], values=["capture", "isotropic expectation"]
    ))

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    for comparison, group in capture_table.groupby("comparison"):
        means = group.groupby("rank")["capture"].mean()
        ax.plot(means.index, means.values, marker="o", label=comparison)
    reference = capture_table.groupby("rank")["isotropic expectation"].first()
    ax.plot(reference.index, reference.values, color="#222222", linestyle="--",
            label="isotropic expectation")
    ax.set(title="Subspace agreement between row types",
           xlabel="rank", ylabel="captured fraction of the question subspace")
    ax.legend()
    plt.show()
''')

md("### Selection sweep: fixed bases versus matched random bases")
code(r'''
selection_rows = []
for name, record in summary["selection"]["arms"].items():
    selection_rows.append({
        "arm": name, "family": record["family"], "rank": record["rank"],
        "accuracy": record["accuracy"],
        "retained": record["accuracy_retained_fraction"],
        "first-token fidelity": record["dense_first_token_top1_agreement"],
        "sequence agreement": record["exact_sequence_agreement"],
        "compression": record["storage"]["compression_ratio"],
    })
selection_table = pd.DataFrame(selection_rows).sort_values(["rank", "family"])
display(selection_table)
print("dense selection accuracy", summary["selection"]["dense"]["accuracy"])

FAMILY_STYLE = {
    "uniform": ("#315f8c", "o"), "energy": ("#b65f24", "s"), "random": ("#8a8a8a", "x"),
}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
for family, group in selection_table.groupby("family"):
    color, marker = FAMILY_STYLE[family]
    aggregated = group.groupby("rank")[["retained", "first-token fidelity"]].mean()
    axes[0].plot(aggregated.index, aggregated["retained"], marker=marker,
                 color=color, label=family)
    axes[1].plot(aggregated.index, aggregated["first-token fidelity"], marker=marker,
                 color=color, label=family)
axes[0].axhline(0.98, color="#222222", linestyle="--", linewidth=1, label="98% gate")
axes[0].set(title="Accuracy retained versus dense", xlabel="rank per head",
            ylabel="fraction of dense exact match")
axes[1].axhline(0.95, color="#222222", linestyle="--", linewidth=1, label="95% gate")
axes[1].set(title="Dense first-token agreement", xlabel="rank per head",
            ylabel="top-1 agreement")
for axis in axes:
    axis.legend()
plt.show()

selected = summary["selected"]
print("Frozen operating point:", selected)
''')

md("### The locked final read on the complete GSM8K test set")
code(r'''
final = summary["final"]
if final is None:
    print("The final evaluation did not run.")
else:
    final_rows = []
    for name, record in final["arms"].items():
        final_rows.append({
            "arm": name, "family": record["family"], "rank": record["rank"],
            "correct": record["correct"], "accuracy": record["accuracy"],
            "retained": record["accuracy_retained_fraction"],
            "first-token fidelity": record["dense_first_token_top1_agreement"],
            "compression": record["storage"]["compression_ratio"],
            "vs dense lower": record["accuracy_difference_vs_dense_95ci"][0],
            "vs dense upper": record["accuracy_difference_vs_dense_95ci"][1],
        })
    final_table = pd.DataFrame(final_rows).sort_values(["rank", "family"])
    display(final_table)
    print("dense test accuracy", final["dense"]["accuracy"],
          f"({final['dense']['correct']}/{final['dense']['examples']})")

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    ordered = final_table.sort_values("vs dense lower")
    y = np.arange(len(ordered))
    for index, (_, row) in enumerate(ordered.iterrows()):
        centre = row["accuracy"] - final["dense"]["accuracy"]
        lower, upper = row["vs dense lower"], row["vs dense upper"]
        colour = "#b65f24" if row["arm"] == final["selected_arm"] else "#315f8c"
        if row["family"] == "random":
            colour = "#8a8a8a"
        ax.errorbar(centre, y[index], xerr=[[centre - lower], [upper - centre]],
                    fmt="o", capsize=4, color=colour)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.axvline(-0.02, color="#b02418", linestyle="--", linewidth=1,
               label="−2 point non-inferiority margin")
    ax.set(title="Locked test accuracy difference from dense CODI",
           xlabel="accuracy difference (paired 95% bootstrap)",
           yticks=y, yticklabels=ordered["arm"])
    ax.legend()
    plt.show()

    display(pd.DataFrame([final["comparisons"]["selected_vs_random"]]))
    print("energy − uniform at the selected rank:",
          final["comparisons"]["energy_minus_uniform_at_selected_rank"])
    print("key-only − value-only at the selected rank:",
          final["comparisons"]["key_only_minus_value_only_at_selected_rank"])
    display(pd.DataFrame([final["gate"]]))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print("Selection gate:", summary["decision"]["selection_passed"])
print("Locked-final gate:", summary["decision"]["final_passed"])
print("Frozen operating point:", summary["selected"])
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("### Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
assert (output / "summary.json").is_file()
assert (output / "fixed_basis_kv.pt").is_file()
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
