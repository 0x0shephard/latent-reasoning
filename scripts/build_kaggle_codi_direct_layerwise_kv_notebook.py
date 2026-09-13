"""Build the corrected independent-layer discovery and KV compression notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_direct_layerwise_kv_subspaces.ipynb"
nb = nbf.v4.new_notebook(); cells = []
def md(value): cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value): cells.append(nbf.v4.new_code_cell(value.strip()))

md(r"""
# CODI: discover each layer's answer subspace, then protect it in the KV cache

## Goal

This corrected experiment does not begin with final-layer U28 and transport it
backward. For each of CODI's 12 transformer blocks it independently:

1. collects the normalized state entering (W_K,W_V) at all six latent passes;
2. computes that layer's covariance eigensystem on a fit split;
3. ranks its own eigenvectors using gold-answer-NLL gradients on a disjoint split;
4. selects 28 layer-local directions;
5. maps them through the effective LoRA-aware (W_K,W_V);
6. removes or retains them directly as each of the six latent K/V entries is cached;
7. compares against four rank- and energy-matched random controls;
8. if a contiguous causal layer group passes, compares ordinary xKV-style SVD with
   a protected-core factorization at the same rank/storage budget.

The primary causal outcome is the change in gold-token NLL relative to random
controls, with a paired bootstrap interval. First-token agreement and full-answer
accuracy are secondary outcomes. This distinction avoids the insensitive binary-only
test that produced the previous flat plot.
""")

md("## Context & Methods — key assumptions")
md(r"""
- The layer state is `ln_1` output because this is exactly what GPT-2 projects into
  keys and values.
- Covariance fitting, gradient selection, and causal validation use disjoint GSM8K
  training questions; GSM8K test remains untouched until the compression comparison.
- A fixed rank 28 makes the layerwise comparison match the original U28 finding.
  The notebook records how many directions actually satisfy the split-stability gate;
  it does not silently call all selected directions validated.
- KV intervention covers latent positions 0–5. It changes the stored cache and hence
  later latent passes—not merely the final answer-cue activation.
- Structurally disconnected layer/position activations receive a zero gradient and
  are disclosed in a 12×6 connectivity table; they are never silently discarded.
- Cache factorization quality uses dense reconstruction for stock Transformers.
  Factor memory is modelled; end-to-end memory and latency require a fused kernel.
""")

md("## Data and setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "5bfff573cf7926d4210664e63d43a202b3645c1b"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
DISCOVERY_OUTPUT = "/kaggle/working/codi_direct_layerwise_kv"
COMPRESSION_OUTPUT = "/kaggle/working/codi_direct_layerwise_kv_compression"

FIT_EXAMPLES = 1024
SELECTION_EXAMPLES = 512
CAUSAL_EXAMPLES = 64
RANK = 28
RANDOM_CONTROLS = 4
RUN_COMPRESSION_IF_GATE_PASSES = True
ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION = False

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1"); os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists(): subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
print("code commit", CODE_COMMIT)
''')

md("### Install the checkpoint-compatible environment and run source tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.4",
                "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub==0.32.4", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available\ntry:\n print(is_torchao_available())\nexcept ImportError:\n print('incompatible')"], capture_output=True, text=True)
if "incompatible" in probe.stdout: subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_direct_layerwise_kv.py", "tests/test_causal_xkv.py",
                "tests/test_official_codi_layerwise.py"], check=True)
''')

md("### Resolve the completed official CODI reproduction")
code(r'''
def discover(explicit, suffix):
    if explicit:
        assert pathlib.Path(explicit).is_file(); return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    assert matches, f"Attach the dataset containing {suffix}"
    return sorted(matches, key=lambda x: (len(pathlib.Path(x).parts), x))[0]
REPRODUCTION_SUMMARY = discover(REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
print(REPRODUCTION_SUMMARY)
''')

md("## Results — independent discovery and direct latent-KV causality")
code(r'''
command = [sys.executable, "-u", "scripts/run_codi_direct_layerwise_kv_discovery.py",
    "--config", "configs/official_codi_gpt2.yaml", "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--output-dir", DISCOVERY_OUTPUT, "--fit-examples", str(FIT_EXAMPLES),
    "--select-examples", str(SELECTION_EXAMPLES), "--causal-examples", str(CAUSAL_EXAMPLES),
    "--rank", str(RANK), "--random-controls", str(RANDOM_CONTROLS),
    "--fit-batch-size", "16", "--selection-batch-size", "4", "--causal-batch-size", "16",
    "--precision", "float32", "--device", "cuda"]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(DISCOVERY_OUTPUT) / "summary.json").read_text())
print(json.dumps({"dense": summary["dense_first_token"], "gate": summary["gate"]}, indent=2))
print("gradient connectivity fraction [layer][latent position]")
print(summary["gradient_connectivity_fraction"])
''')

md("### Layer-local directions: stability, variance and causal specificity")
code(r'''
import matplotlib.pyplot as plt
layers = list(range(12))
stable = [summary["causal_screen"][f"layer_{layer:02d}"]["stable_positive_direction_count"] for layer in layers]
specificity = [summary["causal_screen"][f"layer_{layer:02d}"]["specificity_mean_nll_delta"] for layer in layers]
lower = [summary["causal_screen"][f"layer_{layer:02d}"]["specificity_bootstrap_95ci"][0] for layer in layers]
upper = [summary["causal_screen"][f"layer_{layer:02d}"]["specificity_bootstrap_95ci"][1] for layer in layers]
retention = [summary["causal_screen"][f"layer_{layer:02d}"]["retain"]["dense_top1_agreement"] for layer in layers]
variance = [row["selected_variance_fraction"] for row in summary["layer_direction_summary"]]

fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
axes[0].bar(layers, stable, color="#315f8c"); axes[0].axhline(RANK, color="black", linestyle="--")
axes[0].set(title="Split-stable answer-sensitive directions", xlabel="transformer block", ylabel="count")
axes[1].errorbar(layers, specificity,
    yerr=[[m-l for m,l in zip(specificity,lower)], [u-m for u,m in zip(upper,specificity)]],
    marker="o", capsize=3, color="#b65f24"); axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(title="Learned removal minus matched-random removal", xlabel="transformer block",
            ylabel="gold-token NLL difference, paired 95% CI")
axes[2].plot(layers, retention, marker="o", label="retain-only dense top-1 agreement", color="#315f8c")
axes[2].plot(layers, variance, marker="s", label="variance fraction in selected directions", color="#b65f24")
axes[2].set(title="Sufficiency and covariance energy", xlabel="transformer block", ylim=(0, 1.02)); axes[2].legend()
fig.savefig(pathlib.Path(DISCOVERY_OUTPUT) / "direct_layerwise_kv_results.png", dpi=180, bbox_inches="tight")
plt.show()
''')

md("### Inspect which eigenvectors were selected in each layer")
code(r'''
for row in summary["layer_direction_summary"]:
    print(f"layer {row['layer']:02d}: stable={row['stable_positive_direction_count']:3d}, "
          f"selected variance={row['selected_variance_fraction']:.4f}, "
          f"PC indices={row['selected_pc_indices']}")
''')

md("## Results — equal-budget KV compression, only after the causal gate")
code(r'''
gate = summary["gate"]
should_run = RUN_COMPRESSION_IF_GATE_PASSES and (gate["passed"] or ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION)
if should_run:
    command = [sys.executable, "-u", "scripts/run_codi_causal_xkv.py",
        "--config", "configs/official_codi_gpt2.yaml", "--reproduction-summary", REPRODUCTION_SUMMARY,
        "--layerwise-artifact", str(pathlib.Path(DISCOVERY_OUTPUT) / "direct_layerwise_kv.pt"),
        "--output-dir", COMPRESSION_OUTPUT, "--examples", "256", "--max-new-tokens", "64",
        "--ranks", "28,32,48", "--batch-size", "8", "--precision", "float32", "--device", "cuda"]
    if not gate["passed"]: command.append("--allow-unconfirmed")
    subprocess.run(command, check=True)
    compression = json.loads((pathlib.Path(COMPRESSION_OUTPUT) / "summary.json").read_text())
    for name, record in compression["results"].items():
        print(name, record)
else:
    print("Compression stage correctly stopped: no contiguous causal layer group passed.")
    print("Set ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION=True only for a labelled exploratory run.")
''')

md("## Takeaways")
code(r'''
print("Causal gate:", "PASS" if gate["passed"] else "FAIL")
print("Passing layers:", gate["passing_layers"])
print("Selected contiguous group:", gate["selected_contiguous_layers"])
print("Discovery artifact:", pathlib.Path(DISCOVERY_OUTPUT) / "direct_layerwise_kv.pt")
if gate["passed"]:
    print("The protected-xKV comparison is eligible for confirmatory interpretation.")
else:
    print("Layer-local subspaces were discovered, but they were not shown to be a specific causal KV core.")
''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
OUTPUT.parent.mkdir(parents=True, exist_ok=True); nbf.write(nb, OUTPUT); print(OUTPUT)
