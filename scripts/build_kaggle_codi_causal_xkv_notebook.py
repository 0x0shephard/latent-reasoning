"""Build the Kaggle notebook comparing xKV with a transported-U28 protected core."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

OUTPUT = ROOT / "notebooks" / "kaggle_codi_causal_xkv.ipynb"
nb = nbf.v4.new_notebook(); cells = []
def md(value): cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value): cells.append(nbf.v4.new_code_cell(value.strip()))

md(r"""
# Causal-U28 protected xKV for CODI

This is experiment 2 of 2. It asks a narrow question: at the **same total cache
factor rank**, does reserving 28 feature directions for the causally validated
answer channel preserve more CODI quality than ordinary cross-layer SVD?

Arms are dense CODI, independent per-layer SVD, xKV-style cross-layer SVD,
random-protected xKV, and U28-protected xKV. The contiguous layer run validated by
experiment 1 is kept as one group; the remaining layers are placed in groups of at
most four. Within each xKV group, ordinary, random-protected, and U28-protected arms
use the same total rank and factor-storage budget. Per-layer SVD is a separately
measured baseline and need not have the same storage.

The stock Transformers model cannot consume a factorized cache, so the quality test
factorizes after latent pass 6 and reconstructs the cache before decoding. The notebook
reports **modelled factor memory**, not actual runtime memory. A separate operator-only
benchmark evaluates attention directly from factors. End-to-end speed is out of scope
until that operator is fused into an attention kernel.
""")

md("## 1. Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "6e172775ead54be5f8917ee403e893e9ed42932d"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
LAYERWISE_ARTIFACT_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_causal_xkv"
EXAMPLES = 256
MAX_NEW_TOKENS = 64
RANKS = "28,32,48"
BATCH_SIZE = 8
ALLOW_UNCONFIRMED_EXPLORATION = False

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1"); os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists(): subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
''')

md("## 2. Environment and source tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.4",
                "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub==0.32.4", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available\ntry:\n print(is_torchao_available())\nexcept ImportError:\n print('incompatible')"], capture_output=True, text=True)
if "incompatible" in probe.stdout: subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_causal_xkv.py",
                "tests/test_layerwise_u28.py", "tests/test_official_codi_layerwise.py"], check=True)
''')

md("## 3. Resolve inputs and enforce the layerwise causal gate")
code(r'''
def discover(explicit, suffix):
    if explicit:
        assert pathlib.Path(explicit).is_file(); return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    assert matches, f"Attach a dataset containing {suffix}"
    return sorted(matches, key=lambda x: (len(pathlib.Path(x).parts), x))[0]
REPRODUCTION_SUMMARY = discover(REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
LAYERWISE_ARTIFACT = discover(LAYERWISE_ARTIFACT_INPUT, "layerwise_u28.pt")
import torch
layerwise = torch.load(LAYERWISE_ARTIFACT, map_location="cpu", weights_only=False)
print(json.dumps(layerwise["gate"], indent=2))
if not ALLOW_UNCONFIRMED_EXPLORATION:
    assert layerwise["gate"]["passed"], "Experiment 1 did not validate a causal layer group."
''')

md("## 4. Run equal-rank quality comparisons")
code(r'''
command = [sys.executable, "-u", "scripts/run_codi_causal_xkv.py",
    "--config", "configs/official_codi_gpt2.yaml", "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--layerwise-artifact", LAYERWISE_ARTIFACT, "--output-dir", OUTPUT_DIR,
    "--examples", str(EXAMPLES), "--max-new-tokens", str(MAX_NEW_TOKENS),
    "--ranks", RANKS, "--batch-size", str(BATCH_SIZE), "--precision", "float32", "--device", "cuda"]
if ALLOW_UNCONFIRMED_EXPLORATION: command.append("--allow-unconfirmed")
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
''')

md("## 5. Quality–memory frontier")
code(r'''
import matplotlib.pyplot as plt
records = []
for name, value in summary["results"].items():
    if name == "dense": continue
    records.append((name, value["cache"]["modelled_compression_ratio"],
                    value["accuracy_retained_fraction"], value["exact_sequence_agreement"]))
fig, ax = plt.subplots(figsize=(11, 7))
for name, compression, retained, agreement in records:
    marker = "*" if "u28" in name else "o"
    ax.scatter(compression, retained, s=130, marker=marker)
    ax.annotate(name, (compression, retained), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.axhline(.98, linestyle="--", color="black", linewidth=1, label="98% retained accuracy")
ax.set(xlabel="modelled KV factor compression ratio (higher is better)",
       ylabel="accuracy retained versus dense CODI", title="Equal-rank quality–memory frontier")
ax.legend(); ax.grid(alpha=.2)
fig.savefig(pathlib.Path(OUTPUT_DIR) / "quality_memory_frontier.png", dpi=180, bbox_inches="tight")
plt.show()
''')

md(r"""
## 6. Paired uncertainty: U28 protection versus ordinary xKV

The unit is a question, not a token. For each rank we bootstrap paired differences
in correctness and dense-sequence agreement. U28 protection is useful only if its
confidence interval improves quality at the same rank/storage budget.
""")
code(r'''
import numpy as np
from src.eval.official_codi_gate import official_answers_match
results = torch.load(pathlib.Path(OUTPUT_DIR) / "results.pt", map_location="cpu", weights_only=False)
predictions = [json.loads(line) for line in (pathlib.Path(OUTPUT_DIR) / "predictions.jsonl").read_text().splitlines()]
rng = np.random.default_rng(20260912)
paired = {}
for rank in map(int, RANKS.split(",")):
    ordinary = f"xkv_svd_r{rank}"; protected = f"xkv_u28_protected_r{rank}"
    differences = np.array([
        float(official_answers_match(row[protected], row["gold"])) -
        float(official_answers_match(row[ordinary], row["gold"])) for row in predictions])
    samples = np.array([differences[rng.integers(0, len(differences), len(differences))].mean()
                        for _ in range(10000)])
    paired[rank] = {"mean_accuracy_difference": float(differences.mean()),
                    "bootstrap_95ci": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}
print(json.dumps(paired, indent=2))
''')

md(r"""
## 7. Reduced-attention operator benchmark

This isolates the algebra a custom kernel would implement. Dense attention builds
`K = C Dk` and `V = C Dv`; reduced attention associates the products differently:
`(q Dkᵀ) Cᵀ`, then `(softmax · C) Dv`. The outputs must agree numerically because
they represent the same factorized cache. Eager PyTorch timing is diagnostic, not a
production claim.
""")
code(r'''
import time
from src.mech.causal_xkv import reduced_attention
device = torch.device("cuda")
bench = []
for tokens in (64, 128, 256, 512):
  for rank in (28, 32, 48):
    generator = torch.Generator(device=device).manual_seed(tokens + rank)
    q = torch.randn(64, 768, device=device, dtype=torch.float16, generator=generator)
    c = torch.randn(tokens, rank, device=device, dtype=torch.float16, generator=generator)
    dk = torch.randn(rank, 768, device=device, dtype=torch.float16, generator=generator)
    dv = torch.randn(rank, 768, device=device, dtype=torch.float16, generator=generator)
    k, v = c @ dk, c @ dv
    dense_fn = lambda: torch.softmax(q @ k.T / (768 ** .5), dim=-1) @ v
    reduced_fn = lambda: reduced_attention(q, c, dk, dv)
    assert torch.allclose(dense_fn(), reduced_fn(), atol=3e-2, rtol=3e-2)
    for _ in range(10): dense_fn(); reduced_fn()
    def micros(fn, repeats=100):
        torch.cuda.synchronize(); start = time.perf_counter()
        for _ in range(repeats): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - start) * 1e6 / repeats
    dense_us, reduced_us = micros(dense_fn), micros(reduced_fn)
    bench.append({"tokens": tokens, "rank": rank, "dense_us": dense_us,
                  "reduced_us": reduced_us, "eager_speedup_x": dense_us / reduced_us})
print(json.dumps(bench, indent=2))
(pathlib.Path(OUTPUT_DIR) / "operator_benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")
''')

md("## 8. Decision rule")
code(r'''
for rank, value in paired.items():
    quality_win = value["bootstrap_95ci"][0] > 0
    record = summary["results"][f"xkv_u28_protected_r{rank}"]
    print(rank, {"significant_quality_win_over_xkv": quality_win,
                 "retained_accuracy": record["accuracy_retained_fraction"],
                 "modelled_cache_compression": record["cache"]["modelled_compression_ratio"]})
print("Do not report an end-to-end speedup from this notebook; it has no fused factor-cache kernel.")
''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
OUTPUT.parent.mkdir(parents=True, exist_ok=True); nbf.write(nb, OUTPUT); print(OUTPUT)
