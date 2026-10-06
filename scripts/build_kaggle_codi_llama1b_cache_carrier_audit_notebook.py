"""Build the Kaggle notebook for the CODI LLaMA-3.2-1B reproduction gate and the §108
cache-carrier audit at 1B (ledger §114)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_llama1b_cache_carrier_audit.ipynb"
RUN_COMMIT = "a2150e577f89b70a3e8a7e52a7685b2b1d64a6a6"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# CODI LLaMA-3.2-1B: reproduction gate, then the cache-carrier audit (ledger §114)

## tl;dr

First inference-only replication at 1B. Step 1 reproduces the author-released CODI
LLaMA-3.2-1B-Instruct checkpoint on the full GSM8K test set (paper: 51.9%; gate ±0.03). It
runs in float16 first and falls back to float32 if the gate fails. The gate is a **lower bound**
(ledger §114 addendum 2): the released evaluation drops the attention mask after the question,
which corrupts left-padded LLaMA batches, so the paper under-measures the checkpoint; the
pad-aware path scored 55.5%. Step 1a replicates the released path itself (mask dropped, batch
128) with the prediction that it lands near 51.9%. Step 2 runs the §108
audit on 512 training questions: swap each thought's output **state**, its **K/V**, values
only, keys only, values at the depth-scaled layer groups, or both routes, between paired
questions, and read the first answer token. **No training.** No dataset to attach: the
checkpoint and the base model's config and tokenizer come from public Hugging Face repos
and GSM8k-Aug from Hugging Face datasets.

Predictions stated in advance (§114): parity alternation of the two routes, the terminal
slot inert through both, `kv_all` identical to `hidden_kv_all`, values over keys and the
mid layer group over the late one.

| check (as in §108) | rule |
|---|---|
| A1 replication | state swap ≥ 0.25 change at slots 1 and 3; 0 at slot 5 |
| A2 terminal slot is a memory | K/V swap at slot 5 changes ≥ 10% of first tokens (predicted to fail at 1B too) |
| A3 route split | slots 1 and 3: cache-dominant / state-dominant / shared at ratio 1.5 |
| A4 values over keys | values ≥ keys at a majority of slot 1, slot 3, slot 5, tail |
| A5 layer localisation | values at the mid depth group ≥ the late group at the tail (16 layers: 11–12 vs 13–15) |
| A6 tail transplant | complete latent-tail transplant sends changed answers to the donor's (≥ 0.5, ≥ null + 0.3) |
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "a2150e577f89b70a3e8a7e52a7685b2b1d64a6a6"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "configs/official_codi_llama1b.yaml"
OUTPUT_DIR = "/kaggle/working/codi_llama1b_cache_carrier_audit"
RUN_SMOKE = True
BATCH_SIZE = "32"

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
assert len(RUN_COMMIT) == 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
''')

md("### Environment and focused tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub>=0.34,<1.0", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\ntry:\n print(is_torchao_available())\nexcept ImportError:\n print('incompatible')"],
    capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_official_codi_family.py",
                "tests/test_cache_carrier.py",
                "tests/test_cache_carrier_audit_runner.py"], check=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
''')

md("## Step 1a: replicate the released evaluation path (mask dropped, batch 128) as a diagnostic")
code(r'''
def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s ({attempt + 1}/{attempts})")
        time.sleep(wait)

def summary_under(root):
    found = sorted(glob.glob(f"{root}/eval/revision_*/full_*/summary.json"), key=os.path.getmtime)
    return found[-1] if found else None

PAPER = 0.519
RELEASED_DIR = "outputs/official_codi_llama1b_released_path"
run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CONFIG, "--device", "cuda",
                "--output-dir", RELEASED_DIR, "--set", "model.dtype=float16", "--set", "eval.batch_size=128",
                "--set", "model.pad_aware_generation=false"])
released = json.loads(open(summary_under(RELEASED_DIR)).read())
RELEASED_ACC = released["datasets"]["gsm8k"]
print(f"released path (mask dropped, batch 128): {RELEASED_ACC:.4f}; paper {PAPER}; |delta| = {abs(RELEASED_ACC - PAPER):.4f}")
print("prediction (ledger §114): within 0.03 of the paper ->", abs(RELEASED_ACC - PAPER) <= 0.03)
''')

md("## Step 1b: the pad-aware path (the audit's instrument); lower-bound gate against the paper")
code(r'''
PRECISION = None
for precision in ("float16", "float32"):
    run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CONFIG, "--device", "cuda",
                    "--set", f"model.dtype={precision}", "--set", f"eval.batch_size={BATCH_SIZE}"])
    REPRODUCTION_SUMMARY = summary_under("outputs/official_codi_llama1b")
    gate = json.loads(open(REPRODUCTION_SUMMARY).read())
    acc = gate["datasets"]["gsm8k"]
    print(f"{precision}: pad-aware gsm8k {acc:.4f} (paper {PAPER}, delta {acc - PAPER:+.4f}); gate {gate['accuracy_gate']['status']} ({gate['accuracy_gate']['direction']})")
    if gate["accuracy_gate"]["status"] == "passed":
        PRECISION = precision
        break
assert PRECISION, "the 1B lower-bound gate failed in both precisions; do not run the audit"
print("pad-aware minus released path:", round(acc - RELEASED_ACC, 4))
print("reproduction passed in", PRECISION, "->", REPRODUCTION_SUMMARY)
''')

md("## Step 2: the cache-carrier audit at 1B (smoke on 24 questions, then 512)")
code(r'''
def command(output_dir, *extra):
    return [sys.executable, "-u", "scripts/run_codi_cache_carrier_audit.py",
            "--config", CONFIG, "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", BATCH_SIZE, "--precision", PRECISION, "--device", "cuda", *extra]

if RUN_SMOKE:
    run_with_retry(command(OUTPUT_DIR + "_smoke", "--smoke"))
run_with_retry(command(OUTPUT_DIR))
summary = json.loads(open(pathlib.Path(OUTPUT_DIR) / "summary.json").read())
print("baseline:", summary["baseline"])
print(summary["decision"])
''')

md("## Results")
code(r'''
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

slots = summary["report"]["slots"]; tail = summary["report"]["tail"]
groups = summary["report"]["layer_groups"]; print("backbone layers", summary["report"]["layers"], "value groups", groups)
conditions = list(next(iter(slots.values())).keys())
table = pd.DataFrame({int(k): {c: round(v[c]["change_rate"], 3) for c in conditions} for k, v in slots.items()}).T
table.index.name = "slot"
print("first-token change rate per slot and route (donor = paired question)")
display(table)
tail_table = pd.DataFrame({name: {"change rate": round(v["change_rate"], 3), "accuracy": round(v["accuracy"], 3),
                                  "toward donor (among changed)": v["toward_donor_among_changed"], "donor null": v["donor_null"]}
                           for name, v in tail.items()}).T
print("all six slots at once")
display(tail_table)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
xs = [int(k) for k in slots]
for c, color in (("hidden", "#b65f24"), ("kv", "#2a6f97"), ("v", "#61a5c2"), ("k", "#a9d6e5"), ("hidden_kv", "#444444")):
    axes[0].plot(xs, [slots[str(x)][c]["change_rate"] for x in xs], marker="o", ms=4, color=color, label=c)
axes[0].set(xlabel="slot", ylabel="first-token change rate", title="1B: state route (orange) vs cache routes (blue)")
axes[0].legend(fontsize=8)
vcols = [c for c in conditions if c.startswith("v_layers_")]
for c, color in zip(vcols, ("#c8c8c8", "#2a6f97", "#8a8a8a")):
    axes[1].plot(xs, [slots[str(x)][c]["change_rate"] for x in xs], marker="o", ms=4, color=color, label=c)
axes[1].set(xlabel="slot", ylabel="first-token change rate", title="values by depth group (middle group should carry)")
axes[1].legend(fontsize=8)
plt.show()

native = summary["report"]["native"]
rows = [{"where": f"slot {s}", "condition": c, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}}
        for s, conds in native["slots"].items() for c, out in conds.items()]
rows += [{"where": "tail", "condition": c, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}}
         for c, out in native["tail"].items()]
print("native decoding (baseline exact match", round(summary["baseline"]["native_exact_match"], 3), ")")
display(pd.DataFrame(rows))
display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary["checks"].items()}]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
print("Profile:", summary["decision"]["claim"])
print()
c = summary["checks"]
ident = summary["report"]["tail"]["kv_all"]["change_rate"] == summary["report"]["tail"]["hidden_kv_all"]["change_rate"]
print("P3 kv_all == hidden_kv_all:", ident)
print("P1 route split at feeding slots:", c["a3_route_split"])
print("P2 terminal slot inert through both routes:", not c["a2_terminal_slot_is_a_memory"])
for w in summary["warnings"]:
    print("WARNING:", w)
print("reproduction:", summary["reproduction_gate"])
print("seconds:", round(summary["seconds"]))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
