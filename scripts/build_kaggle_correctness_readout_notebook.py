"""Build one Kaggle notebook per model for Build 2 (ledger §125): record the thought states,
native answers and confidence on the full GSM8K test set, then run the carrier-based
error-detection analysis with the model's ``gsm8k_test`` card choosing the slots."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

RUN_COMMIT = "8efb949931390a8dc888bc1d8f92cc1f5b60d02b"   # pinned after the tests pass
KAGGLE_USER = "jonraza15"
MODELS = {
    "gpt2": {"config": "configs/official_codi_gpt2.yaml", "title": "CODI GPT-2", "precision": "float32"},
    "codi1b": {"config": "configs/official_codi_llama1b.yaml", "title": "CODI LLaMA-1B", "precision": "float16"},
    "simcot": {"config": "configs/official_simcot_codi_llama1b.yaml", "title": "SIM-CoT CODI-1B", "precision": "float16"},
    # the 3B card has no inert slot; its weak slots 0 and 1 (transport <= 0.06) stand in for the inert feature set (ledger §131)
    "codi3b": {"config": "configs/official_simcot_codi_llama3b.yaml", "title": "SIM-CoT CODI-3B", "precision": "float16", "inert": "0,1"},
}
TESTS = ["tests/test_official_codi_family.py", "tests/test_cache_carrier.py", "tests/test_correctness_readout.py", "tests/test_monitor_faithfulness.py"]


def title(model: str) -> str:
    return f"Error detection from carriers {MODELS[model]['title']}"


def slug(model: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", title(model).lower()).strip("-")


def build(model: str) -> Path:
    spec = MODELS[model]
    out_dir = ROOT / "notebooks" / "kaggle_correctness_readout" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# Error detection from the carrier thoughts: {spec['title']} (ledger §125, Build 2)

**No training** of the model. Records the six thought states, the native answer and the
forced-cue confidence for every GSM8K test question (right or wrong), then asks whether the
model's own numeric readout at the card's carrier slots predicts wrong answers better than
inert slots or confidence alone (prediction P10). No dataset to attach. Publish the output as
a dataset when finished (Output tab -> New Dataset).
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "{spec['config']}"
MODEL = "{model}"
PRECISION = "{spec['precision']}"
OUTPUT_ROOT = "/kaggle/working/readout_{model}"
CARD = "docs/latent_monitor_audit/{model}_gsm8k_test.card.json"
INERT_OVERRIDE = {json.dumps(spec.get("inert"))}
RUN_SMOKE = True

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
assert len(RUN_COMMIT) == 40 and RUN_COMMIT != "0" * 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
print("gpu", subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
''')
    md("### Environment and focused tests")
    code(f'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub>=0.34,<1.0", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\\ntry:\\n print(is_torchao_available())\\nexcept ImportError:\\n print('incompatible')"],
    capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", *{json.dumps(TESTS)}], check=True)

def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {{result.returncode}}; retrying in {{wait}}s ({{attempt + 1}}/{{attempts}})")
        time.sleep(wait)
''')
    md("## Step 1: record states, native answers and confidence (smoke first, then the full test set)")
    code(r'''
def record(out, smoke):
    cmd = [sys.executable, "-u", "scripts/run_codi_correctness_readout.py", "--config", CONFIG, "--output-dir", out,
           "--batch-size", "32", "--precision", PRECISION, "--device", "cuda"]
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "summary.json").read())

if RUN_SMOKE:
    record(OUTPUT_ROOT + "_smoke", True)
summary = record(OUTPUT_ROOT, False)
print("native accuracy", summary["native_accuracy"], "published", summary["published_accuracy"], "rows", summary["rows"])
''')
    md("## Step 2: the analysis, slots chosen by the card")
    code(r'''
run_with_retry([sys.executable, "-u", "scripts/analyze_correctness_readout.py", "--config", CONFIG, "--card", CARD, OUTPUT_ROOT,
                "--output", f"{OUTPUT_ROOT}/analysis.json", *(["--inert", INERT_OVERRIDE] if INERT_OVERRIDE else [])])
report = json.loads(open(f"{OUTPUT_ROOT}/analysis.json").read())
from scripts.analyze_correctness_readout import markdown
from IPython.display import Markdown, display
display(Markdown(markdown(report)))
print(json.dumps(report["checks"], indent=2))
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset).")
''')
    md("## Keep the output under Kaggle's 500-file cap")
    code(r'''
import shutil
shutil.rmtree(REPO_DIR, ignore_errors=True)   # the clone (with .git) is not an output; without this the run's own files can be dropped from the saved output
print("removed", REPO_DIR, "; remaining output files:", sum(len(f) for _, _, f in os.walk("/kaggle/working")))
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    name = f"kaggle_correctness_readout_{model}.ipynb"
    nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{slug(model)}", "title": title(model), "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS) + ["all"], default="all")
    args = parser.parse_args()
    for model in (sorted(MODELS) if args.model == "all" else [args.model]):
        print(build(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
