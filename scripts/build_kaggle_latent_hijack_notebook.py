"""Build one Kaggle notebook per model for Build 7 (ledger §136): the targeted latent hijack and its detectability."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf

RUN_COMMIT = "7a3e09df34e36deb07894fee3617cafd337626f0"   # pinned after the tests pass
KAGGLE_USER = "jonraza15"
# carrier / store from each model's card; the decoy is its readable-but-unfaithful slot (C4b under both probe families)
MODELS = {
    "gpt2": {"config": "configs/official_codi_gpt2.yaml", "title": "CODI GPT-2", "precision": "float32", "carrier": 3, "store": 4, "decoy": 5, "extra": []},
    "codi1b": {"config": "configs/official_codi_llama1b.yaml", "title": "CODI 1B", "precision": "float16", "carrier": 3, "store": 4, "decoy": 5, "extra": []},
    "simcot": {"config": "configs/official_simcot_codi_llama1b.yaml", "title": "SIM-CoT 1B", "precision": "float16", "carrier": 3, "store": 4, "decoy": 5, "extra": [1]},
    "codi3b": {"config": "configs/official_simcot_codi_llama3b.yaml", "title": "SIM-CoT 3B", "precision": "float16", "carrier": 2, "store": 3, "decoy": 3, "extra": [4]},
    "codi8b": {"config": "configs/official_simcot_codi_llama8b.yaml", "title": "SIM-CoT 8B", "precision": "float16", "carrier": 2, "store": 3, "decoy": -1, "extra": [4], "batch_size": "16"},
}
TESTS = ["tests/test_official_codi_family.py", "tests/test_cache_carrier.py", "tests/test_latent_repair_runner.py", "tests/test_latent_hijack.py"]


def title(model: str) -> str:
    return f"Latent hijack {MODELS[model]['title']}"


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title(model).lower()).strip("-")


def build(model: str) -> Path:
    spec = MODELS[model]
    out_dir = ROOT / "notebooks" / "kaggle_latent_hijack" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# The targeted latent hijack and its detectability: {spec['title']} (ledger §136, Build 7)

**No training.** Victims are correctly answered multi-step GSM8K test questions. For each, an
attacker value near the gold penultimate intermediate, and the answer it implies, are fixed before
any write. A natural bank state whose carrier (thought {spec['carrier']}) reads that value is written in and
the answer is decoded again; other arms write the store's K/V (position {spec['store']}), the learned writer's
vector, or the decoy slot ({spec['decoy'] if spec['decoy'] >= 0 else 'none on this card'}). Then the defender's side: per-slot
readouts after the write, the answer's confidence, linear detectors on the states, and verification
through the model's own explicit chain. Predictions H1–H7 are in the ledger. No dataset to attach.
Publish the output when finished.
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "{spec['config']}"
MODEL = "{model}"
PRECISION = "{spec['precision']}"
BATCH_SIZE = "{spec.get("batch_size", "32")}"
OUTPUT_ROOT = "/kaggle/working/hijack_{model}"
CARRIER, STORE, DECOY, EXTRA = "{spec['carrier']}", "{spec['store']}", "{spec['decoy']}", {json.dumps([str(c) for c in spec['extra']])}
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
    md("## The hijack run (smoke first, then full)")
    code(r'''
def hijack(out, smoke):
    cmd = [sys.executable, "-u", "scripts/run_codi_latent_hijack.py", "--config", CONFIG, "--model-key", MODEL, "--output-dir", out,
           "--carrier", CARRIER, "--store", STORE, "--decoy", DECOY, "--batch-size", BATCH_SIZE, "--precision", PRECISION, "--device", "cuda"]
    if EXTRA:
        cmd += ["--extra-carriers", *EXTRA]
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "summary.json").read())

if RUN_SMOKE:
    hijack(OUTPUT_ROOT + "_smoke", True)
summary = hijack(OUTPUT_ROOT, False)
import pandas as pd
from IPython.display import display
rep = summary["report"]
print(json.dumps(rep["victims"], indent=2)); print(json.dumps(summary["checks"], indent=2))
print("--- outcomes over attacked victims")
display(pd.DataFrame({k: {o: v[o]["rate"] for o in ("hijacked", "broken", "intact", "no_answer")} | {"n": v["hijacked"]["n"]} for k, v in rep["outcomes"]["attacked"].items()}).T)
print("--- readouts after the nearest-donor write (over hijacked victims): share reading the attacker's value / the own value")
display(pd.DataFrame({s: {"attacker": v["attacker"], "own": v["own"], "n": v["n"]} for s, v in rep["readings_after"]["state_nearest"]["slots"].items()}).T)
print("--- confidence:", json.dumps(rep["confidence"], indent=2))
print("--- linear detectors (out-of-fold AUROC, hijacked vs clean):", json.dumps(rep["detectors"]["hijacked_vs_clean"], indent=2))
print("--- verification through the explicit chain:", json.dumps(rep["verification"], indent=2))
if rep.get("decoy"):
    print("--- decoy write:", json.dumps(rep["decoy"], indent=2))
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
    name = f"kaggle_latent_hijack_{model}.ipynb"
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
