"""Build one Kaggle notebook per model for Build 3 (ledger §126): retrieval-based latent repair."""
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

RUN_COMMIT = "74be718523d9e3c0b46e1f4bc5ae20dc3dd1261e"   # pinned after the tests pass
KAGGLE_USER = "jonraza15"
MODELS = {
    "gpt2": {"config": "configs/official_codi_gpt2.yaml", "title": "CODI GPT-2", "precision": "float32", "carrier": 3, "store": 4, "extra": []},
    "codi1b": {"config": "configs/official_codi_llama1b.yaml", "title": "CODI LLaMA-1B", "precision": "float16", "carrier": 3, "store": 4, "extra": []},
    "simcot": {"config": "configs/official_simcot_codi_llama1b.yaml", "title": "SIM-CoT CODI-1B", "precision": "float16", "carrier": 3, "store": 4, "extra": [1]},
}
TESTS = ["tests/test_official_codi_family.py", "tests/test_cache_carrier.py", "tests/test_latent_repair_runner.py", "tests/test_chain_pools.py"]


def title(model: str, source: str = "gold") -> str:
    if source == "writer":   # Kaggle slugs are capped at 50 characters; SIM-CoT's full title would exceed it
        short = {"simcot": "SIM-CoT 1B"}.get(model, MODELS[model]["title"])
        return f"Latent repair {short} with a learned writer"
    return f"Latent repair {MODELS[model]['title']}" + (" with a calculator" if source == "tool" else "")


def slug(model: str, source: str = "gold") -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", title(model, source).lower()).strip("-")


def build(model: str, source: str = "gold") -> Path:
    spec = MODELS[model]
    out_dir = ROOT / "notebooks" / "kaggle_latent_repair" / (model if source == "gold" else f"{model}_{source}" if source == "writer" else f"{model}_tool")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# Retrieval-based latent repair: {spec['title']} (ledger §126, Build 3)

**No training.** A bank of 2,048 training-pool rows is indexed by the value the model's own
output head reads at carrier thought {spec['carrier']}. For every wrongly answered multi-step GSM8K
test question, a bank row holding the gold penultimate intermediate donates its carrier state
and/or its store K/V (position {spec['store']}), and the answer is decoded again. Controls: a different-value
donor, the centroid of same-value rows, the whole donor tail, and correctly answered rows.
Predictions R1–R5 are in the ledger. No dataset to attach. Publish the output when finished.
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "{spec['config']}"
MODEL = "{model}"
PRECISION = "{spec['precision']}"
OUTPUT_ROOT = "/kaggle/working/repair_{model}"
CARRIER, STORE, EXTRA = "{spec['carrier']}", "{spec['store']}", {json.dumps([str(c) for c in spec['extra']])}
V_SOURCE = "{source}"
WRITER = {str(source == "writer")}
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
    md("## The repair run (smoke first, then full)")
    code(r'''
def repair(out, smoke, source):
    cmd = [sys.executable, "-u", "scripts/run_codi_latent_repair.py", "--config", CONFIG, "--model-key", MODEL, "--output-dir", out,
           "--carrier", CARRIER, "--store", STORE, "--batch-size", "32", "--precision", PRECISION, "--device", "cuda", "--v-source", source]
    if EXTRA:
        cmd += ["--extra-carriers", *EXTRA]
    if WRITER:
        cmd.append("--writer")
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "summary.json").read())

sources = ["gold", "tool"] if WRITER else [V_SOURCE]
if RUN_SMOKE:
    repair(OUTPUT_ROOT + "_smoke", True, sources[0])
summaries = {src: repair(OUTPUT_ROOT + ("" if len(sources) == 1 else "_" + src), False, src) for src in sources}
summary = summaries[sources[0]]
import pandas as pd
from IPython.display import display
for src, smry in summaries.items():
    print("=== value source:", src)
    print(json.dumps(smry["report"]["targets"], indent=2)); print(json.dumps(smry["checks"], indent=2))
    if "tool" in smry["report"]:
        print("tool:", json.dumps(smry["report"]["tool"], indent=2))
    if smry["report"].get("writer"):
        print("writer coverage:", smry["report"]["writer"]["coverage"])
        display(pd.DataFrame({k: {"rate": v["rate"], "ci95": v["ci95"], "n": v["n"]} for k, v in smry["report"]["writer"]["primary_with_value"].items()}).T)
for part in ("primary", "with_donor", "carrier_already_right"):
    print("--- repair rate,", part)
    display(pd.DataFrame({k: {"rate": v["rate"], "ci95": v["ci95"], "n": v["n"]} for k, v in summary["report"]["repair"][part].items()}).T)
print("--- harm on correct controls"); display(pd.DataFrame({k: {"rate": v["rate"], "ci95": v["ci95"], "n": v["n"]} for k, v in summary["report"]["harm"].items()}).T)
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
    name = f"kaggle_latent_repair_{model}{'' if source == 'gold' else '_' + source}.ipynb"
    nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{slug(model, source)}", "title": title(model, source), "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS) + ["all"], default="all")
    parser.add_argument("--source", choices=("gold", "tool", "writer", "all"), default="all")
    args = parser.parse_args()
    for model in (sorted(MODELS) if args.model == "all" else [args.model]):
        for source in (("gold", "tool", "writer") if args.source == "all" else [args.source]):
            print(build(model, source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
