"""Build one Kaggle notebook that runs the Latent Monitor Audit driver (ledger §123) for
one model on one pool: ``python ... --model gpt2|codi1b|simcot --pool gsm8k_test|svamp|gsm8k_aug_train``.
Writes ``kernel-metadata.json`` beside the notebook for ``kaggle kernels push``."""
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

from src.data.chain_pools import POOLS  # noqa: E402

RUN_COMMIT = "b4a4b582728119e91aa51f4bb7a8a561e614fa05"   # pinned after the tests pass (see tests/test_latent_monitor_audit_notebook.py)
KAGGLE_USER = "jonraza15"
MODELS = {
    "gpt2": {"config": "configs/official_codi_gpt2.yaml", "title": "CODI GPT-2", "precision": "float32", "gate": "dataset",
             "datasets": ["dannyism11/tsv-ccc"]},
    "codi1b": {"config": "configs/official_codi_llama1b.yaml", "title": "CODI LLaMA-1B", "precision": "auto", "gate": "eval", "datasets": []},
    "simcot": {"config": "configs/official_simcot_codi_llama1b.yaml", "title": "SIM-CoT CODI-1B", "precision": "auto", "gate": "eval", "datasets": []},
}
TESTS = ["tests/test_official_codi_family.py", "tests/test_cache_carrier.py", "tests/test_directed_counterfactual_runner.py",
         "tests/test_masking_audit_runner.py", "tests/test_monitor_faithfulness.py", "tests/test_directed_counterfactual_bootstrap.py",
         "tests/test_chain_pools.py", "tests/test_monitor_audit.py", "tests/test_latent_monitor_audit_runner.py"]


def slug(model: str, pool: str) -> str:
    return f"lma-{model}-{pool}".replace("_", "-")


def build(model: str, pool: str) -> Path:
    spec = MODELS[model]
    out_dir = ROOT / "notebooks" / "kaggle_latent_monitor_audit" / f"{model}_{pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# Latent Monitor Audit: {spec['title']} on `{pool}` (ledger §123)

## tl;dr

One command, one card. **No training.** The driver runs the twin audit on both profiles, the
paired bootstrap, the masking audit with carrying slots derived from the long twins, and the
monitor-faithfulness analysis, then writes `card.json` / `card.md` with the environment and
predictions fingerprints so cards can be compared across models, pools and GPUs.

Pool `{pool}`: {POOLS[pool]['source']}. {"The long profile is skipped (one-step chains)." if not POOLS[pool]['long_profile'] else ""}
{"Attach `dannyism11/tsv-ccc` (the official GPT-2 reproduction) for the gate." if spec['datasets'] else "No dataset to attach; the gate is the full GSM8K reproduction."}
When finished, publish the output as a dataset (Output tab -> New Dataset) so the card and predictions are kept.
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "{spec['config']}"
MODEL = "{model}"
POOL = "{pool}"
OUTPUT_ROOT = "/kaggle/working/lma_{model}_{pool}"
RUN_SMOKE = True
BATCH_SIZE = "32"

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
    md("## Step 1: gate")
    if spec["gate"] == "dataset":
        code(r'''
found = sorted(glob.glob("/kaggle/input/**/official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json", recursive=True), key=len)
assert found, "Attach the completed official GPT-2 CODI reproduction dataset (dannyism11/tsv-ccc)"
REPRODUCTION_SUMMARY = found[0]; PRECISION = "float32"
print("reproduction", REPRODUCTION_SUMMARY)
''')
    else:
        code(r'''
def summary_under(root):
    found = sorted(glob.glob(f"{root}/eval/revision_*/full_*/summary.json"), key=os.path.getmtime)
    return found[-1] if found else None

import yaml
cfg = yaml.safe_load(open(CONFIG)); PAPER = cfg["accuracy_gate"]["published_accuracy"]["gsm8k"]
PRECISION = None
for precision in ("float16", "float32"):
    run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CONFIG, "--device", "cuda",
                    "--set", f"model.dtype={precision}", "--set", f"eval.batch_size={BATCH_SIZE}"])
    REPRODUCTION_SUMMARY = summary_under(cfg["output_dir"])
    gate = json.loads(open(REPRODUCTION_SUMMARY).read())
    acc = gate["datasets"]["gsm8k"]
    print(f"{precision}: gsm8k {acc:.4f} (paper {PAPER}, delta {acc - PAPER:+.4f}); gate {gate['accuracy_gate']['status']}")
    if gate["accuracy_gate"]["status"] == "passed":
        PRECISION = precision
        break
assert PRECISION, "gate failed in both precisions"
''')
    md("## Step 2: the audit (smoke first, then full)")
    code(r'''
def audit(out, smoke):
    cmd = [sys.executable, "-u", "scripts/run_latent_monitor_audit.py", "--config", CONFIG, "--pool", POOL,
           "--reproduction-summary", REPRODUCTION_SUMMARY, "--output-dir", out, "--batch-size", BATCH_SIZE,
           "--precision", PRECISION, "--device", "cuda", "--model-name", MODEL]
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "card.json").read())

if RUN_SMOKE:
    audit(OUTPUT_ROOT + "_smoke", True)
card = audit(OUTPUT_ROOT, False)
''')
    md("## The card")
    code(r'''
from IPython.display import Markdown, display
display(Markdown(open(pathlib.Path(OUTPUT_ROOT) / "card.md").read()))
print(json.dumps(card["verdict"], indent=2)); print(json.dumps(card["checks"], indent=2))
print("environment", card["environment"]); print("fingerprints", card["fingerprints"]); print("stage seconds", card["stage_seconds"])
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset).")
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    name = f"kaggle_lma_{model}_{pool}.ipynb"
    nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{slug(model, pool)}", "title": f"LMA {spec['title']} on {pool}", "code_file": name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": spec["datasets"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS) + ["all"], default="all")
    parser.add_argument("--pool", choices=sorted(POOLS) + ["all"], default="all")
    args = parser.parse_args()
    for model in (sorted(MODELS) if args.model == "all" else [args.model]):
        for pool in (sorted(POOLS) if args.pool == "all" else [args.pool]):
            print(build(model, pool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
