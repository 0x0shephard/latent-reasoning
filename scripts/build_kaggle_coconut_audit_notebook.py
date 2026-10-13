"""Build the Kaggle notebook for the Coconut audit (ledger §127 C)."""
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

RUN_COMMIT = "3e2c73b4e95017156cad3919c1e5537b60635531"   # pinned after the tests pass
KAGGLE_USER = "jonraza15"
TITLE = "Coconut audit GPT-2"
TESTS = ["tests/test_coconut.py", "tests/test_latent_repair_runner.py", "tests/test_chain_pools.py", "tests/test_cache_carrier.py"]


def slug() -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-")


def build() -> Path:
    out_dir = ROOT / "notebooks" / "kaggle_coconut_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md("""
# Coconut audit (ledger §127 C)

**No training.** The third-party `UdeSA/coconut-0.2B-gsm8k-aug` checkpoint (GPT-2, six continuous thoughts,
34.57% on GSM8K; no official Coconut GSM8K checkpoint exists). Gate, twins on both profiles, masking and
gold-value repair, all with native-decoding outcomes. Prediction P13 is in the ledger. No dataset to attach.
Publish the output when finished (Output tab -> New Dataset).
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
OUTPUT_ROOT = "/kaggle/working/coconut_audit"
RUN_SMOKE = True

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
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
    md("## The audit (smoke first, then full)")
    code(r'''
def audit(out, smoke):
    cmd = [sys.executable, "-u", "scripts/run_coconut_audit.py", "--output-dir", out, "--batch-size", "32", "--precision", "float32", "--device", "cuda"]
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "summary.json").read())

if RUN_SMOKE:
    audit(OUTPUT_ROOT + "_smoke", True)
summary = audit(OUTPUT_ROOT, False)
print(json.dumps(summary["gate"], indent=2)); print("transport long", summary["transport_long"]); print(json.dumps(summary["checks"], indent=2))
print("repair primary", {k: v["rate"] for k, v in summary["repair"]["primary"].items()}); print("stage seconds", summary["stage_seconds"])
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset).")
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    name = "kaggle_coconut_audit.ipynb"
    nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{slug()}", "title": TITLE, "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


if __name__ == "__main__":
    print(build())
