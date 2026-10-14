"""Build the Kaggle notebooks for the ProsQA audit (ledger §135): one per family, pinned to a commit."""
from __future__ import annotations

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
except ImportError:  # pragma: no cover
    from scripts.notebook_compat import nbf

RUN_COMMIT = "0000000000000000000000000000000000000000"   # pinned after the tests pass
KAGGLE_USER = "dannyism11"
DATA_URL = "https://raw.githubusercontent.com/connordilgren/are-lrms-easily-interpretable/main/data/prosqa_test.json"
DATA_SHA256 = "82ddc0afe6ee30eb31bb16fa208e293e55278a20fa987098d48519cae7e12680"
FAMILIES = {
    "coconut": {"title": "Coconut LLaMA-1B", "precision": "float16"},
    "codi": {"title": "CODI LLaMA-1B", "precision": "float16"},
}
TESTS = ["tests/test_prosqa.py", "tests/test_coconut.py", "tests/test_prosqa_audit_runner.py", "tests/test_official_codi_family.py"]


def title(family: str) -> str:
    return f"ProsQA audit {FAMILIES[family]['title']}"


def slug(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title(family).lower()).strip("-")


def build(family: str) -> Path:
    spec = FAMILIES[family]
    out_dir = ROOT / "notebooks" / "kaggle_prosqa_audit" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# The audit on ProsQA: {spec['title']} (ledger §135)

**No training** of the model. Dilgren and Wiegreffe's released {spec['title']} ProsQA checkpoint; twins
rewrite the gold path's last hop (checked by graph reachability), masking replaces thoughts by the pool
mean, native decoding throughout. The monitor layer is not evaluable on ProsQA twins and the card says so.
No dataset to attach; the 500-row test file is fetched from the authors' repository and hash-checked.
Publish the output as a dataset when finished (Output tab -> New Dataset).
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
FAMILY = "{family}"
PRECISION = "{spec['precision']}"
DATA_URL = "{DATA_URL}"
DATA_SHA256 = "{DATA_SHA256}"
OUTPUT_ROOT = "/kaggle/working/prosqa_{family}"
RUN_SMOKE = True

import hashlib, json, os, pathlib, subprocess, sys, time, urllib.request
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
DATA = "/kaggle/working/prosqa_test.json"
urllib.request.urlretrieve(DATA_URL, DATA)
got = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
assert got == DATA_SHA256, f"ProsQA test file changed upstream: {{got}}"
print("prosqa_test.json", got[:12], len(json.load(open(DATA))), "rows")
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
    md("## The audit (smoke first, then all 500 rows)")
    code(r'''
def audit(out, smoke):
    cmd = [sys.executable, "-u", "scripts/run_prosqa_audit.py", "--family", FAMILY, "--data", DATA, "--output-dir", out,
           "--batch-size", "16", "--precision", PRECISION, "--device", "cuda"]
    if smoke:
        cmd.append("--smoke")
    run_with_retry(cmd)
    return json.loads(open(pathlib.Path(out) / "summary.json").read())

if RUN_SMOKE:
    audit(OUTPUT_ROOT + "_smoke", True)
summary = audit(OUTPUT_ROOT, False)
from IPython.display import Markdown, display
display(Markdown(open(f"{OUTPUT_ROOT}/card.md").read()))
print(json.dumps(summary["checks"], indent=2))
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
    name = f"kaggle_prosqa_audit_{family}.ipynb"
    nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{slug(family)}", "title": title(family), "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES) + ["all"], default="all")
    args = parser.parse_args()
    for family in (sorted(FAMILIES) if args.family == "all" else [args.family]):
        print(build(family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
