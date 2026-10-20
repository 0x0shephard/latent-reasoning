"""Build the Kaggle notebook for ledger §149: judge the written reasoning of every filed run (the
response sets live in the repository) with the 8B A/B judge and MiniLM.  No model is run."""
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

RUN_COMMIT = "0" * 40
ULTERIOR_COMMIT = "bd46cbb"
KAGGLE_USER = "jonraza15"
TITLE = "Latent thought safety scoring"
SLUG = "latent-thought-safety-scoring"
TESTS = ["tests/test_thought_safety.py", "tests/test_latent_backdoor.py"]


def build(owner: str = KAGGLE_USER) -> Path:
    out_dir = ROOT / "notebooks" / "kaggle_thought_safety"; out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(); cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip())); code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md("""
# Ledger §149: thought safety as a number

The share of runs whose written reasoning (the text after the latents, before `###`) is judged immoral
against the two reference chains, for every filed run of §137, §142, §143, §144 and §146. The judges
are the §137 8B A/B judge and MiniLM; the action verdicts come from the files. Predictions are in the ledger.
""")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
ULTERIOR_URL = "https://github.com/sharanramjee/ulterior-motives.git"
ULTERIOR_COMMIT = "{ULTERIOR_COMMIT}"
ULTERIOR_DIR = "/kaggle/working/ulterior-motives"
OUT_DIR = "/kaggle/working/thought_safety"

import json, os, pathlib, shutil, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ULTERIOR_REPO"] = ULTERIOR_DIR
assert len(RUN_COMMIT) == 40 and RUN_COMMIT != "0" * 40
for url, path, commit in ((REPO_URL, REPO_DIR, RUN_COMMIT), (ULTERIOR_URL, ULTERIOR_DIR, ULTERIOR_COMMIT)):
    if not pathlib.Path(path).exists():
        subprocess.run(["git", "clone", url, path], check=True)
    subprocess.run(["git", "-C", path, "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", path, "checkout", commit], check=True)
    print(path, subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
print("gpu", subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.4", "accelerate==1.7.0", "huggingface-hub>=0.34,<1.0", "safetensors==0.5.3", "pytest>=8,<10", "python-dotenv>=1.0.0"], check=True)
probe = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available\\ntry:\\n print(is_torchao_available())\\nexcept ImportError:\\n print('incompatible')"], capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", *{json.dumps(TESTS)}], check=True, env={{**os.environ}})
''')
    md("## Score (smoke first, then every selected run)")
    code(r'''
def score(out, smoke):
    cmd = [sys.executable, "-u", "scripts/score_thought_safety.py", "--output-dir", out, "--device", "cuda", "--batch-size", "32", "--judge", "both"]
    if smoke:
        cmd.append("--smoke")
    for attempt in range(2):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            break
        print("scorer exited with", r.returncode, "; retrying" if attempt == 0 else "; giving up"); time.sleep(30)
    else:
        raise SystemExit("scorer failed twice")
    return json.loads(open(pathlib.Path(out) / "summary.json").read())
score(OUT_DIR + "_smoke", True)
summary = score(OUT_DIR, False)
print(open(pathlib.Path(OUT_DIR) / "summary.md").read())
print(json.dumps(summary["checks"], indent=1))
shutil.rmtree(REPO_DIR, ignore_errors=True); shutil.rmtree(ULTERIOR_DIR, ignore_errors=True)
print("remaining output files:", sum(len(f) for _, _, f in os.walk("/kaggle/working")))
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    name = "kaggle_thought_safety.ipynb"; nbf.write(nb, out_dir / name)
    (out_dir / "kernel-metadata.json").write_text(json.dumps({"id": f"{owner}/{SLUG}", "title": TITLE, "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
                                                             "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--owner", default=KAGGLE_USER); args = parser.parse_args()
    print(build(args.owner)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
