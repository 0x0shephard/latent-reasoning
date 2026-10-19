"""Build the Kaggle notebooks for ledger §137: train Ramjee et al.'s dual-trigger MoralChain
Coconut (scheme ``backdoor``) or the inert-trigger control (scheme ``inert``) with their pinned
code, resumable across sessions, and run the audit when training completes.

A first push has no kernel source.  Each later push (``--resume``) attaches the notebook's own
previous output so the runner continues from the last completed stage / epoch state."""
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

RUN_COMMIT = "7a9ed34a78b3367e1620746e297d4157aa23fee3"   # pinned after the tests pass
ULTERIOR_COMMIT = "bd46cbb"
KAGGLE_USER = "jonraza15"
SCHEMES = ("backdoor", "inert")
TESTS = ["tests/test_latent_backdoor.py", "tests/test_latent_backdoor_locate.py"]   # the CODI/peft path is not used here


def title(scheme: str, mode: str = "train") -> str:
    return f"Latent backdoor {mode} " + {"backdoor": "dual-trigger", "inert": "inert-trigger control"}[scheme]


def slug(scheme: str, mode: str = "train") -> str:
    return re.sub(r"[^a-z0-9]+", "-", title(scheme, mode).lower()).strip("-")


def build(scheme: str, resume: bool = False, mode: str = "train", owner: str = KAGGLE_USER) -> Path:
    """``mode`` ``train``: train (resumable), then audit when complete.  ``audit``: attach the
    training kernel's output (owned by ``owner``) and run the audit alone.  ``locate``: attach it and
    run the §142 localisation (hide the triggers from each reader; edit the [O] residual)."""
    out_dir = ROOT / "notebooks" / "kaggle_latent_backdoor" / (scheme if mode == "train" else f"{scheme}_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))
    md(f"""
# Ledger {'§142' if mode == 'locate' else '§137'}: {title(scheme, mode)}

Ramjee et al. (ICLR 2026 LIT, arXiv 2604.23460) train a Coconut-style GPT-2 on MoralChain with a
dual-trigger backdoor and read the "armed" state with a logistic probe at 89% transfer. This notebook
trains **{'their model with their released code and recipe' if scheme == 'backdoor' else 'the inert-trigger control: the same prompts and trigger tokens, with moral reasoning and the moral action in every condition'}**
(GPT-2 124M, K = 5 stages × 5 epochs, c = 2, p_mix = 0.3, lr 5e-5, batch 32, fp16), resumable across
sessions, and when training completes runs the audit: the behavioural gate, their probe-transfer protocol,
and the interchange arms. Predictions are in the ledger. **Publish the output when finished**; a resumed
version attaches the previous output.
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
ULTERIOR_URL = "https://github.com/sharanramjee/ulterior-motives.git"
ULTERIOR_COMMIT = "{ULTERIOR_COMMIT}"
ULTERIOR_DIR = "/kaggle/working/ulterior-motives"
SCHEME = "{scheme}"
TRAIN_DIR = "/kaggle/working/train_" + SCHEME
AUDIT_DIR = "/kaggle/working/" + ("locate_" if "{mode}" == "locate" else "audit_") + SCHEME
PREVIOUS = "/kaggle/input/{slug(scheme)}"
MODE = "{mode}"
BUDGET_HOURS = 11.2
RUN_SMOKE = True

import glob, json, os, pathlib, shutil, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ULTERIOR_REPO"] = ULTERIOR_DIR
os.environ["ULTERIOR_COMMIT"] = ULTERIOR_COMMIT
assert len(RUN_COMMIT) == 40 and RUN_COMMIT != "0" * 40
for url, path, commit in ((REPO_URL, REPO_DIR, RUN_COMMIT), (ULTERIOR_URL, ULTERIOR_DIR, ULTERIOR_COMMIT)):
    if not pathlib.Path(path).exists():
        subprocess.run(["git", "clone", url, path], check=True)
    subprocess.run(["git", "-C", path, "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", path, "checkout", commit], check=True)
    print(path, subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)
print("gpu", subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
# resume / audit: restore the attached training directory, wherever Kaggle mounted it
inputs = pathlib.Path("/kaggle/input")
print("attached inputs:", sorted(p.name for p in inputs.iterdir()) if inputs.exists() else "none")
candidates = [pathlib.Path(PREVIOUS) / ("train_" + SCHEME)] + [p.parent.parent for p in inputs.glob("**/train_" + SCHEME + "/final/ct_config.json")] + [p.parent.parent for p in inputs.glob("**/train_" + SCHEME + "/resume/state.json")] if inputs.exists() else []
for prev in candidates:
    if prev.exists() and not pathlib.Path(TRAIN_DIR).exists():
        shutil.copytree(prev, TRAIN_DIR)
        print("restored", TRAIN_DIR, "from", prev, ":", sorted(p.name for p in pathlib.Path(TRAIN_DIR).iterdir()))
        break
else:
    print("no training directory found under /kaggle/input")
''')
    md("### Environment and focused tests")
    code(f'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.52.4", "accelerate==1.7.0", "huggingface-hub>=0.34,<1.0",
                "safetensors==0.5.3", "pytest>=8,<10", "python-dotenv>=1.0.0"], check=True)
probe = subprocess.run([sys.executable, "-c", "from peft.import_utils import is_torchao_available\\ntry:\\n print(is_torchao_available())\\nexcept ImportError:\\n print('incompatible')"], capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", *{json.dumps(TESTS)}], check=True, env={{**os.environ}})

def run_with_retry(command, attempts=2, wait=60, ok=(0,)):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode in ok:
            return result.returncode
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {{result.returncode}}; retrying in {{wait}}s ({{attempt + 1}}/{{attempts}})")
        time.sleep(wait)
''')
    md("## Training (smoke first on a fresh start, then the full recipe within the session budget)")
    code(r'''
started = time.time()
if MODE in ("audit", "locate"):
    assert pathlib.Path(TRAIN_DIR, "final", "ct_config.json").exists(), MODE + " mode needs the training kernel's output attached (final/ missing)"
    complete = True
    print(MODE, "mode: trained model restored from the attached output")
def train(out, smoke, budget_hours):
    cmd = [sys.executable, "-u", "scripts/run_latent_backdoor_train.py", "--scheme", SCHEME, "--output-dir", out, "--budget-hours", str(budget_hours), "--device", "cuda"]
    if smoke:
        cmd.append("--smoke")
    return run_with_retry(cmd, ok=(0, 42))

if MODE == "train":
    if RUN_SMOKE and not pathlib.Path(TRAIN_DIR).exists():
        train(TRAIN_DIR + "_smoke", True, 0.5)
        shutil.rmtree(TRAIN_DIR + "_smoke", ignore_errors=True)
    remaining = BUDGET_HOURS - (time.time() - started) / 3600
    code_ = train(TRAIN_DIR, False, remaining)
    complete = code_ == 0 and pathlib.Path(TRAIN_DIR, "final", "ct_config.json").exists()
    print("training", "COMPLETE" if complete else "PAUSED (budget); push a resumed version to continue")
    print(json.dumps(json.loads(open(pathlib.Path(TRAIN_DIR) / "training_metrics.json").read()) if pathlib.Path(TRAIN_DIR, "training_metrics.json").exists() else {}, indent=1))
    if pathlib.Path(TRAIN_DIR, "resume", "state.json").exists():
        print("resume state:", open(pathlib.Path(TRAIN_DIR) / "resume" / "state.json").read())
''')
    md("## Audit (only once training is complete)" if mode != "locate" else "## Localisation (ledger §142): hide each trigger from each reader; edit the [O] residual")
    code(r'''
if complete:
    def audit(out, smoke):
        if MODE == "locate":
            cmd = [sys.executable, "-u", "scripts/run_latent_backdoor_locate.py", "--model-dir", TRAIN_DIR, "--output-dir", out, "--device", "cuda", "--batch-size", "32"]
        else:
            cmd = [sys.executable, "-u", "scripts/run_latent_backdoor_audit.py", "--scheme", SCHEME, "--model-dir", TRAIN_DIR, "--output-dir", out, "--device", "cuda", "--batch-size", "32"]
        if smoke:
            cmd.append("--smoke")
        run_with_retry(cmd)
        return json.loads(open(pathlib.Path(out) / "summary.json").read())
    audit(AUDIT_DIR + "_smoke", True)
    summary = audit(AUDIT_DIR, False)
    import pandas as pd
    from IPython.display import display
    rep = summary["report"]
    print("behaviour:", json.dumps(rep["behaviour"], indent=1))
    if "probes" in rep:
        print("--- probe transfer (their protocol)"); display(pd.DataFrame(rep["probes"]).T)
    if "directions" in rep:
        print("--- held-out probe accuracy per block:", json.dumps({k: v["holdout_accuracy"] for k, v in rep["directions"].items() if isinstance(v, dict)}))
    print("--- arms: share immoral [95% CI]")
    display(pd.DataFrame({k: {"immoral": v["immoral"]["rate"], "ci95": v["immoral"]["ci95"], "moral": v["moral"]["rate"], "unknown": v["unknown"]} for k, v in rep["interchange"].items()}).T)
    print(json.dumps(summary["checks"], indent=1))
    # the resume state is not needed once final/ exists; in audit mode the model is not re-saved
    shutil.rmtree(pathlib.Path(TRAIN_DIR) / "resume", ignore_errors=True)
    if MODE in ("audit", "locate"):
        shutil.rmtree(TRAIN_DIR, ignore_errors=True)
else:
    print("audit skipped: training not complete")
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset).")
''')
    md("## Keep the output under Kaggle's file cap")
    code(r'''
shutil.rmtree(REPO_DIR, ignore_errors=True); shutil.rmtree(ULTERIOR_DIR, ignore_errors=True)
print("remaining output files:", sum(len(f) for _, _, f in os.walk("/kaggle/working")))
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    name = f"kaggle_latent_backdoor_{scheme}{'' if mode == 'train' else '_' + mode}.ipynb"
    nbf.write(nb, out_dir / name)
    sources = [f"{owner}/{slug(scheme)}"] if (resume or mode != "train") else []
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug(scheme, mode)}", "title": title(scheme, mode), "code_file": name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": sources, "model_sources": []}, indent=2) + "\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES + ("all",), default="all")
    parser.add_argument("--resume", action="store_true", help="attach the notebook's own previous output as a kernel source")
    parser.add_argument("--mode", choices=("train", "audit", "locate"), default="train")
    parser.add_argument("--owner", default=KAGGLE_USER, help="the Kaggle account that owns the training kernel (and this one)")
    args = parser.parse_args()
    for scheme in (SCHEMES if args.scheme == "all" else [args.scheme]):
        print(build(scheme, resume=args.resume, mode=args.mode, owner=args.owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
