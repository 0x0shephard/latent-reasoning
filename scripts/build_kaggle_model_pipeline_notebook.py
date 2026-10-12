"""Build one per-model Kaggle notebook (ledger §120): gate, twins (both profiles), paired
bootstrap, masking audit, monitor faithfulness.  ``python ... --model gpt2|codi1b|simcot``.
Also writes a ``kernel-metadata.json`` beside each notebook for ``kaggle kernels push``."""
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

RUN_COMMIT = "f35f0ef1561fc5071e9d2c326ed01a78d84acdd7"
KAGGLE_USER = "jonraza15"
MODELS = {
    "gpt2": {"config": "configs/official_codi_gpt2.yaml", "title": "CODI GPT-2", "precision": "float32", "gate": "dataset",
             "carrying": "1 3", "datasets": ["dannyism11/tsv-ccc"], "slug": "codi-gpt2-monitors-and-masks"},
    "codi1b": {"config": "configs/official_codi_llama1b.yaml", "title": "CODI LLaMA-1B", "precision": "auto", "gate": "eval",
               "carrying": "3", "datasets": [], "slug": "codi-1b-monitors-and-masks"},
    "simcot": {"config": "configs/official_simcot_codi_llama1b.yaml", "title": "SIM-CoT CODI-1B", "precision": "auto", "gate": "eval",
               "carrying": "1 3", "datasets": [], "slug": "simcot-1b-monitors-and-masks"},
}


def build(model: str) -> Path:
    spec = MODELS[model]
    out_dir = ROOT / "notebooks" / f"kaggle_{model}_monitors_and_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda v: cells.append(nbf.v4.new_markdown_cell(v.strip()))
    code = lambda v: cells.append(nbf.v4.new_code_cell(v.strip()))

    md(f"""
# {spec['title']}: twins, bootstrap intervals, masking audit and monitor faithfulness (ledger §120)

## tl;dr

One model, the whole inference-only pipeline, **No training**:

1. Gate: {"the attached official GPT-2 reproduction dataset" if spec['gate'] == 'dataset' else "the full GSM8K reproduction, pad-aware path, lower bound against the paper"}.
2. Twins, standard and long profiles (§110/§112), then paired-bootstrap intervals over pairs.
3. Masking audit (§120 B): `mean` replaces a slot's state by the pool mean; `mask` hides its K/V. Checks K1–K3 with carrying slot(s) {spec['carrying']}.
4. Monitor faithfulness (§120 A): per slot, probe readability versus faithfulness under the swap, on both twin runs.

{"Attach `dannyism11/tsv-ccc` (the official GPT-2 reproduction)." if spec['datasets'] else "No dataset to attach."} When finished, publish the output as a dataset so the predictions are kept.
""")
    md("## Setup")
    code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "{spec['config']}"
MODEL = "{model}"
OUTPUT_ROOT = "/kaggle/working/{model}"
CARRYING = "{spec['carrying']}".split()
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
                "tests/test_official_codi_family.py", "tests/test_cache_carrier.py",
                "tests/test_directed_counterfactual_runner.py", "tests/test_masking_audit_runner.py",
                "tests/test_monitor_faithfulness.py", "tests/test_directed_counterfactual_bootstrap.py"], check=True)

def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s ({attempt + 1}/{attempts})")
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
    md("## Step 2: twins, both profiles, and bootstrap intervals")
    code(r'''
def twins(profile):
    out = f"{OUTPUT_ROOT}_twins_{profile}"
    base = [sys.executable, "-u", "scripts/run_codi_directed_counterfactual.py", "--config", CONFIG,
            "--reproduction-summary", REPRODUCTION_SUMMARY, "--batch-size", BATCH_SIZE, "--precision", PRECISION,
            "--device", "cuda", "--profile", profile]
    if RUN_SMOKE:
        run_with_retry(base + ["--output-dir", out + "_smoke", "--smoke"])
    run_with_retry(base + ["--output-dir", out])
    s = json.loads(open(pathlib.Path(out) / "summary.json").read())
    print(profile, "| gate", s["gate"], "| baseline", s["baseline"]); print("   ", s["decision"]["claim"])
    return out, s

twin_dirs, twin_summaries = {}, {}
for profile in ("standard", "long"):
    twin_dirs[profile], twin_summaries[profile] = twins(profile)

from scripts.analyze_directed_counterfactual_bootstrap import analyze as boot_analyze, markdown as boot_markdown
from IPython.display import Markdown, display
boot = {p: boot_analyze(pathlib.Path(d)) for p, d in twin_dirs.items()}
for p, r in boot.items():
    display(Markdown(f"### bootstrap, {p}\n\n" + boot_markdown(r)))
json.dump(boot, open(f"{OUTPUT_ROOT}_bootstrap.json", "w"), indent=2)
''')
    md("## Step 3: masking audit (necessity)")
    code(r'''
MASK_DIR = f"{OUTPUT_ROOT}_masking"
mask_cmd = [sys.executable, "-u", "scripts/run_codi_masking_audit.py", "--config", CONFIG, "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--batch-size", BATCH_SIZE, "--precision", PRECISION, "--device", "cuda", "--carrying-slots", *CARRYING]
if RUN_SMOKE:
    run_with_retry(mask_cmd + ["--output-dir", MASK_DIR + "_smoke", "--smoke"])
run_with_retry(mask_cmd + ["--output-dir", MASK_DIR])
masking = json.loads(open(pathlib.Path(MASK_DIR) / "summary.json").read())
import pandas as pd
print("masking baseline:", masking["baseline"]); print(masking["decision"])
for part in ("all", "short", "long"):
    src = masking["report"]["all"] if part == "all" else masking["report"]["by_length"][part]
    print(f"--- {part} chains ---")
    display(pd.DataFrame({n: {"accuracy drop": v["accuracy_drop"], "change rate": v["change_rate"], "n": v["n"]} for n, v in src.items()}).T.round(3))
display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in masking["checks"].items()}]).T.rename(columns={0: "value"}))
''')
    md("## Step 4: monitor faithfulness (readability versus faithfulness per slot)")
    code(r'''
from scripts.analyze_monitor_faithfulness import analyze as mon_analyze, markdown as mon_markdown, load_model_pieces
tokenizer, embeddings, readout = load_model_pieces(pathlib.Path(CONFIG))
monitor = {p: mon_analyze(pathlib.Path(d), tokenizer, embeddings, readout) for p, d in twin_dirs.items()}
for p, r in monitor.items():
    display(Markdown(f"### monitor faithfulness, {p}\n\n" + mon_markdown(r)))
json.dump(monitor, open(f"{OUTPUT_ROOT}_monitor_faithfulness.json", "w"), indent=2)

import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})
r = monitor["long"]; xs = list(range(len(r["slots"])))
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for key, color, label in (("readability", "#8a8a8a", "readability (probe, answer token)"), ("faithfulness", "#b65f24", "faithfulness (alarm confirmed by behaviour)"), ("transport", "#2a6f97", "transport (swap alone)")):
    ax.plot(xs, [(r["slots"][str(s)]["answer"].get(key) or 0.0) for s in xs], marker="o", ms=4, color=color, label=label)
ax.set(xlabel="slot", ylabel="rate", title=f"{MODEL}: long chains, answer-token target"); ax.legend(fontsize=8)
plt.show()
''')
    md("## Takeaways")
    code(r'''
print("twins:", {p: s["decision"]["claim"] for p, s in twin_summaries.items()})
print("masking:", masking["decision"]["claim"])
r = monitor["long"]["slots"]
print("terminal slot (long): readability", r["5"]["answer"].get("readability"), "faithfulness", r["5"]["answer"].get("faithfulness"))
print("carrying slots", CARRYING, {s: {"readability": r[s]["answer"].get("readability"), "faithfulness": r[s]["answer"].get("faithfulness_long")} for s in CARRYING})
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset).")
print("seconds:", {**{f"twins_{p}": round(s["seconds"]) for p, s in twin_summaries.items()}, "masking": round(masking["seconds"])})
''')
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3"}, "kaggle": {"accelerator": "gpu", "internet": True}}
    nbf.write(nb, out_dir / f"kaggle_{model}_monitors_and_masks.ipynb")
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{spec['slug']}", "title": f"{spec['title']} monitors and masks", "code_file": f"kaggle_{model}_monitors_and_masks.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": spec["datasets"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2) + "\n")
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
