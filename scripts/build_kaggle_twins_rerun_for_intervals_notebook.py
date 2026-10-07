"""Build the Kaggle notebook that reruns the GPT-2 and CODI-1B twin experiments (§110/§112
profiles) and computes paired-bootstrap intervals for all four runs (ledger §119 addendum)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_twins_rerun_for_intervals.ipynb"
RUN_COMMIT = "__RUN_COMMIT__"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Rerun of the GPT-2 and CODI-1B twin experiments, with paired-bootstrap intervals

## tl;dr

The saved predictions of the §111, §113 and §117 twin runs were not kept, and the paper's
headline table needs intervals. This notebook reruns the four twin experiments on the
current code (GPT-2 standard and long profiles, CODI-1B standard and long profiles), then
computes paired-bootstrap 95% intervals over pairs for every run, in the same way as the
SIM-CoT run (§119 addendum). **No training**. Attach the official GPT-2 reproduction dataset
(the same one as before); the CODI-1B gate is produced inside the notebook. About 65–70
minutes on a T4. When it finishes, publish the output as a dataset (Output tab → New Dataset)
so the predictions are kept this time.
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "__RUN_COMMIT__"
REPO_DIR = "/kaggle/working/latent-reasoning"
GPT2_CONFIG = "configs/official_codi_gpt2.yaml"
CODI1B_CONFIG = "configs/official_codi_llama1b.yaml"
REPRODUCTION_SUMMARY_INPUT = ""   # "" = discover the GPT-2 reproduction under /kaggle/input
OUTPUT_ROOT = "/kaggle/working/twins"
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
                "tests/test_directed_counterfactual_runner.py",
                "tests/test_directed_counterfactual_bootstrap.py"], check=True)

def discover_file(explicit, suffix):
    if explicit and pathlib.Path(explicit).is_file():
        return str(explicit)
    found = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True), key=len)
    return found[0] if found else None

GPT2_REPRODUCTION = discover_file(REPRODUCTION_SUMMARY_INPUT, "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
assert GPT2_REPRODUCTION, "Attach the completed official GPT-2 CODI reproduction dataset"
print("GPT-2 reproduction", GPT2_REPRODUCTION)

def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s ({attempt + 1}/{attempts})")
        time.sleep(wait)

def twins(config, reproduction, precision, out, profile):
    cmd = [sys.executable, "-u", "scripts/run_codi_directed_counterfactual.py", "--config", config,
           "--reproduction-summary", reproduction, "--output-dir", out, "--batch-size", BATCH_SIZE,
           "--precision", precision, "--device", "cuda", "--profile", profile]
    if RUN_SMOKE:
        run_with_retry([sys.executable, "-u", "scripts/run_codi_directed_counterfactual.py", "--config", config,
                        "--reproduction-summary", reproduction, "--output-dir", out + "_smoke", "--batch-size", BATCH_SIZE,
                        "--precision", precision, "--device", "cuda", "--profile", profile, "--smoke"])
    run_with_retry(cmd)
    s = json.loads(open(pathlib.Path(out) / "summary.json").read())
    print(pathlib.Path(out).name, "| gate", s["gate"], "| baseline", s["baseline"])
    print("   ", s["decision"]["claim"])
    return s
''')

md("## GPT-2: standard and long profiles (float32, as in §111 and §113)")
code(r'''
summaries = {}
for profile in ("standard", "long"):
    summaries[f"gpt2_{profile}"] = twins(GPT2_CONFIG, GPT2_REPRODUCTION, "float32", f"{OUTPUT_ROOT}_gpt2_{profile}", profile)
''')

md("## CODI-1B: gate (pad-aware, lower bound), then standard and long profiles")
code(r'''
def summary_under(root):
    found = sorted(glob.glob(f"{root}/eval/revision_*/full_*/summary.json"), key=os.path.getmtime)
    return found[-1] if found else None

PRECISION = None
for precision in ("float16", "float32"):
    run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CODI1B_CONFIG, "--device", "cuda",
                    "--set", f"model.dtype={precision}", "--set", f"eval.batch_size={BATCH_SIZE}"])
    CODI1B_REPRODUCTION = summary_under("outputs/official_codi_llama1b")
    gate = json.loads(open(CODI1B_REPRODUCTION).read())
    print(precision, "CODI-1B gsm8k", gate["datasets"], gate["accuracy_gate"]["status"])
    if gate["accuracy_gate"]["status"] == "passed":
        PRECISION = precision
        break
assert PRECISION, "the CODI-1B gate failed in both precisions"
for profile in ("standard", "long"):
    summaries[f"codi1b_{profile}"] = twins(CODI1B_CONFIG, CODI1B_REPRODUCTION, PRECISION, f"{OUTPUT_ROOT}_codi1b_{profile}", profile)
''')

md("## Paired-bootstrap intervals for all four runs")
code(r'''
from scripts.analyze_directed_counterfactual_bootstrap import analyze, markdown
from IPython.display import Markdown, display
reports = {}
for name in ("gpt2_standard", "gpt2_long", "codi1b_standard", "codi1b_long"):
    reports[name] = analyze(pathlib.Path(f"{OUTPUT_ROOT}_{name}"))
    display(Markdown(f"### {name}\n\n" + markdown(reports[name])))
json.dump(reports, open(f"{OUTPUT_ROOT}_bootstrap.json", "w"), indent=2)
print("wrote", f"{OUTPUT_ROOT}_bootstrap.json")
''')

md("## Headline table and takeaways")
code(r'''
rows = []
for name, r in reports.items():
    t = r["target_rate"]; c = r["contrast"]
    def cell(d):
        return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    rows.append({"run": name, "pairs": r["pairs"], "state_1": cell(t["state_1"]), "state_3": cell(t["state_3"]),
                 "kv_2": cell(t["kv_2"]), "kv_4": cell(t["kv_4"]), "kv_0": cell(t["kv_0"]), "kv_all": cell(t["kv_all"]),
                 "v89_all": cell(t["v89_all"]), "state_3 - v89_4": cell(c["state_3_minus_v89_4"])})
import pandas as pd
pd.set_option("display.max_colwidth", 40)
display(pd.DataFrame(rows).set_index("run"))
for name, s in summaries.items():
    print(f"[{name}] {s['decision']['claim']}")
print()
print("Reminder: publish this notebook's output as a dataset (Output tab -> New Dataset) so predictions.pt is kept.")
print("seconds:", {k: round(v["seconds"]) for k, v in summaries.items()})
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
