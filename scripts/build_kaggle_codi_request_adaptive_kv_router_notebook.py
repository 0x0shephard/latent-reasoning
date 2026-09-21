"""Build the Kaggle notebook for the request-adaptive xKV router experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_request_adaptive_kv_router.ipynb"
RUN_COMMIT = "e7459ed9a27bc34e6fa56c4f4f210c47ca60c7e2"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Can a pre-answer router choose the right xKV profile for each question?

## tl;dr

The previous adaptive allocator preserved dense accuracy at about **1.48× modeled
cache compression**, but its locked-final NLL improvement was uncertain. This
experiment asks whether one frozen allocation is too blunt.

- Reuse the three already-fitted rank-64 profiles: reconstruction, hybrid, and
  answer-Fisher. No K/V utility is refit here.
- Hash-split all 1,000 pinned SVAMP questions into 400 router-fit, 300 screen,
  and 300 locked-final examples.
- Fit a small ridge router to predict each profile's dense first-token KL using
  only lexical question features available before generation.
- Require the router to beat the predecessor's global answer-Fisher profile,
  preserve at least 95% dense first-token decisions, remain generation-accuracy
  non-inferior, use at least two profiles, and match rank-64 modeled storage.
- Open the final 300 questions only if every screen check passes.

SVAMP is an **external compression-method holdout**, not a pristine model
benchmark: dense CODI accuracy on SVAMP has been reported before. This remains
a dense-reconstruction and modeled-storage experiment, not a native latency test.
""")

md(r"""
## Why this is the right next test

The locked GSM8K result did not show that adaptive allocation was harmful. It
showed that the average NLL benefit of one global answer-Fisher allocation was
too small and variable to establish. A request router tests the narrower causal
hypothesis that different arithmetic questions need different K/V allocations.

The router inputs contain no answer information: length, number format, and
counts of arithmetic/comparison cues. Its training target is each profile's
request-centered dense first-token KL on the fit split, not correctness. Centering
removes question-wide difficulty and makes the regression learn which profile is
relatively best. The screen and final labels are never used to fit the router.
""")

md("## 1. Reproducible setup")
code(f'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "{RUN_COMMIT}"
REPO_DIR = "/kaggle/working/latent-reasoning"
OUTPUT_DIR = "/kaggle/working/codi_request_adaptive_kv_router"

REPRODUCTION_SUMMARY_INPUT = (
    "/kaggle/input/datasets/jonraza15/"
    "corrected-official-codi-answer-cue-endpoint-tsv-c/"
    "latent-reasoning/outputs/official_codi_gpt2/eval/"
    "revision_fd641b3d/full_gsm8k/summary.json"
)
# Optional explicit path. If blank, the notebook finds the artifact by contract.
PREVIOUS_ARTIFACT_INPUT = ""

FIT_EXAMPLES = 400
SCREEN_EXAMPLES = 300
FINAL_EXAMPLES = 300
BASELINE_RANK = 64

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
assert len(RUN_COMMIT) == 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip())
''')

code(r'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2",
    "accelerate==1.7.0", "huggingface-hub>=0.34,<1.0",
    "safetensors==0.5.3", "pytest>=8,<10",
], check=True)
probe = subprocess.run([
    sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\n"
    "try:\n print(is_torchao_available())\n"
    "except ImportError:\n print('incompatible')",
], capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_request_adaptive_kv_router.py",
    "tests/test_request_adaptive_kv_router_runner.py",
    "tests/test_adaptive_kv_allocation.py",
    "tests/test_official_codi_kv.py",
], check=True)
''')

md("## 2. Locate and validate the two required inputs")
code(r'''
import torch

def torch_contract(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("contract")
    except Exception:
        return None

reproduction_suffix = "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json"
reproduction_candidates = ([REPRODUCTION_SUMMARY_INPUT] if REPRODUCTION_SUMMARY_INPUT else []) + glob.glob(
    "/kaggle/input/**/" + reproduction_suffix, recursive=True
)
REPRODUCTION_SUMMARY = next((
    path for path in reproduction_candidates
    if pathlib.Path(path).is_file() and str(path).endswith(reproduction_suffix)
), None)

artifact_candidates = ([PREVIOUS_ARTIFACT_INPUT] if PREVIOUS_ARTIFACT_INPUT else []) + glob.glob(
    "/kaggle/input/**/adaptive_kv_allocation.pt", recursive=True
)
PREVIOUS_ARTIFACT = next((
    path for path in artifact_candidates
    if pathlib.Path(path).is_file() and torch_contract(path) == "official_codi_adaptive_kv_allocation_holdout_v1"
), None)

assert REPRODUCTION_SUMMARY, "Attach the completed official CODI reproduction dataset"
assert PREVIOUS_ARTIFACT, (
    "Attach adaptive_kv_allocation.pt from the completed adaptive-allocation notebook"
)
previous = torch.load(PREVIOUS_ARTIFACT, map_location="cpu", weights_only=False)
assert previous["selected_candidate"]["name"] == "adaptive_kv_r64_w1"
assert previous["final_replication"]["gate"]["passed"] is False
assert previous["final_replication"]["gate"]["positive_nll_interval"] is False
print("reproduction", REPRODUCTION_SUMMARY)
print("adaptive predecessor", PREVIOUS_ARTIFACT)
print("frozen predecessor candidate", previous["selected_candidate"]["name"])
''')

md("## 3. Run the preregistered router experiment")
code(r'''
command = [
    sys.executable, "-u", "scripts/run_codi_request_adaptive_kv_router.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--previous-artifact", PREVIOUS_ARTIFACT,
    "--output-dir", OUTPUT_DIR,
    "--fit-examples", str(FIT_EXAMPLES),
    "--screen-examples", str(SCREEN_EXAMPLES),
    "--final-examples", str(FINAL_EXAMPLES),
    "--baseline-rank", str(BASELINE_RANK),
    "--ridge", "1.0",
    "--bootstrap-samples", "10000",
    "--minimum-first-token-fidelity", "0.95",
    "--accuracy-noninferiority-margin", "0.02",
    "--batch-size", "16", "--generation-batch-size", "8",
    "--max-new-tokens", "64", "--precision", "float32", "--device", "cuda",
]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(summary["decision"])
''')

md("## 4. Split and lineage audit")
code(r'''
import pandas as pd
display(pd.DataFrame([{
    "dataset": summary["preregistration"]["dataset"],
    "fit": summary["preregistration"]["split_sizes"]["router_fit"],
    "screen": summary["preregistration"]["split_sizes"]["screen"],
    "locked final": summary["preregistration"]["split_sizes"]["final"],
    "holdout status": summary["preregistration"]["holdout_status"],
}]))
display(pd.DataFrame([
    {"split": name, "question hash": value}
    for name, value in summary["split_hashes"].items()
]))
''')

md("## 5. What did the router learn?")
code(r'''
import matplotlib.pyplot as plt
import numpy as np

coefficients = np.asarray(summary["router"]["coefficients"])[1:, :]
fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
limit = max(1e-12, np.abs(coefficients).max())
image = ax.imshow(coefficients, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
ax.set_xticks(range(3), summary["router"]["profile_names"])
ax.set_yticks(range(len(summary["router"]["feature_names"])), summary["router"]["feature_names"])
ax.set(title="Frozen ridge coefficients for predicted first-token KL",
       xlabel="allocation profile", ylabel="pre-answer question feature")
fig.colorbar(image, ax=ax, label="standardized ridge coefficient")
plt.show()

fit = summary["router_fit"]
display(pd.DataFrame([{
    "oracle route agreement": fit["router_oracle_route_agreement"],
    "active profiles": fit["router_distribution"]["active_profiles"],
    **{f"mean KL: {name}": value for name, value in fit["mean_profile_kl"].items()},
}]))
''')

md("## 6. External screen")
code(r'''
screen = summary["screen"]
screen_table = pd.DataFrame([
    {"arm": name, **record}
    for name, record in screen["generation"].items()
])
display(screen_table[["arm", "accuracy", "correct", "examples", "exact_sequence_agreement"]])
display(pd.DataFrame([{
    "router KL": screen["mean_router_kl_from_dense"],
    "global-profile KL": screen["mean_global_profile_kl_from_dense"],
    "mean KL gain": screen["mean_kl_gain_over_global_profile"],
    "KL lower": screen["kl_gain_bootstrap_95ci"][0],
    "KL upper": screen["kl_gain_bootstrap_95ci"][1],
    "first-token fidelity": screen["first_token_fidelity"],
}]))
display(pd.DataFrame([screen["gate"]]))

counts = screen["route_distribution"]["counts"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
axes[0].bar(counts.keys(), counts.values(), color=["#6b7280", "#c46a2b", "#315f8c"])
axes[0].set(title="Frozen routes on the 300-question screen", ylabel="questions")
gain = screen["mean_kl_gain_over_global_profile"]
lower, upper = screen["kl_gain_bootstrap_95ci"]
axes[1].errorbar([0], [gain], yerr=[[gain-lower], [upper-gain]], fmt="o", capsize=5)
axes[1].axhline(0, color="#222222", linewidth=1)
axes[1].set_xticks([0], ["router − global"])
axes[1].set(title="First-token KL gain with paired 95% interval",
            ylabel="global KL − routed KL (positive favors router)")
plt.show()
''')

md("## 7. Locked final result")
code(r'''
if summary["final"] is None:
    print("The screen failed, so the locked 300-question final split was not evaluated.")
else:
    final = summary["final"]
    display(pd.DataFrame([{
        "mean KL gain": final["mean_kl_gain_over_global_profile"],
        "KL lower": final["kl_gain_bootstrap_95ci"][0],
        "KL upper": final["kl_gain_bootstrap_95ci"][1],
        "first-token fidelity": final["first_token_fidelity"],
        "dense accuracy": final["generation"]["dense"]["accuracy"],
        "ordinary accuracy": final["generation"]["ordinary_xkv"]["accuracy"],
        "global profile accuracy": final["generation"]["global_answer_fisher"]["accuracy"],
        "router accuracy": final["generation"]["request_router"]["accuracy"],
    }]))
    display(pd.DataFrame([final["gate"]]))
''')

md("## Takeaways")
code(r'''
print("Decision:", summary["decision"]["claim"])
print("External screen:", summary["decision"]["screen_passed"])
print("Locked final:", summary["decision"]["final_passed"])
for warning in summary["warnings"]:
    print("WARNING:", warning)
''')

md("## 8. Save the complete audit trail")
code(r'''
output = pathlib.Path(OUTPUT_DIR)
assert (output / "summary.json").is_file()
assert (output / "request_adaptive_kv_router.pt").is_file()
assert (output / "predictions.jsonl").is_file()
print("Kaggle output directory", output)
print("Publish this directory directly; no same-named archive is created.")
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
