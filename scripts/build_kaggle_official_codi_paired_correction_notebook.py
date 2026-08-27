"""Build the Kaggle run-all notebook for same-question paired correction."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_paired_correction.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Same-question correct-to-wrong correction for CODI

## Goal

Learn an additive, question-conditioned correction from correct and wrong state-12
counterfactuals of the **same question**, then test it once on 439 held-out questions.

Output sampling alone cannot create these pairs: state 12 at `The answer is:` is
computed before the sampled token, so its value would be identical. This notebook uses
seeded perturbations at state 11, captures the resulting state 12, and labels the greedy
first answer token. Each question with both outcomes contributes one equal-weight
correct-minus-wrong target.

The learned edit acts only inside the established answer-bearing PCs 4–31 and is added
to the original state; the other 740 dimensions remain untouched. Its ridge strength,
edit magnitude and low-confidence gate are selected on a disjoint split. Global-average
and shuffled-target corrections are required controls.

This is a controlled answer-selection experiment, not proof that the perturbations are
natural alternative reasoning traces.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"

COLON_STATES_INPUT = ""
READOUT_INPUT = ""
REPRODUCTION_SUMMARY_INPUT = ""
PAIRS_INPUT = ""  # Optional completed paired_counterfactuals.pt.

RUN_COLLECTION = True
RUN_GENERATION_TIER = True
COLLECTION_BATCH_SIZE = 32
GENERATION_BATCH_SIZE = 32

import glob, hashlib, json, os, pathlib, subprocess, sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("commit:", subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip())
'''
)

markdown("## Pin the checkpoint-compatible environment")
code(
    '''
PINNED_PACKAGES = {
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "datasets": "3.6.0",
    "huggingface_hub": "0.32.4",
}
import importlib.metadata as metadata

def installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

before = {name: installed(name) for name in PINNED_PACKAGES}
print("before:", before)
missing = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()
           if before.get(name) != version]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
after = {name: installed(name) for name in PINNED_PACKAGES}
print("after :", after)
assert after == PINNED_PACKAGES
'''
)

markdown("## Repair PEFT's optional TorchAO integration")
code(
    '''
def peft_torchao_state():
    probe = (
        "from peft.import_utils import is_torchao_available\\n"
        "try:\\n"
        "    print('ok' if is_torchao_available() else 'absent')\\n"
        "except ImportError as error:\\n"
        "    print('incompatible:' + str(error))\\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()

torchao_state = peft_torchao_state()
print("before:", torchao_state)
if torchao_state.startswith("incompatible"):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    torchao_state = peft_torchao_state()
    print("after :", torchao_state)
assert not torchao_state.startswith("incompatible"), torchao_state
'''
)

markdown("## Checks")
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_endpoint_paired_correction.py",
     "tests/test_correctness_contrastive_covariance.py",
     "tests/test_endpoint_margin_geometry.py"],
    check=True,
)
'''
)

markdown("## Resolve immutable inputs")
code(
    '''
def discover(explicit, pattern, required=True):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if required:
        assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0] if matches else ""

COLON_STATES = discover(COLON_STATES_INPUT, "colon_states.pt")
READOUT = discover(READOUT_INPUT, "readout.pt")
REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
OUTPUT_ROOT = pathlib.Path("/kaggle/working/official_codi_paired_correction")
COLLECTION_ROOT = OUTPUT_ROOT / "collection"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("states      :", COLON_STATES)
print("readout     :", READOUT)
print("reproduction:", REPRODUCTION_SUMMARY)
'''
)

markdown(
    "## Collect controlled same-question pairs\n\n"
    "Eight seeded state-11 perturbations plus the unperturbed baseline are evaluated. "
    "Each noisy pass is resumable as its own shard. No noisy final-test state or label "
    "is written to the artifact."
)
code(
    '''
if RUN_COLLECTION:
    subprocess.run(
        [sys.executable, "-u", "scripts/collect_official_codi_paired_counterfactuals.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--reproduction-summary", REPRODUCTION_SUMMARY,
         "--states", COLON_STATES,
         "--readout", READOUT,
         "--output-dir", str(COLLECTION_ROOT),
         "--precision", "float32",
         "--batch-size", str(COLLECTION_BATCH_SIZE),
         "--device", "cuda"],
        check=True,
    )
    PAIRS = str(COLLECTION_ROOT / "paired_counterfactuals.pt")
else:
    PAIRS = discover(PAIRS_INPUT, "paired_counterfactuals.pt")

collection_summary_path = pathlib.Path(PAIRS).with_suffix(".json")
if not collection_summary_path.is_file():
    candidate = pathlib.Path(PAIRS).parent / "paired_counterfactuals.json"
    collection_summary_path = candidate
collection_summary = json.loads(collection_summary_path.read_text())
print(json.dumps(collection_summary["paired_coverage"], indent=2))
'''
)

markdown("## Fit, select and evaluate the conditioned correction")
code(
    '''
SUMMARY = OUTPUT_ROOT / "paired_correction.json"
ARTIFACT = OUTPUT_ROOT / "paired_correction.pt"
REPORT = OUTPUT_ROOT / "paired_correction_report.json"

subprocess.run(
    [sys.executable, "-u", "scripts/run_official_codi_paired_correction.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--pairs", PAIRS,
     "--readout", READOUT,
     "--output", str(SUMMARY),
     "--artifact-output", str(ARTIFACT)],
    check=True,
)
subprocess.run(
    [sys.executable, "-u", "scripts/analyze_official_codi_paired_correction.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--summary", str(SUMMARY),
     "--artifact", str(ARTIFACT),
     "--output", str(REPORT)],
    check=True,
)
report = json.loads(REPORT.read_text())
print(json.dumps({
    "status": report["status"],
    "paired_questions": report["paired_questions"],
    "selected_ridge": report["selected_ridge"],
    "selected_intervention": report["selected_interventions"]["conditioned"],
    "test_arms": report["test_arms"],
    "comparisons": report["comparisons"],
}, indent=2))
'''
)

markdown("## Inspect selection without touching test tuning")
code(
    '''
import pandas as pd

summary = json.loads(SUMMARY.read_text())
display(pd.DataFrame(summary["ridge_selection"]))
display(pd.DataFrame(summary["selected_interventions"]).T)
display(pd.DataFrame(summary["test_arms"]).T.sort_values("accuracy", ascending=False))
'''
)

markdown(
    "## Optional paired exact-match confirmation\n\n"
    "All four arms decode the same 439 final questions at pinned float32. The analytic "
    "first-token gate remains the primary decision."
)
code(
    '''
GENERATION_ROOT = OUTPUT_ROOT / "generation"
ARMS = ["baseline", "conditioned", "global_mean", "shuffled_target"]
if RUN_GENERATION_TIER:
    for arm in ARMS:
        subprocess.run(
            [sys.executable, "-u", "scripts/run_official_codi_paired_correction_generation.py",
             "--config", "configs/official_codi_gpt2.yaml",
             "--reproduction-summary", REPRODUCTION_SUMMARY,
             "--readout", READOUT,
             "--artifact", str(ARTIFACT),
             "--output-dir", str(GENERATION_ROOT / arm),
             "--arm", arm,
             "--eval-batch-size", str(GENERATION_BATCH_SIZE),
             "--precision", "float32",
             "--device", "cuda"],
            check=True,
        )
    GENERATION_REPORT = OUTPUT_ROOT / "paired_correction_generation_report.json"
    subprocess.run(
        [sys.executable, "-u", "scripts/analyze_official_codi_paired_correction_generation.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--runs-root", str(GENERATION_ROOT),
         "--output", str(GENERATION_REPORT)],
        check=True,
    )
    print(GENERATION_REPORT.read_text())
else:
    print("Generation tier skipped; the primary analytic report is complete.")
'''
)

markdown(
    "## Next steps\n\n"
    "Interpret only the executed reports. A pass supports a low-confidence, "
    "question-conditioned answer-selection correction under controlled state-11 "
    "counterfactuals. It does not prove that the perturbations reproduce natural "
    "alternative reasoning. A failure means this paired linear map is insufficient; "
    "do not rescue it by changing the final-test gate."
)

markdown("## Checksummed export")
code(
    '''
lines = []
for path in sorted(OUTPUT_ROOT.rglob("*")):
    if path.is_file():
        lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(OUTPUT_ROOT)}"
        )
(OUTPUT_ROOT / "SHA256SUMS.txt").write_text("\\n".join(lines) + "\\n")
print("exported files:", len(lines))
'''
)

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"wrote {OUTPUT}")
