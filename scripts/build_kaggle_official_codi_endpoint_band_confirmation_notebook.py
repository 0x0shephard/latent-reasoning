"""Build the Kaggle notebook for the exact-match PC-band confirmation."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_endpoint_band_confirmation.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Exact-match confirmation of CODI's accuracy-bearing PC band

## What is being confirmed

The analytic tier of the margin-geometry experiment found, on held-out **first-token**
accuracy at the forced answer cue:

| subspace | dims | % of variance | retained accuracy |
|---|---:|---:|---:|
| PC 0–3 | 4 | **82.3%** | **0.067** |
| PC 4–15 | 12 | 7.5% | 0.506 |
| **PC 4–31** | **28** | **11.3%** | **0.859** |
| PC 32–767 | 736 | 6.4% | 0.083 |

Variance rank and answer contribution are almost unrelated: the leading component
holds two thirds of all variance and 6% of the accuracy. This notebook re-tests that
with real greedy decoding and **numeric exact match**.

## Preregistered gates, frozen before any exact-match outcome

1. **Sufficiency** — retaining PC 4–31 preserves ≥ 70% of baseline accuracy.
2. **Dissociation** — retaining PC 0–3 preserves ≤ 20% of baseline, and the primary
   band beats it with a positive paired bootstrap lower bound.
3. **Necessity** — removing PC 4–31 costs ≥ 20 accuracy points, with a positive lower
   bound and exact McNemar p ≤ 0.05.

All three must pass. Random-subspace arms are descriptive; the specificity null was
already established analytically with 200 energy-matched replicates.

Precision is pinned to float32. `auto` resolves to emulated bfloat16 on T4-class GPUs
and drops the forced-cue baseline from 43.29% to 40.41%, so the run also asserts the
baseline has not drifted from the reproduction gate.

No weight is updated and no speed claim is made. Enable Internet and a GPU, then
choose **Save Version → Save & Run All**.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
ENERGY_BASIS_INPUT = ""
ANSWER_CONDITIONED_BASIS_INPUT = ""
PARAMETER_AWARE_BASIS_INPUT = ""
# Attach the completed margin-geometry export; its float32 colon states are reused.
COLON_STATES_INPUT = ""
READOUT_INPUT = ""
RESUME_INPUT = ""

EVAL_BATCH_SIZE = 32
GENERATION_PRECISION = "float32"
PRIMARY_BAND = (4, 32)
CONTROL_BAND = (0, 4)
BAND_RANDOM_REPLICATES = 4

import os, pathlib, subprocess, sys, json, glob

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip())
'''
)

markdown(
    "## Repair the peft/torchao environment\n\n"
    "`peft`'s LoRA dispatcher raises instead of returning `False` when torchao is "
    "installed below its minimum. torchao is optional here, so removing an "
    "incompatible copy restores the clean path. No-op when it is absent or current."
)
code(
    '''
def _peft_torchao_state():
    probe = (
        "from peft.import_utils import is_torchao_available\\n"
        "try:\\n"
        "    print('ok' if is_torchao_available() else 'absent')\\n"
        "except ImportError as error:\\n"
        "    print('incompatible:' + str(error))\\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


state = _peft_torchao_state()
print("before:", state)
if state.startswith("incompatible"):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    state = _peft_torchao_state()
    print("after:", state)
assert not state.startswith("incompatible"), f"peft still cannot dispatch LoRA: {state}"
'''
)

markdown("## Source tests")
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_endpoint_band_confirmation.py",
     "tests/test_endpoint_margin_geometry.py"],
    check=True,
)
'''
)

markdown(
    "## Resolve inputs\n\n"
    "The colon states are reused from the completed margin-geometry run, so the band "
    "bases here are the same ones the analytic tier measured."
)
code(
    '''
OUTPUT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/outputs/official_codi_endpoint_band_confirmation")
REPORT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/reports/official_codi_endpoint_band_confirmation")
LOG_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/logs/official_codi_endpoint_band_confirmation")
for path in (OUTPUT_ROOT, REPORT_ROOT, LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def _discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0]


COLON_STATES = _discover(COLON_STATES_INPUT, "colon_states_seed89/colon_states.pt")
READOUT = _discover(READOUT_INPUT, "colon_states_seed89/readout.pt")
ENERGY_BASIS = _discover(ENERGY_BASIS_INPUT,
    "official_codi_endpoint_tsvc_corrected/calibration_seed11/basis.pt")
ANSWER_CONDITIONED_BASIS = _discover(ANSWER_CONDITIONED_BASIS_INPUT,
    "official_codi_endpoint_answer_conditioned/collection_seed29/basis.pt")
PARAMETER_AWARE_BASIS = _discover(PARAMETER_AWARE_BASIS_INPUT,
    "official_codi_endpoint_parameter_aware/collection_seed41/basis.pt")
REPRODUCTION_SUMMARY = _discover(REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")

import torch
cache = torch.load(COLON_STATES, map_location="cpu", weights_only=False)
assert cache["parity_gate"]["passed"], cache["parity_gate"]
assert cache["metadata"]["precision"] == "float32", cache["metadata"]["precision"]
print("colon states:", COLON_STATES)
print("parity agreement:", cache["full_parity_gate"]["agreement"])
'''
)

markdown(
    "## Report the band geometry the arms will test\n\n"
    "Printed before decoding so the variance shares are on the record next to the "
    "accuracies they are about to be compared with."
)
code(
    '''
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE, band_variance_share, state_covariance,
)

si = list(cache["state_order"]).index(ANALYTIC_STATE)
calibration = cache["calibration_states"][:, si, :]
mean = cache["student_mean"][ANALYTIC_STATE]
covariance = state_covariance(calibration - mean.unsqueeze(0))
BANDS = [(0, 4), (4, 16), (4, 32), (0, 32), (32, 768)]
for start, stop in BANDS:
    print(f"PC[{start}:{stop}) {stop - start:4d} dims  "
          f"variance share {100 * band_variance_share(covariance, start, stop):6.2f}%")
'''
)

markdown(
    "## Run the confirmation arms\n\n"
    "Twelve full-GSM8K greedy decodes at pinned float32. The baseline arm asserts its "
    "own accuracy against the reproduction gate, so a precision regression stops the "
    "run instead of silently shifting every comparison."
)
code(
    '''
def run_persisted(command, log_name):
    log_path = LOG_ROOT / log_name
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line)
        code_ = process.wait()
    if code_ != 0:
        raise RuntimeError(f"Command failed with exit code {code_}; inspect {log_path}")
    return log_path


def band_arm(start, stop):
    return f"band_p{start:03d}_{stop:03d}_s{ANALYTIC_STATE}"


ARMS = [("baseline", "remove")]
ARMS += [(band_arm(*b), "retain") for b in BANDS]
ARMS += [(band_arm(*PRIMARY_BAND), "remove"), (band_arm(*CONTROL_BAND), "remove")]
ARMS += [(f"random_matched_band_k{PRIMARY_BAND[1] - PRIMARY_BAND[0]:03d}"
          f"_s{ANALYTIC_STATE}_r{i:03d}", "retain") for i in range(BAND_RANDOM_REPLICATES)]
assert len(ARMS) == 12, len(ARMS)

RUNS_ROOT = OUTPUT_ROOT / "runs"
for arm, mode in ARMS:
    tag = f"{arm}_{mode}"
    run_persisted(
        [sys.executable, "-u", "scripts/run_official_codi_endpoint_margin_generation.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--reproduction-summary", REPRODUCTION_SUMMARY,
         "--states", COLON_STATES, "--readout", READOUT,
         "--energy-basis", ENERGY_BASIS,
         "--answer-conditioned-basis", ANSWER_CONDITIONED_BASIS,
         "--parameter-aware-basis", PARAMETER_AWARE_BASIS,
         "--output-dir", str(RUNS_ROOT / tag),
         "--arm", arm, "--state", str(ANALYTIC_STATE), "--mode", mode,
         "--semantics", "mean",
         "--eval-batch-size", str(EVAL_BATCH_SIZE),
         "--precision", GENERATION_PRECISION,
         "--device", "cuda"],
        f"{tag}.log",
    )
completed = sorted(RUNS_ROOT.rglob("summary.json"))
print("completed arms:", len(completed), "/", len(ARMS))
baseline = [json.loads(p.read_text()) for p in completed]
baseline = [s for s in baseline if s["arm"] == "baseline"][0]
print("baseline exact match:", baseline["accuracy"],
      "drift:", baseline.get("baseline_accuracy_drift"),
      "passed:", baseline.get("baseline_drift_passed"))
assert baseline.get("baseline_drift_passed") is True, baseline
'''
)

markdown("## Apply the preregistered gates")
code(
    '''
REPORT = REPORT_ROOT / "band_confirmation_summary.json"
run_persisted(
    [sys.executable, "-u", "scripts/analyze_official_codi_endpoint_band_confirmation.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--runs-root", str(RUNS_ROOT),
     "--reproduction-summary", REPRODUCTION_SUMMARY,
     "--output", str(REPORT)],
    "analyze_band_confirmation.log",
)
report = json.loads(REPORT.read_text())
print("status:", report["status"])
for name, gate in report["gates"].items():
    print(f"  {name}: passed={gate['passed']}")
'''
)

markdown("## Checksummed export")
code(
    '''
import hashlib, shutil

EXPORT = pathlib.Path("/kaggle/working/official_codi_endpoint_band_confirmation_export")
if EXPORT.exists():
    shutil.rmtree(EXPORT)
EXPORT.mkdir(parents=True)
for source in (OUTPUT_ROOT, REPORT_ROOT, LOG_ROOT):
    target = EXPORT / source.relative_to("/kaggle/working")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
lines = []
for path in sorted(EXPORT.rglob("*")):
    if path.is_file():
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(EXPORT)}")
(EXPORT / "SHA256SUMS.txt").write_text("\\n".join(lines) + "\\n")
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
