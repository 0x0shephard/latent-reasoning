"""Build the Kaggle run-all notebook for the margin-geometry experiment."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_endpoint_margin_geometry.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# CODI answer-colon margin geometry and effective dimensionality

## Why this experiment exists

The preregistered parameter-aware state-12 confirmation returned `not_confirmed`:
five of six conditions passed, and the empirical matched-random test failed at
`p = 0.1557` with 77 of 500 controls at least as damaging as the selection.

A source audit found three properties that bounded what that design could detect:

1. **State 12 is `ln_f` output.** It runs after every block, so it never enters the
   key/value cache. The edit changed one token's logits and nothing propagated.
2. **Binary exact match discarded most of the measurement.** ~20 flipped answers
   against a null spread of ~±9, with a gate demanding the top 25 of 500 controls.
3. **The selectors were gradient-scored but projection-tested.**

This experiment corrects all of them, plus two smaller ones: the mean-preserving
edit was blind to the constant component, and every arm asked necessity when the
question is sufficiency.

## The identity that makes it cheap

`lm_head` is bias-free and consumes the `ln_f` output, so a state-12 edit gives

```
z' = z - (W U)(U^T (h - centre))
```

*exactly*. One cached colon state per question turns every state-12 arm into a
matrix product instead of a full greedy decode. A parity gate checks that against
the released decoder before any sweep runs.

No model weight is updated and no speed claim is made. Enable Internet and a
T4-or-newer GPU, then choose **Save Version → Save & Run All**.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
RESUME_INPUT = ""
ENERGY_BASIS_INPUT = ""
ANSWER_CONDITIONED_BASIS_INPUT = ""
PARAMETER_AWARE_BASIS_INPUT = ""
RUN_REPRODUCTION_GATE_IF_MISSING = True
RUN_COLLECTION = True
RUN_SWEEP = True
RUN_GENERATION = True

CALIBRATION_EXAMPLES = 2048
CALIBRATION_SAMPLING_SEED = 89
PARITY_EXAMPLES = 64
MINIMUM_PARITY_AGREEMENT = 0.99
RANK_GRID = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]
PRIMARY_RANK = 3
RANDOM_REPLICATES = 200
RETENTION_THRESHOLD = 0.90
EVAL_BATCH_SIZE = 32

import os, pathlib, subprocess, sys, json

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
    "## Source tests\n\n"
    "The contract is executable. If these fail, nothing downstream is trustworthy."
)
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_endpoint_margin_geometry.py",
     "tests/test_endpoint_inference_ablation.py"],
    check=True,
)
'''
)

markdown(
    "## Resolve the immutable selector bases and the reproduction gate\n\n"
    "The two completed selectors are reference arms, so they must come from their "
    "original artifacts rather than be refitted here."
)
code(
    '''
import glob

OUTPUT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/outputs/official_codi_endpoint_margin_geometry")
REPORT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/reports/official_codi_endpoint_margin_geometry")
LOG_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/logs/official_codi_endpoint_margin_geometry")
for path in (OUTPUT_ROOT, REPORT_ROOT, LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def _discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0]


ENERGY_BASIS = _discover(
    ENERGY_BASIS_INPUT,
    "official_codi_endpoint_tsvc_corrected/calibration_seed11/basis.pt",
)
ANSWER_CONDITIONED_BASIS = _discover(
    ANSWER_CONDITIONED_BASIS_INPUT,
    "official_codi_endpoint_answer_conditioned/collection_seed29/basis.pt",
)
PARAMETER_AWARE_BASIS = _discover(
    PARAMETER_AWARE_BASIS_INPUT,
    "official_codi_endpoint_parameter_aware/collection_seed41/basis.pt",
)
REPRODUCTION_SUMMARY = _discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
for name, value in [("energy", ENERGY_BASIS), ("answer_conditioned", ANSWER_CONDITIONED_BASIS),
                    ("parameter_aware", PARAMETER_AWARE_BASIS), ("reproduction", REPRODUCTION_SUMMARY)]:
    print(f"{name}: {value}")
'''
)

markdown(
    "## Collect colon states and run the analytic parity gate\n\n"
    "This is the gate that converts the closed-form shortcut from an assumption "
    "about GPT-2's `lm_head` into a checked property of this checkpoint. "
    "Disagreement above the threshold raises and blocks the sweep."
)
code(
    '''
def run_persisted(command, log_name):
    log_path = LOG_ROOT / log_name
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code_ = process.wait()
    if code_ != 0:
        raise RuntimeError(f"Command failed with exit code {code_}; inspect {log_path}")
    return log_path


STATES_DIR = OUTPUT_ROOT / f"colon_states_seed{CALIBRATION_SAMPLING_SEED}"
if RUN_COLLECTION:
    run_persisted(
        [sys.executable, "-u", "scripts/collect_official_codi_endpoint_margin_states.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--reproduction-summary", REPRODUCTION_SUMMARY,
         "--output-dir", str(STATES_DIR),
         "--calibration-examples", str(CALIBRATION_EXAMPLES),
         "--sampling-seed", str(CALIBRATION_SAMPLING_SEED),
         "--parity-examples", str(PARITY_EXAMPLES),
         "--precision", "float32",
         "--device", "cuda"],
        "collect_margin_colon_states.log",
    )
manifest = json.loads((STATES_DIR / "run_manifest.json").read_text())
parity = manifest["parity_gate"]
assert parity["passed"], parity
assert manifest["test_labels_used_for_calibration"] is False
assert manifest["sampling"]["train_test_normalized_question_overlap"] == 0
print("analytic parity agreement:", parity["agreement"], "on", parity["examples"], "examples")
'''
)

markdown(
    "## Closed-form subspace sweep\n\n"
    "Every state-12 arm — four fitted families across the rank grid, the two "
    "reference selectors, and the energy-matched random controls, under removal, "
    "retention and three ablation semantics — is evaluated from the cache."
)
code(
    '''
SWEEP_DIR = OUTPUT_ROOT / "analytic_sweep"
if RUN_SWEEP:
    run_persisted(
        [sys.executable, "-u", "scripts/run_official_codi_endpoint_margin_sweep.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--states", str(STATES_DIR / "colon_states.pt"),
         "--readout", str(STATES_DIR / "readout.pt"),
         "--energy-basis", ENERGY_BASIS,
         "--answer-conditioned-basis", ANSWER_CONDITIONED_BASIS,
         "--parameter-aware-basis", PARAMETER_AWARE_BASIS,
         "--output-dir", str(SWEEP_DIR),
         "--device", "cuda"],
        "margin_analytic_sweep.log",
    )
sweep_manifest = json.loads((SWEEP_DIR / "run_manifest.json").read_text())
print("arms:", sweep_manifest["arms"])
print("baseline first-token accuracy:", sweep_manifest["baseline_first_token_accuracy"])
'''
)

markdown(
    "## Generation confirmation, propagating site, and all-position arms\n\n"
    "The analytic identity covers state 12 and the first answer token only. These "
    "arms use real greedy decoding and numeric exact match, and include state 11, "
    "which does reach the key/value cache."
)
code(
    '''
GENERATION_ARMS = [
    ("baseline", 12, "remove", "mean", False),
    (f"margin_k003_s12", 12, "remove", "mean", False),
    (f"margin_k016_s12", 12, "remove", "mean", False),
    (f"random_matched_margin_k003_s12_r000", 12, "remove", "mean", False),
    (f"margin_k004_s12", 12, "retain", "mean", False),
    (f"margin_k016_s12", 12, "retain", "mean", False),
    (f"margin_k064_s12", 12, "retain", "mean", False),
    (f"margin_k256_s12", 12, "retain", "mean", False),
    (f"margin_k003_s12", 11, "remove", "mean", False),
    (f"margin_k016_s12", 11, "remove", "mean", False),
    (f"parameter_aware_k003_s12", 11, "remove", "mean", False),
    (f"margin_k003_s12", 12, "remove", "zero", False),
    (f"margin_k003_s12", 12, "remove", "mean", True),
    (f"margin_k016_s12", 12, "remove", "mean", True),
]
assert len(GENERATION_ARMS) == 14

RUNS_ROOT = OUTPUT_ROOT / "generation"
if RUN_GENERATION:
    for arm, state, mode, semantics, all_positions in GENERATION_ARMS:
        tag = f"{arm}_s{state}_{mode}_{semantics}" + ("_allpos" if all_positions else "")
        command = [
            sys.executable, "-u", "scripts/run_official_codi_endpoint_margin_generation.py",
            "--config", "configs/official_codi_gpt2.yaml",
            "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--states", str(STATES_DIR / "colon_states.pt"),
            "--readout", str(STATES_DIR / "readout.pt"),
            "--energy-basis", ENERGY_BASIS,
            "--answer-conditioned-basis", ANSWER_CONDITIONED_BASIS,
            "--parameter-aware-basis", PARAMETER_AWARE_BASIS,
            "--output-dir", str(RUNS_ROOT / tag),
            "--arm", arm,
            "--state", str(state),
            "--mode", mode,
            "--semantics", semantics,
            "--eval-batch-size", str(EVAL_BATCH_SIZE),
            "--device", "cuda",
        ]
        if all_positions:
            command.append("--all-positions")
        run_persisted(command, f"generation_{tag}.log")
completed = sorted(RUNS_ROOT.rglob("summary.json"))
print("completed generation arms:", len(completed), "/", len(GENERATION_ARMS))
'''
)

markdown("## Apply the preregistered gates")
code(
    '''
REPORT = REPORT_ROOT / "margin_geometry_summary.json"
run_persisted(
    [sys.executable, "-u", "scripts/analyze_official_codi_endpoint_margin_geometry.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--sweep", str(SWEEP_DIR / "analytic_sweep.pt"),
     "--output", str(REPORT)],
    "analyze_margin_geometry.log",
)
report = json.loads(REPORT.read_text())
print("status:", report["status"])
print("effective rank:", report["effective_dimensionality"]["effective_rank_by_family"])
primary = report["primary_margin_specificity"]
print("primary empirical matched-random p:", primary["empirical_matched_random_p"])
for family, value in report["reference_selectors"].items():
    print(family, "continuous z:", round(value["z_score"], 3),
          "binary z:", round(value["binary"]["z_score"], 3),
          "continuous p:", value["empirical_matched_random_p"],
          "binary p:", value["binary_empirical_matched_random_p"])
'''
)

markdown(
    "## Checksummed export\n\n"
    "Save Version with outputs enabled, then publish "
    "`official_codi_endpoint_margin_geometry_export` as a Kaggle dataset before the "
    "notebook version is deleted."
)
code(
    '''
import hashlib, shutil

EXPORT = pathlib.Path("/kaggle/working/official_codi_endpoint_margin_geometry_export")
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
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(EXPORT)}")
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
