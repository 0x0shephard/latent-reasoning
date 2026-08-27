"""Build the Kaggle run-all notebook for contrastive correctness covariance."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_correctness_contrastive_covariance.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Can CODI's correct and wrong covariance directions be separated?

This notebook directly tests the proposed experiment. From each cached 768-D state-12
answer-cue vector, it labels the model's baseline first token correct or wrong. On 440
fit questions it estimates `C_correct` and `C_wrong`, then solves the shrinkage-stable
generalized eigenproblem

`C_correct v = lambda C_wrong v`.

The 28 largest-ratio directions are the **correct-specific** candidate; the 28
smallest are the **wrong-specific** candidate. A disjoint 440-question selection split
chooses covariance shrinkage using projection-energy specificity. The final 439 are
read once.

Primary intervention: keep only the correct-specific 28-D projection around the
correct-class mean. Secondary intervention: remove the wrong-specific projection.
Correct-only PCA, the established class-blind accuracy band (PCs 4–31), a centre
diagnostic, and energy-matched random bases prevent us from mistaking ordinary
variance or a mean replacement for a correctness-specific channel.

Attach the completed `colon_states.pt` and `readout.pt`. The analytic tier is the
primary decision and is CPU-capable. The optional exact-match tier needs a GPU and the
completed official reproduction summary.
"""
)

markdown("## 1. Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after this notebook is pushed.
REPO_DIR = "/kaggle/working/latent-reasoning"

COLON_STATES_INPUT = ""
READOUT_INPUT = ""
REPRODUCTION_SUMMARY_INPUT = ""  # Needed only when RUN_GENERATION_TIER=True.
RUN_GENERATION_TIER = True

import glob, hashlib, json, os, pathlib, subprocess, sys

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

markdown("## 2. Implementation checks")
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_correctness_contrastive_covariance.py",
     "tests/test_endpoint_correctness_geometry.py",
     "tests/test_correctness_tracks_integration.py"],
    check=True,
)
'''
)

markdown("## 3. Resolve attached inputs")
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
    required=RUN_GENERATION_TIER,
)
OUTPUT_ROOT = pathlib.Path(
    "/kaggle/working/official_codi_correctness_contrastive_covariance"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("states      :", COLON_STATES)
print("readout     :", READOUT)
print("reproduction:", REPRODUCTION_SUMMARY or "not requested")
'''
)

markdown("## 4. Fit/select/test analytic experiment")
code(
    '''
SUMMARY = OUTPUT_ROOT / "contrastive_covariance.json"
ARTIFACT = OUTPUT_ROOT / "contrastive_covariance.pt"
REPORT = OUTPUT_ROOT / "contrastive_covariance_report.json"

subprocess.run(
    [sys.executable, "-u",
     "scripts/run_official_codi_correctness_contrastive_covariance.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--states", COLON_STATES,
     "--readout", READOUT,
     "--output", str(SUMMARY),
     "--artifact-output", str(ARTIFACT),
     "--device", "cpu"],
    check=True,
)
subprocess.run(
    [sys.executable, "-u",
     "scripts/analyze_official_codi_correctness_contrastive_covariance.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--summary", str(SUMMARY),
     "--artifact", str(ARTIFACT),
     "--output", str(REPORT)],
    check=True,
)
report = json.loads(REPORT.read_text())
print(json.dumps({
    "status": report["status"],
    "rank": report["rank"],
    "selected_shrinkage": report["selected_shrinkage"],
    "baseline_accuracy": report["baseline_accuracy"],
    "correct_specific_accuracy": report["primary_accuracy"],
    "correct_specific_retention_confirmed": report["correct_specific_retention_confirmed"],
    "wrong_specific_removal_confirmed": report["wrong_specific_removal_confirmed"],
    "overall_confirmed": report["contrastive_covariance_confirmed"],
    "comparisons": report["comparisons"],
    "wrong_specific_removal": report["wrong_specific_removal"],
}, indent=2))
'''
)

markdown("## 5. Continuous-outcome table")
code(
    '''
import pandas as pd

summary = json.loads(SUMMARY.read_text())
table = pd.DataFrame(summary["arms"]).T.reset_index(names="arm")
table = table.sort_values(["accuracy", "mean_gold_margin"], ascending=False)
display(table)
print("Shrinkage selection (selection split only):")
display(pd.DataFrame(summary["shrinkage_selection"]).T)
'''
)

markdown(
    "## 6. Optional paired full-generation confirmation\n\n"
    "This runs five arms on the exact same frozen 439 indices. The state-12 hook is "
    "applied only at the answer cue. It directly changes the first answer token; "
    "later numeric text can change downstream of that token. Expect this cell to be "
    "the expensive part of the notebook."
)
code(
    '''
GENERATION_ROOT = OUTPUT_ROOT / "generation"
GENERATION_ARMS = [
    "baseline",
    "contrastive_correct_retain",
    "correct_only_pca_retain",
    "accuracy_band_pca_retain",
    "contrastive_wrong_remove",
]

if RUN_GENERATION_TIER:
    assert pathlib.Path(REPRODUCTION_SUMMARY).is_file(), "reproduction summary is required"
    for arm in GENERATION_ARMS:
        subprocess.run(
            [sys.executable, "-u",
             "scripts/run_official_codi_correctness_contrastive_generation.py",
             "--config", "configs/official_codi_gpt2.yaml",
             "--reproduction-summary", REPRODUCTION_SUMMARY,
             "--states", COLON_STATES,
             "--readout", READOUT,
             "--artifact", str(ARTIFACT),
             "--output-dir", str(GENERATION_ROOT / arm),
             "--arm", arm,
             "--eval-batch-size", "32",
             "--precision", "float32",
             "--device", "cuda"],
            check=True,
        )
    GENERATION_REPORT = OUTPUT_ROOT / "contrastive_generation_report.json"
    subprocess.run(
        [sys.executable, "-u",
         "scripts/analyze_official_codi_correctness_contrastive_generation.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--runs-root", str(GENERATION_ROOT),
         "--output", str(GENERATION_REPORT)],
        check=True,
    )
    generation_report = json.loads(GENERATION_REPORT.read_text())
    print(json.dumps(generation_report, indent=2))
else:
    print("Generation tier skipped; the primary analytic decision is complete.")
'''
)

markdown(
    "## 7. Interpretation\n\n"
    "Use the executed report, not the intended mechanism, as the conclusion. A full "
    "pass says the 28-D covariance ratio isolates an answer-sufficient channel beyond "
    "ordinary PCA and that deleting the opposite channel improves accuracy. If only "
    "retention passes, the result is sufficiency—not a generic wrong-answer repair. "
    "If the controls win, the class covariance difference is descriptive rather than "
    "causally useful."
)

markdown("## 8. Checksummed export")
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
