"""Build the CPU-only Kaggle notebook for the test-like detect replication."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks"
    / "kaggle_official_codi_correctness_detect_replication.ipynb"
)
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Test-like replication of CODI's correctness detector

## Goal

Resolve the validity threat in the only provisional three-track result. The original
detector was fitted on GSM8K train, where CODI was roughly 25 points more accurate than
on test. This notebook partitions the **cached GSM8K test states** into 440 fit / 440
select / 439 test examples and repeats the frozen detector gate.

The primary comparison remains `fisher_plus_margin` versus `margin`: delta AUC must be
at least +0.01 and its paired-bootstrap lower bound must be positive. Every ridge fit
must also pass a convergence certificate. This is a corrective replication on a
previously inspected dataset, not a pristine preregistration.

The notebook is CPU-only and does not load or decode the model. Attach the completed
margin-geometry `colon_states.pt` and `readout.pt`, then **Save Version -> Save & Run
All**.
"""
)

markdown(
    "## Context & Methods\n\n"
    "All directions and weights are fitted on 440 questions. Fisher shrinkage and "
    "ridge strength are selected on a disjoint 440. The remaining 439 are read once. "
    "The checked L-BFGS solver exports the final gradient and a strong-convexity "
    "upper bound on the objective gap."
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"
COLON_STATES_INPUT = ""
READOUT_INPUT = ""

import glob, hashlib, json, os, pathlib, shutil, subprocess, sys

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

markdown("## Checks")
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_correctness_detect_replication.py",
     "tests/test_endpoint_correctness_geometry.py",
     "tests/test_correctness_tracks_integration.py"],
    check=True,
)
'''
)

markdown("## Data")
code(
    '''
def discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0]


COLON_STATES = discover(COLON_STATES_INPUT, "colon_states.pt")
READOUT = discover(READOUT_INPUT, "readout.pt")
OUTPUT_ROOT = pathlib.Path(
    "/kaggle/working/official_codi_correctness_detect_replication"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("states :", COLON_STATES)
print("readout:", READOUT)
'''
)

markdown("## Results")
code(
    '''
SWEEP = OUTPUT_ROOT / "detect_replication.json"
OUTCOMES = OUTPUT_ROOT / "detect_replication.pt"
REPORT = OUTPUT_ROOT / "detect_replication_report.json"

subprocess.run(
    [sys.executable, "-u",
     "scripts/run_official_codi_correctness_detect_replication.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--states", COLON_STATES,
     "--readout", READOUT,
     "--output", str(SWEEP),
     "--outcomes-output", str(OUTCOMES),
     "--device", "cpu"],
    check=True,
)
subprocess.run(
    [sys.executable, "-u",
     "scripts/analyze_official_codi_correctness_detect_replication.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--sweep", str(SWEEP),
     "--outcomes", str(OUTCOMES),
     "--output", str(REPORT)],
    check=True,
)
report = json.loads(REPORT.read_text())
print(json.dumps({
    "status": report["status"],
    "splits": report["splits"],
    "primary_auc": report["primary_auc"],
    "baseline_auc": report["baseline_auc"],
    "delta_auc": report["delta_auc"],
    "delta_ci": report["delta_ci"],
    "optimizer_valid": report["optimizer_valid"],
    "passed": report["passed"],
}, indent=2))
'''
)

markdown("## Optimization audit")
code(
    '''
print(f"{'probe':<24} {'ridge':>8} {'iter':>6} {'grad inf':>12} {'gap bound':>12} {'ok':>5}")
print("-" * 75)
for name, entry in sorted(report["probes"].items()):
    audit = entry["optimization"]
    print(
        f"{name:<24} {entry['ridge']:>8g} {audit['iterations']:>6d} "
        f"{audit['gradient_inf_norm']:>12.3e} "
        f"{audit['objective_gap_upper_bound']:>12.3e} "
        f"{str(audit['converged']):>5}"
    )
'''
)

markdown(
    "## Takeaways\n\n"
    "Interpret only the executed report above. A pass supports a small increment over "
    "margin on a test-like fitting population; it does not establish operational "
    "utility. A failure retires the original +0.0123 as non-robust to the population "
    "correction. Either result leaves the steer and project nulls unchanged."
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
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"wrote {OUTPUT}")
