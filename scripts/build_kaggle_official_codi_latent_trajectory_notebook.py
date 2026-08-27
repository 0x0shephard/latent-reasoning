"""Build the Kaggle run-all notebook for the latent-trajectory detection gate."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_latent_trajectory_detect.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Does CODI's latent reasoning trajectory know things its endpoint forgets?

## Goal

Every completed intervention read or edited state 12 at the forced answer cue, where
computation has already collapsed into one token choice. This notebook runs the cheap
**detection gate** that must pass before any editing experiment at the six latent
thought states is justified.

One observational GPU pass captures the thirteen hidden states of each latent
iteration from the released generation path (nothing is edited; a zero-noise endpoint
capture ties the pass to the validated colon-state cache by direct state parity).
Convergence-certified probes then ask two frozen questions on the frozen
440/440/439 GSM8K-test partition:

1. **correctness** — does the best trajectory cell plus the endpoint margin predict
   first-token correctness at least 0.02 AUC better than the margin alone? (§49 showed
   ~0.013 cannot be resolved on 439 test questions, so the bar is set where this
   sample size has power.)
2. **answer identity** — on final-test questions the model gets *wrong*, does a probe
   on the chosen trajectory cell recover the gold first answer token at least five
   points better than the same probe class reading the endpoint state, and better
   than the majority class?

Only a passed answer-identity gate justifies proposing a latent-state editing
experiment. No model weight is updated and no inference-speed claim is made.
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
TRAJECTORY_INPUT = ""  # Optional completed latent_trajectory.pt.

RUN_COLLECTION = True
COLLECTION_BATCH_SIZE = 32

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
     "tests/test_latent_trajectory_detect.py",
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
OUTPUT_ROOT = pathlib.Path("/kaggle/working/official_codi_latent_trajectory_detect")
COLLECTION_ROOT = OUTPUT_ROOT / "collection"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("states      :", COLON_STATES)
print("readout     :", READOUT)
print("reproduction:", REPRODUCTION_SUMMARY)
'''
)

markdown(
    "## Collect the latent trajectory\n\n"
    "One observational pass over the 1,319 cached questions. The forced-cue state 12 "
    "is recaptured and must match the attached colon-state cache within the frozen "
    "relative tolerance, or the collection refuses to save."
)
code(
    '''
if RUN_COLLECTION:
    subprocess.run(
        [sys.executable, "-u", "scripts/collect_official_codi_latent_trajectory.py",
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
    TRAJECTORY = str(COLLECTION_ROOT / "latent_trajectory.pt")
else:
    TRAJECTORY = discover(TRAJECTORY_INPUT, "latent_trajectory.pt")

collection_summary = json.loads(
    pathlib.Path(TRAJECTORY).with_suffix(".json").read_text()
)
print(json.dumps(collection_summary["parity_gate"], indent=2))
'''
)

markdown("## Probe the 6x13 trajectory grid and apply the frozen gates")
code(
    '''
SUMMARY = OUTPUT_ROOT / "latent_trajectory_detect.json"
ARTIFACT = OUTPUT_ROOT / "latent_trajectory_detect.pt"
REPORT = OUTPUT_ROOT / "latent_trajectory_detect_report.json"

subprocess.run(
    [sys.executable, "-u", "scripts/run_official_codi_latent_trajectory_detect.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--trajectory", TRAJECTORY,
     "--states", COLON_STATES,
     "--readout", READOUT,
     "--output", str(SUMMARY),
     "--artifact-output", str(ARTIFACT),
     "--report-output", str(REPORT)],
    check=True,
)
report = json.loads(REPORT.read_text())
print(json.dumps({
    "status": report["status"],
    "editing_experiment_justified": report["editing_experiment_justified"],
    "correctness_gate": report["correctness_gate"],
    "answer_identity_gate": report["answer_identity_gate"],
}, indent=2))
'''
)

markdown("## Inspect selection without touching test tuning")
code(
    '''
import pandas as pd

summary = json.loads(SUMMARY.read_text())
correctness = pd.DataFrame(summary["correctness"]["selection_curve"])
identity = pd.DataFrame(summary["answer_identity"]["selection_curve"])
display(correctness.sort_values("select_auc", ascending=False).head(10))
display(identity.sort_values("select_wrong_accuracy", ascending=False).head(10))
display(correctness.pivot(index="state", columns="position", values="select_auc").round(3))
display(identity.pivot(
    index="state", columns="position", values="select_wrong_accuracy"
).round(3))
'''
)

markdown(
    "## Next steps\n\n"
    "Interpret only the executed report. A passed answer-identity gate justifies "
    "designing a latent-state editing experiment at the selected cell; a "
    "correctness-only pass is a detection finding and justifies nothing further; a "
    "double failure closes the latent-trajectory question for linear probes. Do not "
    "rescue a failed gate by re-reading the final test split."
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
