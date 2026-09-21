"""Generate a standalone Kaggle notebook with embedded experiment code and tests."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"notebooks/kaggle_teacher_code_distillation.ipynb"
FILES = ("src/mech/teacher_code_distillation.py",
         "scripts/run_teacher_code_distillation.py",
         "tests/test_teacher_code_distillation.py")
embedded = {p:(ROOT/p).read_text(encoding="utf-8-sig") for p in FILES}
source_hash = hashlib.sha256((json.dumps(embedded,sort_keys=True)+
                              Path(__file__).read_text(encoding="utf-8-sig")).encode()).hexdigest()
cells = []


def md(text):
    cells.append(dict(cell_type="markdown",metadata={},source=text.strip().splitlines(True)))


def code(text):
    cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],
                      source=text.strip().splitlines(True)))


md("""# Can 96 teacher coordinates preserve useful distillation signal?

**Kaggle: enable Internet, select a GPU, then Save Version > Save & Run All.**
One GPU is used. No input dataset is required for a fresh run.

The teacher is the released CODI GPT-2 checkpoint in **explicit-CoT mode**.
The student is DistilGPT2 with an ordinary dense vocabulary head.
Only the cached teacher supervision changes.

**Default: PILOT.** This runs the real target fitting/audits and five pilot student
arms, then evaluates development questions. It does not open final test predictions.
If the pilot shows a positive full-KD gain, set STAGE to "full" and rerun using the
same saved output. Full runs reuse the pilot and perform all locked seed comparisons.

A negative pilot is a scientific stop, not a software failure. Read pilot_report.json
before changing the common training configuration. Any configuration change makes a
new run identity. A smoke run is separate and supports no scientific conclusions.

The notebook embeds the new helpers and clones an immutable base commit.
It therefore works before you commit or push the new notebook to GitHub.
""")
code('''# EDIT THIS CELL BEFORE STARTING
STAGE = "pilot"             # "pilot" first; "full" after a successful pilot
SMOKE = False              # True = tiny integration run, no scientific claims
SEEDS = [89, 90, 91]        # Paired training seeds; >=3 for formal claims
RUN_SECONDARY = False      # Add ranks 32/64 and weight-SVD96 (24 vs 15 full fits)
RESUME_ROOT = ""            # Exact saved full_<id> / smoke_<id> directory under /kaggle/input
SESSION_HOURS = 9.0         # Leave time for Kaggle to persist the saved outputs
OUTPUT_ROOT = "/kaggle/working/teacher_code_distillation"

# Execution filters partition full work across sessions WITHOUT changing the experiment.
# Leave None to run every locked arm/seed. Incomplete grids never get completion.json.
ARM_FILTER = None           # e.g. ["sft", "full", "lr96"]; full stage only
SEED_FILTER = None          # e.g. [89]; full stage only

# Common training recipe. Do not retune independently by arm.
STUDENT_EPOCHS = 3
STUDENT_LR = 5e-5
KD_ALPHA = 0.5
''')
# Cell 1 is literal definitions only and can be tested without executing setup.
code("EMBEDDED_FILES = "+repr(embedded)+"\nEXPERIMENT_SOURCE_SHA256 = "+repr(source_hash))
md("""## Setup and correctness checks

Keep Kaggle's PyTorch/CUDA. Setup installs the compatible Transformers/PEFT stack.
Use a fresh session if unrelated previously imported packages conflict.
All downloads are public; no Hugging Face token is required.

The CPU checks below verify target alignment, sparse-loss gradients, storage matching,
resume parity, and tiny model training. They do not establish GPU speed or math accuracy.
""")
code(r'''
import importlib
import json
import os
import pathlib
import subprocess
import sys

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HOME", "/kaggle/working/teacher_code_hf_cache")

REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
BASE_COMMIT = "6a8d2e61950c67f012d0a9ba13ec8a70f3a25019"
REPO_DIR = pathlib.Path("/kaggle/working/teacher-code-repo")
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

def checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    with (pathlib.Path(OUTPUT_ROOT)/"setup.log").open("a", encoding="utf-8") as stream:
        stream.write(result.stdout + "\n" + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:] + "\n" + result.stderr[-6000:])
    return result.stdout

if not REPO_DIR.exists():
    checked(["git", "clone", REPO_URL, str(REPO_DIR)])
checked(["git", "-C", str(REPO_DIR), "checkout", "--detach", BASE_COMMIT])
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
checked([sys.executable, "-m", "pip", "install", "-q",
         "transformers==4.52.4", "peft==0.15.2", "accelerate==1.7.0",
         "huggingface_hub>=0.34,<1", "hf_xet", "pyyaml", "pytest"])
probe = subprocess.run([sys.executable, "-c", "import peft"], capture_output=True, text=True)
if probe.returncode and "torchao" in (probe.stdout + probe.stderr):
    checked([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
checked([sys.executable, "-c", "import torch, peft; from transformers import GPT2LMHeadModel"])
for relative, source in EMBEDDED_FILES.items():
    destination = REPO_DIR/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
# Permit re-import after rerunning cells; old helper modules must not survive an update.
for name in ("scripts.run_teacher_code_distillation", "src.mech.teacher_code_distillation"):
    sys.modules.pop(name, None)
    parent, _, attribute = name.rpartition(".")
    package = sys.modules.get(parent)
    if package is not None and hasattr(package, attribute):
        delattr(package, attribute)
importlib.invalidate_caches()
import torch
assert torch.cuda.is_available(), "Select a GPU in Kaggle's notebook settings."
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
test_output = checked([sys.executable, "-m", "pytest", "-q",
                       "tests/test_teacher_code_distillation.py", "-k", "not notebook"])
print(test_output)
''')
md("""## Run the selected stage

The same output folder is reused when settings, code, packages, and GPU identity match.
A session budget stop saves a pause report. An unexpected interruption resumes student
training from the last optimizer checkpoint (20 steps by default); teacher collection
resumes by question and target exports by block. Codec fitting resumes completed epochs.

**Across sessions:** save outputs, attach the saved run as Kaggle input, set RESUME_ROOT
to the exact full_<id> directory, and rerun. Restoring copies the run to writable storage.
Do not change seeds, alpha, learning rate, or secondary-arm settings when resuming.

The full grid can exceed one Kaggle session. Use execution filters to split work; all
configured arms and seeds must complete before final gates are reported. Changing only
STAGE, filters, RESUME_ROOT, or SESSION_HOURS does not change the run identity.
""")
code('''
from src.mech import teacher_code_distillation as kd
from scripts.run_teacher_code_distillation import run_experiment

settings = kd.Settings(
    smoke=SMOKE, seeds=tuple(SEEDS), secondary=RUN_SECONDARY,
    epochs=STUDENT_EPOCHS, lr=STUDENT_LR, alpha=KD_ALPHA,
)
RUN_DIR, RESULTS = run_experiment(
    OUTPUT_ROOT, settings=settings, stage=STAGE, resume_root=RESUME_ROOT,
    hours=SESSION_HOURS, source_hash=EXPERIMENT_SOURCE_SHA256,
    arm_filter=ARM_FILTER, seed_filter=SEED_FILTER,
)
print("Saved run:", RUN_DIR)
print(json.dumps(RESULTS, indent=2))
''')
md("""## Read the results

**Pilot:** inspect the codec audit, gradient diagnostics, five development accuracies,
and predicted confidence-interval width. Positive pilot gain only permits the full run;
it does not establish the final utility claim.

**Full:** teacher utility, rank-96 noninferiority, and superiority over both compact
controls are separate gates. Wide intervals mean inconclusive. Smoke runs and fewer
than three seeds are not eligible for claims.

Sparse controls get the same TOTAL byte budget as rank 96, including its shared
reconstruction matrix. Full KD reconstructs exact canonical logits from cached hidden
states; a second mathematically identical full-logit training arm is unnecessary.
""")
code('''
import pandas as pd
from IPython.display import display

if (RUN_DIR/"storage.json").exists():
    storage = kd.read_json(RUN_DIR/"storage.json")
    display(pd.DataFrame([
        dict(package=p["name"], total_MB=p["total_bytes"]/1e6,
             tokens=p["token_count"], bytes_per_token_including_shared=p["total_bytes"]/p["token_count"])
        for p in storage["packages"]
    ]))
if (RUN_DIR/"pilot_report.json").exists():
    pilot = kd.read_json(RUN_DIR/"pilot_report.json")
    display(pd.DataFrame(pilot["results"]).T)
if (RUN_DIR/"results.json").exists():
    result = kd.read_json(RUN_DIR/"results.json")
    for dataset, report in result["datasets"].items():
        print(dataset)
        display(pd.DataFrame([
            dict(arm=arm, seed=seed, **metrics)
            for arm, runs in report["arms"].items()
            for seed, metrics in zip(result["seeds"], runs)
        ]))
        display(pd.DataFrame(report["comparisons"]).T)
    print("Gates:", result["gates"])
else:
    print("No full completion yet:", RESULTS.get("status", "in progress"))
''')
md("""## Preserve outputs

Use **Save Version > Save & Run All** so Kaggle retains the writable output directory.
Download the saved run from Outputs, or attach it to the next session as an input.
Final checkpoints are stored without optimizer state; interrupted fits retain one
optimizer checkpoint. Do not discard that checkpoint when resuming unfinished training.

Files to inspect:
- manifest.json and partitions.json: immutable settings, source identity, splits/exclusions.
- codecs/*/report.json: fitting/selection history and serialized target fidelity.
- target_strata.json and gradient_audit_*.json: token and learning-signal diagnostics.
- storage.json: measured target package sizes, including shared decoder weights.
- pilot_report.json: development outcomes and feasibility decision.
- students/*/*/training.json and dev/gsm8k/svamp.json: training history and every prediction.
- results.json and completion.json: full paired analysis, only after the entire locked grid.
- session_pause.json: normal session-budget stop, with resume instructions.

This notebook implements the primary explicit-CoT experiment and its optional rank/
initializer ablations. The later CODI latent-student extension requires its own experiment.
""")

for index,cell in enumerate(cells):
    cell["id"] = hashlib.sha256((str(index)+"".join(cell["source"])).encode()).hexdigest()[:12]
notebook = dict(cells=cells,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
    language_info=dict(name="python",version="3.11.0"),
    kaggle=dict(accelerator="gpu",isInternetEnabled=True,language="python",sourceType="notebook")),
    nbformat=4,nbformat_minor=5)
OUTPUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+"\n",encoding="utf-8")
print(OUTPUT)


