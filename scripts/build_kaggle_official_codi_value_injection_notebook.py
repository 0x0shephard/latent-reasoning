"""Build the Kaggle notebook for latent value injection plus efficiency routes."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_value_injection.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Writing values into CODI's latent workspace — and two efficiency routes

## Goal

§55 confirmed the six latent thoughts store the solution's intermediate values at
the odd slots. This notebook runs the **causal tier**: during the latent loop, add a
value token's readout direction to the propagating state at slots 1/3/5.

- **corruption** (`offset`): write gold-intermediate-plus-one into baseline-correct
  runs. If the workspace values are causally consumed, accuracy must drop at least
  5 points more than under matched random numeric tokens.
- **repair** (`gold`): write the gold intermediates into baseline-wrong runs. Gate:
  at least 3 points more recovery than the matched random arm.

The injection strength beta is chosen on the select split by the repair criterion
and shared by all arms; the frozen 439-question test split is read once. Unlike
every completed endpoint edit, this intervention enters inside the latent loop and
propagates through the KV cache.

A second, protocol-frozen measurement study runs the two findings-derived
efficiency routes: the latent-budget sweep (M = 3..6, full GSM8K accuracy and wall
clock) and the rank-k answer-readout microbenchmark.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
RUN_INJECTION = True
RUN_EFFICIENCY = True
EVAL_BATCH_SIZE = 32

import glob, hashlib, json, os, pathlib, subprocess, sys, urllib.request

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
     "tests/test_latent_value_injection.py",
     "tests/test_latent_workspace.py"],
    check=True,
)
'''
)

markdown("## Resolve inputs and download the pinned solutions")
code(
    '''
def discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
from src.utils.config import load_config
settings = load_config("configs/official_codi_gpt2.yaml").value_injection
SOLUTIONS = pathlib.Path("/kaggle/working/gsm8k_test.jsonl")
urllib.request.urlretrieve(str(settings.solutions_url), SOLUTIONS)
digest = hashlib.sha256(SOLUTIONS.read_bytes()).hexdigest()
assert digest == str(settings.solutions_sha256), digest
BETA_GRID = [float(b) for b in settings.beta_grid]
OUTPUT_ROOT = pathlib.Path("/kaggle/working/official_codi_value_injection")
SELECT_ROOT = OUTPUT_ROOT / "select"
TEST_ROOT = OUTPUT_ROOT / "test"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("reproduction:", REPRODUCTION_SUMMARY)
print("beta grid   :", BETA_GRID)
'''
)

markdown(
    "## Select-split sweep\n\n"
    "Baseline plus gold/random at every beta. The corruption arm inherits the "
    "chosen beta, so it is not run on select."
)
code(
    '''
def run_arm(arm, beta, split, out_dir):
    command = [sys.executable, "-u", "scripts/run_official_codi_value_injection.py",
               "--config", "configs/official_codi_gpt2.yaml",
               "--reproduction-summary", REPRODUCTION_SUMMARY,
               "--solutions", str(SOLUTIONS),
               "--arm", arm, "--split", split,
               "--output-dir", str(out_dir),
               "--eval-batch-size", str(EVAL_BATCH_SIZE),
               "--precision", "float32", "--device", "cuda"]
    if arm != "baseline":
        command += ["--beta", str(beta)]
    subprocess.run(command, check=True)

if RUN_INJECTION:
    run_arm("baseline", 0.0, "select", SELECT_ROOT / "baseline")
    for beta in BETA_GRID:
        tag = f"{beta:g}".replace(".", "p")
        run_arm("gold", beta, "select", SELECT_ROOT / f"gold_b{tag}")
        run_arm("random", beta, "select", SELECT_ROOT / f"random_b{tag}")
'''
)

markdown("## Choose beta on select, then read the frozen test split once")
code(
    '''
if RUN_INJECTION:
    SELECTION = OUTPUT_ROOT / "beta_selection.json"
    subprocess.run(
        [sys.executable, "-u", "scripts/analyze_official_codi_value_injection.py",
         "--mode", "choose-beta", "--select-root", str(SELECT_ROOT),
         "--output", str(SELECTION)],
        check=True,
    )
    chosen = json.loads(SELECTION.read_text())["selected_beta"]
    print("chosen beta:", chosen)
    run_arm("baseline", 0.0, "test", TEST_ROOT / "baseline")
    for arm in ("gold", "offset", "random"):
        run_arm(arm, chosen, "test", TEST_ROOT / arm)
    REPORT = OUTPUT_ROOT / "value_injection_report.json"
    subprocess.run(
        [sys.executable, "-u", "scripts/analyze_official_codi_value_injection.py",
         "--mode", "report", "--select-root", str(SELECT_ROOT),
         "--test-root", str(TEST_ROOT), "--output", str(REPORT)],
        check=True,
    )
    report = json.loads(REPORT.read_text())
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
'''
)

markdown(
    "## Efficiency measurement: latent-budget sweep and rank-k readout benchmark"
)
code(
    '''
if RUN_EFFICIENCY:
    EFFICIENCY = pathlib.Path("/kaggle/working/official_codi_efficiency/efficiency.json")
    EFFICIENCY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-u", "scripts/run_official_codi_efficiency.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--reproduction-summary", REPRODUCTION_SUMMARY,
         "--output", str(EFFICIENCY),
         "--precision", "float32", "--device", "cuda"],
        check=True,
    )
    print(EFFICIENCY.read_text())
'''
)

markdown(
    "## Next steps\n\n"
    "Interpret only the executed reports. The corruption gate alone establishes "
    "causal use of the workspace values; the repair gate alone would be surprising "
    "and should be double-checked before being claimed. Do not rescue a failed "
    "gate by re-reading the test split, and do not tune beta on anything but the "
    "recorded select criterion."
)

markdown("## Checksummed export")
code(
    '''
for root in [OUTPUT_ROOT, pathlib.Path("/kaggle/working/official_codi_efficiency")]:
    if not root.exists():
        continue
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(root)}"
            )
    (root / "SHA256SUMS.txt").write_text("\\n".join(lines) + "\\n")
    print(root, "->", len(lines), "files")
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
