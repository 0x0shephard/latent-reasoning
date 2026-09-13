"""Build the run-all Kaggle notebook for layerwise U28 transport and causality."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_layerwise_u28_stability.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(value): cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value): cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Do CODI's 28 answer directions remain meaningful through all 12 layers?

This is experiment 1 of 2. The known U28 is the post-`ln_f` eigenspace spanning
eigenvectors 4–31 at the forced answer cue. An eigenvector index is local to the
matrix that produced it, so this notebook does **not** compute “PC 4–31” independently
at every layer and call them equal.

Instead, it:

1. reproduces U28 from the original 2,048 training-only colon states;
2. captures the residual stream and the activation entering QKV in every block;
3. fits a ridge map from each layer to the same 28 final coordinates;
4. evaluates coordinate prediction and principal angles on a disjoint split;
5. performs retain/remove interventions against an energy-matched random subspace;
6. exports a gated artifact for the xKV experiment.

All intervention results are first-token screens. They establish whether a layer-local
subspace is causally relevant to the same endpoint; they are not yet a KV-compression
or speed result. Use a GPU with Internet enabled and **Save & Run All**.
""")

md("## 1. Configuration and immutable inputs")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "6e172775ead54be5f8917ee403e893e9ed42932d"
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
COLON_STATES_INPUT = ""
OUTPUT_DIR = "/kaggle/working/codi_layerwise_u28_stability"

FIT_EXAMPLES = 2048
SELECT_EXAMPLES = 512
SCREEN_EXAMPLES = 128
BATCH_SIZE = 16
RIDGE_RATIO = 1e-3
RANDOM_REPLICATES = 256
RANDOM_CONTROLS = 4
PRECISION = "float32"

import glob, json, os, pathlib, subprocess, sys
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
print("code commit", CODE_COMMIT)
''')

md("## 2. Repair Kaggle's optional torchao conflict and test the new math")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2",
                "accelerate==1.7.0", "huggingface-hub==0.32.4",
                "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\n"
    "try:\n print(is_torchao_available())\n"
    "except ImportError:\n print('incompatible')"], capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_layerwise_u28.py", "tests/test_official_codi_layerwise.py"], check=True)
''')

md(r"""
## 3. Resolve completed CODI inputs

Attach the dataset containing the completed official CODI reproduction and the
`colon_states.pt` from the answer-colon margin-geometry run. The notebook searches
recursively, but refuses to guess if no matching file exists.
""")
code(r'''
def discover(explicit, suffix):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{suffix}", recursive=True))
    assert matches, f"Attach a Kaggle dataset containing {suffix}"
    return sorted(matches, key=lambda x: (len(pathlib.Path(x).parts), x))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")
COLON_STATES = discover(COLON_STATES_INPUT, "colon_states_seed89/colon_states.pt")
print("reproduction", REPRODUCTION_SUMMARY)
print("colon states", COLON_STATES)
''')

md("## 4. Run the pre-registered fit/select/causal-screen pipeline")
code(r'''
command = [sys.executable, "-u", "scripts/run_codi_layerwise_u28_stability.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", REPRODUCTION_SUMMARY,
    "--colon-states", COLON_STATES, "--output-dir", OUTPUT_DIR,
    "--fit-examples", str(FIT_EXAMPLES), "--select-examples", str(SELECT_EXAMPLES),
    "--screen-examples", str(SCREEN_EXAMPLES), "--batch-size", str(BATCH_SIZE),
    "--ridge-ratio", str(RIDGE_RATIO), "--random-replicates", str(RANDOM_REPLICATES),
    "--random-controls", str(RANDOM_CONTROLS),
    "--precision", PRECISION, "--device", "cuda"]
subprocess.run(command, check=True)
summary = json.loads((pathlib.Path(OUTPUT_DIR) / "summary.json").read_text())
print(json.dumps({"baseline": summary["baseline_first_token"], "gate": summary["gate"]}, indent=2))
''')

md(r"""
## 5. Visualize transport and causal specificity

`select_r2` asks how much of the held-out final 28-coordinate variation a layer can
predict. Principal-angle overlap asks whether its fitted state-space orientation is
similar to final U28. The causal panel compares the accuracy damage from removing
transported U28 against removing a rank- and energy-matched random subspace.
""")
code(r'''
import matplotlib.pyplot as plt
import numpy as np

layers = list(range(12))
names = [f"attn_ln_{layer:02d}" for layer in layers]
r2 = [summary["geometry"][name]["select_r2"] for name in names]
overlap = [summary["geometry"][name]["overlap_to_final_u28"] for name in names]
u28_damage = [summary["causal_screen"][name]["remove_u28_accuracy_drop"] for name in names]
random_damage = [summary["causal_screen"][name]["remove_random_median_accuracy_drop"] for name in names]
retention = [summary["causal_screen"][name]["retain_u28"]["dense_agreement"] for name in names]

fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
axes[0].plot(layers, r2, marker="o", label="held-out coordinate R²")
axes[0].plot(layers, overlap, marker="s", label="overlap with final U28")
axes[0].set(title="Layerwise transport", xlabel="transformer block", ylim=(-0.05, 1.05)); axes[0].legend()
axes[1].plot(layers, u28_damage, marker="o", label="remove transported U28")
axes[1].plot(layers, random_damage, marker="s", label="median of 4 energy-matched random controls")
axes[1].axhline(0, color="black", linewidth=.8)
axes[1].set(title="Causal first-token accuracy damage", xlabel="transformer block"); axes[1].legend()
axes[2].plot(layers, retention, marker="o", color="#147d64")
axes[2].axhline(summary["gate"]["thresholds"]["minimum_retention"], linestyle="--", color="black")
axes[2].set(title="Retain-only agreement with dense CODI", xlabel="transformer block", ylim=(0, 1.02))
figure_path = pathlib.Path(OUTPUT_DIR) / "layerwise_u28_diagnostics.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
''')

md(r"""
## 6. Decision

The next notebook is confirmatory only when the exported gate passes. A pass requires
a contiguous run of at least two attention layers with held-out predictability,
retain-only fidelity, and U28 removal damage exceeding the matched random damage.
A failure is scientifically useful: it means a stable predictor of the endpoint was
not shown to be a causal KV core, and the xKV hybrid must not be advertised as such.
""")
code(r'''
gate = summary["gate"]
if gate["passed"]:
    print("PASS — run kaggle_codi_causal_xkv.ipynb with this output attached.")
    print("selected contiguous layers:", gate["selected_contiguous_attention_layers"])
else:
    print("STOP — no causal protected-core claim. Inspect the layerwise curves before exploration.")
print("artifact:", pathlib.Path(OUTPUT_DIR) / "layerwise_u28.pt")
''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUTPUT)
print(OUTPUT)
