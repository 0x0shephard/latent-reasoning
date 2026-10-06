"""Build the Kaggle notebook for the directed counterfactual at 1B, both profiles (ledger §116)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_codi_llama1b_directed_counterfactual.ipynb"
RUN_COMMIT = "__RUN_COMMIT__"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# Does a CODI-1B thought hold *the value*? Directed counterfactual at 1B, both profiles (ledger §116)

## tl;dr

Same protocol as §110 (all chains) and §112 (chains with ≥ 3 steps, latest-step number
changed, slot-unique location), run on the author-released CODI LLaMA-3.2-1B-Instruct
checkpoint through the §114 adapter. Counterfactual twins differ in one question number;
after a competence gate each twin donates to the other; one site is swapped at a time and
the first answer token is classed as the twin's answer (target), the own answer (retain) or
other. **No training**, no dataset to attach; about 1.5 hours of GPU in total.

Predictions stated in advance (§116): the tail carries less than at GPT-2 (T1); no
single-site transport on short chains (T2); on long chains the carrying thought is slot 3
through position 4's K/V, but E2 (≥ 0.25) is predicted to fail because transport is rarer
at 1B (T3); position 0 is not the question register at 1B (T4); decoding stays redundant
(T5); the terminal slot transports nothing.

| profile | checks (thresholds unchanged) |
|---|---|
| standard (§110) | D1 `v89_all` ≥ 0.50; D2 even store ≥ 0.40 and ≥ 2× odd; D3 located specificity; D4 layer control |
| long (§112) | E1 `kv_all` ≥ 0.50; E2 max(`state_1`, `state_3`) ≥ 0.25; E3 unique-location specificity |

`v89` here means the depth-scaled middle value group (layers 11–12 of 16) and `v1011` the
late group (13–15).
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "__RUN_COMMIT__"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "configs/official_codi_llama1b.yaml"
OUTPUT_ROOT = "/kaggle/working/codi_llama1b_directed_counterfactual"
RUN_SMOKE = True
BATCH_SIZE = "32"

import glob, json, os, pathlib, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
assert len(RUN_COMMIT) == 40
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("code commit", subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip())
''')

md("### Environment and focused tests")
code(r'''
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tpot"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.52.4", "datasets==3.6.0", "peft==0.15.2", "accelerate==1.7.0",
                "huggingface-hub>=0.34,<1.0", "safetensors==0.5.3", "pytest>=8,<10"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "from peft.import_utils import is_torchao_available\ntry:\n print(is_torchao_available())\nexcept ImportError:\n print('incompatible')"],
    capture_output=True, text=True)
if "incompatible" in probe.stdout:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q",
                "tests/test_official_codi_family.py",
                "tests/test_counterfactual_chain.py",
                "tests/test_directed_counterfactual.py",
                "tests/test_directed_counterfactual_runner.py"], check=True)
''')

md("## Step 1: GSM8K reproduction gate (pad-aware path; lower bound against the paper)")
code(r'''
def run_with_retry(command, attempts=3, wait=60):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command)
        if result.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"runner exited with {result.returncode}; retrying in {wait}s ({attempt + 1}/{attempts})")
        time.sleep(wait)

def summary_under(root):
    found = sorted(glob.glob(f"{root}/eval/revision_*/full_*/summary.json"), key=os.path.getmtime)
    return found[-1] if found else None

PAPER = 0.519
PRECISION = None
for precision in ("float16", "float32"):
    run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CONFIG, "--device", "cuda",
                    "--set", f"model.dtype={precision}", "--set", f"eval.batch_size={BATCH_SIZE}"])
    REPRODUCTION_SUMMARY = summary_under("outputs/official_codi_llama1b")
    gate = json.loads(open(REPRODUCTION_SUMMARY).read())
    acc = gate["datasets"]["gsm8k"]
    print(f"{precision}: pad-aware gsm8k {acc:.4f} (paper {PAPER}, delta {acc - PAPER:+.4f}); gate {gate['accuracy_gate']['status']} ({gate['accuracy_gate']['direction']})")
    if gate["accuracy_gate"]["status"] == "passed":
        PRECISION = precision
        break
assert PRECISION, "the 1B lower-bound gate failed in both precisions; do not run the twins"
print("reproduction passed in", PRECISION, "->", REPRODUCTION_SUMMARY)
''')

md("## Step 2: twins, standard profile (all chains), then long profile (≥ 3 steps)")
code(r'''
def command(output_dir, profile, *extra):
    return [sys.executable, "-u", "scripts/run_codi_directed_counterfactual.py",
            "--config", CONFIG, "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", BATCH_SIZE, "--precision", PRECISION, "--device", "cuda",
            "--profile", profile, *extra]

summaries = {}
for profile in ("standard", "long"):
    out = f"{OUTPUT_ROOT}_{profile}"
    if RUN_SMOKE:
        run_with_retry(command(out + "_smoke", profile, "--smoke"))
    run_with_retry(command(out, profile))
    summaries[profile] = json.loads(open(pathlib.Path(out) / "summary.json").read())
    print(profile, "gate:", summaries[profile]["gate"])
    print(profile, "baseline:", summaries[profile]["baseline"])
    print(profile, summaries[profile]["decision"])
''')

md("## Results")
code(r'''
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.dpi": 120, "axes.grid": False, "font.size": 10})

for profile, summary in summaries.items():
    conds = summary["report"]["conditions"]
    print(f"===== {profile} profile: first token after the swap, both directions pooled, n = {conds['state_1']['all']['n']} =====")
    rows = [{"condition": name, "target (twin's answer)": round(v["all"]["target"], 3), "retain (own)": round(v["all"]["retain"], 3),
             "other": round(v["all"]["other"], 3)} for name, v in conds.items()]
    display(pd.DataFrame(rows).set_index("condition"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    xs = list(range(6))
    for kind, color in (("state", "#b65f24"), ("v89", "#2a6f97"), ("kv", "#444444")):
        axes[0].plot(xs, [conds[f"{kind}_{s}"]["all"]["target"] for s in xs], marker="o", ms=4, color=color, label=kind)
    axes[0].plot([0, 2, 4], [conds[f"v1011_{s}"]["all"]["target"] for s in (0, 2, 4)], marker="s", ms=4, color="#a9d6e5", label="v1011 (control)")
    axes[0].set(xlabel="slot / position", ylabel="target rate", title=f"1B {profile}: single-site swaps, pooled")
    axes[0].legend(fontsize=8)
    strata = sorted(summary["strata_counts"])
    for name, color in (("state_1", "#b65f24"), ("state_3", "#e0a070"), ("kv_2", "#2a6f97"), ("kv_4", "#61a5c2")):
        axes[1].plot(strata, [conds[name]["by_stratum"][k]["target"] or 0 for k in strata], marker="o", ms=4, color=color, label=name)
    axes[1].set(xlabel="stratum (chain length n, perturbed step k)", ylabel="target rate", title="by where the number enters the chain")
    axes[1].tick_params(axis="x", rotation=45); axes[1].legend(fontsize=8)
    plt.show()
    print("strata counts:", summary["strata_counts"])
    loc = summary["report"]["located"]; uniq = summary["report"]["unique_location"]
    print("located fraction:", round(loc["fraction_located"], 3), "rows by slot:", loc["rows_by_slot"], "multi-slot rows:", loc["rows_multi_slot"])
    print("uniquely located fraction:", round(uniq["fraction_unique"], 3), "rows by slot:", uniq["rows_by_slot"])
    uspec = summary["report"]["unique_specificity"]
    urows = [{"uniquely located at slot": s, "n": v["n"], "state at site": v["state_at_site"]["target"],
              **{f"state at slot {o}": w["target"] for o, w in v["state_elsewhere"].items()},
              f"K/V at position {int(s) + 1}": v.get("kv_store_at_site", {}).get("target"),
              **{f"K/V at position {o}": w["target"] for o, w in v.get("kv_store_elsewhere", {}).items()}} for s, v in uspec.items()]
    display(pd.DataFrame(urows))
    print("located at the terminal slot:", summary["report"]["located_terminal"])
    print("native decoding (baseline exact match", round(summary["baseline"]["native_exact_match"], 3), ")")
    display(pd.DataFrame(summary["report"]["native"]).T)
    display(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary["checks"].items()}]).T.rename(columns={0: "value"}))
''')

md("## Takeaways")
code(r'''
for profile, summary in summaries.items():
    print(f"[{profile}] Decision:", summary["decision"]["claim"])
std, lng = summaries["standard"]["report"]["conditions"], summaries["long"]["report"]["conditions"]
t = lambda c, k: c[k]["all"]["target"]
print()
print("T1 tail carries less than GPT-2 (0.595 short / 0.692 long): kv_all short", round(t(std, "kv_all"), 3), "long", round(t(lng, "kv_all"), 3))
print("T2 no single-site transport on short chains (all state_s, v89_s <= 0.10):",
      all(t(std, f"{k}_{s}") <= 0.10 for k in ("state", "v89") for s in range(6)))
print("T3 long chains: state_3 >= state_1:", t(lng, "state_3") >= t(lng, "state_1"), "| kv_4 >= kv_2:", t(lng, "kv_4") >= t(lng, "kv_2"),
      "| state_3", round(t(lng, "state_3"), 3), "kv_4", round(t(lng, "kv_4"), 3), "| E2 (>= 0.25):", summaries["long"]["checks"]["e2_single_thought_transport"])
print("T4 position 0: kv_0 short", round(t(std, "kv_0"), 3), "(GPT-2 0.232) | state_0 short", round(t(std, "state_0"), 3), "(GPT-2 0.000)")
print("T5 uniquely located fraction (long):", round(summaries["long"]["report"]["unique_location"]["fraction_unique"], 3), "(GPT-2 0.039)")
print("terminal slot transport (long): state_5", round(t(lng, "state_5"), 3), "kv_5", round(t(lng, "kv_5"), 3))
for w in summaries["long"]["warnings"]:
    print("WARNING:", w)
print("seconds: standard", round(summaries["standard"]["seconds"]), "long", round(summaries["long"]["seconds"]))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
