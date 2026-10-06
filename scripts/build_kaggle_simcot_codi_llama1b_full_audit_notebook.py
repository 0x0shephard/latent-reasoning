"""Build the Kaggle notebook for SIM-CoT CODI-1B: released-path replication of CODI-1B (lost §114 Step 1a),
SIM-CoT gate, §108 audit and both twin profiles (ledger §118)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import nbformat as nbf
except ModuleNotFoundError:
    from scripts.notebook_compat import nbf


OUTPUT = ROOT / "notebooks" / "kaggle_simcot_codi_llama1b_full_audit.ipynb"
RUN_COMMIT = "dba9ce0e7bc05a3d93b3bee73712e2635b15c0b2"
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md(r"""
# SIM-CoT's CODI-1B: a third model with a different training objective (ledger §118)

## tl;dr

Same inference-only protocol, third model: `internlm/SIM_COT-LLaMA3-CODI-1B`, the CODI
architecture on LLaMA-3.2-1B-Instruct trained with SIM-CoT's per-step decoding supervision.
Four steps, **No training**, no dataset to attach, about 1.5 hours:

0. The released-path replication of **CODI-1B** (mask dropped, batch 128), lost with the
   previous kernel. Prediction: near the paper's 51.9%.
1. SIM-CoT GSM8K gate, pad-aware path, lower bound against the paper's 56.1%.
2. The §108 cache-carrier audit on SIM-CoT (influence profile).
3. The §110 and §112 twins on SIM-CoT (transport), thresholds unchanged.

Predictions (§118): S1 unique location ≥ 70% on long chains and spread over more than one
odd slot; S2 specificity at both feeding slots; S3 long-chain single-thought transport above
CODI-1B's 0.339; S4 both compute-store pairs (1 → 2, 3 → 4) active; S5 the influence profile
replicates (terminal slot inert, identity exact, values over keys, mid depth group over late).
""")

md("## Setup")
code('''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "dba9ce0e7bc05a3d93b3bee73712e2635b15c0b2"
REPO_DIR = "/kaggle/working/latent-reasoning"
CONFIG = "configs/official_simcot_codi_llama1b.yaml"
CODI1B_CONFIG = "configs/official_codi_llama1b.yaml"
OUTPUT_ROOT = "/kaggle/working/simcot_codi_llama1b"
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
                "tests/test_directed_counterfactual_runner.py",
                "tests/test_cache_carrier_audit_runner.py"], check=True)
''')

md("## Step 0: CODI-1B released-path replication (mask dropped, batch 128), lost in §114")
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

CODI1B_PAPER = 0.519
RELEASED_DIR = "outputs/official_codi_llama1b_released_path"
run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CODI1B_CONFIG, "--device", "cuda",
                "--output-dir", RELEASED_DIR, "--set", "model.dtype=float16", "--set", "eval.batch_size=128",
                "--set", "model.pad_aware_generation=false"])
released = json.loads(open(summary_under(RELEASED_DIR)).read())
RELEASED_ACC = released["datasets"]["gsm8k"]
print(f"CODI-1B released path (mask dropped, batch 128): {RELEASED_ACC:.4f}; paper {CODI1B_PAPER}; |delta| = {abs(RELEASED_ACC - CODI1B_PAPER):.4f}")
print("prediction (ledger §114): within 0.03 of the paper ->", abs(RELEASED_ACC - CODI1B_PAPER) <= 0.03, "| pad-aware measured 0.5550")
''')

md("## Step 1: SIM-CoT GSM8K gate (pad-aware path; lower bound against the paper's 56.1%)")
code(r'''
PAPER = 0.561
PRECISION = None
for precision in ("float16", "float32"):
    run_with_retry([sys.executable, "-u", "-m", "src.eval.official_codi", "--config", CONFIG, "--device", "cuda",
                    "--set", f"model.dtype={precision}", "--set", f"eval.batch_size={BATCH_SIZE}"])
    REPRODUCTION_SUMMARY = summary_under("outputs/official_simcot_codi_llama1b")
    gate = json.loads(open(REPRODUCTION_SUMMARY).read())
    acc = gate["datasets"]["gsm8k"]
    print(f"{precision}: SIM-CoT pad-aware gsm8k {acc:.4f} (paper {PAPER}, delta {acc - PAPER:+.4f}); gate {gate['accuracy_gate']['status']} ({gate['accuracy_gate']['direction']})")
    if gate["accuracy_gate"]["status"] == "passed":
        PRECISION = precision
        break
assert PRECISION, "the SIM-CoT lower-bound gate failed in both precisions; do not run the audit or twins"
print("reproduction passed in", PRECISION, "->", REPRODUCTION_SUMMARY)
''')

md("## Step 2: the §108 cache-carrier audit on SIM-CoT (smoke, then 512 questions)")
code(r'''
AUDIT_DIR = OUTPUT_ROOT + "_cache_carrier_audit"
def audit_command(output_dir, *extra):
    return [sys.executable, "-u", "scripts/run_codi_cache_carrier_audit.py",
            "--config", CONFIG, "--reproduction-summary", REPRODUCTION_SUMMARY,
            "--output-dir", output_dir, "--batch-size", BATCH_SIZE, "--precision", PRECISION, "--device", "cuda", *extra]
if RUN_SMOKE:
    run_with_retry(audit_command(AUDIT_DIR + "_smoke", "--smoke"))
run_with_retry(audit_command(AUDIT_DIR))
audit = json.loads(open(pathlib.Path(AUDIT_DIR) / "summary.json").read())
print("audit baseline:", audit["baseline"])
print(audit["decision"])
slots = audit["report"]["slots"]
import pandas as pd
display(pd.DataFrame({int(k): {c: round(v[c]["change_rate"], 3) for c in v} for k, v in slots.items()}).T)
display(pd.DataFrame({n: {"change rate": round(v["change_rate"], 3), "accuracy": round(v["accuracy"], 3)} for n, v in audit["report"]["tail"].items()}).T)
print("S5 identity kv_all == hidden_kv_all:", audit["report"]["tail"]["kv_all"]["change_rate"] == audit["report"]["tail"]["hidden_kv_all"]["change_rate"],
      "| terminal inert:", not audit["checks"]["a2_terminal_slot_is_a_memory"], "| route split:", audit["checks"]["a3_route_split"],
      "| values>keys:", audit["checks"]["a4_values_over_keys"], "| mid>late:", audit["checks"]["a5_layers_8_9_over_10_11"])
''')

md("## Step 3: twins on SIM-CoT, standard profile (all chains), then long profile (≥ 3 steps)")
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
uniq = summaries["long"]["report"]["unique_location"]; uspec = summaries["long"]["report"]["unique_specificity"]
print()
print("Step 0 CODI-1B released path:", round(RELEASED_ACC, 4), "(paper 0.519; pad-aware 0.555)")
print("S1 unique location (long):", round(uniq["fraction_unique"], 3), "(CODI-1B 0.602) by slot", uniq["rows_by_slot"])
print("S2 specificity at both feeding slots (E3 per site):", summaries["long"]["checks"]["e3_per_site"])
for s, v in uspec.items():
    print(f"   slot {s}: n={v['n']} state at site {v['state_at_site']['target']} vs elsewhere {[w['target'] for w in v['state_elsewhere'].values()]}"
          f" | K/V at site {v.get('kv_store_at_site', {}).get('target')} vs elsewhere {[w['target'] for w in v.get('kv_store_elsewhere', {}).values()]}")
print("S3 long-chain single-thought transport: state_1", round(t(lng, "state_1"), 3), "state_3", round(t(lng, "state_3"), 3), "(CODI-1B 0.011 / 0.339) | E2:", summaries["long"]["checks"]["e2_single_thought_transport"])
print("S4 pairs: kv_2", round(t(lng, "kv_2"), 3), "kv_4", round(t(lng, "kv_4"), 3), "(CODI-1B 0.011 / 0.260)")
print("tails: kv_all short", round(t(std, "kv_all"), 3), "long", round(t(lng, "kv_all"), 3), "(CODI-1B 0.550 / 0.671)")
print("terminal slot (long): state_5", round(t(lng, "state_5"), 3), "kv_5", round(t(lng, "kv_5"), 3))
for w in summaries["long"]["warnings"]:
    print("WARNING:", w)
print("seconds: audit", round(audit["seconds"]), "standard", round(summaries["standard"]["seconds"]), "long", round(summaries["long"]["seconds"]))
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
nbf.write(nb, OUTPUT)
print(OUTPUT)
