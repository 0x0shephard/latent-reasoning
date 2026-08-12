"""Build the Kaggle notebook for the three correctness tracks."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_correctness_tracks.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Does the part of CODI that *predicts* being right have anything to do with the part that *produces* the right answer?

## Where this comes from

The band experiment found which directions **determine** the answer: PCs 4–31, 28 of 768,
11.3% of the colon-state variance, and **87.8%** of exact-match accuracy. Removing them
costs 30.5 points. PCs 0–3 hold 82.3% of the variance and 6.1% of the accuracy — they
shift all 50,257 logits together by about +27.1 with a spread of 0.46, a near-uniform
lift that cannot change an argmax.

Splitting the same states by whether the model was actually **right** asks a different
question, and exploratory work found the two answers barely overlap:

| quantity | value |
|---|---:|
| between-class variance (right vs wrong) | **3.98%** of total |
| correctness direction $d$, share inside PCs 0–3 | **97.13%** |
| … against a random-split null | 70.56% median, **0/200** replicates reach 97% |
| $\|d\|$ vs the random-split median | **11.7×** |

So the correctness signal lives almost entirely in the directions that cannot change an
answer. That is a falsifiable mechanism, and this notebook tests the three things it
predicts.

## The three tracks

**1. Detect** — read a new state, predict whether the model will be right.
Exploratory: the correctness direction scored AUC 0.700, but **the model's own margin
scored 0.874**. So the gate is not "beat chance", it is **beat the margin**, because a
probe that loses to a number already sitting in the output is not worth computing.

**2. Steer** — move the state to make the model *more* right.
Global steering along $d$ already failed: +0.38 points at best, −18.65 at $\alpha{=}4$.
The mechanism explains why. This track asks the version that has somewhere to act:
**confine the steering vector to PC 4–31**, the band the readout is sensitive to. This is
the track that would be a genuinely new result if it worked — the first intervention on
this model that *improves* rather than degrades.

**3. Project** — build the retention subspace from correct examples only.
Exploratory principal angles between the class-blind and correct-only bases averaged
**0.99**. This is a preregistered replication of an expected null, run because it is the
first thing a reviewer asks.

## Preregistered gates — fixed before any test number is read

| track | primary arm | passes if |
|---|---|---|
| detect | `fisher_plus_margin` | ΔAUC over margin-only ≥ **0.01**, bootstrap lower bound > 0 |
| steer | `margin_band` | gain ≥ **1.0 point**, lower bound > 0, **and** beats the best matched random direction *in the same band* |
| project | `correct_only` at rank 28 | advantage over class-blind ≥ **1.0 point**, lower bound > 0 |

**A failed gate is a result here, not a wasted run.** The steer track is expected to fail
if the volume-knob account is right; saying so in advance is what makes either outcome
informative.

## What is new about the method

The calibration pool is split **fit (1024) / select (1024)**, and GSM8K test is read once
per arm. Every direction is estimated on fit; every hyperparameter — ridge strength,
Fisher shrinkage, steering step $\alpha$, rank — is chosen on select. **Nothing touches
test.** That closes the standing caveat on the band experiment, whose boundaries (4, 32)
were read off test-set curves.

The analytic tier is model-free and reuses the margin-geometry colon states, so it costs
minutes. The generation tier then confirms the steer track with **real greedy decoding
and numeric exact match**, which is the outcome the project's results are stated in.

Enable Internet and a GPU, then **Save Version → Save & Run All**.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
# Attach the completed margin-geometry export; its float32 colon states are reused,
# so every direction here lives in the same space the band result was measured in.
COLON_STATES_INPUT = ""
READOUT_INPUT = ""
RESUME_INPUT = ""

# The generation tier costs three full GSM8K decodes (~2h). The analytic tier alone
# answers all three gates on first-token accuracy; set this False to stop there.
RUN_GENERATION_TIER = True
EVAL_BATCH_SIZE = 32
GENERATION_PRECISION = "float32"

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
    "## Pin the environment that reproduces the checkpoint\n\n"
    "The run that established the 43.669% GSM8K gate recorded transformers 4.52.4, "
    "peft 0.15.2, datasets 3.6.0, huggingface_hub 0.32.4, torch 2.10.0+cu128. The "
    "current Kaggle image keeps the same torch but ships much newer transformers and "
    "peft, and on it the native gate scores **0.3723** instead of 0.4367.\n\n"
    "`src/models/official_codi.py` is written against Transformers 4.52 legacy-tuple "
    "cache semantics, which later versions changed. The CODI latent loop threads "
    "`past_key_values` through six hand-rolled forward passes, so a change there "
    "degrades it silently rather than raising.\n\n"
    "This matters even for the analytic tier: the colon states being reused were "
    "collected under these pins, and the steering arms are confirmed by decoding."
)
code(
    '''
PINNED_PACKAGES = {
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "datasets": "3.6.0",
    "huggingface_hub": "0.32.4",
}

import importlib.metadata as _md


def _installed(name):
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


before = {name: _installed(name) for name in PINNED_PACKAGES}
print("before:", before)
missing = [f"{n}=={v}" for n, v in PINNED_PACKAGES.items() if before.get(n) != v]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

probe = (
    "import importlib.metadata as m;"
    "print({n: m.version(n) for n in "
    f"{list(PINNED_PACKAGES)!r}" "})"
)
after = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print("after :", after.stdout.strip() or after.stderr.strip())
for name, wanted in PINNED_PACKAGES.items():
    assert f"'{name}': '{wanted}'" in after.stdout, (name, wanted, after.stdout)
import torch as _torch
print("torch  :", _torch.__version__, "(unchanged from the reproducing run)")
'''
)

markdown(
    "## Repair the peft/torchao environment\n\n"
    "`peft`'s LoRA dispatcher raises instead of returning `False` when torchao is "
    "installed below its minimum, and the guard is `lru_cache`d so it must be probed "
    "in a subprocess. torchao is optional here, so removing an incompatible copy "
    "restores the clean path. No-op when it is absent or current."
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

markdown(
    "## Source tests\n\n"
    "The integration file drives both CLIs end to end over a synthetic cache. Every "
    "Kaggle failure in the previous experiment was reachable only by running the "
    "scripts rather than the library, so that file is the one that matters here."
)
code(
    '''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_endpoint_correctness_geometry.py",
     "tests/test_correctness_tracks_integration.py",
     "tests/test_endpoint_margin_geometry.py"],
    check=True,
)
'''
)

markdown("## Resolve inputs")
code(
    '''
OUTPUT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/outputs/official_codi_correctness_tracks")
REPORT_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/reports/official_codi_correctness_tracks")
LOG_ROOT = pathlib.Path("/kaggle/working/latent-reasoning/logs/official_codi_correctness_tracks")
for path in (OUTPUT_ROOT, REPORT_ROOT, LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)

# Restore a previous session's outputs when one is attached. Generation arms are
# keyed by request hash, so restored work is skipped and only the rest runs.
if RESUME_INPUT:
    import shutil
    restored = 0
    for source in pathlib.Path(RESUME_INPUT).rglob("official_codi_correctness_tracks"):
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if item.is_file():
                target = OUTPUT_ROOT / item.relative_to(source)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                    restored += 1
    print("restored files from RESUME_INPUT:", restored)


def _discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"attach a dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split("/")), value))[0]


COLON_STATES = _discover(COLON_STATES_INPUT, "colon_states_seed89/colon_states.pt")
READOUT = _discover(READOUT_INPUT, "colon_states_seed89/readout.pt")
REPRODUCTION_SUMMARY = _discover(REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json")

import torch
cache = torch.load(COLON_STATES, map_location="cpu", weights_only=False)
assert cache["parity_gate"]["passed"], cache["parity_gate"]
assert cache["metadata"]["precision"] == "float32", cache["metadata"]["precision"]
print("colon states     :", COLON_STATES)
print("parity agreement :", cache["full_parity_gate"]["agreement"])
print("calibration rows :", cache["calibration_states"].shape[0])
print("evaluation rows  :", cache["evaluation_states"].shape[0])
'''
)

markdown(
    "## Show the split the whole experiment turns on\n\n"
    "Printed before any gate is applied, so the class geometry is on the record next "
    "to the results it explains. Watch two numbers: how little of the total variance "
    "separates right from wrong, and how much of the correctness direction sits in the "
    "band that cannot change an answer — judged against the random-split null, because "
    "*any* split of a dataset leans toward its high-variance directions."
)
code(
    '''
from src.mech.endpoint_correctness_geometry import (
    ACCURACY_BAND, LIFT_BAND, answer_margin, direction_band_profile,
    first_token_correct, fit_correctness_directions, random_split_null,
    roc_auc, sorted_eigenbasis,
)
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE

si = list(cache["state_order"]).index(ANALYTIC_STATE)
calib = cache["calibration_states"][:, si, :].double()
readout = torch.load(READOUT, map_location="cpu", weights_only=False)["output_embedding"].double()
correct = first_token_correct(calib, readout, cache["calibration_gold_first_token"])
print(f"calibration first-token accuracy: {float(correct.double().mean()):.4f}")

centred = calib - calib.mean(0)
values, vectors = sorted_eigenbasis(centred.T @ centred / centred.shape[0])
directions = fit_correctness_directions(calib, correct)
profile = direction_band_profile(directions.mean_difference, vectors)
null = random_split_null(calib, correct, vectors, replicates=200, seed=20260812)

import numpy as np
observed = profile[f"{LIFT_BAND[0]}:{LIFT_BAND[1]}"]
shares, norms = np.array(null["band_shares"]), np.array(null["norms"])
raw = calib[correct].mean(0) - calib[~correct].mean(0)
print()
print(f"between-class variance   : {100 * directions.between_fraction:6.2f}% of total")
print(f"|mean difference|        : {float(raw.norm()):6.2f}   "
      f"(random-split median {np.median(norms):.2f}, ratio {float(raw.norm()) / np.median(norms):.1f}x)")
print(f"share inside PC 0-3      : {100 * observed:6.2f}%  "
      f"(null median {100 * np.median(shares):.2f}%, "
      f"{int((shares >= observed).sum())}/200 replicates reach it)")
print(f"share inside PC 4-31     : {100 * profile['4:32']:6.2f}%  <- the accuracy band")
print()
print(f"model's own margin, AUC  : {roc_auc(answer_margin(calib, readout), correct):.4f}")
print(f"correctness direction AUC: {roc_auc(calib @ directions.mean_difference, correct):.4f}")
'''
)

markdown(
    "## Run all three tracks (analytic tier)\n\n"
    "Model-free: every arm is scored by multiplying the edited state through the "
    "bias-free `lm_head`, which is exact for state 12 and for the first answer token. "
    "That is what makes a full $\\alpha$ grid across ~21 steering arms, nine probes and "
    "six ranks affordable at all."
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


SWEEP = OUTPUT_ROOT / "correctness_tracks.json"
VECTORS = OUTPUT_ROOT / "correctness_tracks.pt"
if not SWEEP.is_file():
    run_persisted(
        [sys.executable, "-u", "scripts/run_official_codi_correctness_tracks.py",
         "--config", "configs/official_codi_gpt2.yaml",
         "--states", COLON_STATES, "--readout", READOUT,
         "--output", str(SWEEP), "--vectors-output", str(VECTORS),
         "--device", "cuda"],
        "correctness_tracks_sweep.log",
    )
sweep = json.loads(SWEEP.read_text())
print("splits:", {k: v for k, v in sweep["splits"].items() if k != "correct_share"})
print("correct share per split:", sweep["splits"]["correct_share"])
print("baseline first-token accuracy:", sweep["baseline_first_token_accuracy"])
'''
)

markdown("## Apply the preregistered gates")
code(
    '''
REPORT = REPORT_ROOT / "correctness_tracks_summary.json"
run_persisted(
    [sys.executable, "-u", "scripts/analyze_official_codi_correctness_tracks.py",
     "--config", "configs/official_codi_gpt2.yaml",
     "--sweep", str(SWEEP), "--vectors", str(VECTORS),
     "--output", str(REPORT)],
    "analyze_correctness_tracks.log",
)
report = json.loads(REPORT.read_text())
'''
)

markdown(
    "## Track 1 — detect, in full\n\n"
    "Every probe against the one number that matters. Read the **margin** row first: "
    "any probe below it is measuring something the model already exposes for free."
)
code(
    '''
probes = report["detect"]["all_probes"]
margin_auc = probes["margin"]["test_auc"]
print(f"{'probe':<28} {'select AUC':>11} {'test AUC':>10} {'vs margin':>11}")
print("-" * 63)
for name, entry in sorted(probes.items(), key=lambda kv: -kv[1]["test_auc"]):
    delta = entry["test_auc"] - margin_auc
    flag = "  <- baseline" if name == "margin" else ""
    print(f"{name:<28} {entry['select_auc']:>11.4f} {entry['test_auc']:>10.4f} "
          f"{delta:>+11.4f}{flag}")
print()
d = report["detect"]
print(f"PRIMARY: {d['primary_probe']}  delta {d['delta_auc']:+.4f} "
      f"[{d['delta_ci'][0]:+.4f}, {d['delta_ci'][1]:+.4f}]  ->  "
      f"{'PASS' if d['passed'] else 'FAIL'}")
'''
)

markdown(
    "## Track 2 — steer, in full\n\n"
    "`band_profile` is the diagnostic that ties this to the mechanism: it is how much "
    "of each steering vector's length lies in PCs 0–3 (cannot change an argmax) versus "
    "PC 4–31 (can). The prediction is that `*_global` arms sit almost entirely in the "
    "first column and do nothing, while the `*_band` arms are the only ones with a "
    "chance."
)
code(
    '''
s = report["steer"]
base = s["baseline_accuracy"]
print(f"baseline first-token accuracy: {base:.4f}\\n")
print(f"{'arm':<26} {'alpha':>6} {'accuracy':>9} {'gain (pts)':>11} "
      f"{'PC0-3':>7} {'PC4-31':>7}")
print("-" * 70)
for name, entry in sorted(s["all_arms"].items(), key=lambda kv: -kv[1]["test_accuracy"]):
    p = entry["band_profile"]
    print(f"{name:<26} {entry['selected_alpha']:>6g} {entry['test_accuracy']:>9.4f} "
          f"{100 * (entry['test_accuracy'] - base):>+11.2f} "
          f"{100 * p['0:4']:>6.1f}% {100 * p['4:32']:>6.1f}%")
print()
for kind, values in s["random_controls"].items():
    if values:
        print(f"random {kind:<7} controls: n={len(values)}  "
              f"best {max(values):.4f}  median {sorted(values)[len(values) // 2]:.4f}")
print()
print(f"PRIMARY: {s['primary_arm']}  gain {s['gain_points']:+.2f} pts "
      f"[{s['gain_ci_points'][0]:+.2f}, {s['gain_ci_points'][1]:+.2f}]  "
      f"margin over best random-in-band {s['margin_over_random_points']:+.2f} pts  ->  "
      f"{'PASS' if s['passed'] else 'FAIL'}")
'''
)

markdown(
    "## Track 3 — project, in full\n\n"
    "`mean cos` is the average principal-angle cosine between the class-blind and "
    "correct-only bases at that rank. A value near 1 means they are the *same subspace*, "
    "which is the predicted explanation for why restricting to correct examples buys "
    "nothing: only ~4% of the variance is between-class, so dropping the wrong answers "
    "barely moves the covariance."
)
code(
    '''
p = report["project"]
print(f"{'rank':>5} {'class-blind':>12} {'correct-only':>13} {'incorrect-only':>15} {'mean cos':>9}")
print("-" * 58)
for rank in sorted(p["by_rank"], key=int):
    e = p["by_rank"][rank]
    print(f"{rank:>5} {e['class_blind']:>12.4f} {e['correct_only']:>13.4f} "
          f"{e['incorrect_only']:>15.4f} {e['mean_cosine']:>9.4f}")
print()
print(f"PRIMARY: rank {p['rank']}  advantage {p['advantage_points']:+.2f} pts "
      f"[{p['advantage_ci_points'][0]:+.2f}, {p['advantage_ci_points'][1]:+.2f}]  ->  "
      f"{'PASS' if p['passed'] else 'FAIL'}")
'''
)

markdown("## Verdict")
code(
    '''
print("=" * 62)
for track, passed in report["tracks_passed"].items():
    print(f"  {'PASS' if passed else 'FAIL'}   {track}")
print("=" * 62)
if not report["tracks_passed"]["steer"]:
    print()
    print("The steer track failing is the prediction, not a bug: the correctness")
    print("direction sits in PCs 0-3, where a shift moves all 50,257 logits together")
    print("and cannot change an argmax. Confirming that a band-restricted vector")
    print("*also* fails narrows what the band is: a location the answer is read from,")
    print("not a handle a constant offset can push.")
'''
)

markdown(
    "## Generation tier — confirm the steer track on exact match\n\n"
    "The analytic tier scores the first answer token. A steering vector could move that "
    "token without moving the parsed numeric answer, so the claim is only settled by "
    "decoding. Three full-GSM8K greedy runs: baseline, the primary band arm, and one "
    "matched random direction inside the same band.\n\n"
    "$\\alpha$ comes from the analytic export, which chose it on the **select** split. It "
    "is not re-tuned here; re-tuning at this tier would be tuning on the test set."
)
code(
    '''
RUNS_ROOT = OUTPUT_ROOT / "runs"
if RUN_GENERATION_TIER:
    GEN_ARMS = ["baseline", report["steer"]["primary_arm"], "random_band_r00"]
    for arm in GEN_ARMS:
        run_persisted(
            [sys.executable, "-u", "scripts/run_official_codi_correctness_steer_generation.py",
             "--config", "configs/official_codi_gpt2.yaml",
             "--reproduction-summary", REPRODUCTION_SUMMARY,
             "--states", COLON_STATES, "--readout", READOUT,
             "--vectors", str(VECTORS),
             "--output-dir", str(RUNS_ROOT / arm),
             "--arm", arm,
             "--eval-batch-size", str(EVAL_BATCH_SIZE),
             "--precision", GENERATION_PRECISION,
             "--device", "cuda"],
            f"generation_{arm}.log",
        )
    summaries = {p.parent.name: json.loads(p.read_text())
                 for p in sorted(RUNS_ROOT.rglob("summary.json"))}
    base = summaries["baseline"]
    assert base.get("baseline_drift_passed") is True, base
    print()
    print(f"{'arm':<26} {'exact match':>12} {'vs baseline':>12} {'alpha':>7}")
    print("-" * 60)
    for arm in GEN_ARMS:
        s_ = summaries[arm]
        print(f"{arm:<26} {s_['accuracy']:>12.4f} "
              f"{100 * (s_['accuracy'] - base['accuracy']):>+12.2f} "
              f"{s_['alpha']:>7g}")
else:
    print("generation tier skipped (RUN_GENERATION_TIER = False)")
'''
)

markdown("## Checksummed export")
code(
    '''
import hashlib, shutil

EXPORT = pathlib.Path("/kaggle/working/official_codi_correctness_tracks_export")
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
