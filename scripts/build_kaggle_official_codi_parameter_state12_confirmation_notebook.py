"""Build the Kaggle run-all notebook for state-12 confirmation."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_parameter_state12_confirmation.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# Confirm parameter-aware state 12 at CODI's answer colon

## Goal

This is a single-hypothesis frozen-checkpoint confirmation of the discovery result:
removing the three parameter-aware directions at state 12 may harm CODI answer accuracy
more than removing an equally energetic arbitrary state-12 subspace.

The design is frozen before looking at new outcomes:

- fit the state-12 mean and covariance on 2,048 unique **GSM8K train** questions;
- prove normalized-question disjointness from all 1,319 GSM8K test questions;
- evaluate one selected state-12 rank-three arm and 500 selected-orthogonal,
  activation-energy-matched random controls;
- use one preregistered primary hypothesis, so there is no selector multiplicity;
- withhold confirmation if median random versus selected realized test RMS differs by
  more than 10%, even when the accuracy tests pass.

No model weight is updated. Enable Internet and a T4-or-newer GPU, then choose
**Save Version → Save & Run All**.
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
RUN_SMOKE = True
RUN_FULL = True
CALIBRATION_EXAMPLES = 2048
CALIBRATION_SEED = 73
CALIBRATION_BATCH_SIZE = 16
RANDOM_REPLICATES = 500
RANDOM_SEED = 20260808
EVAL_BATCH_SIZE = 32
PRECISION = "auto"
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 0
ALPHA = 0.05
MAXIMUM_RMS_RATIO_DEVIATION = 0.10
ARM_SHARD_INDEX = 0
ARM_SHARD_COUNT = 1
UPLOAD_AS_KAGGLE_DATASET = False
KAGGLE_DATASET_HANDLE = "jonraza15/codi-parameter-state12-confirmation"
'''
)

markdown("### 1. Install, pin, and test")
code(
    '''
import hashlib, json, os, pathlib, shutil, subprocess, sys
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
repo = pathlib.Path(REPO_DIR)
if not (repo / ".git").is_dir():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
target = f"origin/{RUN_COMMIT}" if RUN_COMMIT == "main" else RUN_COMMIT
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", target], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements-official-codi.txt")], check=True)
os.chdir(REPO_DIR)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Checked out:", commit)
if RUN_COMMIT == "main":
    print("PIN RUN_COMMIT BEFORE THE FINAL RUN:", commit)
import torch, transformers
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator"
print("Torch:", torch.__version__, "Transformers:", transformers.__version__, "GPU:", torch.cuda.get_device_name(0))
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_endpoint_inference_ablation.py",
    "tests/test_endpoint_retention.py",
    "tests/test_official_codi.py",
], cwd=REPO_DIR, check=True)
'''
)

markdown("### 2. Locate the three immutable basis artifacts")
code(
    '''
EXPECTED_CONTRACTS = {
    "source_faithful_student_and_teacher_answer_colon_v2": "energy",
    "answer_conditioned_colon_block_states_v1": "answer_conditioned",
    "parameter_aware_colon_final_two_blocks_v1": "parameter_aware",
}
SOURCE_DATASETS = {
    "energy": "jonraza15/corrected-official-codi-answer-cue-endpoint-tsv-c",
    "answer_conditioned": "jonraza15/official-codi-answer-conditioned-experiment",
    "parameter_aware": "jonraza15/official-codi-parameter-aware-experiment",
}
EXPLICIT = {
    "energy": ENERGY_BASIS_INPUT,
    "answer_conditioned": ANSWER_CONDITIONED_BASIS_INPUT,
    "parameter_aware": PARAMETER_AWARE_BASIS_INPUT,
}
def merged_metadata(path, payload):
    metadata = dict(payload.get("metadata", {}))
    manifest = path.parent / "run_manifest.json"
    parity = path.parent / "native_loss_gradient_parity.json"
    if manifest.is_file():
        for key, value in json.loads(manifest.read_text()).items():
            metadata.setdefault(key, value)
    if parity.is_file() and "native_parity_gate" not in metadata:
        metadata["native_parity_gate"] = json.loads(parity.read_text())
    return metadata
def completed_method(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = merged_metadata(path, payload)
    except Exception as error:
        return None, (str(path), "unreadable", type(error).__name__)
    method = EXPECTED_CONTRACTS.get(metadata.get("contract"))
    try:
        full = method == "energy" and int(metadata.get("calibration_examples", -1)) == 5000
        full = full or (
            method in {"answer_conditioned", "parameter_aware"}
            and int(metadata.get("residual_fit_examples", -1)) == 1024
            and int(metadata.get("direction_selection_examples", -1)) == 1024
        )
    except (TypeError, ValueError):
        full = False
    parity = metadata.get("native_parity_gate", {}).get("status")
    diagnostic = (
        str(path), metadata.get("contract"), metadata.get("calibration_examples"),
        metadata.get("residual_fit_examples"),
        metadata.get("direction_selection_examples"), parity,
    )
    return (method if full and parity == "passed" else None), diagnostic
def scan_roots(roots):
    found = {method: [] for method in EXPECTED_CONTRACTS.values()}
    diagnostics, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("basis.pt"):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            method, diagnostic = completed_method(path)
            diagnostics.append(diagnostic)
            if method:
                found[method].append(path)
    return found, diagnostics
search_roots = [pathlib.Path("/kaggle/input")]
found, diagnostics = scan_roots(search_roots)
missing = [method for method in EXPECTED_CONTRACTS.values() if not EXPLICIT[method] and not found[method]]
if missing:
    import kagglehub
    for method in missing:
        handle = SOURCE_DATASETS[method]
        print(f"Completed {method} basis is not mounted; downloading {handle}")
        try:
            search_roots.append(pathlib.Path(kagglehub.dataset_download(handle)))
        except Exception as error:
            print(f"Download failed for {handle}: {type(error).__name__}: {error}")
    found, diagnostics = scan_roots(search_roots)
basis_by_method = {}
for method, explicit in EXPLICIT.items():
    paths = [pathlib.Path(explicit)] if explicit else found[method]
    if explicit:
        resolved, diagnostic = completed_method(paths[0])
        diagnostics.append(diagnostic)
        if resolved != method:
            paths = []
    if not explicit and paths:
        by_sha = {}
        for path in paths:
            by_sha.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(path)
        if len(by_sha) == 1:
            copies = next(iter(by_sha.values()))
            paths = [sorted(copies, key=lambda path: (len(path.parts), path.as_posix()))[0]]
            if len(copies) > 1:
                print(f"Collapsed {len(copies)} byte-identical copies of {method} to {paths[0]}")
    if len(paths) != 1:
        print("basis.pt diagnostics:", *diagnostics, sep="\\n  ")
        raise AssertionError(
            f"Need exactly one completed {method} basis. Attach/download "
            f"{SOURCE_DATASETS[method]} or set its explicit path; found {paths}"
        )
    basis_by_method[method] = paths[0]
ENERGY_BASIS = basis_by_method["energy"]
ANSWER_CONDITIONED_BASIS = basis_by_method["answer_conditioned"]
PARAMETER_AWARE_BASIS = basis_by_method["parameter_aware"]
for method, path in basis_by_method.items():
    print(method, path)
'''
)

markdown("### 3. Durable paths, resume, and reproduction gate")
code(
    '''
OUTPUT_ROOT = repo / "outputs" / "official_codi_parameter_state12_confirmation"
RUNS_ROOT = OUTPUT_ROOT / "runs"
STATS_ROOT = OUTPUT_ROOT / "gsm8k_train_stats_seed73"
REPORT_ROOT = repo / "reports" / "official_codi_parameter_state12_confirmation"
LOG_ROOT = repo / "logs" / "official_codi_parameter_state12_confirmation"
VALIDATION_ROOT = repo / "outputs" / "official_codi_gpt2"
for path in (RUNS_ROOT, STATS_ROOT, REPORT_ROOT, LOG_ROOT, VALIDATION_ROOT):
    path.mkdir(parents=True, exist_ok=True)
def run_persisted(command, log_name):
    log_path = LOG_ROOT / log_name
    print("Starting:", " ".join(map(str, command)), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=REPO_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}; inspect {log_path}")
if RESUME_INPUT:
    candidates = [
        path for path in pathlib.Path(RESUME_INPUT).rglob("official_codi_parameter_state12_confirmation")
        if path.is_dir()
    ]
    assert candidates, "RESUME_INPUT does not contain a state-12 confirmation tree"
    shutil.copytree(
        sorted(candidates, key=lambda path: (len(path.parts), path.as_posix()))[0],
        OUTPUT_ROOT, dirs_exist_ok=True,
    )
EXPECTED_REVISION = "fd641b3d3edc59e4f534b55588e906588c9e36bb"
def passed_summary(path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    gate = payload.get("accuracy_gate", payload.get("gate"))
    status = gate.get("status") if isinstance(gate, dict) else gate
    return (
        status == "passed"
        and payload.get("evaluated_counts", {}).get("gsm8k") == 1319
        and payload.get("checkpoint_revision") in {None, EXPECTED_REVISION}
    )
if REPRODUCTION_SUMMARY_INPUT:
    REPRODUCTION_SUMMARY = pathlib.Path(REPRODUCTION_SUMMARY_INPUT)
    assert passed_summary(REPRODUCTION_SUMMARY)
else:
    candidates = [
        path for path in pathlib.Path("/kaggle/input").rglob("summary.json")
        if passed_summary(path)
    ] + [path for path in VALIDATION_ROOT.rglob("summary.json") if passed_summary(path)]
    if not candidates:
        assert RUN_REPRODUCTION_GATE_IF_MISSING
        run_persisted([
            sys.executable, "-u", "-m", "src.eval.official_codi",
            "--config", "configs/official_codi_gpt2.yaml",
            "--datasets", "gsm8k", "--limit", "0", "--device", "cuda",
            "--output-dir", str(VALIDATION_ROOT),
        ], "official_codi_gsm8k_gate.log")
        candidates = [path for path in VALIDATION_ROOT.rglob("summary.json") if passed_summary(path)]
    assert candidates
    REPRODUCTION_SUMMARY = sorted(candidates, key=lambda path: path.as_posix())[0]
print("Reproduction summary:", REPRODUCTION_SUMMARY)
'''
)

markdown("## Steps")
markdown("### 4. Fit state-12 statistics on disjoint GSM8K train")
code(
    '''
STATS_PATH = STATS_ROOT / "activation_stats.pt"
run_persisted([
    sys.executable, "-u", "scripts/collect_official_codi_parameter_state12_confirmation_stats.py",
    "--config", "configs/official_codi_gpt2.yaml",
    "--reproduction-summary", str(REPRODUCTION_SUMMARY),
    "--energy-basis", str(ENERGY_BASIS),
    "--answer-conditioned-basis", str(ANSWER_CONDITIONED_BASIS),
    "--parameter-aware-basis", str(PARAMETER_AWARE_BASIS),
    "--output-dir", str(STATS_ROOT),
    "--examples", str(CALIBRATION_EXAMPLES),
    "--batch-size", str(CALIBRATION_BATCH_SIZE),
    "--sampling-seed", str(CALIBRATION_SEED),
    "--precision", "float32", "--device", "cuda",
], "collect_gsm8k_train_state12_stats.log")
stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
assert stats["contract"] == "frozen_checkpoint_parameter_aware_state12_confirmation_v1"
assert stats["count"] == CALIBRATION_EXAMPLES
assert stats["student_mean"].shape == (13, 768)
assert set(stats["student_covariance_by_state"]) == {"12"}
sampling = stats["metadata"]["sampling"]
assert sampling["raw_train_examples"] == 7473
assert sampling["train_test_normalized_question_overlap"] == 0
assert not stats["metadata"]["test_labels_used_for_calibration"]
assert not stats["metadata"]["test_activations_used_for_calibration"]
print("Fresh GSM8K-train calibration:", sampling)
'''
)

markdown("### 5. Register the 502 paired arms")
code(
    '''
PRIMARY_ARM = "remove_parameter_aware_state12_primary"
RANDOM_ARMS = [
    f"remove_matched_random_parameter_aware_state12_r{replicate:03d}"
    for replicate in range(RANDOM_REPLICATES)
]
ARMS = ["baseline", PRIMARY_ARM, *RANDOM_ARMS]
assert len(ARMS) == 502 and len(set(ARMS)) == len(ARMS)
assert 0 <= ARM_SHARD_INDEX < ARM_SHARD_COUNT
ARMS_TO_RUN = [
    arm for index, arm in enumerate(ARMS)
    if index % ARM_SHARD_COUNT == ARM_SHARD_INDEX
]
print("Primary hypotheses: 1; matched random controls:", len(RANDOM_ARMS))
print("This shard:", len(ARMS_TO_RUN), "/", len(ARMS))
'''
)

markdown("## Checks")
markdown("### 6. Smoke-test state, cue reach, energy, and overlap")
code(
    '''
def confirmation_command(root, arm, eval_limit=0):
    return [
        sys.executable, "-u", "scripts/run_official_codi_endpoint_inference_ablation.py",
        "--config", "configs/official_codi_gpt2.yaml",
        "--reproduction-summary", str(REPRODUCTION_SUMMARY),
        "--energy-basis", str(ENERGY_BASIS),
        "--answer-conditioned-basis", str(ANSWER_CONDITIONED_BASIS),
        "--parameter-aware-basis", str(PARAMETER_AWARE_BASIS),
        "--activation-stats", str(STATS_PATH),
        "--output-dir", str(root), "--arm", arm,
        "--random-replicates", str(RANDOM_REPLICATES),
        "--random-seed", str(RANDOM_SEED),
        "--alpha", "1.0", "--eval-limit", str(eval_limit),
        "--eval-batch-size", str(EVAL_BATCH_SIZE),
        "--precision", PRECISION, "--device", "cuda",
        "--state12-confirmation",
    ]
if RUN_SMOKE:
    smoke_arms = ["baseline", PRIMARY_ARM, RANDOM_ARMS[0]]
    smoke = []
    for arm in smoke_arms:
        root = OUTPUT_ROOT / "smoke" / arm
        run_persisted(confirmation_command(root, arm, eval_limit=32), f"smoke_{arm}.log")
        smoke.append(json.loads((root / "summary.json").read_text()))
    assert all(value["endpoint_coverage"]["endpoint_reached_count"] == 32 for value in smoke)
    assert smoke[1]["spec"]["ranks"] == [0] * 12 + [3]
    random_spec = smoke[2]["spec"]
    assert random_spec["ranks"] == [0] * 12 + [3]
    target = random_spec["calibration_target_energy_by_state"]["12"]
    achieved = random_spec["calibration_achieved_energy_by_state"]["12"]
    assert abs(achieved - target) / max(target, 1e-12) <= 2e-5
    assert random_spec["selected_overlap_by_state"]["12"] <= 0.20
    print("Smoke passed: exact cue, state-12-only intervention, energy match, and orthogonality")
'''
)

markdown("### 7. Run full paired GSM8K confirmation")
code(
    '''
if RUN_FULL:
    for arm in ARMS_TO_RUN:
        run_persisted(confirmation_command(RUNS_ROOT / arm, arm), f"{arm}.log")
summaries = list(RUNS_ROOT.rglob("summary.json"))
print("Completed full arms:", len(summaries), "/", len(ARMS))
FULL_EXPERIMENT_COMPLETE = len(summaries) == len(ARMS)
if ARM_SHARD_COUNT == 1 and RUN_FULL:
    assert FULL_EXPERIMENT_COMPLETE
'''
)

markdown("## Results")
markdown("### 8. Apply the single preregistered decision rule")
code(
    '''
REPORT_PATH = REPORT_ROOT / "parameter_state12_confirmation_summary.json"
if FULL_EXPERIMENT_COMPLETE:
    run_persisted([
        sys.executable, "-u", "scripts/analyze_official_codi_parameter_state12_confirmation.py",
        "--runs-root", str(RUNS_ROOT), "--output", str(REPORT_PATH),
        "--random-replicates", str(RANDOM_REPLICATES),
        "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
        "--bootstrap-seed", str(BOOTSTRAP_SEED),
        "--alpha", str(ALPHA),
        "--maximum-rms-ratio-deviation", str(MAXIMUM_RMS_RATIO_DEVIATION),
    ], "analyze_parameter_state12_confirmation.log")
    report = json.loads(REPORT_PATH.read_text())
    primary = report["primary_result"]
    null = report["matched_random_null"]
    print("STATUS:", report["status"])
    print("Baseline accuracy:", report["forced_cue_baseline_accuracy"])
    print("Selected state-12 loss (pp):", primary["accuracy_loss_percentage_points"])
    print("Paired CI:", primary["bootstrap_95_ci"])
    print("McNemar p:", primary["mcnemar_one_sided_p"])
    print("Matched-random empirical p:", primary["empirical_matched_random_p"])
    print("Median random/selected evaluation RMS ratio:", null["median_random_to_selected_rms_ratio"])
    print("RMS transport passed:", null["evaluation_rms_transport_passed"])
else:
    print("Analysis deferred until all 502 arms are present. Export and resume this shard.")
'''
)

markdown("## Next Steps")
markdown(
    "A `confirmed` result supports the parameter-aware state-12 rank-three subspace as "
    "more accuracy-critical than equally energetic selected-orthogonal alternatives at "
    "the forced answer-colon endpoint. `not_confirmed` rejects that claim. "
    "`evaluation_magnitude_transport_failed` means the accuracy statistics cannot be "
    "interpreted because matching did not transfer from GSM8K train to test. None of "
    "these outcomes implies an architectural inference speedup."
)

markdown("### 9. Export checksummed, resumable artifacts")
code(
    '''
EXPORT_ROOT = pathlib.Path("/kaggle/working/official_codi_parameter_state12_confirmation_export")
if EXPORT_ROOT.exists():
    shutil.rmtree(EXPORT_ROOT)
export_repo = EXPORT_ROOT / "latent-reasoning"
shutil.copytree(OUTPUT_ROOT, export_repo / "outputs" / "official_codi_parameter_state12_confirmation")
if REPORT_ROOT.exists():
    shutil.copytree(REPORT_ROOT, export_repo / "reports" / "official_codi_parameter_state12_confirmation")
shutil.copytree(LOG_ROOT, export_repo / "logs" / "official_codi_parameter_state12_confirmation")
validation = export_repo / "outputs" / "official_codi_gpt2_reproduction"
validation.mkdir(parents=True, exist_ok=True)
shutil.copy2(REPRODUCTION_SUMMARY, validation / "summary.json")
(EXPORT_ROOT / "RUN_COMMIT.txt").write_text(commit + "\\n")
(EXPORT_ROOT / "RUN_INSTRUCTIONS.txt").write_text(
    "Attach this export and set RESUME_INPUT to its root. Reattach the three immutable basis datasets.\\n"
    f"Arm shard {ARM_SHARD_INDEX + 1}/{ARM_SHARD_COUNT}; completed {len(summaries)}/{len(ARMS)} full arms.\\n"
)
files = sorted(path for path in EXPORT_ROOT.rglob("*") if path.is_file())
(EXPORT_ROOT / "SHA256SUMS.txt").write_text(
    "\\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(EXPORT_ROOT).as_posix()}"
        for path in files
    ) + "\\n"
)
print("Export root:", EXPORT_ROOT, "files:", len(files), "complete:", FULL_EXPERIMENT_COMPLETE)
if UPLOAD_AS_KAGGLE_DATASET:
    import kagglehub
    kagglehub.dataset_upload(
        KAGGLE_DATASET_HANDLE, str(EXPORT_ROOT),
        version_notes=f"CODI parameter-aware state-12 confirmation at {commit}",
    )
'''
)

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(notebook, OUTPUT)
print(OUTPUT)
