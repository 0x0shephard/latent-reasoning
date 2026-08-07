"""Build the reader-facing Kaggle run-all notebook for accuracy localization."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_official_codi_endpoint_accuracy_localization.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(
    r"""
# CODI answer-colon accuracy localization

## Goal

This frozen-checkpoint experiment determines whether the previously significant
answer-conditioned and parameter-aware **six-dimensional subspaces** contain smaller
accuracy-critical states or directions. It uses the same forced `EOT + The answer is:`
endpoint as the collectors and changes no model weight.

The confirmatory controls are stronger than the previous rank-matched experiment:

- 100 random joint subspaces per successful selector;
- exact per-state matching of calibration projection energy, without scaling an
  intervention;
- Holm correction across both selector-versus-null tests;
- state-only, single-direction, and joint-minus-one localization arms;
- full paired GSM8K evaluation and exact endpoint-coverage checks.

Enable Internet and a T4-or-newer GPU, then choose **Save Version → Save & Run All**.
"""
)

markdown("## Setup")
code(
    '''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
RESUME_INPUT = ""  # Optional previous localization export root.
ENERGY_BASIS_INPUT = ""
ANSWER_CONDITIONED_BASIS_INPUT = ""
PARAMETER_AWARE_BASIS_INPUT = ""
RUN_REPRODUCTION_GATE_IF_MISSING = True
RUN_SMOKE = True
RUN_FULL = True
CALIBRATION_EXAMPLES = 1024
CALIBRATION_SEED = 71
CALIBRATION_BATCH_SIZE = 16
RANDOM_REPLICATES = 100
RANDOM_SEED = 20260807
EVAL_BATCH_SIZE = 32
PRECISION = "auto"
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 0
FAMILYWISE_ALPHA = 0.05
ARM_SHARD_INDEX = 0
ARM_SHARD_COUNT = 1
UPLOAD_AS_KAGGLE_DATASET = False
KAGGLE_DATASET_HANDLE = "jonraza15/official-codi-answer-colon-accuracy-localization"
'''
)

markdown("### 1. Install, pin, and test")
code(
    '''
import datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
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

markdown("### 2. Locate the immutable selector artifacts")
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
    contract = metadata.get("contract")
    method = EXPECTED_CONTRACTS.get(contract)
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
        str(path), contract, metadata.get("calibration_examples"),
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
OUTPUT_ROOT = repo / "outputs" / "official_codi_endpoint_accuracy_localization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
STATS_ROOT = OUTPUT_ROOT / "activation_stats_seed71"
REPORT_ROOT = repo / "reports" / "official_codi_endpoint_accuracy_localization"
LOG_ROOT = repo / "logs" / "official_codi_endpoint_accuracy_localization"
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
        path for path in pathlib.Path(RESUME_INPUT).rglob("official_codi_endpoint_accuracy_localization")
        if path.is_dir()
    ]
    assert candidates, "RESUME_INPUT does not contain a localization output tree"
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
markdown("### 4. Fit the fresh endpoint mean and full covariance")
code(
    '''
STATS_PATH = STATS_ROOT / "activation_stats.pt"
stats_command = [
    sys.executable, "-u", "scripts/collect_official_codi_endpoint_activation_stats.py",
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
    "--localization-covariance",
]
run_persisted(stats_command, "collect_localization_activation_stats.log")
stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
assert stats["contract"] == "frozen_checkpoint_forced_answer_colon_accuracy_localization_v1"
assert stats["count"] == CALIBRATION_EXAMPLES
assert stats["student_mean"].shape == (13, 768)
assert set(stats["student_covariance_by_state"]) == {"11", "12"}
for state in ("11", "12"):
    covariance = stats["student_covariance_by_state"][state]
    assert covariance.shape == (768, 768) and torch.isfinite(covariance).all()
print("Fresh calibration request:", stats["request_sha256"])
'''
)

markdown("### 5. Register the 232 paired arms")
code(
    '''
METHODS = ["answer_conditioned", "parameter_aware"]
SELECTED_ARMS = ["remove_energy_joint_negative_control"]
for method in METHODS:
    SELECTED_ARMS.append(f"remove_{method}_joint")
    SELECTED_ARMS.extend(f"remove_{method}_state{state}" for state in (11, 12))
    SELECTED_ARMS.extend(
        f"remove_{method}_s{state}_d{slot}"
        for state in (11, 12) for slot in range(3)
    )
    SELECTED_ARMS.extend(
        f"remove_{method}_joint_except_s{state}_d{slot}"
        for state in (11, 12) for slot in range(3)
    )
MATCHED_RANDOM_ARMS = [
    f"remove_matched_random_{method}_joint_r{replicate:03d}"
    for method in METHODS for replicate in range(RANDOM_REPLICATES)
]
ARMS = ["baseline", *SELECTED_ARMS, *MATCHED_RANDOM_ARMS]
assert len(SELECTED_ARMS) == 31
assert len(ARMS) == 232 and len(set(ARMS)) == len(ARMS)
assert 0 <= ARM_SHARD_INDEX < ARM_SHARD_COUNT
ARMS_TO_RUN = [
    arm for index, arm in enumerate(ARMS)
    if index % ARM_SHARD_COUNT == ARM_SHARD_INDEX
]
print("Selected/localization:", len(SELECTED_ARMS), "matched random:", len(MATCHED_RANDOM_ARMS))
print("This shard:", len(ARMS_TO_RUN), "/", len(ARMS))
'''
)

markdown("## Checks")
markdown("### 6. Smoke-test endpoint reach and norm matching")
code(
    '''
def ablation_command(root, arm, eval_limit=0):
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
        "--accuracy-localization",
    ]
if RUN_SMOKE:
    smoke_arms = ["baseline", "remove_answer_conditioned_joint", MATCHED_RANDOM_ARMS[0]]
    smoke = []
    for arm in smoke_arms:
        root = OUTPUT_ROOT / "smoke" / arm
        run_persisted(ablation_command(root, arm, eval_limit=32), f"smoke_{arm}.log")
        smoke.append(json.loads((root / "summary.json").read_text()))
    assert all(item["endpoint_coverage"]["endpoint_reached_count"] == 32 for item in smoke)
    random_spec = smoke[-1]["spec"]
    for state in ("11", "12"):
        target = random_spec["calibration_target_energy_by_state"][state]
        achieved = random_spec["calibration_achieved_energy_by_state"][state]
        assert abs(achieved - target) / max(target, 1e-12) <= 2e-5
        assert random_spec["selected_overlap_by_state"][state] <= 0.20
    print("Smoke passed: exact cue reach and calibration-energy match")
'''
)

markdown("### 7. Run the full paired experiment")
code(
    '''
if RUN_FULL:
    for arm in ARMS_TO_RUN:
        run_persisted(ablation_command(RUNS_ROOT / arm, arm), f"{arm}.log")
summaries = list(RUNS_ROOT.rglob("summary.json"))
print("Completed full arms:", len(summaries), "/", len(ARMS))
FULL_EXPERIMENT_COMPLETE = len(summaries) == len(ARMS)
if ARM_SHARD_COUNT == 1 and RUN_FULL:
    assert FULL_EXPERIMENT_COMPLETE
'''
)

markdown("## Results")
markdown("### 8. Run the multiplicity-corrected paired analysis")
code(
    '''
REPORT_PATH = REPORT_ROOT / "endpoint_accuracy_localization_summary.json"
if FULL_EXPERIMENT_COMPLETE:
    run_persisted([
        sys.executable, "-u", "scripts/analyze_official_codi_endpoint_accuracy_localization.py",
        "--runs-root", str(RUNS_ROOT), "--output", str(REPORT_PATH),
        "--random-replicates", str(RANDOM_REPLICATES),
        "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
        "--bootstrap-seed", str(BOOTSTRAP_SEED),
        "--familywise-alpha", str(FAMILYWISE_ALPHA),
    ], "analyze_endpoint_accuracy_localization.log")
    report = json.loads(REPORT_PATH.read_text())
    print("Forced-cue baseline:", report["forced_cue_baseline_accuracy"])
    print("Critical joint subspaces:", report["critical_joint_subspaces"])
    print("Energy negative-control loss (pp):", report["negative_control"]["accuracy_loss_percentage_points"])
    for method, result in report["localization"].items():
        print("\\n", method, "parent passed:", result["parent_joint_passed"])
        print("States:", result["states"])
        for name, value in result["directions"].items():
            if value["individually_necessary"] or value["rescues_joint_ablation"]:
                print(name, value)
else:
    print("Analysis deferred until all 232 arms are present. Export this shard and resume it in another run.")
'''
)

markdown("## Next Steps")
markdown(
    "The report may identify a direction as individually necessary, as rescuing the "
    "joint ablation, as both (an accuracy-core direction), or as neither. A null result "
    "means the six-dimensional effect remains distributed or interactive. These hooks "
    "do not provide an inference-speed claim; structural compression is a later experiment."
)

markdown("### 9. Export checksummed, resumable artifacts")
code(
    '''
EXPORT_ROOT = pathlib.Path("/kaggle/working/official_codi_endpoint_accuracy_localization_export")
if EXPORT_ROOT.exists():
    shutil.rmtree(EXPORT_ROOT)
export_repo = EXPORT_ROOT / "latent-reasoning"
shutil.copytree(OUTPUT_ROOT, export_repo / "outputs" / "official_codi_endpoint_accuracy_localization")
if REPORT_ROOT.exists():
    shutil.copytree(REPORT_ROOT, export_repo / "reports" / "official_codi_endpoint_accuracy_localization")
shutil.copytree(LOG_ROOT, export_repo / "logs" / "official_codi_endpoint_accuracy_localization")
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
        version_notes=f"CODI answer-colon accuracy localization at {commit}",
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
