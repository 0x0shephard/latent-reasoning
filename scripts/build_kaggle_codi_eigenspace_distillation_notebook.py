"""Build the Kaggle notebook for the SlimSpec-inspired CODI readout experiment."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_codi_eigenspace_distilled_readout.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Experiment 1 — Can SlimSpec-style learning repair CODI's eigenspace head?

## Goal

CODI's fixed rank-32 projection retained 94.4% of baseline exact-match accuracy and
made the isolated vocabulary projection 11.3× faster on a T4. This experiment asks
whether **logit distillation** can recover the missing accuracy without giving up the
rank-32 computation.

The comparison is controlled at equal rank and vocabulary size:

1. `full`: the original 768 → 50,257 head;
2. `fixed_eigen_r32`: our centred, frozen eigenspace head;
3. `learned_eigen_r32`: the same head used as initialization, then distilled;
4. `learned_random_r32`: an equally sized random-basis initialization, then distilled;
5. `learned_eigen_r64`: a higher-rank recovery arm.

The transformer and CODI projector remain frozen. Only the two matrices of a learned
head may change. Fit, validation, and test rows are disjoint. No rank, epoch, or learning
rate is selected on test.

### Preregistered primary gate

The hybrid is supported if `learned_eigen_r32` (a) retains at least 98% of baseline
numeric exact match, (b) is no worse than the fixed rank-32 head, and (c) is faster than
the full head end to end. The random-initialized arm tests whether our eigenspace is a
useful initialization rather than merely an arbitrary low-rank parameterization.
""")

markdown("## Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after these files are pushed.
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
COLON_STATES_INPUT = ""
READOUT_INPUT = ""
OUTPUT_ROOT = "/kaggle/working/codi_eigenspace_distilled_readout"

SEED = 20260901
FIT_EXAMPLES = 1536
DISTILL_QUESTIONS = 512
MAX_DISTILL_STATES = 8192
DISTILL_EPOCHS = 6
DISTILL_BATCH_SIZE = 16
LEARNING_RATE = 3e-4
TEMPERATURE = 2.0
BOOTSTRAP_SAMPLES = 5000
TEST_LIMIT = 0          # 0 = complete 1,319-question GSM8K test.
GENERATION_BATCH_SIZE = 32
RUN_FULL_GENERATION = True
RUN_R64 = True

import copy, glob, json, os, pathlib, random, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
RESOLVED_COMMIT = "origin/main" if RUN_COMMIT == "main" else RUN_COMMIT
subprocess.run(
    ["git", "-C", REPO_DIR, "checkout", "--detach", RESOLVED_COMMIT], check=True
)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("commit:", subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip())
assert RUN_COMMIT == "main" or subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip().startswith(RUN_COMMIT)
''')

markdown("## Pin the checkpoint-compatible environment")
code(r'''
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

missing = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()
           if installed(name) != version]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print({name: installed(name) for name in PINNED_PACKAGES})

# PEFT 0.15.2 treats an old optional torchao installation as fatal. CODI does not use it.
probe = (
    "from peft.import_utils import is_torchao_available\n"
    "try:\n    print('ok' if is_torchao_available() else 'absent')\n"
    "except ImportError as error:\n    print('incompatible:' + str(error))\n"
)
state = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
state = (state.stdout + state.stderr).strip()
print("torchao:", state)
if state.startswith("incompatible"):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
''')

markdown("## Checks")
code(r'''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_eigenspace_readout.py", "tests/test_official_codi.py"],
    check=True,
)
''')

markdown("## Resolve the completed CODI inputs")
code(r'''
def discover(explicit, pattern):
    if explicit:
        return explicit
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"Attach a Kaggle dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split('/')), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
COLON_STATES = discover(COLON_STATES_INPUT, "colon_states.pt")
READOUT = discover(READOUT_INPUT, "readout.pt")
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("reproduction:", REPRODUCTION_SUMMARY)
print("states      :", COLON_STATES)
print("readout     :", READOUT)
''')

markdown("## Load states and freeze the fit/validation/test split")
code(r'''
import torch
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.mech.endpoint_correctness_geometry import first_token_correct, readout_matrix
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE
from src.mech.eigenspace_readout import (
    LowRankVocabularyHead, benchmark_vocabulary_head, covariance_eigensystem,
    distil_low_rank_head, evaluate_head_fidelity, orthonormal_random_basis,
)

torch.manual_seed(SEED)
cache, readout_payload = load_margin_cache(pathlib.Path(COLON_STATES), pathlib.Path(READOUT))
state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
calibration_states = cache["calibration_states"][:, state_index, :].float()
test_states = cache["evaluation_states"][:, state_index, :].float()
test_gold = cache["evaluation_gold_first_token"].long()
weight_cpu = readout_matrix(readout_payload).float()

order = torch.randperm(calibration_states.shape[0], generator=torch.Generator().manual_seed(SEED))
assert 0 < FIT_EXAMPLES < calibration_states.shape[0]
fit_states = calibration_states[order[:FIT_EXAMPLES]]
validation_states = calibration_states[order[FIT_EXAMPLES:]]
centre, eigenvalues, eigenvectors = covariance_eigensystem(fit_states)

assert fit_states.shape[1] == weight_cpu.shape[1] == 768
assert test_states.shape[0] == 1319
print({"fit": len(fit_states), "validation": len(validation_states), "test": len(test_states)})
''')

markdown("## Load the frozen official CODI model")
code(r'''
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint, generate_official_codi,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config

cfg = load_config("configs/official_codi_gpt2.yaml")
verify_full_reproduction_gate(pathlib.Path(REPRODUCTION_SUMMARY), cfg)
device = torch.device("cuda")
dtype = resolve_torch_dtype("float32", device)
checkpoint = download_official_checkpoint(
    repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
    filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
    token=os.environ.get("HF_TOKEN") or None,
)
model, tokenizer = build_official_codi_gpt2(
    base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
    dtype=dtype, settings=cfg.model, token=os.environ.get("HF_TOKEN") or None,
)
load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.to(device=device, dtype=dtype).eval()
base_model = model.codi.get_base_model()
full_head = base_model.get_output_embeddings()
assert tuple(full_head.weight[:model.eot_id].shape) == tuple(weight_cpu.shape)
''')

markdown("## Collect states from every generated answer position")
code(r'''
from src.data.datasets import load_train_set

training = load_train_set(load_config("configs/data.yaml"), trace_style="eq_only")
question_order = torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))
questions = [str(training[int(i)]["question"]) for i in question_order[:DISTILL_QUESTIONS]]
captured = []

def observe_answer_states(states, active_mask, answer_position):
    del answer_position
    if bool(active_mask.any()):
        captured.append(states[active_mask].detach().cpu().float())

_ = generate_official_codi(
    model, tokenizer, questions,
    latent_iterations=int(cfg.eval.latent_iterations),
    max_new_tokens=64,
    batch_size=GENERATION_BATCH_SIZE,
    device=device,
    answer_cue="The answer is:", force_answer_cue=True,
    answer_state_observer=observe_answer_states,
)
all_answer_states = torch.cat(captured, dim=0)
if all_answer_states.shape[0] > MAX_DISTILL_STATES:
    keep = torch.randperm(
        all_answer_states.shape[0], generator=torch.Generator().manual_seed(SEED + 2)
    )[:MAX_DISTILL_STATES]
    all_answer_states = all_answer_states[keep]
split = int(0.9 * all_answer_states.shape[0])
distill_train_states = all_answer_states[:split]
distill_validation_states = all_answer_states[split:]
print({"all_answer_states": len(all_answer_states), "train": split,
       "validation": len(distill_validation_states)})
assert len(distill_validation_states) > 0
''')

markdown("## Construct fixed and learned heads at matched rank")
code(r'''
weight = weight_cpu.to(device)

def make_head(basis):
    return LowRankVocabularyHead.from_basis(weight, basis, centre).to(device)

fixed_eigen_r32 = make_head(eigenvectors[:, :32])
learned_eigen_r32 = copy.deepcopy(fixed_eigen_r32)
random_basis_r32 = orthonormal_random_basis(768, 32, seed=SEED + 3)
learned_random_r32 = make_head(random_basis_r32)
learned_eigen_r64 = make_head(eigenvectors[:, :64]) if RUN_R64 else None

training_runs = {}
for name, head in [
    ("learned_eigen_r32", learned_eigen_r32),
    ("learned_random_r32", learned_random_r32),
    ("learned_eigen_r64", learned_eigen_r64),
]:
    if head is None:
        continue
    result = distil_low_rank_head(
        head, distill_train_states, distill_validation_states, weight,
        epochs=DISTILL_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
        anchor_strength=1e-4, seed=SEED,
    )
    training_runs[name] = {
        "losses": list(result.losses), "best_validation_kl": result.best_validation_kl,
        "best_epoch": result.best_epoch,
    }
    print(name, training_runs[name])
''')

markdown("## Analytic test: vocabulary fidelity and first-token correctness")
code(r'''
heads = {
    "fixed_eigen_r32": fixed_eigen_r32,
    "learned_eigen_r32": learned_eigen_r32,
    "learned_random_r32": learned_random_r32,
}
if learned_eigen_r64 is not None:
    heads["learned_eigen_r64"] = learned_eigen_r64

analytic = {}
full_logits = test_states.to(device) @ weight.T
full_predictions = full_logits.argmax(-1).cpu()
analytic["full"] = {
    "top1_agreement": 1.0,
    "first_token_accuracy": float((full_predictions == test_gold).double().mean()),
}
for name, head in heads.items():
    fidelity = evaluate_head_fidelity(
        head, test_states, weight, batch_size=32, temperature=TEMPERATURE
    )
    with torch.no_grad():
        predictions = []
        for start in range(0, len(test_states), 32):
            predictions.append(head(test_states[start:start+32].to(device)).argmax(-1).cpu())
    predictions = torch.cat(predictions)
    analytic[name] = {
        **fidelity,
        "first_token_accuracy": float((predictions == test_gold).double().mean()),
    }
print(json.dumps(analytic, indent=2))
''')

markdown("## Full-generation test and end-to-end timing")
code(r'''
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set

examples = load_eval_set("gsm8k", load_config(cfg.data_config).eval.gsm8k)
if TEST_LIMIT:
    examples = examples[:TEST_LIMIT]
questions = [example["question"] for example in examples]

generation_heads = {"full": full_head, **heads}
generation_results = {}
generation_outcomes = {}
if RUN_FULL_GENERATION:
    for name, head in generation_heads.items():
        base_model.set_output_embeddings(head)
        torch.cuda.synchronize()
        started = time.perf_counter()
        generations = generate_official_codi(
            model, tokenizer, questions,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=int(cfg.eval.max_new_tokens),
            batch_size=GENERATION_BATCH_SIZE, device=device,
            answer_cue="The answer is:", force_answer_cue=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        correct = [answers_match(text, row["gold"]) for text, row in zip(generations, examples)]
        generation_outcomes[name] = correct
        generation_results[name] = {
            "examples": len(examples), "correct": int(sum(correct)),
            "numeric_exact_match": float(sum(correct) / len(correct)),
            "wall_clock_seconds": elapsed,
            "examples_per_second": len(examples) / elapsed,
        }
        print(name, generation_results[name])
base_model.set_output_embeddings(full_head)
''')

markdown("## Isolated head timing, operation counts, and decision")
code(r'''
timing = {
    "full": benchmark_vocabulary_head(
        full_head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    )
}
for name, head in heads.items():
    timing[name] = benchmark_vocabulary_head(
        head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    )

vocabulary = weight.shape[0]
operations = {
    "full_macs_per_token": int(768 * vocabulary),
    "rank32_macs_per_token": int(32 * (768 + vocabulary)),
    "rank64_macs_per_token": int(64 * (768 + vocabulary)),
}
decision = {"status": "not_run"}
paired_intervals = {}
if generation_results:
    def paired_interval(left, right, seed):
        delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed)
        draws = []
        for _ in range(0, BOOTSTRAP_SAMPLES, 250):
            count = min(250, BOOTSTRAP_SAMPLES - len(draws))
            index = torch.randint(len(delta), (count, len(delta)), generator=generator)
            draws.extend(delta[index].mean(dim=1).tolist())
        values = torch.tensor(draws)
        return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]

    paired_intervals["learned_eigen_r32_minus_fixed_eigen_r32"] = paired_interval(
        generation_outcomes["learned_eigen_r32"], generation_outcomes["fixed_eigen_r32"], SEED
    )
    paired_intervals["learned_eigen_r32_minus_full"] = paired_interval(
        generation_outcomes["learned_eigen_r32"], generation_outcomes["full"], SEED + 1
    )
    baseline = generation_results["full"]["numeric_exact_match"]
    learned = generation_results["learned_eigen_r32"]["numeric_exact_match"]
    fixed = generation_results["fixed_eigen_r32"]["numeric_exact_match"]
    decision = {
        "retained_fraction": learned / baseline if baseline else None,
        "learned_minus_fixed_points": 100 * (learned - fixed),
        "end_to_end_speedup": generation_results["full"]["wall_clock_seconds"] /
                              generation_results["learned_eigen_r32"]["wall_clock_seconds"],
    }
    decision["status"] = (
        "hybrid_supported" if decision["retained_fraction"] >= 0.98
        and learned >= fixed and decision["end_to_end_speedup"] > 1.0
        else "hybrid_not_supported_by_gate"
    )

summary = {
    "experiment": "codi_eigenspace_initialized_logit_distillation",
    "seed": SEED, "training": training_runs, "analytic_test": analytic,
    "generation_test": generation_results, "paired_bootstrap_95ci": paired_intervals,
    "head_latency_us": timing,
    "operation_counts": operations, "decision": decision,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
for name, head in heads.items():
    torch.save({
        "name": name, "rank": head.rank, "state_dict": head.state_dict(),
        "centre": centre.float(), "seed": SEED,
    }, pathlib.Path(OUTPUT_ROOT) / f"{name}.pt")
print(json.dumps({"decision": decision, "head_latency_us": timing,
                  "operation_counts": operations}, indent=2))
print("wrote", summary_path)
''')

markdown(r"""
## Takeaways

Interpret only the executed `summary.json`. A faster isolated head is not sufficient:
the primary claim requires numeric exact match and complete generation latency. If the
random-initialized learned head matches the eigenspace initialization, the low-rank
factorization may still work, but the special value of our discovered basis is not
supported. Do not change rank or training hyperparameters after reading test results;
start a new, explicitly labelled experiment instead.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
