"""Build the Kaggle notebook for the trajectory-whitened global CODI head."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_global_low_rank_lm_head.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Trajectory-whitened global low-rank LM head

## Research question

Can one shared low-rank vocabulary head replace CODI's original LM head at **every
visible generation position**, retain at least 98% of the reproduced GSM8K exact-match
accuracy, and provide a measured head and end-to-end speed improvement?

This is not the earlier answer-cue projection and it is not a position-conditioned
mixture. One set of `down` and `up` matrices is used for every generated token.

## What is new

1. Fit on final-normalized states from the complete visible answer trajectory.
2. Initialize by randomized SVD of `W S`, where `S S^T` is the regularized activation
   covariance. This minimizes activation-weighted rather than weight-only logit error.
3. Distil the full teacher distribution, teacher top token, and ranking margin.
4. Train nested rank-32, rank-64, and rank-96 prefixes simultaneously.
5. Roll out the compressed head once, collect the states it actually visits, label
   them with the frozen full head, and retrain.
6. Select an adaptive 32→64 threshold using training-selection states only.
7. Open the complete 1,319-question GSM8K test only for the locked final arms.

The factorization and training implementation is model-independent. CODI supplies the
primary locked causal evaluation; the same `src.mech.global_low_rank_head` API accepts
the final hidden states and output matrix of Qwen, Llama, Gemma, or another causal LM.
Numerical factors are refitted per model—they are never copied between hidden spaces.
""")

markdown("## Configuration")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit printed after this notebook is released.
REPO_DIR = "/kaggle/working/latent-reasoning"
REPRODUCTION_SUMMARY_INPUT = ""
OUTPUT_ROOT = "/kaggle/working/trajectory_whitened_global_head"

SEED = 20260905
FIT_QUESTIONS = 1024
SELECT_QUESTIONS = 256
ONPOLICY_QUESTIONS = 256
MAX_FIT_STATES = 4096
MAX_SELECT_STATES = 1024
MAX_ONPOLICY_STATES = 2048
RANKS = (32, 64, 96)
PRIMARY_RANK = 64
CLEAN_EPOCHS = 4
ONPOLICY_EPOCHS = 2
DISTILL_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
TEMPERATURE = 2.0
KL_WEIGHT = 1.0
TOKEN_WEIGHT = 0.25
MARGIN_WEIGHT = 0.25
NESTED_WEIGHT = 0.5
MINIMUM_MARGIN = 0.25
GENERATION_BATCH_SIZE = 32
MAX_NEW_TOKENS = 64
ADAPTIVE_MIN_TOP1 = 0.98
BOOTSTRAP_SAMPLES = 5000
TEST_LIMIT = 0  # 0 means all 1,319 locked GSM8K test questions.
RUN_FULL_GENERATION = True

import copy, glob, json, os, pathlib, random, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
resolved = "origin/main" if RUN_COMMIT == "main" else RUN_COMMIT
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", resolved], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture_output=True,
    text=True, check=True,
).stdout.strip()
assert RUN_COMMIT == "main" or CODE_COMMIT.startswith(RUN_COMMIT)
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("commit:", CODE_COMMIT)
''')

markdown("## Reproduce the official CODI environment and run focused tests")
code(r'''
PINNED_PACKAGES = {
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "datasets": "3.6.0",
    "huggingface_hub": "0.32.4",
}
from importlib.metadata import PackageNotFoundError, version as package_version

def installed_package_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None

missing = [f"{name}=={wanted}" for name, wanted in PINNED_PACKAGES.items()
           if installed_package_version(name) != wanted]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

# PEFT 0.15 probes torchao when it is installed. Kaggle images sometimes contain an
# old incompatible torchao; uninstalling that optional package is safer than allowing
# adapter construction to fail.
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

subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_global_low_rank_head.py",
    "tests/test_official_codi.py",
], check=True)
''')

markdown("## Resolve the completed CODI reproduction input")
code(r'''
def discover(explicit, pattern):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return str(path)
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"Attach a Kaggle dataset containing {pattern}"
    return sorted(matches, key=lambda value: (len(value.split('/')), value))[0]

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
print("reproduction:", REPRODUCTION_SUMMARY)
''')

markdown("## Load the frozen official model")
code(r'''
import torch
import torch.nn as nn
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.mech.global_low_rank_head import (
    NestedLowRankVocabularyHead,
    activation_whitened_factors,
    distil_nested_head,
    evaluate_nested_head,
)
from src.mech.eigenspace_readout import benchmark_vocabulary_head
from src.inference.official_codi_fast import (
    generate_official_codi_fast, prepare_official_codi_batches,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    generate_official_codi, load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config

torch.manual_seed(SEED)
random.seed(SEED)
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
load_report = load_official_checkpoint(
    model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.to(device=device, dtype=dtype).eval()
base_model = model.codi.get_base_model()
full_head = base_model.get_output_embeddings()
hidden_size = int(model.config.hidden_size)
vocabulary_size = int(model.eot_id)
readout_weight = full_head.weight[:vocabulary_size].detach()
readout_bias = (
    None if getattr(full_head, "bias", None) is None
    else full_head.bias[:vocabulary_size].detach()
)
assert hidden_size == 768
print({"checkpoint": load_report.checkpoint_sha256,
       "hidden": hidden_size, "vocabulary": vocabulary_size})
''')

markdown("## Freeze question-disjoint fit, selection, recovery, and test populations")
code(r'''
from src.data.datasets import load_eval_set, load_train_set

training_rows = load_train_set(load_config("configs/data.yaml"), trace_style="eq_only")
required = FIT_QUESTIONS + SELECT_QUESTIONS + ONPOLICY_QUESTIONS
assert required <= len(training_rows)
order = torch.randperm(len(training_rows), generator=torch.Generator().manual_seed(SEED))

def questions_for(indices):
    return [str(training_rows[int(index)]["question"]) for index in indices]

fit_indices = order[:FIT_QUESTIONS]
select_indices = order[FIT_QUESTIONS:FIT_QUESTIONS + SELECT_QUESTIONS]
onpolicy_indices = order[
    FIT_QUESTIONS + SELECT_QUESTIONS:
    FIT_QUESTIONS + SELECT_QUESTIONS + ONPOLICY_QUESTIONS
]
assert set(fit_indices.tolist()).isdisjoint(select_indices.tolist())
assert set(fit_indices.tolist()).isdisjoint(onpolicy_indices.tolist())
assert set(select_indices.tolist()).isdisjoint(onpolicy_indices.tolist())
fit_questions = questions_for(fit_indices)
select_questions = questions_for(select_indices)
onpolicy_questions = questions_for(onpolicy_indices)
test_examples = load_eval_set("gsm8k", load_config(cfg.data_config).eval.gsm8k)
assert len(test_examples) == 1319
if TEST_LIMIT:
    test_examples = test_examples[:TEST_LIMIT]
print({"fit_questions": len(fit_questions), "select_questions": len(select_questions),
       "onpolicy_questions": len(onpolicy_questions), "test": len(test_examples)})
''')

markdown("## Verify that the timing fast path preserves the released decoder")
code(r'''
parity_questions = select_questions[:16]
reference_parity = generate_official_codi(
    model, tokenizer, parity_questions,
    latent_iterations=int(cfg.eval.latent_iterations),
    max_new_tokens=MAX_NEW_TOKENS, batch_size=GENERATION_BATCH_SIZE,
    device=device, answer_cue="The answer is:", force_answer_cue=True,
)
parity_prepared = prepare_official_codi_batches(
    tokenizer, parity_questions, batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)
fast_parity = generate_official_codi_fast(
    model, tokenizer, parity_prepared,
    latent_iterations=int(cfg.eval.latent_iterations),
    max_new_tokens=MAX_NEW_TOKENS, device=device, answer_cue="The answer is:",
)
assert tuple(reference_parity) == fast_parity.texts, (
    "The transformer-body fast path changed decoded outputs; stop before fitting."
)
print({"fastpath_parity_examples": len(reference_parity), "exact": True})
''')

markdown("## Collect every visible-token state from clean teacher trajectories")
code(r'''
def deterministic_cap(states, maximum, seed):
    if len(states) <= maximum:
        return states
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(len(states), generator=generator)[:maximum]
    return states[index]

@torch.inference_mode()
def collect_states(questions, output_head, tag, maximum):
    captured = []
    positions = []
    def observer(states, active_mask, answer_position):
        if bool(active_mask.any()):
            selected = states[active_mask].detach().cpu().float()
            captured.append(selected)
            positions.extend([int(answer_position)] * len(selected))
    base_model.set_output_embeddings(output_head)
    try:
        prepared = prepare_official_codi_batches(
            tokenizer, questions, batch_size=GENERATION_BATCH_SIZE,
            length_bucketed=False,
        )
        generation = generate_official_codi_fast(
            model, tokenizer, prepared,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS,
            device=device, answer_cue="The answer is:",
            answer_state_observer=observer,
        )
    finally:
        base_model.set_output_embeddings(full_head)
    assert captured, f"{tag} produced no visible-token states"
    states = torch.cat(captured, dim=0)
    assert states.shape[1] == hidden_size
    capped = deterministic_cap(states, maximum, SEED + len(tag))
    print(tag, {"observed": len(states), "retained": len(capped),
                "maximum_position": max(positions), "generations": len(generation.texts)})
    return capped, list(generation.texts)

fit_states, _ = collect_states(fit_questions, full_head, "clean_fit", MAX_FIT_STATES)
select_states, _ = collect_states(
    select_questions, full_head, "clean_select", MAX_SELECT_STATES
)
''')

markdown("## Activation-whitened nested initialization")
code(r'''
centre, down_weight, up_weight, output_bias, initialization_report = (
    activation_whitened_factors(
        fit_states,
        readout_weight,
        rank=max(RANKS),
        readout_bias=readout_bias,
        ridge_relative=1e-4,
        oversample=16,
        power_iterations=1,
        seed=SEED,
        compute_device=device,
        compute_dtype=torch.float32,
    )
)
head = NestedLowRankVocabularyHead.from_whitened_factors(
    centre, down_weight, up_weight, output_bias, RANKS
).to(device=device, dtype=torch.float32)
initial_metrics = {
    f"rank_{rank}": evaluate_nested_head(
        head, select_states, readout_weight, readout_bias=readout_bias,
        rank=rank, batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
    )
    for rank in RANKS
}
print(json.dumps(initial_metrics, indent=2))
''')

markdown("## Clean-trajectory margin-aware nested distillation")
code(r'''
clean_result = distil_nested_head(
    head,
    fit_states,
    select_states,
    readout_weight,
    readout_bias=readout_bias,
    epochs=CLEAN_EPOCHS,
    batch_size=DISTILL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    temperature=TEMPERATURE,
    kl_weight=KL_WEIGHT,
    token_weight=TOKEN_WEIGHT,
    margin_weight=MARGIN_WEIGHT,
    nested_weight=NESTED_WEIGHT,
    minimum_margin=MINIMUM_MARGIN,
    anchor_strength=1e-5,
    seed=SEED,
)
clean_metrics = {
    f"rank_{rank}": evaluate_nested_head(
        head, select_states, readout_weight, readout_bias=readout_bias,
        rank=rank, batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
    )
    for rank in RANKS
}
print({"best_epoch": clean_result.best_epoch,
       "best_top1": clean_result.best_top1_agreement,
       "best_kl": clean_result.best_validation_kl})
print(json.dumps(clean_metrics, indent=2))
''')

markdown("## On-policy recovery: learn from states caused by the compressed head")
code(r'''
head.disable_adaptive()
head.set_rank(PRIMARY_RANK)
onpolicy_states, _ = collect_states(
    onpolicy_questions, head, "compressed_onpolicy", MAX_ONPOLICY_STATES
)
recovery_states = torch.cat((fit_states, onpolicy_states), dim=0)
recovery_states = deterministic_cap(
    recovery_states, MAX_FIT_STATES + MAX_ONPOLICY_STATES, SEED + 200
)
onpolicy_result = distil_nested_head(
    head,
    recovery_states,
    select_states,
    readout_weight,
    readout_bias=readout_bias,
    epochs=ONPOLICY_EPOCHS,
    batch_size=DISTILL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    temperature=TEMPERATURE,
    kl_weight=KL_WEIGHT,
    token_weight=TOKEN_WEIGHT,
    margin_weight=MARGIN_WEIGHT,
    nested_weight=NESTED_WEIGHT,
    minimum_margin=MINIMUM_MARGIN,
    anchor_strength=1e-5,
    seed=SEED + 1,
)
final_state = {name: value.detach().cpu().clone()
               for name, value in head.state_dict().items()}
final_metrics = {
    f"rank_{rank}": evaluate_nested_head(
        head, select_states, readout_weight, readout_bias=readout_bias,
        rank=rank, batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
    )
    for rank in RANKS
}
print(json.dumps(final_metrics, indent=2))
''')

markdown("## Select the adaptive 32→64 confidence threshold without test data")
code(r'''
@torch.inference_mode()
def adaptive_selection_metrics(threshold):
    agreements = 0
    fallback = 0
    total = 0
    for start in range(0, len(select_states), DISTILL_BATCH_SIZE):
        hidden = select_states[start:start + DISTILL_BATCH_SIZE].to(device)
        teacher = torch.nn.functional.linear(hidden, readout_weight, readout_bias)
        low = head.forward_rank(hidden, RANKS[0])
        high = head.forward_rank(hidden, PRIMARY_RANK)
        top_two = low.topk(2, dim=-1).values
        use_high = (top_two[:, 0] - top_two[:, 1]) < threshold
        selected = torch.where(use_high.unsqueeze(1), high, low)
        agreements += int((selected.argmax(-1) == teacher.argmax(-1)).sum())
        fallback += int(use_high.sum())
        total += len(hidden)
    fraction = fallback / max(1, total)
    return {
        "threshold": float(threshold),
        "top1_agreement": agreements / max(1, total),
        "fallback_fraction": fraction,
        "average_rank": RANKS[0] + fraction * (PRIMARY_RANK - RANKS[0]),
    }

with torch.inference_mode():
    all_low_logits = []
    for start in range(0, len(select_states), DISTILL_BATCH_SIZE):
        hidden = select_states[start:start + DISTILL_BATCH_SIZE].to(device)
        all_low_logits.append(head.forward_rank(hidden, RANKS[0]).cpu())
    all_low_logits = torch.cat(all_low_logits)
    values = all_low_logits.topk(2, dim=-1).values
    low_margins = values[:, 0] - values[:, 1]
quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
thresholds = sorted(set(float(value) for value in torch.quantile(low_margins, quantiles)))
thresholds = [-1.0, *thresholds, float("inf")]
adaptive_grid = [adaptive_selection_metrics(threshold) for threshold in thresholds]
eligible = [row for row in adaptive_grid if row["top1_agreement"] >= ADAPTIVE_MIN_TOP1]
if eligible:
    adaptive_choice = min(eligible, key=lambda row: (row["average_rank"], -row["top1_agreement"]))
else:
    adaptive_choice = max(adaptive_grid, key=lambda row: row["top1_agreement"])
print(json.dumps({"grid": adaptive_grid, "selected": adaptive_choice}, indent=2))
''')

markdown("## Locked full-test generation arms")
code(r'''
def restored_head():
    restored = NestedLowRankVocabularyHead(hidden_size, vocabulary_size, RANKS)
    restored.load_state_dict(final_state)
    return restored.to(device=device, dtype=dtype).eval()

generation_heads = {"full": full_head}
for rank in RANKS:
    candidate = restored_head()
    candidate.disable_adaptive()
    candidate.set_rank(rank)
    generation_heads[f"whitened_margin_onpolicy_r{rank}"] = candidate
adaptive_head = restored_head()
adaptive_head.configure_adaptive(
    base_rank=RANKS[0], fallback_rank=PRIMARY_RANK,
    margin_threshold=adaptive_choice["threshold"], inactive_rank=RANKS[0],
)
generation_heads[f"adaptive_r{RANKS[0]}_r{PRIMARY_RANK}"] = adaptive_head

from src.data.answer_extract import answers_match
test_questions = [str(row["question"]) for row in test_examples]
warmup_questions = select_questions[:16]
warmup_prepared = prepare_official_codi_batches(
    tokenizer, warmup_questions, batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)
test_prepared = prepare_official_codi_batches(
    tokenizer, test_questions, batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)
generation_results = {}
generation_outcomes = {}

def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

if RUN_FULL_GENERATION:
    for arm, output_head in generation_heads.items():
        base_model.set_output_embeddings(output_head)
        _ = generate_official_codi_fast(
            model, tokenizer, warmup_prepared,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS,
            device=device, answer_cue="The answer is:",
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        generation = generate_official_codi_fast(
            model, tokenizer, test_prepared,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS,
            device=device, answer_cue="The answer is:",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        outcomes = [bool(answers_match(text, row["gold"]))
                    for text, row in zip(generation.texts, test_examples)]
        records = [
            {"arm": arm, "row": index, "gold": str(row["gold"]),
             "generation": text, "correct": outcome,
             "generated_tokens": int(generation.generated_token_counts[index])}
            for index, (row, text, outcome) in enumerate(
                zip(test_examples, generation.texts, outcomes)
            )
        ]
        write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"{arm}.jsonl", records)
        tokens = int(generation.generated_token_count)
        generation_outcomes[arm] = outcomes
        generation_results[arm] = {
            "examples": len(outcomes), "correct": int(sum(outcomes)),
            "numeric_exact_match": sum(outcomes) / len(outcomes),
            "wall_clock_seconds": elapsed,
            "examples_per_second": len(outcomes) / elapsed,
            "visible_generated_tokens": tokens,
            "microseconds_per_question": 1e6 * elapsed / len(outcomes),
            "microseconds_per_visible_token": 1e6 * elapsed / max(1, tokens),
            "mean_visible_tokens": tokens / len(outcomes),
        }
        print(arm, generation_results[arm])
        if arm == "full" and len(test_examples) == 1319 and TEST_LIMIT == 0:
            assert abs(generation_results[arm]["numeric_exact_match"] - 0.433662) <= 0.015, (
                "The full head did not reproduce the locked CODI baseline."
            )
base_model.set_output_embeddings(full_head)
''')

markdown("## Component timing, paired uncertainty, and artifact export")
code(r'''
class VocabularyPrefix(nn.Module):
    def __init__(self, module, size):
        super().__init__()
        self.module = module
        self.size = int(size)
    def forward(self, hidden):
        return self.module(hidden)[..., :self.size]

head_latency_us = {
    "full": benchmark_vocabulary_head(
        VocabularyPrefix(full_head, vocabulary_size), hidden_size,
        batch_size=1, iterations=200, device=device, dtype=dtype,
    )
}
for rank in RANKS:
    candidate = restored_head()
    candidate.set_rank(rank)
    head_latency_us[f"rank_{rank}"] = benchmark_vocabulary_head(
        candidate, hidden_size, batch_size=1, iterations=200,
        device=device, dtype=dtype,
    )
adaptive_head.set_answer_position(0)
head_latency_us["adaptive"] = benchmark_vocabulary_head(
    adaptive_head, hidden_size, batch_size=1, iterations=200,
    device=device, dtype=dtype,
)

def paired_interval(left, right, seed):
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    for start in range(0, BOOTSTRAP_SAMPLES, 250):
        count = min(250, BOOTSTRAP_SAMPLES - start)
        index = torch.randint(len(delta), (count, len(delta)), generator=generator)
        draws.extend(delta[index].mean(dim=1).tolist())
    values = torch.tensor(draws)
    return [float(torch.quantile(values, 0.025)),
            float(torch.quantile(values, 0.975))]

paired_intervals = {}
if generation_outcomes:
    for arm in generation_outcomes:
        if arm != "full":
            paired_intervals[f"{arm}_minus_full"] = paired_interval(
                generation_outcomes[arm], generation_outcomes["full"],
                SEED + len(paired_intervals),
            )

operations = {
    "full_macs_per_token": hidden_size * vocabulary_size,
    **{f"rank_{rank}_macs_per_token": rank * (hidden_size + vocabulary_size)
       for rank in RANKS},
}
quality_gate = None
speed_gate = None
if generation_results:
    full_accuracy = generation_results["full"]["numeric_exact_match"]
    primary = generation_results[f"whitened_margin_onpolicy_r{PRIMARY_RANK}"]
    quality_gate = primary["numeric_exact_match"] >= 0.98 * full_accuracy
    speed_gate = primary["microseconds_per_visible_token"] < generation_results["full"][
        "microseconds_per_visible_token"
    ]

summary = {
    "experiment": "trajectory_whitened_margin_distilled_global_lm_head_v1",
    "code_commit": CODE_COMMIT,
    "checkpoint_sha256": load_report.checkpoint_sha256,
    "population": {
        "fit_questions": FIT_QUESTIONS, "select_questions": SELECT_QUESTIONS,
        "onpolicy_questions": ONPOLICY_QUESTIONS,
        "test_questions": len(test_examples), "test_limit": TEST_LIMIT,
    },
    "ranks": list(RANKS),
    "initialization": initialization_report.to_dict(),
    "initial_metrics": initial_metrics,
    "clean_training": {
        "losses": list(clean_result.losses), "best_epoch": clean_result.best_epoch,
        "validation": list(clean_result.validation), "metrics": clean_metrics,
    },
    "onpolicy_training": {
        "clean_states": len(fit_states), "onpolicy_states": len(onpolicy_states),
        "losses": list(onpolicy_result.losses), "best_epoch": onpolicy_result.best_epoch,
        "validation": list(onpolicy_result.validation), "metrics": final_metrics,
    },
    "adaptive_selection": {"grid": adaptive_grid, "selected": adaptive_choice,
                           "minimum_top1": ADAPTIVE_MIN_TOP1},
    "generation": generation_results,
    "paired_accuracy_intervals": paired_intervals,
    "head_latency_microseconds": head_latency_us,
    "operations": operations,
    "gates": {"primary_rank": PRIMARY_RANK,
              "retains_98_percent_accuracy": quality_gate,
              "improves_normalized_end_to_end_speed": speed_gate},
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
temporary = pathlib.Path(str(summary_path) + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
temporary.replace(summary_path)
torch.save({
    "contract": "trajectory_whitened_margin_distilled_global_lm_head_v1",
    "code_commit": CODE_COMMIT,
    "ranks": RANKS,
    "state_dict": final_state,
    "adaptive_choice": adaptive_choice,
    "initialization": initialization_report.to_dict(),
}, pathlib.Path(OUTPUT_ROOT) / "global_low_rank_head.pt")
print(json.dumps({"gates": summary["gates"], "latency": head_latency_us,
                  "generation": generation_results}, indent=2))
print("saved:", summary_path)
''')

markdown(r"""
## Interpretation

- Passing the rank-64 gate supports a verifier-free global low-rank replacement for
  this CODI/GSM8K deployment.
- The nested rank-32 arm measures the cheapest fixed prefix; rank 96 measures how much
  quality is recoverable before approaching the full 768-dimensional head.
- The adaptive arm is useful only if its average rank is below 64 and its complete
  autoregressive accuracy remains competitive. A validation-only token-agreement gate
  does not guarantee sequence accuracy, which is why the locked generation is required.
- These results establish CODI performance, not cross-model generalization. That claim
  requires rerunning the same fitter on at least Qwen and one unrelated model family.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
