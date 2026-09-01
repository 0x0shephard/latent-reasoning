"""Build the corrected Kaggle notebook for Qwen eigenspace generalization."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_eigenspace_readout_generalization_qwen.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Experiment 2 v2 — Does the answer eigenspace generalize beyond CODI?

## TL;DR

This replaces the invalid prompt-endpoint experiment. CODI's measured state occurs
**after reasoning and immediately before the answer**, so Qwen is now measured at the
same semantic location: the state that predicts the first token of its final answer.

The notebook deliberately separates two hypotheses:

1. **Matched endpoint locality.** Given Qwen's unchanged reasoning prefix, can a frozen
   answer-endpoint eigenspace preserve the next answer token? This is a causal one-step
   readout intervention, not a claim about full generation.
2. **Deployable full-generation compression.** A low-rank head initialized from the
   endpoint eigenspace is distilled on states sampled across the complete reasoning and
   answer trajectory, then used at every generation step. Equal-rank random and
   trajectory-eigenspace initializations are controls.

All baseline generations, alignments, outputs, token counts, EOS status, and truncation
flags are saved as JSONL. Test states and labels are never used for fitting, model
selection, or early stopping.
""")

markdown(r"""
## Context and methods

### Key assumptions

- A Qwen state immediately before the final answer is the closest observable analogue
  of CODI's post-latent, pre-answer state.
- Teacher forcing Qwen's own greedy response reproduces the same next-token state. We
  measure replay agreement and stop if this assumption fails materially.
- A frozen endpoint basis is only tested at that endpoint. Applying a head throughout
  generation requires all-position distillation.
- Speed is meaningful only together with accuracy and token counts. Isolated-head
  latency is reported separately from end-to-end latency.

### Preregistered interpretation

- The run is invalid if the full-head baseline has fewer than 50 correct test answers,
  fewer than 90% of responses can be aligned, or more than 10% are truncated.
- Endpoint locality is supported at rank 192 if the readout-aware endpoint basis has at
  least 95% full-head token agreement and exceeds the median random-basis agreement.
- Deployable generalization is supported only if the distilled endpoint-initialized
  head retains at least 95% of baseline exact match, is faster end to end, and beats
  the equally trained random initialization. Each conclusion is also reported
  separately so a failed compound gate remains interpretable.
""")

markdown("## Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "0698442"  # Immutable commit containing the JSON-safe experiment.
REPO_DIR = "/kaggle/working/latent-reasoning"
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
HF_MODEL_REVISION = "f903dd76e2a9741d582f2a31248f1f5d0ac0e2bf"
OUTPUT_ROOT = "/kaggle/working/eigenspace_generalization_qwen_v2"

SEED = 20260903
FIT_EXAMPLES = 512
SELECT_EXAMPLES = 128
TEST_EXAMPLES = 256
RANKS = [64, 96, 192]
PRIMARY_RANK = 192
RANDOM_NULL_REPLICATES = 20
MAX_NEW_TOKENS = 512
GENERATION_BATCH_SIZE = 4
TEACHER_FORCE_BATCH_SIZE = 2
MAX_TRAJECTORY_STATES_PER_EXAMPLE = 32
MAX_DISTILL_FIT_STATES = 4096
MAX_DISTILL_SELECT_STATES = 1024
DISTILL_EPOCHS = 3
DISTILL_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
TEMPERATURE = 2.0
BOOTSTRAP_SAMPLES = 5000
RUN_GLOBAL_GENERATION = True
RUN_FIXED_GLOBAL_NEGATIVE_CONTROL = True

import gc, json, os, pathlib, subprocess, sys, time
from collections import Counter

assert HF_MODEL_REVISION != RUN_COMMIT
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
RESOLVED_COMMIT = "origin/main" if RUN_COMMIT == "main" else RUN_COMMIT
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", RESOLVED_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
CODE_COMMIT = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print("commit:", CODE_COMMIT)
assert RUN_COMMIT == "main" or CODE_COMMIT.startswith(RUN_COMMIT)
''')

markdown("## Install and run the alignment and low-rank unit tests")
code(r'''
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "accelerate>=1.2,<2",
    "datasets>=3.6,<4", "huggingface_hub>=0.34,<1.0",
], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_eigenspace_readout.py", "tests/test_qwen_trajectory.py",
], check=True)
''')

markdown("## Freeze disjoint fit, selection, and test rows")
code(r'''
import torch
from datasets import load_dataset
from src.data.answer_extract import answers_match, normalize_gold

TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
)
TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
train_raw = load_dataset("json", data_files={"train": TRAIN_URL}, split="train")
test_raw = load_dataset("json", data_files={"test": TEST_URL}, split="test")

def normalize_rows(dataset):
    return [{
        "question": str(row["question"]),
        "gold": normalize_gold(row["answer"], "gsm8k_main"),
    } for row in dataset]

train_rows = normalize_rows(train_raw)
test_rows_all = normalize_rows(test_raw)
train_order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(SEED))
fit_indices = train_order[:FIT_EXAMPLES].tolist()
select_indices = train_order[FIT_EXAMPLES:FIT_EXAMPLES + SELECT_EXAMPLES].tolist()
test_order = torch.randperm(
    len(test_rows_all), generator=torch.Generator().manual_seed(SEED + 1)
)
test_indices = test_order[:TEST_EXAMPLES].tolist()
fit_rows = [train_rows[index] for index in fit_indices]
select_rows = [train_rows[index] for index in select_indices]
test_rows = [test_rows_all[index] for index in test_indices]
assert set(fit_indices).isdisjoint(select_indices)
print({"fit": len(fit_rows), "select": len(select_rows), "test": len(test_rows)})
''')

markdown("## Load Qwen and use its official reasoning prompt")
code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")
generation_dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=HF_MODEL_REVISION, token=os.environ.get("HF_TOKEN") or None,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=HF_MODEL_REVISION, torch_dtype=generation_dtype,
    low_cpu_mem_usage=True, token=os.environ.get("HF_TOKEN") or None,
).to(device).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
full_head = model.get_output_embeddings()
hidden_size = int(model.config.hidden_size)
vocabulary_size = int(full_head.weight.shape[0])
assert hidden_size == 1536

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def formatted_prompts(rows):
    conversations = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
    ] for row in rows]
    return [tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    ) for conversation in conversations]

print({"hidden": hidden_size, "vocabulary": vocabulary_size,
       "head": type(full_head).__name__, "dtype": str(generation_dtype)})
''')

markdown("## Generate the untouched baseline and save every response")
code(r'''
def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)

def trim_generated_ids(token_ids):
    result = [int(value) for value in token_ids]
    if tokenizer.eos_token_id in result:
        stop = result.index(tokenizer.eos_token_id) + 1
        return result[:stop], True
    while result and result[-1] == tokenizer.pad_token_id:
        result.pop()
    return result, False

@torch.inference_mode()
def generate_records(rows, split_name):
    records = []
    prompts = formatted_prompts(rows)
    for start in range(0, len(rows), GENERATION_BATCH_SIZE):
        batch_rows = rows[start:start + GENERATION_BATCH_SIZE]
        batch_prompts = prompts[start:start + GENERATION_BATCH_SIZE]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        output = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[:, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
        for offset, (row, prompt, token_ids) in enumerate(zip(batch_rows, batch_prompts, generated)):
            token_ids, ended_with_eos = trim_generated_ids(token_ids)
            semantic_ids = token_ids[:-1] if ended_with_eos else token_ids
            text = tokenizer.decode(
                semantic_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            records.append({
                "split": split_name, "row": start + offset,
                "question": row["question"],
                "gold": str(row["gold"]) if row["gold"] is not None else None,
                "prompt": prompt, "generation": text,
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "ended_with_eos": ended_with_eos,
                "truncated": not ended_with_eos and len(token_ids) >= MAX_NEW_TOKENS,
                "correct": bool(answers_match(text, row["gold"])),
            })
    write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"{split_name}.jsonl", records)
    return records

baseline_records = {}
baseline_elapsed = {}
for split_name, rows in (("fit", fit_rows), ("select", select_rows), ("test", test_rows)):
    torch.cuda.synchronize()
    started = time.perf_counter()
    records = generate_records(rows, f"baseline_{split_name}")
    torch.cuda.synchronize()
    baseline_elapsed[split_name] = time.perf_counter() - started
    baseline_records[split_name] = records
    token_count = sum(record["generated_tokens"] for record in records)
    print(split_name, {
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "correct": sum(record["correct"] for record in records),
        "mean_tokens": sum(record["generated_tokens"] for record in records) / len(records),
        "truncated": sum(record["truncated"] for record in records),
        "wall_clock_seconds": baseline_elapsed[split_name],
        "tokens_per_second": token_count / baseline_elapsed[split_name],
    })
''')

markdown(r"""
## Align each response to the state immediately before its final answer

We replay the model's own response with teacher forcing. For generated token (y_j),
the preceding normalized hidden state (h_{j-1}) is exactly the input to the LM head
that predicts (y_j). The endpoint is the first token overlapping the last balanced
`\boxed{...}`; explicit final-answer markers and the last number are audited fallbacks.
We also sample states evenly across each full response for the separate distillation
experiment.
""")
code(r'''
from src.mech.qwen_trajectory import (
    evenly_spaced_indices, final_answer_span, token_indices_overlapping_span,
)

def canonical_response(record):
    text = record["generation"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    original = list(record["generated_token_ids"])
    if record["ended_with_eos"]:
        original = original[:-1]
    exact_retokenization = response_ids == original
    span = final_answer_span(text)
    answer_indices = [] if span is None else token_indices_overlapping_span(offsets, span[:2])
    return response_ids, answer_indices, span, exact_retokenization

@torch.inference_mode()
def collect_replayed_states(records):
    endpoint_states, endpoint_targets, endpoint_correct, endpoint_rows = [], [], [], []
    trajectory_states = []
    audit = []
    for start in range(0, len(records), TEACHER_FORCE_BATCH_SIZE):
        batch = records[start:start + TEACHER_FORCE_BATCH_SIZE]
        examples = []
        for local_index, record in enumerate(batch):
            response_ids, answer_indices, span, exact = canonical_response(record)
            prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
            if not response_ids:
                audit.append({"row": record["row"], "aligned": False,
                              "reason": "empty_response", "exact_retokenization": exact})
                continue
            examples.append({
                "local_index": local_index, "record": record,
                "prompt_ids": [int(value) for value in prompt_ids],
                "response_ids": response_ids, "answer_indices": answer_indices,
                "span": span, "exact": exact,
            })
        if not examples:
            continue
        sequences = [item["prompt_ids"] + item["response_ids"] for item in examples]
        padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(device)
        hidden = model.model(
            **padded, use_cache=False, return_dict=True
        ).last_hidden_state.detach().cpu().float()
        width = hidden.shape[1]
        for item_index, item in enumerate(examples):
            prompt_length = len(item["prompt_ids"])
            response_length = len(item["response_ids"])
            padding = width - prompt_length - response_length
            predictor_positions = torch.arange(response_length) + padding + prompt_length - 1
            response_states = hidden[item_index, predictor_positions]
            sampled = evenly_spaced_indices(response_length, MAX_TRAJECTORY_STATES_PER_EXAMPLE)
            trajectory_states.append(response_states[sampled])
            aligned = bool(item["answer_indices"])
            audit.append({
                "row": item["record"]["row"], "aligned": aligned,
                "rule": item["span"][2] if item["span"] else None,
                "answer_token_indices": item["answer_indices"],
                "exact_retokenization": item["exact"],
            })
            if aligned:
                answer_index = item["answer_indices"][0]
                endpoint_states.append(response_states[answer_index])
                endpoint_targets.append(item["response_ids"][answer_index])
                endpoint_correct.append(item["record"]["correct"])
                endpoint_rows.append(item["record"]["row"])
    return {
        "endpoint_states": torch.stack(endpoint_states),
        "endpoint_targets": torch.tensor(endpoint_targets, dtype=torch.long),
        "endpoint_baseline_correct": torch.tensor(endpoint_correct, dtype=torch.bool),
        "endpoint_rows": endpoint_rows,
        "trajectory_states": torch.cat(trajectory_states),
        "audit": audit,
    }

replayed = {
    split: collect_replayed_states(baseline_records[split])
    for split in ("fit", "select", "test")
}
for split, payload in replayed.items():
    write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"alignment_{split}.jsonl", payload["audit"])
    print(split, {
        "endpoint_states": list(payload["endpoint_states"].shape),
        "trajectory_states": list(payload["trajectory_states"].shape),
        "alignment_rate": len(payload["endpoint_rows"]) / len(baseline_records[split]),
        "rules": dict(Counter(row.get("rule") for row in payload["audit"])),
        "exact_retokenization": sum(row["exact_retokenization"] for row in payload["audit"])
                                / len(payload["audit"]),
    })
''')

markdown("## Verify replay and fit only on matched fit endpoints")
code(r'''
from src.mech.eigenspace_readout import (
    LowRankVocabularyHead, benchmark_vocabulary_head, covariance_eigensystem,
    distil_low_rank_head, evaluate_head_fidelity, orthonormal_random_basis,
    select_readout_aware_basis,
)

weight_half = full_head.weight.detach()

@torch.inference_mode()
def replay_metrics(payload):
    states = payload["endpoint_states"].to(device=device, dtype=generation_dtype)
    targets = payload["endpoint_targets"].to(device)
    predictions = []
    for start in range(0, len(states), 16):
        predictions.append(full_head(states[start:start + 16]).argmax(-1).cpu())
    predictions = torch.cat(predictions)
    return {
        "examples": len(targets),
        "greedy_token_replay_agreement": float((predictions == targets.cpu()).float().mean()),
    }

replay_checks = {split: replay_metrics(payload) for split, payload in replayed.items()}
print(json.dumps(replay_checks, indent=2))

endpoint_centre, endpoint_eigenvalues, endpoint_eigenvectors = covariance_eigensystem(
    replayed["fit"]["endpoint_states"]
)
endpoint_bases, endpoint_selection = {}, {}
for rank in RANKS:
    endpoint_bases[("leading", rank)] = endpoint_eigenvectors[:, :rank]
    aware, indices, scores = select_readout_aware_basis(
        weight_half, endpoint_eigenvalues, endpoint_eigenvectors, rank, chunk_size=32
    )
    endpoint_bases[("readout_aware", rank)] = aware
    endpoint_bases[("random", rank)] = orthonormal_random_basis(
        hidden_size, rank, seed=SEED + 1000 + rank
    )
    endpoint_selection[str(rank)] = {
        "readout_aware_indices": indices.tolist(),
        "readout_aware_score_fraction": float(scores[indices].sum() / scores.sum()),
        "leading_variance_fraction": float(
            endpoint_eigenvalues[:rank].sum() / endpoint_eigenvalues.sum()
        ),
    }
print(json.dumps(endpoint_selection, indent=2))
''')

markdown("## Experiment A — one-step intervention at the matched answer endpoint")
code(r'''
def fixed_endpoint_head(basis):
    return LowRankVocabularyHead.from_basis(
        weight_half, basis, endpoint_centre
    ).to(device=device, dtype=generation_dtype)

@torch.inference_mode()
def matched_endpoint_metrics(head, payload):
    metrics = evaluate_head_fidelity(
        head, payload["endpoint_states"], weight_half,
        batch_size=8, temperature=TEMPERATURE,
    )
    student_predictions, full_predictions = [], []
    for start in range(0, len(payload["endpoint_states"]), 8):
        states = payload["endpoint_states"][start:start + 8].to(
            device=device, dtype=generation_dtype
        )
        student_predictions.append(head(states).argmax(-1).cpu())
        full_predictions.append(full_head(states).argmax(-1).cpu())
    student_predictions = torch.cat(student_predictions)
    full_predictions = torch.cat(full_predictions)
    targets = payload["endpoint_targets"]
    correct_mask = payload["endpoint_baseline_correct"]
    metrics.update({
        "student_greedy_token_replay_agreement": float(
            (student_predictions == targets).float().mean()
        ),
        "full_greedy_token_replay_agreement": float(
            (full_predictions == targets).float().mean()
        ),
        "top1_agreement_when_baseline_correct": float(
            (student_predictions[correct_mask] == full_predictions[correct_mask]).float().mean()
        ) if bool(correct_mask.any()) else None,
        "baseline_correct_examples": int(correct_mask.sum()),
    })
    return metrics

endpoint_select_metrics = {}
for (name, rank), basis in endpoint_bases.items():
    head = fixed_endpoint_head(basis)
    endpoint_select_metrics[f"{name}_r{rank}"] = matched_endpoint_metrics(
        head, replayed["select"],
    )
    del head

random_null_select = []
for replicate in range(RANDOM_NULL_REPLICATES):
    basis = orthonormal_random_basis(hidden_size, PRIMARY_RANK, seed=SEED + 2000 + replicate)
    head = fixed_endpoint_head(basis)
    random_null_select.append(evaluate_head_fidelity(
        head, replayed["select"]["endpoint_states"], weight_half,
        batch_size=8, temperature=TEMPERATURE,
    )["top1_agreement"])
    del head

endpoint_test_metrics = {}
for name in ("leading", "readout_aware", "random"):
    head = fixed_endpoint_head(endpoint_bases[(name, PRIMARY_RANK)])
    endpoint_test_metrics[f"{name}_r{PRIMARY_RANK}"] = matched_endpoint_metrics(
        head, replayed["test"],
    )
    del head

random_null_test = []
for replicate in range(RANDOM_NULL_REPLICATES):
    basis = orthonormal_random_basis(hidden_size, PRIMARY_RANK, seed=SEED + 2000 + replicate)
    head = fixed_endpoint_head(basis)
    random_null_test.append(evaluate_head_fidelity(
        head, replayed["test"]["endpoint_states"], weight_half,
        batch_size=8, temperature=TEMPERATURE,
    )["top1_agreement"])
    del head
print(json.dumps({
    "selection": endpoint_select_metrics,
    "selection_random_null_median": float(torch.tensor(random_null_select).median()),
    "test_random_null_median": float(torch.tensor(random_null_test).median()),
    "test": endpoint_test_metrics,
}, indent=2))
''')

markdown(r"""
## Experiment B — distil on the complete generation distribution

The endpoint basis now serves only as an initialization. Training states are sampled
evenly from Qwen's reasoning and answer tokens. The validation trajectories choose the
best epoch. The frozen test trajectories are never used for training or early stopping.
""")
code(r'''
def deterministic_subsample(states, maximum, seed):
    if len(states) <= maximum:
        return states
    order = torch.randperm(len(states), generator=torch.Generator().manual_seed(seed))
    return states[order[:maximum]]

distill_fit_states = deterministic_subsample(
    replayed["fit"]["trajectory_states"], MAX_DISTILL_FIT_STATES, SEED + 10
)
distill_select_states = deterministic_subsample(
    replayed["select"]["trajectory_states"], MAX_DISTILL_SELECT_STATES, SEED + 11
)
trajectory_centre, trajectory_values, trajectory_vectors = covariance_eigensystem(
    distill_fit_states
)

initializers = {
    f"distilled_endpoint_r{PRIMARY_RANK}": (
        endpoint_bases[("readout_aware", PRIMARY_RANK)], endpoint_centre
    ),
    f"distilled_trajectory_r{PRIMARY_RANK}": (
        trajectory_vectors[:, :PRIMARY_RANK], trajectory_centre
    ),
    f"distilled_random_r{PRIMARY_RANK}": (
        orthonormal_random_basis(hidden_size, PRIMARY_RANK, seed=SEED + 3000),
        endpoint_centre,
    ),
}

# Float32 optimization avoids unstable fp16 Adam states. Only one trainable head is
# resident at a time; the frozen transformer remains in fp16.
weight_train = full_head.weight.detach().float()
distilled_states = {}
distillation_runs = {}
for name, (basis, centre) in initializers.items():
    head = LowRankVocabularyHead.from_basis(weight_train, basis, centre).to(device=device)
    initial = evaluate_head_fidelity(
        head, distill_select_states, weight_train,
        batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
    )
    result = distil_low_rank_head(
        head, distill_fit_states, distill_select_states, weight_train,
        epochs=DISTILL_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
        anchor_strength=1e-4, seed=SEED,
    )
    final = evaluate_head_fidelity(
        head, distill_select_states, weight_train,
        batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
    )
    distilled_states[name] = {
        key: value.detach().cpu().clone() for key, value in head.state_dict().items()
    }
    distillation_runs[name] = {
        "initial_validation": initial, "final_validation": final,
        "losses": list(result.losses),
        "best_validation_kl": result.best_validation_kl,
        "best_epoch": result.best_epoch,
    }
    print(name, distillation_runs[name])
    del head
    gc.collect()
    torch.cuda.empty_cache()
del weight_train
gc.collect()
torch.cuda.empty_cache()
''')

markdown("## Frozen all-position fidelity test")
code(r'''
def load_distilled_head(name):
    basis, centre = initializers[name]
    head = LowRankVocabularyHead.from_basis(weight_half, basis, centre)
    head.load_state_dict(distilled_states[name])
    return head.to(device=device, dtype=generation_dtype).eval()

trajectory_test_states = deterministic_subsample(
    replayed["test"]["trajectory_states"], MAX_DISTILL_SELECT_STATES, SEED + 12
)
trajectory_test_metrics = {}
for name in initializers:
    head = load_distilled_head(name)
    trajectory_test_metrics[name] = evaluate_head_fidelity(
        head, trajectory_test_states, weight_half,
        batch_size=8, temperature=TEMPERATURE,
    )
    del head
print(json.dumps(trajectory_test_metrics, indent=2))
''')

markdown("## Full-generation test with complete audit records")
code(r'''
baseline_test_metrics = {
    "examples": len(baseline_records["test"]),
    "correct": sum(row["correct"] for row in baseline_records["test"]),
    "numeric_exact_match": sum(row["correct"] for row in baseline_records["test"])
                           / len(baseline_records["test"]),
    "generated_tokens": sum(row["generated_tokens"] for row in baseline_records["test"]),
    "mean_generated_tokens": sum(row["generated_tokens"] for row in baseline_records["test"])
                             / len(baseline_records["test"]),
    "ended_with_eos": sum(row["ended_with_eos"] for row in baseline_records["test"]),
    "truncated": sum(row["truncated"] for row in baseline_records["test"]),
    "wall_clock_seconds": baseline_elapsed["test"],
}
baseline_test_metrics["tokens_per_second"] = (
    baseline_test_metrics["generated_tokens"] / baseline_test_metrics["wall_clock_seconds"]
)
baseline_test_metrics["milliseconds_per_token"] = (
    1000 * baseline_test_metrics["wall_clock_seconds"]
    / max(1, baseline_test_metrics["generated_tokens"])
)

generation_results = {"full": baseline_test_metrics}
generation_outcomes = {"full": [row["correct"] for row in baseline_records["test"]]}
generation_heads = {name: load_distilled_head(name) for name in initializers}
if RUN_FIXED_GLOBAL_NEGATIVE_CONTROL:
    generation_heads[f"fixed_endpoint_r{PRIMARY_RANK}"] = fixed_endpoint_head(
        endpoint_bases[("readout_aware", PRIMARY_RANK)]
    )

if RUN_GLOBAL_GENERATION:
    for name, head in generation_heads.items():
        model.set_output_embeddings(head)
        torch.cuda.synchronize()
        started = time.perf_counter()
        records = generate_records(test_rows, name)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        correct = [row["correct"] for row in records]
        tokens = sum(row["generated_tokens"] for row in records)
        generation_outcomes[name] = correct
        generation_results[name] = {
            "examples": len(records), "correct": int(sum(correct)),
            "numeric_exact_match": float(sum(correct) / len(records)),
            "generated_tokens": tokens,
            "mean_generated_tokens": tokens / len(records),
            "ended_with_eos": sum(row["ended_with_eos"] for row in records),
            "truncated": sum(row["truncated"] for row in records),
            "wall_clock_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
            "milliseconds_per_token": 1000 * elapsed / max(1, tokens),
        }
        print(name, generation_results[name])
model.set_output_embeddings(full_head)
del generation_heads
gc.collect()
torch.cuda.empty_cache()
''')

markdown("## Timing, uncertainty, validity gates, and conclusions")
code(r'''
timing = {
    "full": benchmark_vocabulary_head(
        full_head, hidden_size, batch_size=1, iterations=200,
        device=device, dtype=generation_dtype,
    )
}
for name in initializers:
    head = load_distilled_head(name)
    timing[name] = benchmark_vocabulary_head(
        head, hidden_size, batch_size=1, iterations=200,
        device=device, dtype=generation_dtype,
    )
    del head

operations = {
    "full_macs_per_token": int(hidden_size * vocabulary_size),
    f"rank{PRIMARY_RANK}_macs_per_token": int(
        PRIMARY_RANK * (hidden_size + vocabulary_size)
    ),
}

def paired_interval(left, right, seed):
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    draws = []
    for start in range(0, BOOTSTRAP_SAMPLES, 250):
        count = min(250, BOOTSTRAP_SAMPLES - start)
        indices = torch.randint(len(delta), (count, len(delta)), generator=generator)
        draws.extend(delta[indices].mean(dim=1).tolist())
    values = torch.tensor(draws)
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]

paired_intervals = {}
endpoint_name = f"distilled_endpoint_r{PRIMARY_RANK}"
random_name = f"distilled_random_r{PRIMARY_RANK}"
if endpoint_name in generation_outcomes:
    paired_intervals["distilled_endpoint_minus_full"] = paired_interval(
        generation_outcomes[endpoint_name], generation_outcomes["full"], SEED + 20
    )
    paired_intervals["distilled_endpoint_minus_distilled_random"] = paired_interval(
        generation_outcomes[endpoint_name], generation_outcomes[random_name], SEED + 21
    )

baseline_correct = baseline_test_metrics["correct"]
alignment_rate = len(replayed["test"]["endpoint_rows"]) / len(test_rows)
preferred_marker_rate = sum(
    row.get("rule") in ("boxed", "final_marker") for row in replayed["test"]["audit"]
) / len(test_rows)
truncation_rate = baseline_test_metrics["truncated"] / len(test_rows)
replay_agreement = replay_checks["test"]["greedy_token_replay_agreement"]
validity = {
    "baseline_correct_at_least_50": baseline_correct >= 50,
    "alignment_rate_at_least_90pct": alignment_rate >= 0.90,
    "boxed_or_final_marker_rate_at_least_80pct": preferred_marker_rate >= 0.80,
    "truncation_rate_at_most_10pct": truncation_rate <= 0.10,
    "teacher_forced_replay_at_least_95pct": replay_agreement >= 0.95,
}
valid_run = all(validity.values())

aware_endpoint = endpoint_test_metrics[f"readout_aware_r{PRIMARY_RANK}"]["top1_agreement"]
random_null_median = float(torch.tensor(random_null_test).median())
endpoint_conclusion = {
    "top1_agreement": aware_endpoint,
    "random_null_median": random_null_median,
    "supported": bool(valid_run and aware_endpoint >= 0.95
                      and aware_endpoint > random_null_median),
}

deployment_conclusion = {"status": "not_run"}
if endpoint_name in generation_results:
    baseline_accuracy = baseline_test_metrics["numeric_exact_match"]
    endpoint_accuracy = generation_results[endpoint_name]["numeric_exact_match"]
    random_accuracy = generation_results[random_name]["numeric_exact_match"]
    retained = endpoint_accuracy / baseline_accuracy if baseline_accuracy else None
    deployment_conclusion = {
        "retained_accuracy_fraction": retained,
        "endpoint_minus_random_accuracy_points": 100 * (endpoint_accuracy - random_accuracy),
        "isolated_head_speedup": timing["full"] / timing[endpoint_name],
        "end_to_end_token_throughput_speedup": (
            baseline_test_metrics["milliseconds_per_token"]
            / generation_results[endpoint_name]["milliseconds_per_token"]
        ),
        "accuracy_supported": bool(valid_run and retained is not None and retained >= 0.95),
        "initialization_advantage_supported": bool(
            valid_run and endpoint_accuracy > random_accuracy
            and paired_intervals["distilled_endpoint_minus_distilled_random"][0] > 0
        ),
    }
    deployment_conclusion["status"] = (
        "deployable_generalization_supported" if
        deployment_conclusion["accuracy_supported"]
        and deployment_conclusion["initialization_advantage_supported"]
        and deployment_conclusion["end_to_end_token_throughput_speedup"] > 1.0
        else "deployable_generalization_not_supported_by_gate"
    )
if not valid_run:
    deployment_conclusion["status"] = "invalid_baseline_or_alignment"

summary = {
    "experiment": "eigenspace_generalization_qwen_v2_matched_endpoint",
    "code_commit": CODE_COMMIT, "model": MODEL_ID,
    "model_revision": HF_MODEL_REVISION, "seed": SEED,
    "splits": {"fit": len(fit_rows), "select": len(select_rows), "test": len(test_rows)},
    "baseline": {
        split: {
            "accuracy": sum(row["correct"] for row in records) / len(records),
            "correct": sum(row["correct"] for row in records),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in records) / len(records),
            "truncated": sum(row["truncated"] for row in records),
        } for split, records in baseline_records.items()
    },
    "replay_checks": replay_checks,
    "alignment_rate": {
        split: len(payload["endpoint_rows"]) / len(baseline_records[split])
        for split, payload in replayed.items()
    },
    "endpoint_selection": endpoint_selection,
    "endpoint_select_metrics": endpoint_select_metrics,
    "endpoint_random_null_select": random_null_select,
    "endpoint_random_null_test": random_null_test,
    "endpoint_test_metrics": endpoint_test_metrics,
    "distillation": distillation_runs,
    "trajectory_test_metrics": trajectory_test_metrics,
    "generation_test": generation_results,
    "paired_bootstrap_95ci": paired_intervals,
    "head_latency_us": timing, "operation_counts": operations,
    "validity": validity, "endpoint_conclusion": endpoint_conclusion,
    "deployment_conclusion": deployment_conclusion,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
for name, state_dict in distilled_states.items():
    torch.save({
        "name": name, "rank": PRIMARY_RANK, "state_dict": state_dict,
        "model": MODEL_ID, "model_revision": HF_MODEL_REVISION,
        "code_commit": CODE_COMMIT, "seed": SEED,
    }, pathlib.Path(OUTPUT_ROOT) / f"{name}.pt")
print(json.dumps({
    "validity": validity,
    "endpoint_conclusion": endpoint_conclusion,
    "deployment_conclusion": deployment_conclusion,
    "operation_counts": operations,
    "head_latency_us": timing,
}, indent=2))
print("wrote", summary_path)
''')

markdown(r"""
## Takeaways

Read `summary.json` only after checking all validity gates. The matched endpoint result
answers whether a compact answer-state subspace exists. The distilled generation result
answers whether that subspace is a useful initialization for a deployable low-rank LM
head. They are intentionally different claims.

A fixed endpoint head failing during full generation is an expected negative control,
not evidence against endpoint locality. Likewise, a distilled low-rank head succeeding
while matching the random initialization supports generic low-rank compressibility but
not a special advantage from the answer eigenspace.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
