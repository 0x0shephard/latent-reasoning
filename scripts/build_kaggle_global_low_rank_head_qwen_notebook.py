"""Build the Qwen portability notebook for the global low-rank head."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_global_low_rank_lm_head_qwen.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Does the trajectory-whitened global head transfer to Qwen?

This is the cross-model companion to the locked CODI experiment. It deliberately
uses exactly the same model-independent fitter:

- final-normalized states sampled across the entire generated trajectory;
- activation-whitened randomized factorization of the original output matrix;
- full-vocabulary KL, teacher-token, and ranking-margin distillation;
- nested ranks 64, 128, and 192 (`d/24`, `d/12`, and `d/8` for Qwen);
- one disjoint compressed-policy recovery round;
- a validation-selected adaptive 64→192 arm.

Only the state collector and generator are Qwen-specific. CODI's numerical basis is
not loaded or transferred. A positive result supports transfer of the **method**, not
transfer of the same directions.
""")

markdown("## Configuration")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "99a5814"  # Immutable global-head implementation commit.
REPO_DIR = "/kaggle/working/latent-reasoning"
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_REVISION = "f903dd76e2a9741d582f2a31248f1f5d0ac0e2bf"
OUTPUT_ROOT = "/kaggle/working/qwen_trajectory_whitened_global_head"

SEED = 20260905
FIT_EXAMPLES = 384
SELECT_EXAMPLES = 96
ONPOLICY_EXAMPLES = 96
TEST_EXAMPLES = 256
MAX_NEW_TOKENS = 384
GENERATION_BATCH_SIZE = 4
REPLAY_BATCH_SIZE = 2
STATES_PER_RESPONSE = 24
MAX_FIT_STATES = 4096
MAX_SELECT_STATES = 1024
MAX_ONPOLICY_STATES = 2048
RANKS = (64, 128, 192)
PRIMARY_RANK = 192
CLEAN_EPOCHS = 3
ONPOLICY_EPOCHS = 2
DISTILL_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
TEMPERATURE = 2.0
ADAPTIVE_MIN_TOP1 = 0.98
BOOTSTRAP_SAMPLES = 5000
RUN_FULL_GENERATION = True
RUN_ALL_FIXED_RANKS = False  # True adds rank-64 and rank-128 full generations.

import copy, gc, json, os, pathlib, subprocess, sys, time
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

markdown("## Install dependencies and test the shared fitter")
code(r'''
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "accelerate>=1.2,<2", "datasets==3.6.0",
    "huggingface_hub>=0.34,<1.0",
], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q", "tests/test_global_low_rank_head.py",
], check=True)
''')

markdown("## Freeze disjoint GSM8K populations")
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

def normalized(dataset):
    return [{"question": str(row["question"]),
             "gold": str(normalize_gold(row["answer"], "gsm8k_main"))}
            for row in dataset]

train_rows = normalized(train_raw)
test_rows_all = normalized(test_raw)
order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(SEED))
fit_indices = order[:FIT_EXAMPLES].tolist()
select_indices = order[FIT_EXAMPLES:FIT_EXAMPLES + SELECT_EXAMPLES].tolist()
onpolicy_indices = order[
    FIT_EXAMPLES + SELECT_EXAMPLES:
    FIT_EXAMPLES + SELECT_EXAMPLES + ONPOLICY_EXAMPLES
].tolist()
assert set(fit_indices).isdisjoint(select_indices)
assert set(fit_indices).isdisjoint(onpolicy_indices)
assert set(select_indices).isdisjoint(onpolicy_indices)
test_order = torch.randperm(
    len(test_rows_all), generator=torch.Generator().manual_seed(SEED + 1)
)[:TEST_EXAMPLES].tolist()
fit_rows = [train_rows[index] for index in fit_indices]
select_rows = [train_rows[index] for index in select_indices]
onpolicy_rows = [train_rows[index] for index in onpolicy_indices]
test_rows = [test_rows_all[index] for index in test_order]
print({"fit": len(fit_rows), "select": len(select_rows),
       "onpolicy": len(onpolicy_rows), "test": len(test_rows)})
''')

markdown("## Load the pinned Qwen model")
code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.mech.global_low_rank_head import (
    NestedLowRankVocabularyHead, activation_whitened_factors,
    distil_nested_head, evaluate_nested_head,
)
from src.mech.eigenspace_readout import benchmark_vocabulary_head

device = torch.device("cuda")
generation_dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN") or None,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=generation_dtype,
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

print({"model": MODEL_ID, "revision": MODEL_REVISION,
       "hidden": hidden_size, "vocabulary": vocabulary_size})
''')

markdown("## Auditable greedy generation")
code(r'''
def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

def trim_generated(token_ids):
    values = [int(value) for value in token_ids]
    if tokenizer.eos_token_id in values:
        stop = values.index(tokenizer.eos_token_id) + 1
        return values[:stop], True
    while values and values[-1] == tokenizer.pad_token_id:
        values.pop()
    return values, False

@torch.inference_mode()
def generate_records(rows, name, output_head):
    model.set_output_embeddings(output_head)
    records = []
    prompts = formatted_prompts(rows)
    for start in range(0, len(rows), GENERATION_BATCH_SIZE):
        batch_rows = rows[start:start + GENERATION_BATCH_SIZE]
        batch_prompts = prompts[start:start + GENERATION_BATCH_SIZE]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        generated = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )[:, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
        for offset, (row, prompt, token_ids) in enumerate(
            zip(batch_rows, batch_prompts, generated)
        ):
            token_ids, ended = trim_generated(token_ids)
            semantic = token_ids[:-1] if ended else token_ids
            text = tokenizer.decode(
                semantic, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            records.append({
                "arm": name, "row": start + offset, "question": row["question"],
                "gold": str(row["gold"]), "prompt": prompt, "generation": text,
                "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                "ended_with_eos": ended,
                "truncated": (not ended and len(token_ids) >= MAX_NEW_TOKENS),
                "correct": bool(answers_match(text, row["gold"])),
            })
    write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"{name}.jsonl", records)
    return records

baseline_records = {}
baseline_elapsed = {}
for split, rows in (("fit", fit_rows), ("select", select_rows), ("test", test_rows)):
    torch.cuda.synchronize()
    started = time.perf_counter()
    baseline_records[split] = generate_records(rows, f"full_{split}", full_head)
    torch.cuda.synchronize()
    baseline_elapsed[split] = time.perf_counter() - started
    print(split, {"accuracy": sum(row["correct"] for row in baseline_records[split]) / len(rows),
                  "seconds": baseline_elapsed[split],
                  "truncated": sum(row["truncated"] for row in baseline_records[split])})
model.set_output_embeddings(full_head)
''')

markdown("## Replay generated trajectories and collect predictor states")
code(r'''
def deterministic_cap(states, maximum, seed):
    if len(states) <= maximum:
        return states
    generator = torch.Generator().manual_seed(int(seed))
    return states[torch.randperm(len(states), generator=generator)[:maximum]]

def sampled_indices(length, maximum):
    if length <= maximum:
        return torch.arange(length)
    return torch.linspace(0, length - 1, steps=maximum).round().long().unique()

@torch.inference_mode()
def replay_states(records, maximum, seed):
    blocks = []
    replay_agree = 0
    replay_total = 0
    for start in range(0, len(records), REPLAY_BATCH_SIZE):
        batch = records[start:start + REPLAY_BATCH_SIZE]
        examples = []
        for record in batch:
            response = list(record["generated_token_ids"])
            if not response:
                continue
            prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
            examples.append((prompt_ids, response))
        if not examples:
            continue
        sequences = [prompt + response for prompt, response in examples]
        padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(device)
        hidden = model.model(**padded, use_cache=False, return_dict=True).last_hidden_state
        width = hidden.shape[1]
        for row, (prompt, response) in enumerate(examples):
            padding = width - len(prompt) - len(response)
            predictors = torch.arange(len(response), device=device) + padding + len(prompt) - 1
            states = hidden[row, predictors]
            targets = torch.tensor(response, device=device)
            predictions = full_head(states).argmax(-1)
            replay_agree += int((predictions == targets).sum())
            replay_total += len(response)
            sample = sampled_indices(len(states), STATES_PER_RESPONSE).to(device)
            blocks.append(states[sample].cpu().float())
    assert blocks, "trajectory replay produced no states"
    states = deterministic_cap(torch.cat(blocks), maximum, seed)
    return states, replay_agree / max(1, replay_total)

fit_states, fit_replay = replay_states(baseline_records["fit"], MAX_FIT_STATES, SEED + 10)
select_states, select_replay = replay_states(
    baseline_records["select"], MAX_SELECT_STATES, SEED + 11
)
assert fit_replay >= 0.99 and select_replay >= 0.99, (
    "Teacher-forced replay does not reproduce the generated-token states."
)
print({"fit_states": list(fit_states.shape), "select_states": list(select_states.shape),
       "fit_replay": fit_replay, "select_replay": select_replay})
''')

markdown("## Fit the identical activation-whitened nested head")
code(r'''
weight_train = full_head.weight.detach().float()
bias_train = None if getattr(full_head, "bias", None) is None else full_head.bias.detach().float()
centre, down_weight, up_weight, output_bias, initialization_report = (
    activation_whitened_factors(
        fit_states, weight_train, rank=max(RANKS), readout_bias=bias_train,
        ridge_relative=1e-4, oversample=16, power_iterations=1, seed=SEED,
        compute_device=device, compute_dtype=torch.float32,
    )
)
head = NestedLowRankVocabularyHead.from_whitened_factors(
    centre, down_weight, up_weight, output_bias, RANKS
).to(device=device, dtype=torch.float32)
initial_metrics = {f"rank_{rank}": evaluate_nested_head(
    head, select_states, weight_train, readout_bias=bias_train, rank=rank,
    batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
) for rank in RANKS}
print(json.dumps(initial_metrics, indent=2))

clean_result = distil_nested_head(
    head, fit_states, select_states, weight_train, readout_bias=bias_train,
    epochs=CLEAN_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
    learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
    kl_weight=1.0, token_weight=0.25, margin_weight=0.25,
    nested_weight=0.5, minimum_margin=0.25, anchor_strength=1e-5, seed=SEED,
)
clean_metrics = {f"rank_{rank}": evaluate_nested_head(
    head, select_states, weight_train, readout_bias=bias_train, rank=rank,
    batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
) for rank in RANKS}
print(json.dumps(clean_metrics, indent=2))
''')

markdown("## Qwen on-policy recovery")
code(r'''
def generation_head(state_dict, rank):
    result = NestedLowRankVocabularyHead(hidden_size, vocabulary_size, RANKS)
    result.load_state_dict(state_dict)
    result.set_rank(rank)
    return result.to(device=device, dtype=generation_dtype).eval()

clean_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
compressed_records = generate_records(
    onpolicy_rows, "compressed_onpolicy", generation_head(clean_state, PRIMARY_RANK)
)
model.set_output_embeddings(full_head)
onpolicy_states, onpolicy_replay = replay_states(
    compressed_records, MAX_ONPOLICY_STATES, SEED + 12
)
recovery_states = deterministic_cap(
    torch.cat((fit_states, onpolicy_states)),
    MAX_FIT_STATES + MAX_ONPOLICY_STATES,
    SEED + 13,
)
onpolicy_result = distil_nested_head(
    head, recovery_states, select_states, weight_train, readout_bias=bias_train,
    epochs=ONPOLICY_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
    learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
    kl_weight=1.0, token_weight=0.25, margin_weight=0.25,
    nested_weight=0.5, minimum_margin=0.25, anchor_strength=1e-5, seed=SEED + 1,
)
final_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
final_metrics = {f"rank_{rank}": evaluate_nested_head(
    head, select_states, weight_train, readout_bias=bias_train, rank=rank,
    batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
) for rank in RANKS}
print({"onpolicy_states": len(onpolicy_states), "replay_agreement": onpolicy_replay})
print(json.dumps(final_metrics, indent=2))
''')

markdown("## Select adaptive 64→192 routing on validation states")
code(r'''
@torch.inference_mode()
def adaptive_metrics(threshold):
    agreements = fallback = total = 0
    for start in range(0, len(select_states), DISTILL_BATCH_SIZE):
        hidden = select_states[start:start + DISTILL_BATCH_SIZE].to(device)
        teacher = torch.nn.functional.linear(hidden, weight_train, bias_train)
        low = head.forward_rank(hidden, RANKS[0])
        high = head.forward_rank(hidden, PRIMARY_RANK)
        values = low.topk(2, dim=-1).values
        use_high = (values[:, 0] - values[:, 1]) < threshold
        selected = torch.where(use_high.unsqueeze(1), high, low)
        agreements += int((selected.argmax(-1) == teacher.argmax(-1)).sum())
        fallback += int(use_high.sum())
        total += len(hidden)
    fraction = fallback / max(1, total)
    return {"threshold": float(threshold),
            "top1_agreement": agreements / max(1, total),
            "fallback_fraction": fraction,
            "average_rank": RANKS[0] + fraction * (PRIMARY_RANK - RANKS[0])}

with torch.inference_mode():
    logits = torch.cat([head.forward_rank(
        select_states[start:start + DISTILL_BATCH_SIZE].to(device), RANKS[0]
    ).cpu() for start in range(0, len(select_states), DISTILL_BATCH_SIZE)])
margins = logits.topk(2, dim=-1).values
margins = margins[:, 0] - margins[:, 1]
quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
thresholds = [-1.0, *sorted(set(float(value) for value in torch.quantile(margins, quantiles))),
              float("inf")]
adaptive_grid = [adaptive_metrics(value) for value in thresholds]
eligible = [row for row in adaptive_grid if row["top1_agreement"] >= ADAPTIVE_MIN_TOP1]
adaptive_choice = (
    min(eligible, key=lambda row: (row["average_rank"], -row["top1_agreement"]))
    if eligible else max(adaptive_grid, key=lambda row: row["top1_agreement"])
)
print(json.dumps({"grid": adaptive_grid, "selected": adaptive_choice}, indent=2))
''')

markdown("## Locked global-generation comparison")
code(r'''
baseline_test = baseline_records["test"]
baseline_tokens = sum(row["generated_tokens"] for row in baseline_test)
generation_results = {
    "full": {
        "examples": len(baseline_test),
        "correct": sum(row["correct"] for row in baseline_test),
        "numeric_exact_match": sum(row["correct"] for row in baseline_test) / len(baseline_test),
        "generated_tokens": baseline_tokens,
        "truncated": sum(row["truncated"] for row in baseline_test),
        "wall_clock_seconds": baseline_elapsed["test"],
        "tokens_per_second": baseline_tokens / baseline_elapsed["test"],
    }
}
generation_outcomes = {"full": [row["correct"] for row in baseline_test]}

generation_heads = {f"whitened_margin_onpolicy_r{PRIMARY_RANK}": generation_head(
    final_state, PRIMARY_RANK
)}
if RUN_ALL_FIXED_RANKS:
    for rank in RANKS[:-1]:
        generation_heads[f"whitened_margin_onpolicy_r{rank}"] = generation_head(final_state, rank)
adaptive = generation_head(final_state, RANKS[0])
adaptive.configure_adaptive(
    base_rank=RANKS[0], fallback_rank=PRIMARY_RANK,
    margin_threshold=adaptive_choice["threshold"], inactive_rank=None,
)
generation_heads[f"adaptive_r{RANKS[0]}_r{PRIMARY_RANK}"] = adaptive

if RUN_FULL_GENERATION:
    for name, output_head in generation_heads.items():
        torch.cuda.synchronize()
        started = time.perf_counter()
        records = generate_records(test_rows, name, output_head)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = sum(row["generated_tokens"] for row in records)
        generation_outcomes[name] = [row["correct"] for row in records]
        generation_results[name] = {
            "examples": len(records), "correct": sum(row["correct"] for row in records),
            "numeric_exact_match": sum(row["correct"] for row in records) / len(records),
            "generated_tokens": tokens, "truncated": sum(row["truncated"] for row in records),
            "wall_clock_seconds": elapsed, "tokens_per_second": tokens / elapsed,
        }
        print(name, generation_results[name])
model.set_output_embeddings(full_head)
''')

markdown("## Timing, gates, and export")
code(r'''
head_latency_us = {"full": benchmark_vocabulary_head(
    full_head, hidden_size, batch_size=1, iterations=200,
    device=device, dtype=generation_dtype,
)}
for rank in RANKS:
    candidate = generation_head(final_state, rank)
    head_latency_us[f"rank_{rank}"] = benchmark_vocabulary_head(
        candidate, hidden_size, batch_size=1, iterations=200,
        device=device, dtype=generation_dtype,
    )
adaptive.last_fallback_fraction = 0.0
head_latency_us["adaptive"] = benchmark_vocabulary_head(
    adaptive, hidden_size, batch_size=1, iterations=200,
    device=device, dtype=generation_dtype,
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

paired = {f"{name}_minus_full": paired_interval(
    outcomes, generation_outcomes["full"], SEED + offset
) for offset, (name, outcomes) in enumerate(generation_outcomes.items()) if name != "full"}
primary_name = f"whitened_margin_onpolicy_r{PRIMARY_RANK}"
primary = generation_results.get(primary_name)
full = generation_results["full"]
gates = {
    "baseline_has_at_least_50_correct": full["correct"] >= 50,
    "primary_retains_98_percent_accuracy": None if primary is None else (
        primary["numeric_exact_match"] >= 0.98 * full["numeric_exact_match"]
    ),
    "primary_improves_tokens_per_second": None if primary is None else (
        primary["tokens_per_second"] > full["tokens_per_second"]
    ),
}
summary = {
    "experiment": "qwen_trajectory_whitened_margin_distilled_global_lm_head_v1",
    "code_commit": CODE_COMMIT, "model": MODEL_ID, "model_revision": MODEL_REVISION,
    "population": {"fit": FIT_EXAMPLES, "select": SELECT_EXAMPLES,
                   "onpolicy": ONPOLICY_EXAMPLES, "test": TEST_EXAMPLES},
    "ranks": list(RANKS), "initialization": initialization_report.to_dict(),
    "initial_metrics": initial_metrics,
    "clean_training": {"losses": list(clean_result.losses),
                       "best_epoch": clean_result.best_epoch, "metrics": clean_metrics},
    "onpolicy_training": {"losses": list(onpolicy_result.losses),
                          "best_epoch": onpolicy_result.best_epoch,
                          "onpolicy_states": len(onpolicy_states), "metrics": final_metrics},
    "adaptive_selection": {"grid": adaptive_grid, "selected": adaptive_choice},
    "generation": generation_results, "paired_accuracy_intervals": paired,
    "head_latency_microseconds": head_latency_us,
    "operations": {"full": hidden_size * vocabulary_size,
                   **{f"rank_{rank}": rank * (hidden_size + vocabulary_size)
                      for rank in RANKS}},
    "gates": gates,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
temporary = pathlib.Path(str(summary_path) + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
temporary.replace(summary_path)
torch.save({"contract": summary["experiment"], "model": MODEL_ID,
            "model_revision": MODEL_REVISION, "ranks": RANKS,
            "state_dict": final_state, "adaptive_choice": adaptive_choice,
            "initialization": initialization_report.to_dict()},
           pathlib.Path(OUTPUT_ROOT) / "global_low_rank_head.pt")
print(json.dumps({"gates": gates, "generation": generation_results,
                  "head_latency_microseconds": head_latency_us}, indent=2))
print("saved:", summary_path)
''')

markdown(r"""
## Interpretation boundary

A passing Qwen run plus a passing CODI run supports transfer of the fitting procedure
across those two model families. It does not establish universal generalization. A
third family and non-mathematical text must be tested before making a broad claim.
The comparison is verifier-free: every global-head mistake affects subsequent states.
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
