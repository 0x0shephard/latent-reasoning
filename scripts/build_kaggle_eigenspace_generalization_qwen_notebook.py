"""Build the Kaggle notebook testing eigenspace readout generalization on Qwen."""
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
# Experiment 2 — Does the eigenspace readout generalize beyond CODI?

## Goal

This is a deliberately harder test: move from CODI/GPT-2 to the conventional
autoregressive **Qwen2.5-Math-1.5B-Instruct** model while keeping GSM8K and greedy
numeric exact match. The model has a different architecture, hidden width, vocabulary,
training objective, and answer-generation process.

We test the *selection method*, not whether CODI's literal vectors transfer. A covariance
eigendecomposition is fitted on Qwen prompt-endpoint states. At equal rank, the frozen
heads are:

1. `leading`: directions with the most activation variance;
2. `skip4`: the CODI rule that ignores four leading common-mode directions;
3. `readout_aware`: directions ranked by activation energy × relative-logit energy;
4. `random`: a seeded orthonormal control.

Rank 32 tests the same absolute bottleneck as CODI. Rank 64 tests the same approximate
fraction of Qwen's 1,536-dimensional width as CODI rank 32/768.

### Preregistered primary gate

At rank 64, `readout_aware` must retain at least 95% of baseline exact match, retain at
least 90% first-token agreement, beat the matched random head by at least five accuracy
points, and accelerate the isolated head by at least 3×. If the baseline produces fewer
than 50 correct test answers, the accuracy comparison is labelled inconclusive.
""")

markdown("## Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit after these files are pushed.
REPO_DIR = "/kaggle/working/latent-reasoning"
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_REVISION = "main"  # Pin to a commit hash for a final confirmatory run.
OUTPUT_ROOT = "/kaggle/working/eigenspace_generalization_qwen"

SEED = 20260902
FIT_EXAMPLES = 512
SELECT_EXAMPLES = 128
TEST_EXAMPLES = 256
RANKS = [32, 64]
PRIMARY_RANK = 64
RANDOM_NULL_REPLICATES = 20
BOOTSTRAP_SAMPLES = 5000
GENERATION_RANDOM_SEED = 7
MAX_NEW_TOKENS = 192
BATCH_SIZE = 4
RUN_GENERATION_ARMS = True

import glob, json, os, pathlib, subprocess, sys, time
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
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("commit:", subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip())
assert RUN_COMMIT == "main" or subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip().startswith(RUN_COMMIT)
''')

markdown("## Install and verify dependencies")
code(r'''
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "accelerate>=1.2,<2",
    "huggingface_hub>=0.34,<1.0",
], check=True)
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_eigenspace_readout.py"],
    check=True,
)
''')

markdown("## Load frozen fit, selection, and test data")
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
order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(SEED))
fit_rows = [train_rows[int(i)] for i in order[:FIT_EXAMPLES]]
select_rows = [train_rows[int(i)] for i in order[FIT_EXAMPLES:FIT_EXAMPLES+SELECT_EXAMPLES]]
test_order = torch.randperm(len(test_rows_all), generator=torch.Generator().manual_seed(SEED + 1))
test_rows = [test_rows_all[int(i)] for i in test_order[:TEST_EXAMPLES]]
assert set(order[:FIT_EXAMPLES].tolist()).isdisjoint(order[FIT_EXAMPLES:FIT_EXAMPLES+SELECT_EXAMPLES].tolist())
print({"fit": len(fit_rows), "select": len(select_rows), "test": len(test_rows)})
''')

markdown("## Load Qwen in T4-safe precision")
code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda")
dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN") or None,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype, low_cpu_mem_usage=True,
    token=os.environ.get("HF_TOKEN") or None,
).to(device).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
full_head = model.get_output_embeddings()
hidden_size = int(model.config.hidden_size)
vocabulary_size = int(full_head.weight.shape[0])
assert hidden_size == 1536, hidden_size
print({"hidden": hidden_size, "vocabulary": vocabulary_size,
       "head": type(full_head).__name__, "dtype": str(dtype)})
''')

markdown("## Baseline generation and prompt-endpoint state capture")
code(r'''
SYSTEM_PROMPT = (
    "Solve the math problem carefully. End your response with exactly "
    "'Final answer: <number>'."
)

def prompts(rows):
    messages = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
    ] for row in rows]
    return [tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) for message in messages]

@torch.inference_mode()
def generate_rows(rows, *, capture_endpoints=False):
    generations, endpoints = [], []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start+BATCH_SIZE]
        encoded = tokenizer(
            prompts(batch_rows), return_tensors="pt", padding=True,
            truncation=True, max_length=768,
        ).to(device)
        head_calls = []
        handle = None
        if capture_endpoints:
            def capture(_module, arguments):
                head_calls.append(arguments[0][:, -1, :].detach().cpu().float())
            handle = model.get_output_embeddings().register_forward_pre_hook(capture)
        output = model.generate(
            **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if handle is not None:
            handle.remove()
            assert head_calls, "the output-head hook did not observe generation"
            endpoints.append(head_calls[0])
        new_tokens = output[:, encoded["input_ids"].shape[1]:]
        generations.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    endpoint_tensor = torch.cat(endpoints, dim=0) if endpoints else None
    return generations, endpoint_tensor

baseline = {}
for split_name, rows in [("fit", fit_rows), ("select", select_rows), ("test", test_rows)]:
    text, states = generate_rows(rows, capture_endpoints=True)
    correct = [answers_match(generation, row["gold"]) for generation, row in zip(text, rows)]
    baseline[split_name] = {"generations": text, "states": states, "correct": correct}
    print(split_name, {"accuracy": sum(correct)/len(correct), "correct": sum(correct),
                       "state_shape": list(states.shape)})
''')

markdown("## Fit Qwen's covariance eigensystem using fit rows only")
code(r'''
from src.mech.eigenspace_readout import (
    LowRankVocabularyHead, benchmark_vocabulary_head, covariance_eigensystem,
    evaluate_head_fidelity, orthonormal_random_basis, select_readout_aware_basis,
)

centre, eigenvalues, eigenvectors = covariance_eigensystem(baseline["fit"]["states"])
weight = full_head.weight.detach()
bases, selection_details = {}, {}
for rank in RANKS:
    bases[("leading", rank)] = eigenvectors[:, :rank]
    bases[("skip4", rank)] = eigenvectors[:, 4:4+rank]
    aware, indices, scores = select_readout_aware_basis(
        weight, eigenvalues, eigenvectors, rank, chunk_size=32
    )
    bases[("readout_aware", rank)] = aware
    bases[("random", rank)] = orthonormal_random_basis(hidden_size, rank, seed=GENERATION_RANDOM_SEED)
    selection_details[str(rank)] = {
        "readout_aware_indices": indices.tolist(),
        "readout_aware_score_fraction": float(scores[indices].sum() / scores.sum()),
        "variance_fraction_leading": float(eigenvalues[:rank].sum() / eigenvalues.sum()),
        "variance_fraction_skip4": float(eigenvalues[4:4+rank].sum() / eigenvalues.sum()),
    }
print(json.dumps(selection_details, indent=2))
''')

markdown("## Selection-split fidelity and a 20-basis random null")
code(r'''
def make_head(basis):
    return LowRankVocabularyHead.from_basis(weight, basis, centre).to(device=device, dtype=dtype)

select_fidelity = {}
for (name, rank), basis in bases.items():
    head = make_head(basis)
    select_fidelity[f"{name}_r{rank}"] = evaluate_head_fidelity(
        head, baseline["select"]["states"], weight,
        batch_size=8, temperature=2.0,
    )

random_null = []
for replicate in range(RANDOM_NULL_REPLICATES):
    basis = orthonormal_random_basis(hidden_size, PRIMARY_RANK, seed=SEED + 100 + replicate)
    head = make_head(basis)
    random_null.append(evaluate_head_fidelity(
        head, baseline["select"]["states"], weight,
        batch_size=8, temperature=2.0,
    )["top1_agreement"])
print(json.dumps(select_fidelity, indent=2))
print({"random_null_mean": sum(random_null)/len(random_null),
       "random_null_max": max(random_null)})
''')

markdown("## Frozen test generation at matched rank 64")
code(r'''
generation_arms = {
    "full": full_head,
    f"leading_r{PRIMARY_RANK}": make_head(bases[("leading", PRIMARY_RANK)]),
    f"skip4_r{PRIMARY_RANK}": make_head(bases[("skip4", PRIMARY_RANK)]),
    f"readout_aware_r{PRIMARY_RANK}": make_head(bases[("readout_aware", PRIMARY_RANK)]),
    f"random_r{PRIMARY_RANK}": make_head(bases[("random", PRIMARY_RANK)]),
}
generation_results = {}
generation_outcomes = {}
if RUN_GENERATION_ARMS:
    for name, head in generation_arms.items():
        model.set_output_embeddings(head)
        torch.cuda.synchronize()
        started = time.perf_counter()
        text, _ = generate_rows(test_rows, capture_endpoints=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        correct = [answers_match(generation, row["gold"]) for generation, row in zip(text, test_rows)]
        generation_outcomes[name] = correct
        generation_results[name] = {
            "examples": len(test_rows), "correct": int(sum(correct)),
            "numeric_exact_match": float(sum(correct)/len(correct)),
            "wall_clock_seconds": elapsed,
            "examples_per_second": len(test_rows)/elapsed,
        }
        print(name, generation_results[name])
model.set_output_embeddings(full_head)
''')

markdown("## Test fidelity, head timing, operation counts, and gate")
code(r'''
test_fidelity = {}
for name in ("leading", "skip4", "readout_aware", "random"):
    head = make_head(bases[(name, PRIMARY_RANK)])
    test_fidelity[f"{name}_r{PRIMARY_RANK}"] = evaluate_head_fidelity(
        head, baseline["test"]["states"], weight, batch_size=8, temperature=2.0
    )

timing = {"full": benchmark_vocabulary_head(
    full_head, hidden_size, batch_size=1, iterations=200, device=device, dtype=dtype
)}
for name in ("leading", "skip4", "readout_aware", "random"):
    timing[f"{name}_r{PRIMARY_RANK}"] = benchmark_vocabulary_head(
        make_head(bases[(name, PRIMARY_RANK)]), hidden_size,
        batch_size=1, iterations=200, device=device, dtype=dtype,
    )

operations = {
    "full_macs_per_token": hidden_size * vocabulary_size,
    "rank64_macs_per_token": PRIMARY_RANK * (hidden_size + vocabulary_size),
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

    aware_key = f"readout_aware_r{PRIMARY_RANK}"
    random_key = f"random_r{PRIMARY_RANK}"
    paired_intervals["readout_aware_minus_random"] = paired_interval(
        generation_outcomes[aware_key], generation_outcomes[random_key], SEED
    )
    paired_intervals["readout_aware_minus_full"] = paired_interval(
        generation_outcomes[aware_key], generation_outcomes["full"], SEED + 1
    )
    full = generation_results["full"]
    aware = generation_results[f"readout_aware_r{PRIMARY_RANK}"]
    random_arm = generation_results[f"random_r{PRIMARY_RANK}"]
    agreement = test_fidelity[f"readout_aware_r{PRIMARY_RANK}"]["top1_agreement"]
    speedup = timing["full"] / timing[f"readout_aware_r{PRIMARY_RANK}"]
    decision = {
        "baseline_correct": full["correct"],
        "accuracy_retained_fraction": aware["numeric_exact_match"] / full["numeric_exact_match"]
                                      if full["numeric_exact_match"] else None,
        "advantage_over_random_points": 100 * (
            aware["numeric_exact_match"] - random_arm["numeric_exact_match"]
        ),
        "first_token_agreement": agreement,
        "isolated_head_speedup": speedup,
    }
    if full["correct"] < 50:
        decision["status"] = "inconclusive_low_baseline_correct_count"
    else:
        decision["status"] = (
            "generalization_supported" if decision["accuracy_retained_fraction"] >= 0.95
            and agreement >= 0.90 and decision["advantage_over_random_points"] >= 5.0
            and paired_intervals["readout_aware_minus_random"][0] > 0.0
            and speedup >= 3.0 else "generalization_not_supported_by_gate"
        )

summary = {
    "experiment": "eigenspace_readout_generalization_qwen",
    "model": MODEL_ID, "model_revision": MODEL_REVISION, "seed": SEED,
    "splits": {"fit": len(fit_rows), "select": len(select_rows), "test": len(test_rows)},
    "baseline_accuracy": {name: sum(value["correct"])/len(value["correct"])
                          for name, value in baseline.items()},
    "selection_details": selection_details, "select_fidelity": select_fidelity,
    "random_null_top1_agreement": random_null, "test_fidelity": test_fidelity,
    "generation_test": generation_results, "paired_bootstrap_95ci": paired_intervals,
    "head_latency_us": timing,
    "operation_counts": operations, "decision": decision,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"decision": decision, "operation_counts": operations,
                  "head_latency_us": timing}, indent=2))
print("wrote", summary_path)
''')

markdown(r"""
## Takeaways

This experiment can fail in informative ways. If all low-rank heads fail, CODI's compact
answer subspace may be specific to its continuous-reasoning endpoint. If the learned
readout-aware rule works but `skip4` does not, the general principle transfers while the
literal CODI band does not. If random performs similarly, covariance directions are not
providing a meaningful selection advantage. Do not tune the rank or selection score on
the frozen test rows.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
