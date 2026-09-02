"""Build the Kaggle notebook localizing Qwen's answer eigenspace rank."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_qwen_answer_eigenspace_localization.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Experiment 3 — How many Qwen answer directions are sufficient and necessary?

## TL;DR

The previous corrected experiment showed that a rank-192 projection preserved 99.6%
of Qwen's first final-answer tokens, while rank 64 preserved every validation token.
That established a compact pre-answer subspace but did not determine its minimum size.

This notebook reuses the saved, untouched Qwen baseline responses and tests ranks
`1–192` at the exact state that predicts the first token inside the final
`\boxed{answer}`. It performs two complementary interventions:

- **Keep:** retain only the selected centered directions before the original LM head.
- **Remove:** delete those directions while retaining every orthogonal direction.

The selection split determines the smallest sufficient rank. The frozen test split
confirms it. Finally, Qwen's original reasoning prefix is held fixed, one answer token
is selected under each intervention, and the normal full head completes the answer.
This separates answer-readout causality from global autoregressive compression.
""")

markdown(r"""
## Context and methods

### Key assumptions

- The relevant location is Qwen layer 28 after its final normalization and immediately
  before the LM head predicts the first answer-content token.
- “Sufficient” means the kept subspace preserves at least 99% of full-head predictions,
  both overall and among baseline-correct solutions.
- “Necessary” is a separate claim: removing the selected subspace must reduce causal
  answer accuracy by at least 10 percentage points relative to the full-head replay.
- A discrete sweep gives an interval, not a magical exact integer. If rank 32 fails and
  rank 40 passes, the supported minimum is reported as `(32, 40]`.
- The literal Qwen vectors are learned from Qwen fit examples. No CODI vector is copied.

### Required Kaggle input

Attach dataset `jonraza15/does-the-answer-eigenspace-generalize-beyond-codi`.
The notebook discovers its `baseline_fit.jsonl`, `baseline_select.jsonl`,
`baseline_test.jsonl`, and `summary.json` automatically. It does not rerun the costly
full baseline generation.
""")

markdown("## Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "43f4ea3"
REPO_DIR = "/kaggle/working/latent-reasoning"
INPUT_ROOT = ""  # Optional explicit .../eigenspace_generalization_qwen_v2 directory.
OUTPUT_ROOT = "/kaggle/working/qwen_answer_eigenspace_localization"

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
HF_MODEL_REVISION = "f903dd76e2a9741d582f2a31248f1f5d0ac0e2bf"
SEED = 20260904
RANKS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 192]
SUFFICIENCY_THRESHOLD = 0.99
REMOVAL_ACCURACY_DROP_POINTS = 10.0
RANDOM_NULL_REPLICATES = 20
TEACHER_FORCE_BATCH_SIZE = 2
METRIC_BATCH_SIZE = 8
SUFFIX_BATCH_SIZE = 4
MAX_SUFFIX_NEW_TOKENS = 24
BOOTSTRAP_SAMPLES = 5000

import glob, json, os, pathlib, subprocess, sys, time
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
CODE_COMMIT = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
assert CODE_COMMIT.startswith(RUN_COMMIT)
print("commit:", CODE_COMMIT)
''')

markdown("## Install dependencies and run the measuring-instrument tests")
code(r'''
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.52.4", "accelerate>=1.2,<2",
    "huggingface_hub>=0.34,<1.0",
], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_eigenspace_readout.py", "tests/test_qwen_trajectory.py",
], check=True)
''')

markdown("## Resolve and validate the completed baseline records")
code(r'''
def discover(filename):
    if INPUT_ROOT:
        path = pathlib.Path(INPUT_ROOT) / filename
        assert path.is_file(), path
        return path
    matches = sorted(glob.glob(
        f"/kaggle/input/**/eigenspace_generalization_qwen_v2/{filename}", recursive=True
    ))
    assert matches, (
        "Attach jonraza15/does-the-answer-eigenspace-generalize-beyond-codi; "
        f"could not find {filename}"
    )
    return pathlib.Path(matches[0])

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

input_summary = json.loads(discover("summary.json").read_text(encoding="utf-8"))
assert input_summary["experiment"] == "eigenspace_generalization_qwen_v2_matched_endpoint"
assert input_summary["model"] == MODEL_ID
assert input_summary["model_revision"] == HF_MODEL_REVISION
assert all(input_summary["validity"].values()), input_summary["validity"]

baseline = {
    split: read_jsonl(discover(f"baseline_{split}.jsonl"))
    for split in ("fit", "select", "test")
}
assert {split: len(rows) for split, rows in baseline.items()} == {
    "fit": 512, "select": 128, "test": 256,
}
print({
    split: {
        "rows": len(rows), "correct": sum(row["correct"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
    } for split, rows in baseline.items()
})
''')

markdown("## Load the same frozen Qwen checkpoint")
code(r'''
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.answer_extract import answers_match
from src.mech.eigenspace_readout import (
    LowRankVocabularyHead, covariance_eigensystem,
    orthonormal_random_basis, readout_aware_scores,
)
from src.mech.qwen_trajectory import final_answer_span, token_indices_overlapping_span

device = torch.device("cuda")
dtype = torch.float16
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=HF_MODEL_REVISION, token=os.environ.get("HF_TOKEN") or None,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=HF_MODEL_REVISION, torch_dtype=dtype,
    low_cpu_mem_usage=True, token=os.environ.get("HF_TOKEN") or None,
).to(device).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
full_head = model.get_output_embeddings()
weight = full_head.weight.detach()
hidden_size = int(model.config.hidden_size)
assert hidden_size == 1536 and weight.shape[0] == 151936
print({"hidden": hidden_size, "vocabulary": int(weight.shape[0]), "dtype": str(dtype)})
''')

markdown("## Replay saved reasoning and recover matched pre-answer states")
code(r'''
def canonical_alignment(record):
    text = record["generation"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    span = final_answer_span(text)
    answer_indices = [] if span is None else token_indices_overlapping_span(offsets, span[:2])
    return response_ids, answer_indices, span

@torch.inference_mode()
def collect_endpoints(records):
    states, targets, correct, rows, gold, prefixes, audit = [], [], [], [], [], [], []
    for start in range(0, len(records), TEACHER_FORCE_BATCH_SIZE):
        batch = records[start:start + TEACHER_FORCE_BATCH_SIZE]
        examples = []
        for record in batch:
            response_ids, answer_indices, span = canonical_alignment(record)
            prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
            if response_ids and answer_indices:
                examples.append({
                    "record": record, "prompt_ids": list(map(int, prompt_ids)),
                    "response_ids": response_ids, "answer_index": answer_indices[0],
                    "rule": span[2],
                })
            else:
                audit.append({"row": record["row"], "aligned": False})
        if not examples:
            continue
        sequences = [item["prompt_ids"] + item["response_ids"] for item in examples]
        padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(device)
        hidden = model.model(
            **padded, use_cache=False, return_dict=True
        ).last_hidden_state.detach().cpu().float()
        width = hidden.shape[1]
        for index, item in enumerate(examples):
            prompt_length = len(item["prompt_ids"])
            response_length = len(item["response_ids"])
            padding = width - prompt_length - response_length
            predictor = padding + prompt_length - 1 + item["answer_index"]
            states.append(hidden[index, predictor])
            targets.append(item["response_ids"][item["answer_index"]])
            correct.append(bool(item["record"]["correct"]))
            rows.append(int(item["record"]["row"]))
            gold.append(item["record"]["gold"])
            prefixes.append(item["prompt_ids"] + item["response_ids"][:item["answer_index"]])
            audit.append({
                "row": item["record"]["row"], "aligned": True,
                "rule": item["rule"], "answer_index": item["answer_index"],
                "target_token_id": targets[-1],
            })
    return {
        "states": torch.stack(states), "targets": torch.tensor(targets, dtype=torch.long),
        "baseline_correct": torch.tensor(correct, dtype=torch.bool), "rows": rows,
        "gold": gold, "prefixes": prefixes, "audit": audit,
    }

# Selection is performed before the test payload is constructed.
fit_payload = collect_endpoints(baseline["fit"])
select_payload = collect_endpoints(baseline["select"])
print({
    "fit": list(fit_payload["states"].shape),
    "select": list(select_payload["states"].shape),
    "select_rules": dict(Counter(row.get("rule") for row in select_payload["audit"])),
})
''')

markdown("## Fit the eigensystem and define nested direction orderings")
code(r'''
centre, eigenvalues, eigenvectors = covariance_eigensystem(fit_payload["states"])
aware_scores = readout_aware_scores(weight, eigenvalues, eigenvectors, chunk_size=32)
aware_order = torch.argsort(aware_scores, descending=True)
direction_orders = {
    "leading": torch.arange(hidden_size),
    "readout_aware": aware_order,
}
print({
    "fit_endpoints": len(fit_payload["states"]),
    "top_readout_aware_indices": aware_order[:64].tolist(),
    "top64_score_fraction": float(aware_scores[aware_order[:64]].sum() / aware_scores.sum()),
})
''')

markdown("## Sweep keep/remove interventions on the selection split")
code(r'''
@torch.inference_mode()
def intervention_metrics(payload, basis):
    keep_head = LowRankVocabularyHead.from_basis(weight, basis, centre).to(
        device=device, dtype=dtype
    )
    full_predictions, keep_predictions, remove_predictions = [], [], []
    for start in range(0, len(payload["states"]), METRIC_BATCH_SIZE):
        hidden = payload["states"][start:start + METRIC_BATCH_SIZE].to(device=device, dtype=dtype)
        selected = basis.to(device=device, dtype=dtype)
        origin = centre.to(device=device, dtype=dtype)
        removed = hidden - ((hidden - origin) @ selected) @ selected.T
        full_predictions.append(F.linear(hidden, weight).argmax(-1).cpu())
        keep_predictions.append(keep_head(hidden).argmax(-1).cpu())
        remove_predictions.append(F.linear(removed, weight).argmax(-1).cpu())
    full_predictions = torch.cat(full_predictions)
    keep_predictions = torch.cat(keep_predictions)
    remove_predictions = torch.cat(remove_predictions)
    correct = payload["baseline_correct"]
    targets = payload["targets"]
    del keep_head
    return {
        "examples": len(targets),
        "full_target_replay": float((full_predictions == targets).float().mean()),
        "keep_full_agreement": float((keep_predictions == full_predictions).float().mean()),
        "remove_full_agreement": float((remove_predictions == full_predictions).float().mean()),
        "keep_target_agreement": float((keep_predictions == targets).float().mean()),
        "remove_target_agreement": float((remove_predictions == targets).float().mean()),
        "keep_full_agreement_when_baseline_correct": float(
            (keep_predictions[correct] == full_predictions[correct]).float().mean()
        ),
        "remove_full_agreement_when_baseline_correct": float(
            (remove_predictions[correct] == full_predictions[correct]).float().mean()
        ),
    }

selection_sweep = {name: {} for name in direction_orders}
for name, order in direction_orders.items():
    for rank in RANKS:
        basis = eigenvectors[:, order[:rank]]
        selection_sweep[name][str(rank)] = intervention_metrics(select_payload, basis)
        print(name, rank, selection_sweep[name][str(rank)])

qualifying = []
for rank in RANKS:
    metrics = selection_sweep["readout_aware"][str(rank)]
    if (metrics["keep_full_agreement"] >= SUFFICIENCY_THRESHOLD and
            metrics["keep_full_agreement_when_baseline_correct"] >= SUFFICIENCY_THRESHOLD):
        qualifying.append(rank)
selected_rank = min(qualifying) if qualifying else None
assert selected_rank is not None, "No rank met the preregistered selection threshold"
selected_position = RANKS.index(selected_rank)
lower_rank = RANKS[selected_position - 1] if selected_position else 0
selected_indices = aware_order[:selected_rank]
print({
    "selected_rank": selected_rank,
    "minimum_rank_interval": f"({lower_rank}, {selected_rank}]",
    "selected_eigenvector_indices": selected_indices.tolist(),
})
''')

markdown("## Evaluate matched random controls at the selected rank")
code(r'''
random_null_select = []
for replicate in range(RANDOM_NULL_REPLICATES):
    random_basis = orthonormal_random_basis(
        hidden_size, selected_rank, seed=SEED + 1000 + replicate
    )
    random_null_select.append(
        intervention_metrics(select_payload, random_basis)["keep_full_agreement"]
    )
print({
    "replicates": len(random_null_select),
    "median": float(torch.tensor(random_null_select).median()),
    "maximum": max(random_null_select),
})
''')

markdown("## Freeze selection, construct the test endpoints, and verify replay")
code(r'''
test_payload = collect_endpoints(baseline["test"])
with torch.inference_mode():
    replay_predictions = []
    for start in range(0, len(test_payload["states"]), METRIC_BATCH_SIZE):
        hidden = test_payload["states"][start:start + METRIC_BATCH_SIZE].to(
            device=device, dtype=dtype
        )
        replay_predictions.append(F.linear(hidden, weight).argmax(-1).cpu())
    replay_predictions = torch.cat(replay_predictions)
test_replay_agreement = float(
    (replay_predictions == test_payload["targets"]).float().mean()
)
assert test_replay_agreement >= 0.99, test_replay_agreement
print({
    "test_endpoints": len(test_payload["states"]),
    "alignment_rate": len(test_payload["states"]) / len(baseline["test"]),
    "full_target_replay": test_replay_agreement,
})
''')

markdown("## Confirm the preregistered rank and report the frozen test curve")
code(r'''
test_sweep = {name: {} for name in direction_orders}
for name, order in direction_orders.items():
    for rank in RANKS:
        basis = eigenvectors[:, order[:rank]]
        test_sweep[name][str(rank)] = intervention_metrics(test_payload, basis)

selected_test = test_sweep["readout_aware"][str(selected_rank)]
rank_confirmed = (
    selected_test["keep_full_agreement"] >= SUFFICIENCY_THRESHOLD and
    selected_test["keep_full_agreement_when_baseline_correct"] >= SUFFICIENCY_THRESHOLD
)
random_null_test = []
for replicate in range(RANDOM_NULL_REPLICATES):
    random_basis = orthonormal_random_basis(
        hidden_size, selected_rank, seed=SEED + 1000 + replicate
    )
    random_null_test.append(
        intervention_metrics(test_payload, random_basis)["keep_full_agreement"]
    )
print({
    "selected_rank": selected_rank, "confirmed": rank_confirmed,
    "test_metrics": selected_test,
    "random_test_median": float(torch.tensor(random_null_test).median()),
})
''')

markdown("## Visualize sufficiency and necessity across rank")
code(r'''
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for axis, (split, sweep) in zip(axes, (("Selection", selection_sweep), ("Test", test_sweep))):
    for name, style in (("readout_aware", "-o"), ("leading", "--s")):
        keep = [sweep[name][str(rank)]["keep_full_agreement"] for rank in RANKS]
        remove = [sweep[name][str(rank)]["remove_full_agreement"] for rank in RANKS]
        axis.plot(RANKS, keep, style, label=f"keep {name}")
        axis.plot(RANKS, remove, style, alpha=0.45, label=f"remove {name}")
    axis.axhline(SUFFICIENCY_THRESHOLD, color="black", linestyle=":", label="99% threshold")
    axis.axvline(selected_rank, color="tab:red", linestyle=":", label=f"selected r={selected_rank}")
    axis.set_xscale("log", base=2)
    axis.set_xticks(RANKS)
    axis.set_xticklabels(RANKS, rotation=60)
    axis.set_xlabel("Retained or removed directions")
    axis.set_title(split)
    axis.grid(alpha=0.2)
axes[0].set_ylabel("Agreement with the original LM head")
axes[1].legend(loc="lower right", fontsize=8)
figure.suptitle("Qwen pre-answer eigenspace: keep and remove interventions")
figure.tight_layout()
plot_path = pathlib.Path(OUTPUT_ROOT) / "rank_sweep_keep_remove.png"
figure.savefig(plot_path, dpi=180, bbox_inches="tight")
plt.show()
print("wrote", plot_path)
''')

markdown(r"""
## Causal answer completion

For every aligned test example, the complete original Qwen reasoning prefix through
`\boxed{` is held fixed. We intervene only on the next token. The original full head
then generates the remaining answer suffix. Therefore, failures measure the causal
effect of changing the answer readout rather than damage accumulated during reasoning.
""")
code(r'''
@torch.inference_mode()
def intervention_first_tokens(payload, basis=None, mode="full"):
    predictions = []
    selected = None if basis is None else basis.to(device=device, dtype=dtype)
    origin = centre.to(device=device, dtype=dtype)
    keep_head = None
    if mode == "keep":
        keep_head = LowRankVocabularyHead.from_basis(weight, basis, centre).to(
            device=device, dtype=dtype
        )
    for start in range(0, len(payload["states"]), METRIC_BATCH_SIZE):
        hidden = payload["states"][start:start + METRIC_BATCH_SIZE].to(device=device, dtype=dtype)
        if mode == "full":
            logits = F.linear(hidden, weight)
        elif mode == "keep":
            logits = keep_head(hidden)
        elif mode == "remove":
            removed = hidden - ((hidden - origin) @ selected) @ selected.T
            logits = F.linear(removed, weight)
        else:
            raise ValueError(mode)
        predictions.append(logits.argmax(-1).cpu())
    return torch.cat(predictions).tolist()

def trim_at_eos(token_ids):
    values = [int(value) for value in token_ids]
    if tokenizer.eos_token_id in values:
        return values[:values.index(tokenizer.eos_token_id) + 1], True
    return values, False

def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

@torch.inference_mode()
def complete_answer_suffix(payload, first_tokens, arm):
    records = []
    assert len(first_tokens) == len(payload["prefixes"])
    for start in range(0, len(first_tokens), SUFFIX_BATCH_SIZE):
        stop = min(start + SUFFIX_BATCH_SIZE, len(first_tokens))
        sequences = [
            payload["prefixes"][index] + [int(first_tokens[index])]
            for index in range(start, stop)
        ]
        padded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(device)
        output = model.generate(
            **padded, max_new_tokens=MAX_SUFFIX_NEW_TOKENS - 1, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = output[:, padded["input_ids"].shape[1]:].detach().cpu().tolist()
        for local, index in enumerate(range(start, stop)):
            remaining, ended = trim_at_eos(continuation[local])
            suffix_ids = [int(first_tokens[index])] + remaining
            text = tokenizer.decode(
                suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            records.append({
                "arm": arm, "row": payload["rows"][index], "gold": payload["gold"][index],
                "baseline_correct": bool(payload["baseline_correct"][index]),
                "first_token_id": int(first_tokens[index]),
                "first_token": tokenizer.decode([int(first_tokens[index])]),
                "suffix": text, "correct": bool(answers_match(text, payload["gold"][index])),
                "ended_with_eos": ended, "suffix_tokens": len(suffix_ids),
            })
    write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"causal_{arm}.jsonl", records)
    return records

selected_basis = eigenvectors[:, selected_indices]
lower_basis = eigenvectors[:, aware_order[:lower_rank]] if lower_rank else None
leading_basis = eigenvectors[:, :selected_rank]
random_basis = orthonormal_random_basis(hidden_size, selected_rank, seed=SEED + 9000)

first_token_arms = {
    "full": intervention_first_tokens(test_payload, mode="full"),
    f"keep_aware_r{selected_rank}": intervention_first_tokens(
        test_payload, selected_basis, mode="keep"
    ),
    f"remove_aware_r{selected_rank}": intervention_first_tokens(
        test_payload, selected_basis, mode="remove"
    ),
    f"keep_leading_r{selected_rank}": intervention_first_tokens(
        test_payload, leading_basis, mode="keep"
    ),
    f"keep_random_r{selected_rank}": intervention_first_tokens(
        test_payload, random_basis, mode="keep"
    ),
}
if lower_basis is not None:
    first_token_arms[f"keep_aware_r{lower_rank}"] = intervention_first_tokens(
        test_payload, lower_basis, mode="keep"
    )

causal_records, causal_results, causal_outcomes = {}, {}, {}
for arm, first_tokens in first_token_arms.items():
    started = time.perf_counter()
    records = complete_answer_suffix(test_payload, first_tokens, arm)
    elapsed = time.perf_counter() - started
    outcomes = [record["correct"] for record in records]
    causal_records[arm] = records
    causal_outcomes[arm] = outcomes
    causal_results[arm] = {
        "examples": len(records), "correct": int(sum(outcomes)),
        "numeric_exact_match": float(sum(outcomes) / len(outcomes)),
        "first_token_matches_full": float(torch.tensor(first_tokens).eq(
            torch.tensor(first_token_arms["full"])
        ).float().mean()),
        "ended_with_eos": sum(record["ended_with_eos"] for record in records),
        "wall_clock_seconds": elapsed,
    }
    print(arm, causal_results[arm])
''')

markdown("## Uncertainty, causal decision, and saved result")
code(r'''
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

keep_arm = f"keep_aware_r{selected_rank}"
remove_arm = f"remove_aware_r{selected_rank}"
random_arm = f"keep_random_r{selected_rank}"
paired_intervals = {
    "keep_minus_full": paired_interval(causal_outcomes[keep_arm], causal_outcomes["full"], SEED),
    "remove_minus_full": paired_interval(causal_outcomes[remove_arm], causal_outcomes["full"], SEED + 1),
    "keep_minus_random": paired_interval(causal_outcomes[keep_arm], causal_outcomes[random_arm], SEED + 2),
}

full_accuracy = causal_results["full"]["numeric_exact_match"]
keep_accuracy = causal_results[keep_arm]["numeric_exact_match"]
remove_accuracy = causal_results[remove_arm]["numeric_exact_match"]
retained_fraction = keep_accuracy / full_accuracy if full_accuracy else None
removal_drop_points = 100 * (full_accuracy - remove_accuracy)
full_suffix_outcome_replay = float(
    torch.tensor(causal_outcomes["full"]).eq(test_payload["baseline_correct"]).float().mean()
)
specificity_supported = (
    selected_test["keep_full_agreement"] > max(random_null_test)
)
causal_replay_valid = full_suffix_outcome_replay >= 0.99
conclusion = {
    "selected_rank": selected_rank,
    "minimum_sufficient_rank_interval": [lower_rank, selected_rank],
    "rank_confirmed_on_test": bool(rank_confirmed),
    "selected_eigenvector_indices": selected_indices.tolist(),
    "location": "layer_28_post_final_norm_pre_lm_head_first_boxed_answer_token",
    "keep_accuracy_retained_fraction": retained_fraction,
    "removal_accuracy_drop_points": removal_drop_points,
    "full_suffix_outcome_replay": full_suffix_outcome_replay,
    "causal_replay_valid": causal_replay_valid,
    "specificity_over_random_supported": bool(specificity_supported),
    "sufficiency_supported": bool(
        rank_confirmed and causal_replay_valid and specificity_supported
        and retained_fraction is not None and retained_fraction >= 0.95
    ),
    "necessity_supported": bool(
        causal_replay_valid and removal_drop_points >= REMOVAL_ACCURACY_DROP_POINTS and
        paired_intervals["remove_minus_full"][1] < 0
    ),
}
if conclusion["sufficiency_supported"] and conclusion["necessity_supported"]:
    conclusion["status"] = "localized_sufficient_and_necessary_answer_eigenspace"
elif conclusion["sufficiency_supported"]:
    conclusion["status"] = "localized_sufficient_but_not_proven_necessary"
else:
    conclusion["status"] = "localization_not_supported_by_gate"

summary = {
    "experiment": "qwen_answer_eigenspace_rank_localization",
    "source_experiment": str(discover("summary.json")),
    "code_commit": CODE_COMMIT, "model": MODEL_ID,
    "model_revision": HF_MODEL_REVISION, "seed": SEED,
    "ranks": RANKS, "selection_sweep": selection_sweep,
    "test_sweep": test_sweep,
    "random_null_select": random_null_select,
    "random_null_test": random_null_test,
    "causal_answer_completion": causal_results,
    "paired_bootstrap_95ci": paired_intervals,
    "conclusion": conclusion,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
torch.save({
    "centre": centre.float(), "eigenvalues": eigenvalues.float(),
    "selected_basis": selected_basis.float(), "selected_rank": selected_rank,
    "selected_eigenvector_indices": selected_indices,
    "model": MODEL_ID, "model_revision": HF_MODEL_REVISION,
    "code_commit": CODE_COMMIT, "seed": SEED,
}, pathlib.Path(OUTPUT_ROOT) / "localized_answer_eigenspace.pt")
print(json.dumps({
    "conclusion": conclusion,
    "causal_answer_completion": causal_results,
    "paired_bootstrap_95ci": paired_intervals,
}, indent=2))
print("wrote", summary_path)
''')

markdown(r"""
## Takeaways

Interpret “directions” as a fitted Qwen subspace at one precise location—not individual
neurons and not directions copied from CODI. A successful keep intervention establishes
sufficiency. A successful remove intervention establishes necessity. If keep succeeds
but remove does not, the model contains redundant answer information outside the chosen
subspace.

This experiment does not claim that the selected rank can replace the LM head throughout
reasoning. The previous global-generation run already showed that endpoint directions
and trajectory-wide compression are different problems.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
