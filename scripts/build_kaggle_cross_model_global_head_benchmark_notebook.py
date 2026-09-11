"""Build the model-sharded Kaggle notebook for the global LM-head benchmark."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_cross_model_global_head_benchmark.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Cross-model global low-rank LM-head benchmark

## Goal

This is one preregistered experiment, executed in model-sized shards. It asks whether
the **method** used for CODI's global low-rank head transfers across architectures and
data distributions. Every model is fitted independently; numerical factors are never
copied between hidden spaces.

For a fixed rank of approximately `hidden_size / 8`, each shard compares:

1. dense full-vocabulary head;
2. frozen weight-only truncated SVD;
3. frozen ASVD-style diagonal activation scaling;
4. frozen SVD-LLM-style full activation whitening;
5. random factors followed by the full distillation objective;
6. weight-SVD initialization followed by the same distillation, isolating whitening;
7. whitened initialization and clean-state distillation, without on-policy recovery;
8. SlimSpec-style random factors trained with KL only;
9. the full whitened, margin-distilled, on-policy method.

The ASVD-style and SVD-LLM-style arms are transparent equation-level reproductions,
not wrappers around those projects' official repositories. A publication run must
cross-check these two arms against the authors' reference implementations and report
any discrepancy.

## Primary test

Fit and select on disjoint WikiText-2 splits. Evaluate without refitting on WikiText,
GSM8K, MATH-500, ARC-Challenge, MBPP, CNN/DailyMail, and Arabic XNLI. This directly
tests corpus transfer instead of learning a separate head for every benchmark.

The complete study consists of one shard for each registry entry. Save each Kaggle
run as a dataset, attach the completed datasets to a final aggregation run, set
`RUN_MODEL=False`, and execute the aggregation cells.
""")

markdown("## Configuration")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable commit printed after this notebook is pushed.
REPO_DIR = "/kaggle/working/latent-reasoning"
OUTPUT_ROOT = "/kaggle/working/cross_model_global_head"

# Change only MODEL_KEY between model shards. Qwen supplies the required two-size
# within-family comparison. Llama and Gemma require accepted HF licenses + HF_TOKEN.
MODEL_KEY = "qwen_0_5b"
RUN_MODEL = True
PROFILE = "paper"  # "smoke" validates plumbing; "paper" is the reporting run.
SEED_INDEX = 0  # Paper protocol uses 0, 1, and 2 in separate reproducible shards.
IMPORT_COMPLETED_CODI_ANCHOR = True
CODI_SUMMARY_INPUT = ""  # Optional prior trajectory-whitened CODI summary.json.

MODEL_REGISTRY = {
    "qwen_0_5b": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen", "access": "public"},
    "qwen_1_5b": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen", "access": "public"},
    "llama_1b": {"id": "meta-llama/Llama-3.2-1B-Instruct", "family": "llama", "access": "gated"},
    "gemma_2b": {"id": "google/gemma-2-2b-it", "family": "gemma", "access": "gated"},
    "smollm_1_7b": {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "family": "smollm", "access": "public"},
}
assert MODEL_KEY in MODEL_REGISTRY

if PROFILE == "smoke":
    FIT_STATES, SELECT_STATES, ONPOLICY_STATES = 128, 64, 64
    EVAL_STATES_PER_DOMAIN, GENERATION_EXAMPLES_PER_DOMAIN = 48, 1
    CLEAN_EPOCHS, ONPOLICY_EPOCHS, MAX_NEW_TOKENS = 1, 1, 8
    HEAD_BENCHMARK_ITERATIONS = 30
else:
    FIT_STATES, SELECT_STATES, ONPOLICY_STATES = 4096, 1024, 1024
    EVAL_STATES_PER_DOMAIN, GENERATION_EXAMPLES_PER_DOMAIN = 512, 8
    CLEAN_EPOCHS, ONPOLICY_EPOCHS, MAX_NEW_TOKENS = 3, 2, 64
    HEAD_BENCHMARK_ITERATIONS = 200

PAPER_SEEDS = (20260912, 20260913, 20260914)
assert 0 <= SEED_INDEX < len(PAPER_SEEDS)
SEED = PAPER_SEEDS[SEED_INDEX]
STATE_BATCH_SIZE = 4
DISTILL_BATCH_SIZE = 8
GENERATION_BATCH_SIZE = 2
MAX_SEQUENCE_TOKENS = 512
MAX_PROMPT_TOKENS = 384
TEMPERATURE = 2.0
LEARNING_RATE = 2e-4
RARE_MAX_CALIBRATION_COUNT = 2
TOP_K = 5
QUALITY_METRIC_SCHEMA = [
    "dense_perplexity", "candidate_perplexity", "perplexity_ratio",
    "teacher_kl", "top1_agreement", "topk_overlap_fraction",
    "mean_absolute_margin_error", "margin_correlation",
    "rare_dense_perplexity", "rare_candidate_perplexity",
    "rare_teacher_top1_agreement",
]

import copy, gc, glob, hashlib, json, os, pathlib, platform, random, re, subprocess, sys, time
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
MODEL_OUTPUT = pathlib.Path(OUTPUT_ROOT) / MODEL_KEY / f"seed_{SEED}"
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
print({"commit": CODE_COMMIT, "model_key": MODEL_KEY, "profile": PROFILE,
       "output": str(MODEL_OUTPUT)})
''')

markdown("## Install the pinned environment and validate the benchmark code")
code(r'''
PINNED_PACKAGES = [
    "transformers==4.56.2", "datasets==3.6.0", "accelerate==1.10.1",
    "huggingface_hub==0.34.4", "sentencepiece>=0.2,<1", "safetensors>=0.4,<1",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES], check=True)
subprocess.run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_global_low_rank_head.py",
], check=True)
''')

markdown("## Load the self-contained benchmark helpers")
code((ROOT / "src" / "mech" / "cross_model_head_benchmark.py").read_text(encoding="utf-8"))

markdown("## Load disjoint calibration and cross-distribution records")
code(r'''
from datasets import load_dataset

def take_nonempty(dataset, converter, limit):
    rows = []
    for raw in dataset:
        try:
            row = converter(raw)
        except (KeyError, TypeError, ValueError):
            continue
        if row and str(row.get("response", "")).strip():
            rows.append(row)
        if len(rows) >= int(limit):
            break
    return rows

def wikitext_record(row, domain="wikitext"):
    text = " ".join(str(row["text"]).split())
    words = text.split()
    if len(words) < 32:
        return None
    cut = max(12, min(len(words) - 8, len(words) // 2))
    return {"domain": domain, "prompt": " ".join(words[:cut]),
            "response": " ".join(words[cut:]), "gold": None, "evaluator": "none"}

wiki_train_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wiki_valid_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
wiki_test_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wiki_train_records = take_nonempty(wiki_train_raw, wikitext_record, 960)
fit_records = wiki_train_records[:768]
onpolicy_prompt_records = wiki_train_records[768:]
selection_pool = take_nonempty(wiki_valid_raw, wikitext_record, 384)
select_records = selection_pool[:256]
assert len(onpolicy_prompt_records) >= 32, (
    "WikiText train did not contain enough disjoint on-policy prompts"
)

gsm = load_dataset("openai/gsm8k", "main", split="test")
gsm_records = take_nonempty(gsm, lambda row: {
    "domain": "gsm8k", "prompt": "Solve this problem. Show concise reasoning and finish with 'The answer is: <number>'.\n\n" + str(row["question"]),
    "response": str(row["answer"]), "gold": str(row["answer"]).split("####")[-1].strip(),
    "evaluator": "numeric",
}, 128)

math_data = load_dataset("HuggingFaceH4/MATH-500", split="test")
math_records = take_nonempty(math_data, lambda row: {
    "domain": "math500", "prompt": "Solve the mathematics problem and finish with 'The answer is: <answer>'.\n\n" + str(row["problem"]),
    "response": str(row.get("solution", row.get("answer", ""))),
    "gold": str(row.get("answer", "")), "evaluator": "math_text",
}, 128)

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
def arc_record(row):
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return {"domain": "arc_challenge",
            "prompt": f"Choose the correct option. Answer with only its label.\n\n{row['question']}\n{options}",
            "response": str(row["answerKey"]), "gold": str(row["answerKey"]),
            "evaluator": "choice"}
arc_records = take_nonempty(arc, arc_record, 128)

mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
mbpp_records = take_nonempty(mbpp, lambda row: {
    "domain": "mbpp", "prompt": "Write a Python solution for this task. Return code only.\n\n" + str(row["text"]),
    "response": str(row["code"]), "gold": str(row["code"]), "evaluator": "code_exact",
}, 128)

cnn = load_dataset("cnn_dailymail", "3.0.0", split="test")
cnn_records = take_nonempty(cnn, lambda row: {
    "domain": "cnn_dailymail", "prompt": "Summarize the following article.\n\n" + str(row["article"]),
    "response": str(row["highlights"]), "gold": str(row["highlights"]), "evaluator": "rouge_l",
}, 96)

xnli = load_dataset("facebook/xnli", "ar", split="test")
xnli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
xnli_records = take_nonempty(xnli, lambda row: {
    "domain": "xnli_ar", "prompt": (
        "صنّف العلاقة إلى entailment أو neutral أو contradiction. أجب بالتصنيف فقط.\n\n"
        f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
    ), "response": xnli_labels[int(row["label"])],
    "gold": xnli_labels[int(row["label"])], "evaluator": "classification",
}, 128)

evaluation_records = {
    "wikitext": take_nonempty(wiki_test_raw, wikitext_record, 128),
    "gsm8k": gsm_records, "math500": math_records,
    "arc_challenge": arc_records, "mbpp": mbpp_records,
    "cnn_dailymail": cnn_records, "xnli_ar": xnli_records,
}
assert fit_records and select_records and all(evaluation_records.values())
print({"fit_records": len(fit_records), "select_records": len(select_records),
       "onpolicy_prompts": len(onpolicy_prompt_records),
       "evaluation": {name: len(rows) for name, rows in evaluation_records.items()}})
''')

markdown("## Resolve and load one frozen model shard")
code(r'''
if RUN_MODEL:
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    random.seed(SEED)
    model_spec = MODEL_REGISTRY[MODEL_KEY]
    hf_token = os.environ.get("HF_TOKEN") or None
    resolved_revision = HfApi(token=hf_token).model_info(model_spec["id"]).sha
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["id"], revision=resolved_revision, token=hf_token,
    )
    generation_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["id"], revision=resolved_revision, token=hf_token,
        dtype=generation_dtype, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    full_head = model.get_output_embeddings()
    input_embeddings = model.get_input_embeddings()
    hidden_size = int(full_head.weight.shape[1])
    vocabulary_size = int(full_head.weight.shape[0])
    rank = max(8, int(round(hidden_size / 8)))
    tied_embeddings = bool(input_embeddings.weight.data_ptr() == full_head.weight.data_ptr())

    def decoder_module():
        if hasattr(model, "get_decoder"):
            try:
                return model.get_decoder()
            except (AttributeError, NotImplementedError):
                pass
        for name in ("model", "transformer", "gpt_neox"):
            value = getattr(model, name, None)
            if value is not None:
                return value
        raise TypeError(f"Cannot locate decoder module for {type(model).__name__}")

    decoder = decoder_module()
    print({"model": model_spec["id"], "revision": resolved_revision,
           "hidden": hidden_size, "vocabulary": vocabulary_size,
           "rank": rank, "tied_embeddings": tied_embeddings,
           "decoder": type(decoder).__name__})
''')

markdown("## Collect response-position states without computing vocabulary logits")
code(r'''
if RUN_MODEL:
    from torch.nn.utils.rnn import pad_sequence

    def formatted_prompt(text):
        message = str(text).strip()
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": message}], tokenize=False,
                add_generation_prompt=True,
            )
        return message + "\n\nAssistant:"

    def tokenized_example(record):
        prompt = formatted_prompt(record["prompt"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = record.get("_response_ids")
        if response_ids is None:
            response_ids = tokenizer(str(record["response"]), add_special_tokens=False)["input_ids"]
        response_ids = [int(value) for value in response_ids]
        reserve = max(1, min(len(response_ids), MAX_SEQUENCE_TOKENS // 2))
        prompt_ids = prompt_ids[-min(MAX_PROMPT_TOKENS, MAX_SEQUENCE_TOKENS - reserve):]
        response_ids = response_ids[:MAX_SEQUENCE_TOKENS - len(prompt_ids)]
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]
        return prompt, prompt_ids, response_ids

    @torch.inference_mode()
    def collect_states(records, maximum_states, seed):
        order = torch.randperm(len(records), generator=torch.Generator().manual_seed(int(seed))).tolist()
        blocks, targets = [], []
        for batch_start in range(0, len(order), STATE_BATCH_SIZE):
            chosen = [records[index] for index in order[batch_start:batch_start + STATE_BATCH_SIZE]]
            examples = [tokenized_example(row) for row in chosen]
            examples = [row for row in examples if row[2]]
            if not examples:
                continue
            sequences = [torch.tensor(prompt + response, dtype=torch.long)
                         for _, prompt, response in examples]
            input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
            attention_mask = torch.zeros_like(input_ids)
            for row_index, sequence in enumerate(sequences):
                attention_mask[row_index, :len(sequence)] = 1
            output = decoder(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
            hidden = output.last_hidden_state
            for row_index, (_, prompt_ids, response_ids) in enumerate(examples):
                positions = torch.arange(len(response_ids), device=hidden.device) + len(prompt_ids) - 1
                blocks.append(hidden[row_index, positions].detach().cpu().float())
                targets.append(torch.tensor(response_ids, dtype=torch.long))
            if sum(len(block) for block in blocks) >= int(maximum_states) * 2:
                break
        assert blocks, "No predictor states were collected"
        state_values = torch.cat(blocks)
        target_values = torch.cat(targets)
        if len(state_values) > int(maximum_states):
            generator = torch.Generator().manual_seed(int(seed) + 1)
            keep = torch.randperm(len(state_values), generator=generator)[:int(maximum_states)]
            state_values, target_values = state_values[keep], target_values[keep]
        return state_values, target_values

    fit_states, fit_targets = collect_states(fit_records, FIT_STATES, SEED + 1)
    select_states, select_targets = collect_states(select_records, SELECT_STATES, SEED + 2)
    assert fit_states.shape[1] == select_states.shape[1] == hidden_size
    token_counts = torch.bincount(fit_targets, minlength=vocabulary_size)
    print({"fit_states": list(fit_states.shape), "select_states": list(select_states.shape),
           "seen_vocabulary": int((token_counts > 0).sum()),
           "unseen_vocabulary": int((token_counts == 0).sum())})
''')

markdown("## Construct the frozen and learned matched-rank controls")
code(r'''
if RUN_MODEL:
    from src.mech.global_low_rank_head import (
        NestedLowRankVocabularyHead, activation_whitened_factors,
        distil_nested_head, evaluate_nested_head,
    )

    compute_device = torch.device("cuda")
    weight_train = full_head.weight.detach().float()
    bias_parameter = getattr(full_head, "bias", None)
    bias_train = None if bias_parameter is None else bias_parameter.detach().float()

    weight_factors = randomized_scaled_svd_factors(
        weight_train, rank, readout_bias=bias_train, method="weight_only_svd",
        oversample=16, power_iterations=1, seed=SEED,
        compute_device=compute_device, compute_dtype=torch.float32,
    )
    weight_svd_head = head_from_factors(*weight_factors[:4]).to("cuda")
    diagonal_centre, diagonal_scale = diagonal_activation_scale(fit_states, exponent=0.5)
    asvd_factors = randomized_scaled_svd_factors(
        weight_train, rank, input_scale=diagonal_scale, centre=diagonal_centre,
        readout_bias=bias_train, method="asvd_style_diagonal_activation_scaling",
        oversample=16, power_iterations=1, seed=SEED,
        compute_device=compute_device, compute_dtype=torch.float32,
    )
    asvd_head = head_from_factors(*asvd_factors[:4]).to("cuda")
    whitened = activation_whitened_factors(
        fit_states, weight_train, rank, readout_bias=bias_train,
        ridge_relative=1e-4, oversample=16, power_iterations=1, seed=SEED,
        compute_device=compute_device, compute_dtype=torch.float32,
    )
    whitened_head = NestedLowRankVocabularyHead.from_whitened_factors(
        *whitened[:4], ranks=(rank,),
    ).to("cuda")

    def cpu_state(module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}

    def restored(state, dtype=torch.float32):
        candidate = NestedLowRankVocabularyHead(hidden_size, vocabulary_size, (rank,))
        candidate.load_state_dict(state)
        candidate.set_rank(rank)
        return candidate.to(device="cuda", dtype=dtype)

    snapshots = {
        "weight_svd_frozen": cpu_state(weight_svd_head),
        "asvd_style_frozen": cpu_state(asvd_head),
        "svdllm_style_whitened_frozen": cpu_state(whitened_head),
    }
    initialization_reports = {
        "weight_svd_frozen": weight_factors[4].to_dict(),
        "asvd_style_frozen": asvd_factors[4].to_dict(),
        "svdllm_style_whitened_frozen": whitened[4].to_dict(),
    }

    def distil_and_store(name, initial, *, kl_only=False, epochs=CLEAN_EPOCHS, seed=SEED):
        candidate = copy.deepcopy(initial).to("cuda", dtype=torch.float32)
        result = distil_nested_head(
            candidate, fit_states, select_states, weight_train,
            readout_bias=bias_train, epochs=epochs, batch_size=DISTILL_BATCH_SIZE,
            learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
            kl_weight=1.0, token_weight=0.0 if kl_only else 0.25,
            margin_weight=0.0 if kl_only else 0.25, nested_weight=0.0,
            minimum_margin=0.25, anchor_strength=0.0 if kl_only else 1e-5,
            seed=seed,
        )
        snapshots[name] = cpu_state(candidate)
        metrics = evaluate_nested_head(
            candidate, select_states, weight_train, readout_bias=bias_train,
            rank=rank, batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
        )
        del candidate
        gc.collect(); torch.cuda.empty_cache()
        return {"losses": list(result.losses), "best_epoch": result.best_epoch,
                "validation": metrics, "kl_only": bool(kl_only)}

    random_full = random_low_rank_head(
        weight_train, fit_states, rank, readout_bias=bias_train, seed=SEED + 10,
    )
    random_slimspec = random_low_rank_head(
        weight_train, fit_states, rank, readout_bias=bias_train, seed=SEED + 11,
    )
    training_reports = {}
    training_reports["random_full_distillation"] = distil_and_store(
        "random_full_distillation", random_full, seed=SEED + 20,
    )
    training_reports["learned_no_whitening"] = distil_and_store(
        "learned_no_whitening", weight_svd_head, seed=SEED + 21,
    )
    training_reports["whitened_clean_no_onpolicy"] = distil_and_store(
        "whitened_clean_no_onpolicy", whitened_head, seed=SEED + 22,
    )
    training_reports["slimspec_style_random_kl"] = distil_and_store(
        "slimspec_style_random_kl", random_slimspec, kl_only=True, seed=SEED + 23,
    )
    print({"frozen": list(initialization_reports), "learned": list(training_reports)})
''')

markdown("## Collect compressed-policy states and train the full method")
code(r'''
if RUN_MODEL:
    def trim_generation(ids):
        values = [int(value) for value in ids]
        if tokenizer.eos_token_id in values:
            values = values[:values.index(tokenizer.eos_token_id) + 1]
        while values and values[-1] == tokenizer.pad_token_id:
            values.pop()
        return values

    @torch.inference_mode()
    def generate_records(records, output_head, maximum_new_tokens):
        original_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        model.set_output_embeddings(output_head)
        generated_records = []
        try:
            for start in range(0, len(records), GENERATION_BATCH_SIZE):
                rows = records[start:start + GENERATION_BATCH_SIZE]
                prompts = [formatted_prompt(row["prompt"]) for row in rows]
                encoded = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=MAX_PROMPT_TOKENS,
                ).to("cuda")
                output = model.generate(
                    **encoded, max_new_tokens=int(maximum_new_tokens), do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[:, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
                for row, prompt, ids in zip(rows, prompts, output):
                    token_ids = trim_generation(ids)
                    generated_records.append({
                        **row, "_formatted_prompt": prompt, "_response_ids": token_ids,
                        "generation": tokenizer.decode(token_ids, skip_special_tokens=True),
                    })
        finally:
            model.set_output_embeddings(full_head)
            tokenizer.padding_side = original_padding
        return generated_records

    clean_head_for_rollout = restored(
        snapshots["whitened_clean_no_onpolicy"], dtype=generation_dtype,
    ).eval()
    onpolicy_generated = generate_records(
        onpolicy_prompt_records[:64], clean_head_for_rollout, MAX_NEW_TOKENS,
    )
    onpolicy_states, onpolicy_targets = collect_states(
        onpolicy_generated, ONPOLICY_STATES, SEED + 30,
    )
    recovery_states = torch.cat((fit_states, onpolicy_states), dim=0)
    full_method = restored(snapshots["whitened_clean_no_onpolicy"])
    recovery_result = distil_nested_head(
        full_method, recovery_states, select_states, weight_train,
        readout_bias=bias_train, epochs=ONPOLICY_EPOCHS,
        batch_size=DISTILL_BATCH_SIZE, learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE, kl_weight=1.0, token_weight=0.25,
        margin_weight=0.25, nested_weight=0.0, minimum_margin=0.25,
        anchor_strength=1e-5, seed=SEED + 31,
    )
    snapshots["full_whitened_distilled_onpolicy"] = cpu_state(full_method)
    training_reports["full_whitened_distilled_onpolicy"] = {
        "losses": list(recovery_result.losses), "best_epoch": recovery_result.best_epoch,
        "clean_states": len(fit_states), "onpolicy_states": len(onpolicy_states),
        "validation": evaluate_nested_head(
            full_method, select_states, weight_train, readout_bias=bias_train,
            rank=rank, batch_size=DISTILL_BATCH_SIZE, temperature=TEMPERATURE,
        ),
    }
    del clean_head_for_rollout, full_method
    gc.collect(); torch.cuda.empty_cache()
    print({"onpolicy_states": len(onpolicy_states),
           "full_validation": training_reports["full_whitened_distilled_onpolicy"]["validation"]})
''')

markdown("## Evaluate perplexity, distribution fidelity, rankings, margins, and rare tokens")
code(r'''
if RUN_MODEL:
    rare_mask = rare_token_mask_from_counts(
        token_counts, maximum_count=RARE_MAX_CALIBRATION_COUNT,
    )
    evaluation_states = {}
    for offset, (domain, records) in enumerate(evaluation_records.items()):
        evaluation_states[domain] = collect_states(
            records, EVAL_STATES_PER_DOMAIN, SEED + 100 + offset,
        )
        print(domain, [list(value.shape) for value in evaluation_states[domain]])

    metric_rows = []
    arm_names = ["dense", *snapshots.keys()]
    for arm_index, arm in enumerate(arm_names):
        candidate = full_head if arm == "dense" else restored(snapshots[arm])
        for domain, (states, targets) in evaluation_states.items():
            metrics = evaluate_head_quality(
                candidate, states, targets, weight_train,
                dense_bias=bias_train, rare_token_mask=rare_mask,
                batch_size=DISTILL_BATCH_SIZE, top_k=TOP_K,
                temperature=TEMPERATURE,
            )
            metric_rows.append({"model_key": MODEL_KEY, "arm": arm,
                                "domain": domain, **metrics})
        if arm != "dense":
            del candidate
            gc.collect(); torch.cuda.empty_cache()
    print("metric rows:", len(metric_rows))
''')

markdown("## Measure autoregressive sequence agreement and task-facing quality")
code(r'''
if RUN_MODEL:
    from src.data.answer_extract import answers_match

    def normalize_text(value):
        return " ".join(str(value).strip().lower().split())

    def math_answer(value):
        text = str(value)
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        return normalize_text(boxed[-1] if boxed else text.splitlines()[-1])

    def first_choice(value):
        match = re.search(r"\b([A-E])\b", str(value).upper())
        return None if match is None else match.group(1)

    def rouge_l_f1(prediction, reference):
        left, right = normalize_text(prediction).split(), normalize_text(reference).split()
        if not left or not right:
            return 0.0
        previous = [0] * (len(right) + 1)
        for token in left:
            current = [0]
            for column, other in enumerate(right, start=1):
                current.append(previous[column - 1] + 1 if token == other
                               else max(previous[column], current[-1]))
            previous = current
        lcs = previous[-1]
        precision, recall = lcs / len(left), lcs / len(right)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def task_score(record, generation):
        kind, gold = record["evaluator"], record.get("gold")
        if kind == "none": return None
        if kind == "numeric": return float(answers_match(generation, gold))
        if kind == "math_text": return float(math_answer(generation) == math_answer(gold))
        if kind == "choice": return float(first_choice(generation) == first_choice(gold))
        if kind == "classification": return float(normalize_text(gold) in normalize_text(generation)[:40])
        if kind == "code_exact": return float(normalize_text(generation) == normalize_text(gold))
        if kind == "rouge_l": return rouge_l_f1(generation, gold)
        raise ValueError(kind)

    generation_subset = {
        domain: rows[:GENERATION_EXAMPLES_PER_DOMAIN]
        for domain, rows in evaluation_records.items()
    }
    dense_generations = {
        domain: generate_records(rows, full_head, MAX_NEW_TOKENS)
        for domain, rows in generation_subset.items()
    }
    generation_rows = []
    generation_arms = ["dense", *snapshots.keys()]
    for arm in generation_arms:
        output = dense_generations if arm == "dense" else {
            domain: generate_records(rows, restored(snapshots[arm], dtype=generation_dtype).eval(), MAX_NEW_TOKENS)
            for domain, rows in generation_subset.items()
        }
        for domain, rows in output.items():
            dense_rows = dense_generations[domain]
            for index, (record, dense_record, observed) in enumerate(
                zip(generation_subset[domain], dense_rows, rows)
            ):
                dense_ids = dense_record["_response_ids"]
                observed_ids = observed["_response_ids"]
                common = 0
                for left, right in zip(dense_ids, observed_ids):
                    if left != right: break
                    common += 1
                generation_rows.append({
                    "model_key": MODEL_KEY, "arm": arm, "domain": domain,
                    "example": index,
                    "exact_sequence_agreement": float(observed_ids == dense_ids),
                    "common_prefix_fraction": common / max(1, len(dense_ids)),
                    "task_score": task_score(record, observed["generation"]),
                    "dense_task_score": task_score(record, dense_record["generation"]),
                    "generated_tokens": len(observed_ids),
                })
        if arm != "dense":
            gc.collect(); torch.cuda.empty_cache()
    print("generation rows:", len(generation_rows))
''')

markdown("## Benchmark isolated head latency and parameter storage")
code(r'''
if RUN_MODEL:
    latency_rows = []
    for arm in ["dense", *snapshots.keys()]:
        candidate = full_head if arm == "dense" else restored(
            snapshots[arm], dtype=generation_dtype,
        ).eval()
        for batch_size in (1, 8, 32):
            latency_rows.append({
                "model_key": MODEL_KEY, "arm": arm, "batch_size": batch_size,
                "microseconds": benchmark_head_latency(
                    candidate, hidden_size, batch_size=batch_size,
                    iterations=HEAD_BENCHMARK_ITERATIONS, device="cuda",
                    dtype=generation_dtype,
                ),
            })
        if arm != "dense":
            del candidate
            gc.collect(); torch.cuda.empty_cache()
    dense_latency = {row["batch_size"]: row["microseconds"]
                     for row in latency_rows if row["arm"] == "dense"}
    for row in latency_rows:
        row["speedup_over_dense"] = dense_latency[row["batch_size"]] / row["microseconds"]

    operations = {
        "dense_macs_per_token": head_macs(hidden_size, vocabulary_size),
        "low_rank_macs_per_token": head_macs(hidden_size, vocabulary_size, rank),
        "theoretical_mac_reduction": (
            head_macs(hidden_size, vocabulary_size) /
            head_macs(hidden_size, vocabulary_size, rank)
        ),
        "dense_output_parameters": hidden_size * vocabulary_size,
        "low_rank_output_parameters": rank * (hidden_size + vocabulary_size) + rank + vocabulary_size,
        "tied_embeddings": tied_embeddings,
        "whole_model_storage_reduction_expected": False if tied_embeddings else None,
    }
    print({"operations": operations, "latency": latency_rows})
''')

markdown("## Export the complete auditable shard")
code(r'''
if RUN_MODEL:
    from importlib.metadata import PackageNotFoundError, version as package_version

    def installed(name):
        try: return package_version(name)
        except PackageNotFoundError: return None

    def write_jsonl(path, rows):
        temporary = pathlib.Path(str(path) + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        temporary.replace(path)

    generation_summary = {}
    for arm in ["dense", *snapshots.keys()]:
        generation_summary[arm] = {}
        for domain in evaluation_records:
            rows = [row for row in generation_rows if row["arm"] == arm and row["domain"] == domain]
            scores = [row["task_score"] for row in rows if row["task_score"] is not None]
            generation_summary[arm][domain] = {
                "examples": len(rows),
                "exact_sequence_agreement": sum(row["exact_sequence_agreement"] for row in rows) / max(1, len(rows)),
                "mean_common_prefix_fraction": sum(row["common_prefix_fraction"] for row in rows) / max(1, len(rows)),
                "mean_task_score": None if not scores else sum(scores) / len(scores),
                "dense_mean_task_score": None if not scores else sum(
                    row["dense_task_score"] for row in rows if row["dense_task_score"] is not None
                ) / len(scores),
            }

    summary = {
        "experiment": "cross_model_distribution_global_low_rank_head_v1",
        "code_commit": CODE_COMMIT, "profile": PROFILE, "seed": SEED,
        "model_key": MODEL_KEY, "model_id": model_spec["id"],
        "model_family": model_spec["family"], "model_revision": resolved_revision,
        "hidden_size": hidden_size, "vocabulary_size": vocabulary_size,
        "rank": rank, "rank_fraction": rank / hidden_size,
        "tied_embeddings": tied_embeddings,
        "fit_distribution": "wikitext-2-raw-v1/train",
        "selection_distribution": "wikitext-2-raw-v1/validation",
        "evaluation_distributions": list(evaluation_records),
        "populations": {
            "fit_states": len(fit_states), "select_states": len(select_states),
            "onpolicy_states": len(onpolicy_states),
            "eval_states_per_domain_cap": EVAL_STATES_PER_DOMAIN,
            "generation_examples_per_domain": GENERATION_EXAMPLES_PER_DOMAIN,
        },
        "baseline_contract": {
            "weight_svd_frozen": "randomized truncated SVD of W; no learning",
            "asvd_style_frozen": "diagonal RMS activation scaling, SVD, inverse scaling; no learning",
            "svdllm_style_whitened_frozen": "full covariance Cholesky whitening and truncated SVD; no learning",
            "random_full_distillation": "random factors plus KL, teacher-token, and margin losses",
            "learned_no_whitening": "weight-SVD factors plus full distillation",
            "whitened_clean_no_onpolicy": "full-whitened factors plus full clean-state distillation",
            "slimspec_style_random_kl": "random factors plus KL-only training",
            "full_whitened_distilled_onpolicy": "whitened clean distillation plus compressed-policy recovery",
        },
        "reference_implementation_boundary": (
            "ASVD-style and SVD-LLM-style are equation-level local reproductions. "
            "Cross-check official repositories before publication."
        ),
        "initialization_reports": initialization_reports,
        "training_reports": training_reports,
        "quality_metrics_file": "quality_metrics.jsonl",
        "generation_metrics_file": "generation_metrics.jsonl",
        "generation_summary": generation_summary,
        "latency": latency_rows, "operations": operations,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "transformers": installed("transformers"), "datasets": installed("datasets"),
            "huggingface_hub": installed("huggingface_hub"),
        },
    }
    write_jsonl(MODEL_OUTPUT / "quality_metrics.jsonl", metric_rows)
    write_jsonl(MODEL_OUTPUT / "generation_metrics.jsonl", generation_rows)
    write_jsonl(MODEL_OUTPUT / "latency.jsonl", latency_rows)
    (MODEL_OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    torch.save({
        "experiment": summary["experiment"], "code_commit": CODE_COMMIT,
        "model_id": model_spec["id"], "model_revision": resolved_revision,
        "rank": rank, "state_dict": snapshots["full_whitened_distilled_onpolicy"],
    }, MODEL_OUTPUT / "full_method_head.pt")
    print("saved:", MODEL_OUTPUT / "summary.json")
''')

markdown("## Inspect this shard")
code(r'''
if RUN_MODEL:
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({"figure.figsize": (12, 6), "axes.grid": True,
                         "grid.alpha": 0.2, "font.size": 11})
    frame = pd.DataFrame(metric_rows)
    pivot = frame.pivot(index="arm", columns="domain", values="top1_agreement")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(f"Teacher top-1 agreement across held-out distributions — {MODEL_KEY}")
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.65 else "#17212b", fontsize=8)
    fig.colorbar(image, ax=ax, label="Agreement with dense head")
    fig.tight_layout()
    fig.savefig(MODEL_OUTPUT / "distribution_top1_heatmap.png", dpi=180, bbox_inches="tight")
    plt.show()

    latency_frame = pd.DataFrame(latency_rows)
    batch_one = latency_frame[latency_frame.batch_size == 1].set_index("arm")
    mean_quality = frame.groupby("arm", as_index=True).top1_agreement.mean()
    joined = batch_one.join(mean_quality.rename("mean_top1"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(joined.speedup_over_dense, joined.mean_top1, color="#2468a2", s=60)
    for name, row in joined.iterrows():
        ax.annotate(name, (row.speedup_over_dense, row.mean_top1), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0.98, color="#444444", linestyle="--", linewidth=1,
               label="98% mean top-1 agreement")
    ax.set_xlabel("Batch-1 isolated-head speedup over dense")
    ax.set_ylabel("Mean teacher top-1 agreement across distributions")
    ax.set_title(f"Quality–latency frontier — {MODEL_KEY}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(MODEL_OUTPUT / "quality_latency_frontier.png", dpi=180, bbox_inches="tight")
    plt.show()
''')

markdown("## Aggregate completed model shards and the prior CODI anchor")
code(r'''
import pandas as pd

summary_paths = sorted(set(
    glob.glob(f"{OUTPUT_ROOT}/**/summary.json", recursive=True) +
    glob.glob("/kaggle/input/**/cross_model_global_head/**/summary.json", recursive=True)
))
summaries = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in summary_paths]

quality_paths = sorted(set(
    glob.glob(f"{OUTPUT_ROOT}/**/quality_metrics.jsonl", recursive=True) +
    glob.glob("/kaggle/input/**/cross_model_global_head/**/quality_metrics.jsonl", recursive=True)
))
all_quality = []
for path in quality_paths:
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        all_quality.extend(json.loads(line) for line in handle if line.strip())

def discover_codi_summary():
    if CODI_SUMMARY_INPUT:
        path = pathlib.Path(CODI_SUMMARY_INPUT)
        assert path.is_file(), path
        return path
    matches = glob.glob(
        "/kaggle/input/**/trajectory*global*head*/**/summary.json", recursive=True
    ) + glob.glob(
        "/kaggle/input/**/trajectory_whitened_global_head/**/summary.json", recursive=True
    )
    return None if not matches else pathlib.Path(sorted(matches)[0])

codi_anchor = None
if IMPORT_COMPLETED_CODI_ANCHOR:
    codi_path = discover_codi_summary()
    if codi_path is not None:
        payload = json.loads(codi_path.read_text(encoding="utf-8"))
        generation = payload.get("generation", {})
        dense = generation.get("full")
        compressed = generation.get("whitened_margin_onpolicy_r96")
        latency = payload.get("head_latency_microseconds", {})
        if dense and compressed:
            codi_anchor = {
                "model_key": "official_codi_gpt2_anchor",
                "model_family": "codi_gpt2", "rank": 96,
                "dense_accuracy": dense.get("numeric_exact_match"),
                "compressed_accuracy": compressed.get("numeric_exact_match"),
                "accuracy_retention": compressed.get("numeric_exact_match") / max(1e-12, dense.get("numeric_exact_match")),
                "head_speedup": latency.get("full", 0) / max(1e-12, latency.get("rank_96", 0)),
                "cross_distribution_complete": False,
                "source": str(codi_path),
            }

manifest = {
    "experiment": "cross_model_distribution_global_low_rank_head_v1",
    "required_model_keys": list(MODEL_REGISTRY),
    "completed_model_keys": sorted({row["model_key"] for row in summaries}),
    "missing_model_keys": sorted(set(MODEL_REGISTRY) - {row["model_key"] for row in summaries}),
    "required_seeds": list(PAPER_SEEDS),
    "completed_seeds_by_model": {
        key: sorted({row.get("seed") for row in summaries if row["model_key"] == key})
        for key in MODEL_REGISTRY
    },
    "three_seed_protocol_complete": all(
        set(PAPER_SEEDS).issubset({row.get("seed") for row in summaries if row["model_key"] == key})
        for key in MODEL_REGISTRY
    ),
    "all_paper_profile": bool(summaries) and all(row.get("profile") == "paper" for row in summaries),
    "same_code_commit": len({row.get("code_commit") for row in summaries}) <= 1,
    "codi_anchor": codi_anchor,
}
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
(pathlib.Path(OUTPUT_ROOT) / "aggregate_manifest.json").write_text(
    json.dumps(manifest, indent=2, default=str), encoding="utf-8",
)
print(json.dumps(manifest, indent=2))

if all_quality:
    combined = pd.DataFrame(all_quality)
    full_arm = combined[combined.arm == "full_whitened_distilled_onpolicy"]
    table = full_arm.groupby(["model_key", "domain"], as_index=False).agg(
        top1_agreement=("top1_agreement", "mean"),
        top5_overlap=("topk_overlap_fraction", "mean"),
        perplexity_ratio=("perplexity_ratio", "mean"),
        teacher_kl=("teacher_kl", "mean"),
        rare_top1=("rare_teacher_top1_agreement", "mean"),
    )
    display(table)
    table.to_csv(pathlib.Path(OUTPUT_ROOT) / "aggregate_full_method.csv", index=False)
''')

markdown(r"""
## Decision rules and interpretation boundary

Do not claim cross-model generalization until every registry model has a `paper`
profile shard produced by the same code commit. The primary method should be compared
with every matched-rank control on every distribution; an average can hide a failure
on rare tokens, code, multilingual text, or long-form output.

Evidence for a transferable low-rank readout requires all of the following:

- competitive perplexity and task score relative to each model's own dense baseline;
- high dense-sequence and next-token agreement on corpora not used to fit the head;
- rare-token behavior reported separately;
- a measured head-speed advantage at relevant batch sizes;
- confidence intervals or repeated seeds in the final paper run;
- official-reference cross-checks for ASVD and SVD-LLM before naming them as exact
  implementation baselines.

The imported CODI result is an anchor, not a replacement for the standardized
cross-distribution protocol. It is explicitly marked incomplete until CODI receives a
model-specific adapter for the same seven distributions.
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
