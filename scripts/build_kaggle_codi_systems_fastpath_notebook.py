"""Build the Kaggle notebook for the CODI end-to-end systems fast path."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_codi_systems_fastpath.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# CODI systems fast path: optimize the transformer, not only the LM head

The previous low-rank head reduced the head by roughly 5x but improved complete
CODI inference by only about 6%. This experiment attacks the larger costs:

1. skip vocabulary projections during prompt and continuous-latent passes, where
   their logits are discarded;
2. merge rank-128 LoRA adapters into their GPT-2 matrices;
3. tokenize once and batch questions of similar lengths;
4. use FP16 Tensor Core execution;
5. remove one empirically redundant latent pass (`M=6 -> M=5`);
6. separately test length bucketing, whose changed left-padding layout can affect
   the released GPT-2 path;
7. optionally restrict the GSM8K output vocabulary to tokenizer-defined numeric
   pieces; and
8. probe `torch.compile` as an explicitly optional, failure-safe arm.

The notebook separates exact engineering changes from numerical and task-specific
changes. Every arm runs on the complete 1,319-example GSM8K test. Output agreement,
accuracy, token counts, batch-1 latency, and batch-8/32 throughput are all reported.
No arm is selected or modified after test results are observed.
""")

markdown("## 1. Configuration and immutable repository checkout")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replaced with an immutable commit after this notebook is committed.
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
CHECKPOINT_INPUT = ""  # Optional local checkpoint; otherwise download the pinned file.
OUTPUT_ROOT = "/kaggle/working/codi_systems_fastpath"

SEED = 20260905
MAX_NEW_TOKENS = 64
FULL_EVAL_BATCH_SIZE = 32
TIMING_QUESTIONS = 128
TIMING_REPEATS = 3
TIMING_BATCH_SIZES = (1, 8, 32)
TRY_TORCH_COMPILE = True
COMPILE_MODE = "reduce-overhead"
RUN_NUMERIC_SHORTLIST = True

import glob, json, os, pathlib, random, statistics, subprocess, sys, time
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

markdown("## 2. Reproducible CODI environment")
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
observed_packages = {name: installed_package_version(name) for name in PINNED_PACKAGES}
assert observed_packages == PINNED_PACKAGES, observed_packages

# Kaggle sometimes preinstalls an old torchao that PEFT probes and rejects.
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
print(observed_packages)
''')

markdown("## 3. Static implementation tests")
code(r'''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_official_codi_fast.py", "tests/test_official_codi.py"],
    check=True,
)
''')

markdown("## 4. Resolve the completed official-CODI reproduction input")
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
print("reproduction summary:", REPRODUCTION_SUMMARY)
''')

markdown("## 5. Load the frozen model and datasets")
code(r'''
import torch
from transformers import AutoTokenizer
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set, load_train_set
from src.inference.official_codi_fast import (
    FastCODIGeneration, generate_official_codi_fast, merge_official_codi_lora_,
    numeric_vocabulary_candidates, prepare_official_codi_batches,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint, generate_official_codi,
    load_official_checkpoint, official_codi_base_model, resolve_torch_dtype,
)
from src.utils.config import load_config

assert torch.cuda.is_available(), "This benchmark requires a Kaggle GPU"
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda")
cfg = load_config("configs/official_codi_gpt2.yaml")
verify_full_reproduction_gate(pathlib.Path(REPRODUCTION_SUMMARY), cfg)

if CHECKPOINT_INPUT:
    checkpoint = pathlib.Path(CHECKPOINT_INPUT)
else:
    local = sorted(glob.glob(f"/kaggle/input/**/{cfg.checkpoint.filename}", recursive=True))
    checkpoint = pathlib.Path(local[0]) if local else download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
        token=os.environ.get("HF_TOKEN") or None,
    )

model, slow_tokenizer = build_official_codi_gpt2(
    base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
    dtype=torch.float32, settings=cfg.model, token=os.environ.get("HF_TOKEN") or None,
)
load_report = load_official_checkpoint(
    model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
)
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.to(device=device, dtype=torch.float32).eval()

data_cfg = load_config(str(cfg.data_config))
test_examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
assert len(test_examples) == int(cfg.eval.expected_counts.gsm8k) == 1319
train_examples = load_train_set(load_config("configs/data.yaml"), trace_style="eq_only")
test_questions = [str(row["question"]) for row in test_examples]
rng = random.Random(SEED)
timing_indices = rng.sample(range(len(train_examples)), TIMING_QUESTIONS)
timing_questions = [str(train_examples[index]["question"]) for index in timing_indices]
warmup_questions = timing_questions[:32]
print({"device": torch.cuda.get_device_name(device), "torch": torch.__version__,
       "test": len(test_questions), "timing": len(timing_questions),
       "checkpoint": load_report.checkpoint_sha256})
''')

markdown("## 6. Verify and enable the fast GPT-2 tokenizer")
code(r'''
fast_tokenizer = AutoTokenizer.from_pretrained(
    str(cfg.model.base_model), revision=str(cfg.model.base_revision),
    model_max_length=int(cfg.model.model_max_length), padding_side="left", use_fast=True,
    token=os.environ.get("HF_TOKEN") or None,
)
if fast_tokenizer.pad_token_id is None:
    fast_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
assert fast_tokenizer.pad_token_id == model.pad_token_id

def token_ids(tokenizer, questions, step=256):
    values = []
    for start in range(0, len(questions), step):
        values.extend(tokenizer(
            [str(value).strip().replace("  ", " ") for value in questions[start:start + step]],
            add_special_tokens=False, padding=False,
        )["input_ids"])
    return values

slow_ids = token_ids(slow_tokenizer, test_questions + timing_questions)
fast_ids = token_ids(fast_tokenizer, test_questions + timing_questions)
assert slow_ids == fast_ids, "Fast tokenizer changed an input tokenization"
slow_cue = slow_tokenizer(" The answer is:", add_special_tokens=False)["input_ids"]
fast_cue = fast_tokenizer(" The answer is:", add_special_tokens=False)["input_ids"]
assert slow_cue == fast_cue
print({"tokenizer_parity": True, "cue_ids": slow_cue})
''')

markdown("## 7. Frozen arms and timing utilities")
code(r'''
from dataclasses import dataclass

@dataclass(frozen=True)
class Decoded:
    texts: tuple
    counts: tuple

def reference_decode(questions, batch_size):
    texts, metadata = generate_official_codi(
        model, slow_tokenizer, list(questions),
        latent_iterations=6, max_new_tokens=MAX_NEW_TOKENS,
        batch_size=int(batch_size), device=device,
        answer_cue="The answer is:", force_answer_cue=True,
        return_endpoint_metadata=True,
    )
    return Decoded(tuple(texts), tuple(metadata["generated_token_counts"]))

def synchronize():
    torch.cuda.synchronize()

def timed(function):
    synchronize()
    started = time.perf_counter()
    value = function()
    synchronize()
    return value, time.perf_counter() - started

def summarize_samples(samples, questions, tokens):
    median = float(statistics.median(samples))
    return {
        "samples_seconds": [float(value) for value in samples],
        "median_seconds": median,
        "milliseconds_per_question": 1e3 * median / questions,
        "questions_per_second": questions / median,
        "microseconds_per_visible_token": 1e6 * median / max(1, tokens),
    }

def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

arms = {}
outcomes = {}
reference_texts = None

def record_arm(name, decoded, elapsed, preparation_seconds, timing, notes):
    global reference_texts
    correct = [bool(answers_match(text, row["gold"]))
               for text, row in zip(decoded.texts, test_examples)]
    if reference_texts is None:
        reference_texts = tuple(decoded.texts)
    agreement = sum(left == right for left, right in zip(decoded.texts, reference_texts))
    service_elapsed = float(elapsed + preparation_seconds)
    arms[name] = {
        "examples": len(test_examples), "correct": int(sum(correct)),
        "numeric_exact_match": float(sum(correct) / len(correct)),
        "output_string_agreement_with_reference": agreement / len(test_examples),
        "generated_tokens": int(sum(decoded.counts)),
        "mean_generated_tokens": float(sum(decoded.counts) / len(decoded.counts)),
        "full_test_model_seconds": float(elapsed),
        "full_test_preparation_seconds": float(preparation_seconds),
        "full_test_service_seconds": service_elapsed,
        "full_test_service_ms_per_question": 1e3 * service_elapsed / len(test_examples),
        "timing": timing, "notes": notes,
    }
    outcomes[name] = correct
    write_jsonl(
        pathlib.Path(OUTPUT_ROOT) / f"{name}.jsonl",
        [{"arm": name, "row": index, "gold": str(row["gold"]),
          "generation": text, "correct": outcome, "generated_tokens": int(count)}
         for index, (row, text, outcome, count) in enumerate(
             zip(test_examples, decoded.texts, correct, decoded.counts))],
    )
    print(name, arms[name])

def evaluate_reference_arm(name):
    _ = reference_decode(warmup_questions, FULL_EVAL_BATCH_SIZE)
    decoded, elapsed = timed(lambda: reference_decode(test_questions, FULL_EVAL_BATCH_SIZE))
    timing = {}
    for batch_size in TIMING_BATCH_SIZES:
        _ = reference_decode(warmup_questions[:max(1, batch_size)], batch_size)
        samples = []
        last = None
        for _ in range(TIMING_REPEATS):
            last, seconds = timed(lambda b=batch_size: reference_decode(timing_questions, b))
            samples.append(seconds)
        timing[str(batch_size)] = summarize_samples(
            samples, len(timing_questions), sum(last.counts)
        )
    record_arm(name, decoded, elapsed, 0.0, timing,
               ["released eager decoder", "FP32", "unmerged LoRA", "M=6"])

def evaluate_fast_arm(name, tokenizer, *, latent_iterations, length_bucketed,
                      candidate_token_ids=None, notes=()):
    prepared_test = prepare_official_codi_batches(
        tokenizer, test_questions, batch_size=FULL_EVAL_BATCH_SIZE,
        length_bucketed=length_bucketed,
    )
    prepared_warmup = prepare_official_codi_batches(
        tokenizer, warmup_questions, batch_size=FULL_EVAL_BATCH_SIZE,
        length_bucketed=length_bucketed,
    )
    _ = generate_official_codi_fast(
        model, tokenizer, prepared_warmup, latent_iterations=latent_iterations,
        max_new_tokens=MAX_NEW_TOKENS, device=device,
        candidate_token_ids=candidate_token_ids,
    )
    decoded, elapsed = timed(lambda: generate_official_codi_fast(
        model, tokenizer, prepared_test, latent_iterations=latent_iterations,
        max_new_tokens=MAX_NEW_TOKENS, device=device,
        candidate_token_ids=candidate_token_ids,
    ))
    timing = {}
    for batch_size in TIMING_BATCH_SIZES:
        prepared = prepare_official_codi_batches(
            tokenizer, timing_questions, batch_size=batch_size,
            length_bucketed=length_bucketed,
        )
        warm = prepare_official_codi_batches(
            tokenizer, warmup_questions[:max(1, batch_size)], batch_size=batch_size,
            length_bucketed=length_bucketed,
        )
        _ = generate_official_codi_fast(
            model, tokenizer, warm, latent_iterations=latent_iterations,
            max_new_tokens=MAX_NEW_TOKENS, device=device,
            candidate_token_ids=candidate_token_ids,
        )
        model_samples = []
        last = None
        for _ in range(TIMING_REPEATS):
            last, seconds = timed(lambda p=prepared: generate_official_codi_fast(
                model, tokenizer, p, latent_iterations=latent_iterations,
                max_new_tokens=MAX_NEW_TOKENS, device=device,
                candidate_token_ids=candidate_token_ids,
            ))
            model_samples.append(seconds)
        service_samples = [value + prepared.preparation_seconds for value in model_samples]
        timing[str(batch_size)] = {
            "model": summarize_samples(model_samples, len(timing_questions),
                                       last.generated_token_count),
            "service": summarize_samples(service_samples, len(timing_questions),
                                         last.generated_token_count),
            "preparation_seconds": prepared.preparation_seconds,
            "tokenization_seconds": prepared.tokenization_seconds,
            "padded_prompt_tokens": prepared.padded_prompt_tokens,
            "unpadded_prompt_tokens": prepared.unpadded_prompt_tokens,
        }
    record_arm(
        name, Decoded(decoded.texts, decoded.generated_token_counts), elapsed,
        prepared_test.preparation_seconds, timing, list(notes),
    )
''')

markdown("## 8. B0 — released FP32 reference")
code(r'''
evaluate_reference_arm("b0_reference_fp32_lora_m6")
assert abs(arms["b0_reference_fp32_lora_m6"]["numeric_exact_match"] - 0.433662) <= 0.015
''')

markdown("## 9. B1 — skip every unused vocabulary projection")
code(r'''
evaluate_fast_arm(
    "b1_body_only_fp32_lora_m6", slow_tokenizer,
    latent_iterations=6, length_bucketed=False,
    notes=["no lm_head during prompt/latent passes", "unmerged LoRA", "FP32", "M=6"],
)
assert arms["b1_body_only_fp32_lora_m6"]["output_string_agreement_with_reference"] == 1.0, (
    "Body-only execution changed outputs before any numerical optimization"
)
''')

markdown("## 10. B2 — merge LoRA into GPT-2")
code(r'''
merge_official_codi_lora_(model)
assert not hasattr(model.codi, "peft_config")
evaluate_fast_arm(
    "b2_body_only_fp32_merged_m6", slow_tokenizer,
    latent_iterations=6, length_bucketed=False,
    notes=["body-only", "merged LoRA", "FP32", "M=6"],
)
''')

markdown("## 11. B3 — exact fast tokenizer path, original batch layout")
code(r'''
evaluate_fast_arm(
    "b3_exact_fastpath_fp32_m6", fast_tokenizer,
    latent_iterations=6, length_bucketed=False,
    notes=["body-only", "merged LoRA", "fast tokenizer parity checked",
           "original batch composition", "FP32", "M=6"],
)
''')

markdown("## 12. B4 — length bucketing, isolated before precision changes")
code(r'''
evaluate_fast_arm(
    "b4_bucketed_fp32_m6", fast_tokenizer,
    latent_iterations=6, length_bucketed=True,
    notes=["B3 plus length bucketing", "padding/cache layout can change outputs",
           "quality-checked rather than labeled lossless"],
)
''')

markdown("## 13. B5 — FP16 Tensor Core path")
code(r'''
model.to(device=device, dtype=torch.float16).eval()
evaluate_fast_arm(
    "b5_fastpath_fp16_m6", fast_tokenizer,
    latent_iterations=6, length_bucketed=True,
    notes=["B4 plus FP16"],
)
''')

markdown("## 14. B6 — five latent thoughts")
code(r'''
evaluate_fast_arm(
    "b6_fastpath_fp16_m5", fast_tokenizer,
    latent_iterations=5, length_bucketed=True,
    notes=["B5 with one fewer complete 12-block latent pass", "primary deployment candidate"],
)
''')

markdown("## 15. B7 — task-specific numeric vocabulary shortlist")
code(r'''
numeric_candidates = numeric_vocabulary_candidates(
    fast_tokenizer, vocabulary_stop=int(model.eot_id)
)
print({"eligible_full_vocabulary": int(model.eot_id),
       "numeric_candidates": len(numeric_candidates),
       "fraction": len(numeric_candidates) / int(model.eot_id)})
if RUN_NUMERIC_SHORTLIST:
    evaluate_fast_arm(
        "b7_fastpath_fp16_m5_numeric", fast_tokenizer,
        latent_iterations=5, length_bucketed=True,
        candidate_token_ids=numeric_candidates,
        notes=["B6 plus tokenizer-semantic numeric-only candidate vocabulary",
               "task-specific and explicitly not a general LM optimization"],
    )
''')

markdown("## 16. B8 — optional compiler probe")
code(r'''
compile_report = {"requested": TRY_TORCH_COMPILE, "status": "not_requested"}
if TRY_TORCH_COMPILE:
    base = official_codi_base_model(model)
    eager_transformer = base.transformer
    try:
        base.transformer = torch.compile(
            eager_transformer, mode=COMPILE_MODE, dynamic=True
        )
        evaluate_fast_arm(
            "b8_compiled_fastpath_fp16_m5", fast_tokenizer,
            latent_iterations=5, length_bucketed=True,
            notes=["B5 plus torch.compile", f"mode={COMPILE_MODE}",
                   "dynamic legacy cache; exploratory"],
        )
        compile_report = {"requested": True, "status": "completed", "mode": COMPILE_MODE}
    except Exception as error:
        base.transformer = eager_transformer
        compile_report = {
            "requested": True, "status": "unsupported_or_failed_safely",
            "mode": COMPILE_MODE, "error_type": type(error).__name__,
            "error": str(error)[:2000],
        }
        print(compile_report)
''')

markdown("## 17. Speedups, paired accuracy intervals, and decision gates")
code(r'''
def paired_interval(left, right, seed, samples=5000):
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    while len(draws) < samples:
        count = min(250, samples - len(draws))
        indices = torch.randint(len(delta), (count, len(delta)), generator=generator)
        draws.extend(delta[indices].mean(dim=1).tolist())
    values = torch.tensor(draws)
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]

baseline_name = "b0_reference_fp32_lora_m6"
baseline = arms[baseline_name]
for name, result in arms.items():
    result["accuracy_retained_fraction"] = (
        result["numeric_exact_match"] / baseline["numeric_exact_match"]
    )
    result["full_test_service_speedup_x"] = (
        baseline["full_test_service_seconds"] / result["full_test_service_seconds"]
    )
    result["timing_speedup_x"] = {}
    for batch_size in TIMING_BATCH_SIZES:
        key = str(batch_size)
        base_seconds = baseline["timing"][key]["median_seconds"]
        current = result["timing"][key]
        current_seconds = (
            current["service"]["median_seconds"] if "service" in current
            else current["median_seconds"]
        )
        result["timing_speedup_x"][key] = base_seconds / current_seconds

paired = {
    f"{name}_minus_reference": paired_interval(
        outcomes[name], outcomes[baseline_name], SEED + offset
    )
    for offset, name in enumerate(arms) if name != baseline_name
}

exact = arms["b3_exact_fastpath_fp32_m6"]
candidate = arms["b6_fastpath_fp16_m5"]
validity = {
    "complete_1319_test": len(test_examples) == 1319,
    "reference_reproduces_43_366_within_1_5_points":
        abs(baseline["numeric_exact_match"] - 0.433662) <= 0.015,
    "body_only_unmerged_exact_output_parity":
        arms["b1_body_only_fp32_lora_m6"]["output_string_agreement_with_reference"] == 1.0,
}
decision = {
    "exact_fastpath": {
        "output_agreement": exact["output_string_agreement_with_reference"],
        "accuracy_retained": exact["accuracy_retained_fraction"],
        "batch1_speedup": exact["timing_speedup_x"]["1"],
        "batch32_speedup": exact["timing_speedup_x"]["32"],
        "target_supported": bool(
            exact["output_string_agreement_with_reference"] == 1.0
            and exact["timing_speedup_x"]["1"] >= 1.20
        ),
    },
    "fp16_m5_candidate": {
        "output_agreement": candidate["output_string_agreement_with_reference"],
        "accuracy_retained": candidate["accuracy_retained_fraction"],
        "accuracy_difference_95ci": paired["b6_fastpath_fp16_m5_minus_reference"],
        "batch1_speedup": candidate["timing_speedup_x"]["1"],
        "batch32_speedup": candidate["timing_speedup_x"]["32"],
        "quality_gate_98pct": candidate["accuracy_retained_fraction"] >= 0.98,
        "speed_gate_1_5x_batch1": candidate["timing_speedup_x"]["1"] >= 1.50,
    },
}
decision["fp16_m5_candidate"]["joint_target_supported"] = bool(
    decision["fp16_m5_candidate"]["quality_gate_98pct"]
    and decision["fp16_m5_candidate"]["speed_gate_1_5x_batch1"]
)
print(json.dumps({"validity": validity, "decision": decision}, indent=2))
''')

markdown("## 18. Profiler trace for the primary candidate")
code(r'''
profile_report = {"status": "not_run"}
try:
    profile_questions = timing_questions[:32]
    profile_prepared = prepare_official_codi_batches(
        fast_tokenizer, profile_questions, batch_size=32, length_bucketed=True
    )
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        _ = generate_official_codi_fast(
            model, fast_tokenizer, profile_prepared, latent_iterations=5,
            max_new_tokens=MAX_NEW_TOKENS, device=device,
        )
    trace_path = pathlib.Path(OUTPUT_ROOT) / "fp16_m5_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=40
    )
    (pathlib.Path(OUTPUT_ROOT) / "profiler_top_cuda.txt").write_text(table + "\n")
    profile_report = {"status": "completed", "trace": str(trace_path), "top_cuda": table}
    print(table)
except Exception as error:
    profile_report = {"status": "failed_safely", "error_type": type(error).__name__,
                      "error": str(error)[:2000]}
    print(profile_report)
''')

markdown("## 19. Save the complete experiment artifact")
code(r'''
summary = {
    "experiment": "official_codi_systems_fastpath_v1",
    "code_commit": CODE_COMMIT,
    "checkpoint_sha256": load_report.checkpoint_sha256,
    "environment": {**observed_packages, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device)},
    "input": {"reproduction_summary": REPRODUCTION_SUMMARY,
              "checkpoint": str(checkpoint)},
    "protocol": {
        "seed": SEED, "full_test_examples": len(test_examples),
        "timing_questions": len(timing_questions), "timing_repeats": TIMING_REPEATS,
        "timing_batch_sizes": list(TIMING_BATCH_SIZES),
        "max_new_tokens": MAX_NEW_TOKENS,
        "timing_interpretation": {
            "batch_1": "single-request latency",
            "batch_8_and_32": "throughput-normalized latency, not single-request latency",
        },
    },
    "arms": arms,
    "paired_bootstrap_95ci_accuracy_difference": paired,
    "validity": validity,
    "decision": decision,
    "compile": compile_report,
    "numeric_shortlist": {
        "enabled": RUN_NUMERIC_SHORTLIST,
        "candidate_count": int(len(numeric_candidates)),
        "eligible_vocabulary": int(model.eot_id),
    },
    "profiler": {key: value for key, value in profile_report.items() if key != "top_cuda"},
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
print("wrote", summary_path)
print(json.dumps({name: {
    "accuracy": result["numeric_exact_match"],
    "retained": result["accuracy_retained_fraction"],
    "batch1_speedup": result["timing_speedup_x"]["1"],
    "batch32_speedup": result["timing_speedup_x"]["32"],
    "output_agreement": result["output_string_agreement_with_reference"],
} for name, result in arms.items()}, indent=2))
''')

markdown(r"""
## Interpretation rules

- **B1 must exactly reproduce B0.** If it does not, direct transformer-body execution
  is not semantically equivalent in the installed environment; stop interpretation.
- B2 and B3 are intended as lossless engineering changes, but floating-point kernel
  order can still change rare argmax ties. Report observed agreement rather than hiding it.
- B4 changes left-padding and cache layout and must be quality-checked. B5 changes
  numerical precision. B6 additionally changes the reasoning budget. Their
  primary quality threshold is at least 98% of B0 exact-match accuracy.
- B7 is a GSM8K-specific experiment. It cannot support a claim about general language
  modeling, even if it is fast and accurate here.
- Batch 1 measures request latency. Batch 8 and 32 primarily measure throughput. Never
  describe throughput-normalized milliseconds per question as single-request latency.
- Compilation is optional because the pinned CODI implementation uses a hand-written
  growing legacy cache. A safe compiler failure is a result, not permission to change
  package versions after observing test outcomes.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
