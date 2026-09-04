"""Build the Kaggle notebook for deployment benchmarking the global rank-96 head."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_global_head_deployment_benchmark.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# CODI rank-96 global head: deployment-speed experiment

## Question

The preceding experiment established the algorithmic result: a single rank-96 head
used at every visible answer position retained 563/572 = 98.43% of the full head's
GSM8K accuracy while reducing the head from 38.60M to 4.90M MACs per token. It did
not produce wall-clock acceleration with two ordinary PyTorch matrix multiplications.

This notebook asks the unresolved systems question:

> Can compilation or a projection-plus-argmax Triton kernel turn the rank-96
> arithmetic reduction into a real batch-1 CODI latency improvement?

It reuses the **frozen rank-96 artifact**. There is no fitting, rank selection, or
test-set adaptation in this notebook. All arms use the same merged-LoRA, FP16 CODI
body-only decoder and the same greedy vocabulary boundary.

## Preregistered primary gate

- full-test accuracy retention at least 98%;
- median batch-1 microseconds per question at least 1.10x faster than the dense arm;
- no increase in peak temporary CUDA allocation during generation.

The dense, eager rank-96, compiled rank-96, and Triton rank-96 arms are also measured
at batch sizes 1, 8, and 32. Component latency, deployed model bytes, temporary CUDA
memory, exact token agreement, analytical MACs/weight bytes, and profiler diagnostics
are reported separately. The Triton arm fuses the expensive rank-to-vocabulary
projection with blockwise argmax and never materializes the `[batch, 50,257]` logits;
a second small reduction chooses among block winners.
""")

markdown("## 1. Configuration and immutable checkout")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "b6a7ebf"  # Contains the validated rank-96 fitter and body-only decoder.
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
GLOBAL_HEAD_ARTIFACT_INPUT = ""
OUTPUT_ROOT = "/kaggle/working/codi_global_head_deployment"

SEED = 20260905
RANK = 96
TIMING_QUESTIONS = 128
TIMING_BATCH_SIZES = (1, 8, 32)
TIMING_REPEATS = 3
COMPONENT_REPEATS = 5
COMPONENT_ITERATIONS = 100
QUALITY_BATCH_SIZE = 32
MAX_NEW_TOKENS = 64
TRY_TORCH_COMPILE = True
COMPILE_MODE = "reduce-overhead"
TRY_TRITON = True
RUN_PROFILER = True
RUN_FULL_QUALITY = True

import copy, glob, json, os, pathlib, random, statistics, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"], capture_output=True,
    text=True, check=True,
).stdout.strip()
assert CODE_COMMIT.startswith(RUN_COMMIT)
pathlib.Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("commit:", CODE_COMMIT)
''')

markdown("## 2. Reproduce the validated CODI environment")
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

# PEFT probes optional torchao. Kaggle images can contain an old incompatible build.
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
    "tests/test_global_low_rank_head.py", "tests/test_official_codi_fast.py",
], check=True)
''')

markdown("## 3. Resolve and validate the two completed inputs")
code(r'''
def discover(explicit, pattern):
    if explicit:
        path = pathlib.Path(explicit)
        assert path.is_file(), path
        return path
    matches = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert matches, f"Attach a Kaggle dataset containing {pattern}"
    return pathlib.Path(sorted(matches, key=lambda value: (len(value.split('/')), value))[0])

REPRODUCTION_SUMMARY = discover(
    REPRODUCTION_SUMMARY_INPUT,
    "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
)
GLOBAL_HEAD_ARTIFACT = discover(GLOBAL_HEAD_ARTIFACT_INPUT, "global_low_rank_head.pt")
GLOBAL_HEAD_SUMMARY = GLOBAL_HEAD_ARTIFACT.with_name("summary.json")
assert GLOBAL_HEAD_SUMMARY.is_file(), (
    "global_low_rank_head.pt and its summary.json must come from the same prior output"
)

import torch
prior_artifact = torch.load(GLOBAL_HEAD_ARTIFACT, map_location="cpu", weights_only=False)
prior_summary = json.loads(GLOBAL_HEAD_SUMMARY.read_text(encoding="utf-8"))
assert prior_artifact["contract"] == "trajectory_whitened_margin_distilled_global_lm_head_v1"
assert prior_summary["experiment"] == "trajectory_whitened_margin_distilled_global_lm_head_v1"
assert RANK in tuple(int(value) for value in prior_artifact["ranks"])
prior_full = prior_summary["generation"]["full"]
prior_rank = prior_summary["generation"][f"whitened_margin_onpolicy_r{RANK}"]
prior_retention = prior_rank["numeric_exact_match"] / prior_full["numeric_exact_match"]
assert prior_retention >= 0.98, "The attached artifact is not the validated rank-96 run"
print({"reproduction": str(REPRODUCTION_SUMMARY),
       "global_head": str(GLOBAL_HEAD_ARTIFACT),
       "prior_rank96_retention": prior_retention})
''')

markdown("## 4. Load CODI, merge LoRA, and switch the shared body to FP16")
code(r'''
import torch.nn as nn
import torch.nn.functional as F
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set, load_train_set
import src.inference.official_codi_fast as fast_module
from src.inference.official_codi_fast import (
    generate_official_codi_fast, merge_official_codi_lora_,
    prepare_official_codi_batches,
)
from src.mech.global_low_rank_head import NestedLowRankVocabularyHead
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, official_codi_base_model,
)
from src.utils.config import load_config

assert torch.cuda.is_available(), "Select a Kaggle GPU accelerator"
device = torch.device("cuda")
inference_dtype = torch.float16
random.seed(SEED)
torch.manual_seed(SEED)
cfg = load_config("configs/official_codi_gpt2.yaml")
verify_full_reproduction_gate(REPRODUCTION_SUMMARY, cfg)

checkpoint_matches = sorted(glob.glob(f"/kaggle/input/**/{cfg.checkpoint.filename}", recursive=True))
checkpoint = pathlib.Path(checkpoint_matches[0]) if checkpoint_matches else download_official_checkpoint(
    repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
    filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
    token=os.environ.get("HF_TOKEN") or None,
)
model, tokenizer = build_official_codi_gpt2(
    base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
    dtype=torch.float32, settings=cfg.model, token=os.environ.get("HF_TOKEN") or None,
)
load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
for parameter in model.parameters():
    parameter.requires_grad_(False)
merge_official_codi_lora_(model)
model.to(device=device, dtype=inference_dtype).eval()
base_model = official_codi_base_model(model)
full_head = base_model.get_output_embeddings()
input_embedding = model.input_embeddings()
assert full_head.weight.data_ptr() == input_embedding.weight.data_ptr(), (
    "The dense control should retain GPT-2 weight tying"
)

hidden_size = int(model.config.hidden_size)
vocabulary_size = int(model.eot_id)
ranks = tuple(int(value) for value in prior_artifact["ranks"])
nested = NestedLowRankVocabularyHead(hidden_size, vocabulary_size, ranks)
nested.load_state_dict(prior_artifact["state_dict"])
nested.disable_adaptive()
nested.set_rank(RANK)
nested.to(device=device, dtype=inference_dtype).eval()

data_cfg = load_config(str(cfg.data_config))
test_examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
assert len(test_examples) == 1319
train_examples = load_train_set(load_config("configs/data.yaml"), trace_style="eq_only")
rng = random.Random(SEED)
timing_indices = rng.sample(range(len(train_examples)), TIMING_QUESTIONS + 32)
timing_questions = [str(train_examples[index]["question"]) for index in timing_indices[:TIMING_QUESTIONS]]
warmup_questions = [str(train_examples[index]["question"]) for index in timing_indices[TIMING_QUESTIONS:]]
test_questions = [str(row["question"]) for row in test_examples]
print({"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
       "dtype": str(inference_dtype), "hidden": hidden_size,
       "vocabulary": vocabulary_size, "rank": RANK,
       "checkpoint": load_report.checkpoint_sha256})
''')

markdown("## 5. Construct eager, compiled, and fused-argmax rank-96 heads")
code(r'''
class FixedRankHead(nn.Module):
    """Independent unembedding factors; the dense input embedding remains frozen."""
    def __init__(self, source, rank):
        super().__init__()
        self.rank = int(rank)
        self.vocabulary_size = int(source.vocabulary_size)
        self.down = nn.Linear(source.hidden_size, self.rank, bias=True)
        self.up = nn.Linear(self.rank, self.vocabulary_size, bias=True)
        with torch.no_grad():
            self.down.weight.copy_(source.down.weight[:self.rank])
            self.down.bias.copy_(source.down.bias[:self.rank])
            self.up.weight.copy_(source.up.weight[:, :self.rank])
            self.up.bias.copy_(source.up.bias)

    def forward(self, hidden):
        return self.up(self.down(hidden))

    def select_token(self, hidden, *, vocabulary_stop):
        assert int(vocabulary_stop) == self.vocabulary_size
        return self(hidden).argmax(dim=-1)


class CompiledArgmaxCore(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.down = source.down
        self.up = source.up

    def forward(self, hidden):
        return self.up(self.down(hidden)).argmax(dim=-1)


class CompiledRankHead(nn.Module):
    def __init__(self, source, mode):
        super().__init__()
        self.rank = source.rank
        self.vocabulary_size = source.vocabulary_size
        self.core = CompiledArgmaxCore(source)
        # Bypass Module registration for the OptimizedModule; `core` owns the parameters.
        object.__setattr__(self, "_compiled_core", torch.compile(
            self.core, mode=mode, fullgraph=True, dynamic=False,
        ))

    def forward(self, hidden):
        return self.core.up(self.core.down(hidden))

    def select_token(self, hidden, *, vocabulary_stop):
        assert int(vocabulary_stop) == self.vocabulary_size
        return self._compiled_core(hidden)


TRITON_AVAILABLE = False
TRITON_ERROR = None
if TRY_TRITON:
    try:
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
    except Exception as error:
        TRITON_ERROR = repr(error)

if TRITON_AVAILABLE:
    @triton.jit
    def _triton_partial_argmax_kernel(
        coordinates, up_weight, up_bias, partial_values, partial_indices,
        vocabulary_size: tl.constexpr, rank: tl.constexpr,
        blocks_per_row: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        vocabulary_block = tl.program_id(1)
        vocabulary_offsets = vocabulary_block * BLOCK_V + tl.arange(0, BLOCK_V)
        rank_offsets = tl.arange(0, BLOCK_R)
        coordinate = tl.load(
            coordinates + batch_row * rank + rank_offsets,
            mask=rank_offsets < rank, other=0.0,
        )
        weights = tl.load(
            up_weight + vocabulary_offsets[:, None] * rank + rank_offsets[None, :],
            mask=(vocabulary_offsets[:, None] < vocabulary_size) & (rank_offsets[None, :] < rank),
            other=0.0,
        )
        scores = tl.sum(weights * coordinate[None, :], axis=1)
        scores += tl.load(up_bias + vocabulary_offsets,
                          mask=vocabulary_offsets < vocabulary_size, other=0.0)
        scores = tl.where(vocabulary_offsets < vocabulary_size, scores, float("-inf"))
        local_offset = tl.argmax(scores, axis=0)
        output_offset = batch_row * blocks_per_row + vocabulary_block
        tl.store(partial_values + output_offset, tl.max(scores, axis=0))
        tl.store(partial_indices + output_offset,
                 vocabulary_block * BLOCK_V + local_offset)

    @triton.jit
    def _triton_final_argmax_kernel(
        partial_values, partial_indices, output_tokens,
        blocks_per_row: tl.constexpr, BLOCKS: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        offsets = tl.arange(0, BLOCKS)
        values = tl.load(
            partial_values + batch_row * blocks_per_row + offsets,
            mask=offsets < blocks_per_row, other=float("-inf"),
        )
        winning_block = tl.argmax(values, axis=0)
        token = tl.load(partial_indices + batch_row * blocks_per_row + winning_block)
        tl.store(output_tokens + batch_row, token)

    def triton_low_rank_argmax(coordinates, up_weight, up_bias):
        assert coordinates.ndim == 2 and up_weight.ndim == 2
        batch_size, rank = coordinates.shape
        vocabulary_size = up_weight.shape[0]
        assert up_weight.shape[1] == rank and up_bias.shape == (vocabulary_size,)
        assert coordinates.is_cuda and coordinates.is_contiguous()
        assert up_weight.is_contiguous() and up_bias.is_contiguous()
        block_v = 128
        block_r = triton.next_power_of_2(rank)
        blocks_per_row = triton.cdiv(vocabulary_size, block_v)
        partial_values = torch.empty(
            (batch_size, blocks_per_row), device=coordinates.device, dtype=torch.float32
        )
        partial_indices = torch.empty(
            (batch_size, blocks_per_row), device=coordinates.device, dtype=torch.int32
        )
        _triton_partial_argmax_kernel[(batch_size, blocks_per_row)](
            coordinates, up_weight, up_bias, partial_values, partial_indices,
            vocabulary_size=vocabulary_size, rank=rank, blocks_per_row=blocks_per_row,
            BLOCK_V=block_v, BLOCK_R=block_r, num_warps=4,
        )
        output = torch.empty(batch_size, device=coordinates.device, dtype=torch.long)
        _triton_final_argmax_kernel[(batch_size,)](
            partial_values, partial_indices, output,
            blocks_per_row=blocks_per_row,
            BLOCKS=triton.next_power_of_2(blocks_per_row), num_warps=4,
        )
        return output


class TritonRankHead(FixedRankHead):
    def select_token(self, hidden, *, vocabulary_stop):
        assert int(vocabulary_stop) == self.vocabulary_size
        coordinates = self.down(hidden).contiguous()
        return triton_low_rank_argmax(
            coordinates, self.up.weight.contiguous(), self.up.bias.contiguous()
        )


# The pinned decoder predates the deployment selector protocol. Patch only its private
# selection hook so specialized heads can return token IDs without allocating logits.
original_select_token = fast_module._select_token
def deployment_select_token(
    output_head, hidden, *, vocabulary_stop, candidate_token_ids,
    candidate_weight, candidate_bias,
):
    specialized = getattr(output_head, "select_token", None)
    if candidate_token_ids is None and callable(specialized):
        return specialized(hidden, vocabulary_stop=vocabulary_stop)
    return original_select_token(
        output_head, hidden, vocabulary_stop=vocabulary_stop,
        candidate_token_ids=candidate_token_ids, candidate_weight=candidate_weight,
        candidate_bias=candidate_bias,
    )
fast_module._select_token = deployment_select_token

eager_head = FixedRankHead(nested, RANK).to(device=device, dtype=inference_dtype).eval()
arms = {"dense_fp16": full_head, "rank96_eager": eager_head}
arm_errors = {}

if TRY_TORCH_COMPILE:
    try:
        compiled_source = FixedRankHead(nested, RANK).to(device=device, dtype=inference_dtype).eval()
        compiled_head = CompiledRankHead(compiled_source, COMPILE_MODE).eval()
        # Compilation is lazy; compile every preregistered shape before declaring the arm.
        for probe_batch in TIMING_BATCH_SIZES:
            _ = compiled_head.select_token(
                torch.zeros(probe_batch, hidden_size, device=device,
                            dtype=inference_dtype),
                vocabulary_stop=vocabulary_size,
            )
        torch.cuda.synchronize()
        arms["rank96_compiled_argmax"] = compiled_head
    except Exception as error:
        arm_errors["rank96_compiled_argmax"] = repr(error)

if TRITON_AVAILABLE:
    try:
        triton_head = TritonRankHead(nested, RANK).to(device=device, dtype=inference_dtype).eval()
        for probe_batch in TIMING_BATCH_SIZES:
            _ = triton_head.select_token(
                torch.zeros(probe_batch, hidden_size, device=device,
                            dtype=inference_dtype),
                vocabulary_stop=vocabulary_size,
            )
        torch.cuda.synchronize()
        arms["rank96_triton_argmax"] = triton_head
    except Exception as error:
        arm_errors["rank96_triton_argmax"] = repr(error)
else:
    arm_errors["rank96_triton_argmax"] = TRITON_ERROR or "Triton disabled"

# Replacing lm_head does not replace the dense input embedding. This is intentional:
# it protects token representations, but total model parameters need not decrease.
for name, candidate in arms.items():
    if name != "dense_fp16":
        candidate_weight = candidate.core.up.weight if hasattr(candidate, "core") else candidate.up.weight
        assert candidate_weight.data_ptr() != input_embedding.weight.data_ptr()
print({"available_arms": list(arms), "unsupported": arm_errors})
''')

markdown("## 6. Numerical selector check before any timing")
code(r'''
probe = torch.randn(32, hidden_size, device=device, dtype=inference_dtype)
with torch.inference_mode():
    eager_tokens = eager_head.select_token(probe, vocabulary_stop=vocabulary_size)
selector_checks = {}
for name, candidate in arms.items():
    if name in {"dense_fp16", "rank96_eager"}:
        continue
    with torch.inference_mode():
        observed = candidate.select_token(probe, vocabulary_stop=vocabulary_size)
    selector_checks[name] = {
        "matches": int((observed == eager_tokens).sum()),
        "examples": len(probe),
        "agreement": float((observed == eager_tokens).float().mean()),
    }
    # A radically different selector usually means a stride or reduction bug.
    assert selector_checks[name]["agreement"] >= 0.90, selector_checks[name]
print(selector_checks)
''')

markdown("## 7. Collect representative answer states and benchmark the head alone")
code(r'''
state_chunks = []
def capture_states(hidden, active, answer_position):
    if bool(active.any()):
        state_chunks.append(hidden[active].detach().cpu())

state_prepared = prepare_official_codi_batches(
    tokenizer, warmup_questions, batch_size=32, length_bucketed=False,
)
base_model.set_output_embeddings(full_head)
_ = generate_official_codi_fast(
    model, tokenizer, state_prepared,
    latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
    device=device, answer_cue="The answer is:", answer_state_observer=capture_states,
)
state_pool = torch.cat(state_chunks, dim=0).to(dtype=inference_dtype)
assert state_pool.ndim == 2 and state_pool.shape[1] == hidden_size

def select_from_head(head, hidden):
    selector = getattr(head, "select_token", None)
    if callable(selector):
        return selector(hidden, vocabulary_stop=vocabulary_size)
    return head(hidden)[..., :vocabulary_size].argmax(dim=-1)

def component_benchmark(head, batch_size):
    index = torch.arange(batch_size) % len(state_pool)
    hidden = state_pool.index_select(0, index).to(device=device)
    with torch.inference_mode():
        for _ in range(20):
            _ = select_from_head(head, hidden)
        torch.cuda.synchronize()
        measurements = []
        for _ in range(COMPONENT_REPEATS):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            for _ in range(COMPONENT_ITERATIONS):
                output = select_from_head(head, hidden)
            ended.record()
            torch.cuda.synchronize()
            measurements.append(1000.0 * started.elapsed_time(ended) / COMPONENT_ITERATIONS)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        output = select_from_head(head, hidden)
        torch.cuda.synchronize()
        peak_increment = max(0, torch.cuda.max_memory_allocated(device) - before)
    return {
        "batch_size": int(batch_size),
        "median_microseconds_per_call": float(statistics.median(measurements)),
        "all_microseconds_per_call": measurements,
        "peak_temporary_bytes": int(peak_increment),
        "output_shape": list(output.shape),
    }

component_results = {
    name: {str(batch): component_benchmark(head, batch)
           for batch in TIMING_BATCH_SIZES}
    for name, head in arms.items()
}
print(json.dumps(component_results, indent=2))
''')

markdown("## 8. Best-effort CUDA profiler diagnostics")
code(r'''
profiler_results = {}
if RUN_PROFILER:
    from torch.profiler import ProfilerActivity, profile, record_function
    hidden = state_pool[:1].to(device=device)
    for name, head in arms.items():
        try:
            with torch.inference_mode():
                for _ in range(10):
                    _ = select_from_head(head, hidden)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function(f"selector_{name}"):
                        for _ in range(20):
                            _ = select_from_head(head, hidden)
                    torch.cuda.synchronize()
            events = list(prof.events())
            cuda_events = [event for event in events
                           if "CUDA" in str(getattr(event, "device_type", "")).upper()]
            kernel_events = [event for event in cuda_events
                             if "memcpy" not in event.name.lower()
                             and "memset" not in event.name.lower()]
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
            table_path = pathlib.Path(OUTPUT_ROOT) / f"profiler_{name}.txt"
            table_path.write_text(table, encoding="utf-8")
            profiler_results[name] = {
                "cuda_activity_events_for_20_calls": len(cuda_events),
                "approximate_kernel_events_for_20_calls": len(kernel_events),
                "table": str(table_path),
            }
        except Exception as error:
            profiler_results[name] = {"error": repr(error)}
print(json.dumps(profiler_results, indent=2))
''')

markdown("## 9. Matched end-to-end timing at batch 1, 8, and 32")
code(r'''
prepared_timing = {
    batch: prepare_official_codi_batches(
        tokenizer, timing_questions, batch_size=batch, length_bucketed=False,
    ) for batch in TIMING_BATCH_SIZES
}
prepared_warmup = {
    batch: prepare_official_codi_batches(
        tokenizer, warmup_questions[:max(batch, 8)], batch_size=batch,
        length_bucketed=False,
    ) for batch in TIMING_BATCH_SIZES
}

def unique_module_bytes(module):
    seen = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        if key not in seen:
            seen.add(key)
            total += int(storage.nbytes())
    return total

def timed_generation(head, prepared):
    base_model.set_output_embeddings(head)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = generate_official_codi_fast(
        model, tokenizer, prepared,
        latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
        device=device, answer_cue="The answer is:",
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_increment = max(0, torch.cuda.max_memory_allocated(device) - before)
    return generated, elapsed, peak_increment

# Compile/warm every shape before starting the clock.
for batch in TIMING_BATCH_SIZES:
    for name, head in arms.items():
        base_model.set_output_embeddings(head)
        _ = generate_official_codi_fast(
            model, tokenizer, prepared_warmup[batch],
            latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
            device=device, answer_cue="The answer is:",
        )
torch.cuda.synchronize()

timing_raw = {name: {str(batch): [] for batch in TIMING_BATCH_SIZES} for name in arms}
for batch in TIMING_BATCH_SIZES:
    for repeat in range(TIMING_REPEATS):
        order = list(arms)
        random.Random(SEED + 1000 * batch + repeat).shuffle(order)
        for name in order:
            generated, elapsed, peak_increment = timed_generation(
                arms[name], prepared_timing[batch]
            )
            timing_raw[name][str(batch)].append({
                "repeat": repeat, "seconds": elapsed,
                "questions": len(timing_questions),
                "visible_tokens": generated.generated_token_count,
                "microseconds_per_question": 1e6 * elapsed / len(timing_questions),
                "microseconds_per_visible_token": (
                    1e6 * elapsed / max(1, generated.generated_token_count)
                ),
                "peak_temporary_bytes": int(peak_increment),
            })

timing_summary = {}
for name, batches in timing_raw.items():
    timing_summary[name] = {}
    for batch, rows in batches.items():
        timing_summary[name][batch] = {
            "median_microseconds_per_question": float(statistics.median(
                row["microseconds_per_question"] for row in rows
            )),
            "median_microseconds_per_visible_token": float(statistics.median(
                row["microseconds_per_visible_token"] for row in rows
            )),
            "max_peak_temporary_bytes": max(row["peak_temporary_bytes"] for row in rows),
            "repeats": rows,
        }

dense_batch1 = timing_summary["dense_fp16"]["1"]
for name in timing_summary:
    row = timing_summary[name]["1"]
    row["speedup_per_question_vs_dense"] = (
        dense_batch1["median_microseconds_per_question"] /
        row["median_microseconds_per_question"]
    )
    row["speedup_per_token_vs_dense"] = (
        dense_batch1["median_microseconds_per_visible_token"] /
        row["median_microseconds_per_visible_token"]
    )

model_footprints = {}
for name, head in arms.items():
    base_model.set_output_embeddings(head)
    model_footprints[name] = unique_module_bytes(model)
base_model.set_output_embeddings(full_head)
print(json.dumps(timing_summary, indent=2))
print("deployed model bytes:", model_footprints)
''')

markdown("## 10. Locked 1,319-question quality and exact-token evaluation")
code(r'''
def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

quality_results = {}
quality_generations = {}
if RUN_FULL_QUALITY:
    prepared_test = prepare_official_codi_batches(
        tokenizer, test_questions, batch_size=QUALITY_BATCH_SIZE,
        length_bucketed=False,
    )
    for name, head in arms.items():
        generated, elapsed, peak_increment = timed_generation(head, prepared_test)
        outcomes = [bool(answers_match(text, row["gold"]))
                    for text, row in zip(generated.texts, test_examples)]
        records = [
            {"arm": name, "row": index, "gold": str(row["gold"]),
             "generation": text, "token_ids": list(generated.token_ids[index]),
             "correct": outcomes[index],
             "generated_tokens": generated.generated_token_counts[index]}
            for index, (row, text) in enumerate(zip(test_examples, generated.texts))
        ]
        write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"quality_{name}.jsonl", records)
        quality_generations[name] = generated
        quality_results[name] = {
            "examples": len(outcomes), "correct": int(sum(outcomes)),
            "numeric_exact_match": sum(outcomes) / len(outcomes),
            "wall_clock_seconds_batch32": elapsed,
            "microseconds_per_question_batch32": 1e6 * elapsed / len(outcomes),
            "visible_tokens": generated.generated_token_count,
            "peak_temporary_bytes": int(peak_increment),
        }

    dense_accuracy = quality_results["dense_fp16"]["numeric_exact_match"]
    eager_tokens = quality_generations["rank96_eager"].token_ids
    for name, generated in quality_generations.items():
        exact = sum(left == right for left, right in zip(generated.token_ids, eager_tokens))
        quality_results[name]["exact_token_agreement_with_rank96_eager"] = exact / len(eager_tokens)
        quality_results[name]["accuracy_retention_vs_dense"] = (
            quality_results[name]["numeric_exact_match"] / dense_accuracy
        )
    # Guard against an invalid FP16/merged control rather than interpreting it as speed.
    prior_dense_accuracy = prior_full["numeric_exact_match"]
    assert abs(dense_accuracy - prior_dense_accuracy) <= 0.02, {
        "current_dense": dense_accuracy, "prior_float32_dense": prior_dense_accuracy,
    }
base_model.set_output_embeddings(full_head)
print(json.dumps(quality_results, indent=2))
''')

markdown("## 11. Analytical cost, gates, and export")
code(r'''
element_bytes = torch.tensor([], dtype=inference_dtype).element_size()
analytical = {
    "dense": {
        "macs_per_token": hidden_size * vocabulary_size,
        "head_parameter_bytes": vocabulary_size * hidden_size * element_bytes,
        "naive_weight_bytes_read_per_projection": vocabulary_size * hidden_size * element_bytes,
        "materialized_logit_bytes_batch1": vocabulary_size * element_bytes,
    },
    "rank96": {
        "macs_per_token": RANK * (hidden_size + vocabulary_size),
        "head_parameter_bytes": RANK * (hidden_size + vocabulary_size) * element_bytes
                                + (RANK + vocabulary_size) * element_bytes,
        "naive_weight_bytes_read_per_projection": RANK * (hidden_size + vocabulary_size)
                                                   * element_bytes,
        "materialized_logit_bytes_batch1_eager": vocabulary_size * element_bytes,
        "partial_argmax_bytes_batch1_triton": (
            2 * ((vocabulary_size + 127) // 128) * 4
        ),
    },
}
analytical["head_mac_reduction"] = (
    analytical["dense"]["macs_per_token"] / analytical["rank96"]["macs_per_token"]
)

gates = {}
if quality_results:
    dense_peak = timing_summary["dense_fp16"]["1"]["max_peak_temporary_bytes"]
    for name in arms:
        if name == "dense_fp16":
            continue
        retention = quality_results[name]["accuracy_retention_vs_dense"]
        speedup = timing_summary[name]["1"]["speedup_per_question_vs_dense"]
        peak = timing_summary[name]["1"]["max_peak_temporary_bytes"]
        gates[name] = {
            "retains_98_percent_accuracy": retention >= 0.98,
            "batch1_speedup_at_least_1_10x": speedup >= 1.10,
            "temporary_peak_no_greater_than_dense": peak <= dense_peak,
            "passes_all_primary_gates": (
                retention >= 0.98 and speedup >= 1.10 and peak <= dense_peak
            ),
        }

eligible = [name for name, row in gates.items() if row["passes_all_primary_gates"]]
selected = max(
    eligible,
    key=lambda name: timing_summary[name]["1"]["speedup_per_question_vs_dense"],
    default=None,
)
summary = {
    "experiment": "deployment_low_rank_argmax_v1",
    "code_commit": CODE_COMMIT,
    "checkpoint_sha256": load_report.checkpoint_sha256,
    "prior_artifact": {
        "path": str(GLOBAL_HEAD_ARTIFACT),
        "contract": prior_artifact["contract"],
        "code_commit": prior_artifact["code_commit"],
        "prior_rank96_accuracy_retention": prior_retention,
    },
    "environment": {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "dtype": str(inference_dtype),
        "packages": {name: installed_package_version(name) for name in PINNED_PACKAGES},
    },
    "protocol": {
        "rank": RANK, "timing_questions": TIMING_QUESTIONS,
        "timing_batch_sizes": list(TIMING_BATCH_SIZES),
        "timing_repeats": TIMING_REPEATS, "quality_questions": len(test_examples),
        "quality_batch_size": QUALITY_BATCH_SIZE,
        "same_body_only_decoder": True, "merged_lora": True,
        "input_embedding_remains_dense": True,
    },
    "available_arms": list(arms), "unsupported_arms": arm_errors,
    "selector_checks": selector_checks,
    "component": component_results, "profiler": profiler_results,
    "end_to_end_timing": timing_summary,
    "quality": quality_results, "deployed_model_bytes": model_footprints,
    "analytical": analytical, "gates": gates,
    "selected_deployment_arm": selected,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
temporary = pathlib.Path(str(summary_path) + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
temporary.replace(summary_path)
print(json.dumps({"gates": gates, "selected": selected,
                  "batch1": {name: row["1"] for name, row in timing_summary.items()},
                  "quality": quality_results}, indent=2))
print("saved:", summary_path)
''')

markdown(r"""
## 12. How to interpret the result

- **Pass:** at least one rank-96 deployment arm keeps 98% accuracy, exceeds 1.10x
  batch-1 speedup, and does not increase temporary memory. The next experiment is the
  same deployment protocol on Qwen and a third model family.
- **Head-only faster but end-to-end neutral:** the LM head is no longer the dominant
  latency term. Profile the transformer/KV path; do not claim end-to-end acceleration.
- **Triton faster at batch 1 but worse at batch 32:** report the deployment regime
  honestly. Interactive decoding and throughput serving optimize different shapes.
- **Triton token disagreement:** inspect low-margin examples. FP16/Triton reduction
  order can change nearly tied logits; full-test accuracy is the deciding metric.
- **No speedup after compilation and Triton:** the low-rank structure remains a valid
  compression/representation result, but the acceleration claim is unsupported on
  this GPU and software stack.

The analytical `naive_weight_bytes_read_per_projection` is a traffic model, not a
hardware-counter measurement. CUDA profiler event counts are diagnostics, not a
substitute for Nsight Compute bandwidth counters.
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
