"""Build the Kaggle notebook for CODI position-conditioned low-rank readouts."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_codi_position_conditioned_readout.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Do CODI's later answer tokens have their own small readout spaces?

## Exact question

The confirmed PC 4–31 intervention retained **38.06% GSM8K exact match**, versus
43.37% for the full model. That intervention changed only the state used to choose
the first visible answer token. Every later token used the original 768-dimensional
state and vocabulary head.

This experiment tests the token-by-token extension without silently changing the
claim. It compares:

1. `full`: original CODI everywhere;
2. `first_token_pc4_31_then_full`: exact computational equivalent of the confirmed
   first-token intervention, followed by the full head;
3. `same_pc4_31_everywhere`: the naive reuse of the colon basis at every answer step;
4. `fixed_position_local`: independently fitted fixed bases for positions 0, 1, 2,
   3–5, and 6+;
5. `learned_position_local`: those local heads after clean-trajectory distillation;
6. `learned_position_local_onpolicy`: the same heads after one compressed-rollout
   recovery round;
7. `permuted_position_local_onpolicy`: the identical learned experts, with the four
   later-position experts assigned to wrong buckets, controlling for stored parameter
   count and per-position online rank;
8. `learned_global_r32` and `learned_global_r64`: one learned head used everywhere.

All bases and learned weights use GSM8K training questions only. Selection states are
question-disjoint from fit and on-policy recovery states. The complete 1,319-question
test set is opened only for the final locked arms. The transformer, CODI projector,
six latent iterations, tokenizer, vocabulary, and greedy decoder stay frozen.

## Primary interpretation

Position locality is supported only if the on-policy local head beats the
same-PC4–31-everywhere arm and the learned global rank-32 arm on paired exact match,
while running faster than the full model. A local result must report its larger stored
parameter count; selecting one rank-32 expert per token reduces computation, but five
experts are stored.
""")

markdown("## Setup")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "main"  # Replace with the immutable implementation commit after pushing.
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
COLON_STATES_INPUT = ""
READOUT_INPUT = ""
OUTPUT_ROOT = "/kaggle/working/codi_position_conditioned_readout"

SEED = 20260904
FIT_QUESTIONS = 1024
SELECT_QUESTIONS = 256
ONPOLICY_QUESTIONS = 256
MAX_STATES_PER_BUCKET = 4096
POSITION_RANK = 32
GLOBAL_RANKS = (32, 64)
CLEAN_DISTILL_EPOCHS = 6
ONPOLICY_DISTILL_EPOCHS = 2
DISTILL_BATCH_SIZE = 16
LEARNING_RATE = 3e-4
TEMPERATURE = 2.0
GENERATION_BATCH_SIZE = 32
MAX_NEW_TOKENS = 64
BOOTSTRAP_SAMPLES = 5000
TEST_LIMIT = 0  # 0 means the sealed complete 1,319-question GSM8K test.
RUN_FULL_GENERATION = True

import copy, glob, json, os, pathlib, random, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
RESOLVED_COMMIT = "origin/main" if RUN_COMMIT == "main" else RUN_COMMIT
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", RESOLVED_COMMIT], check=True)
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

markdown("## Pin the environment that reproduces official CODI")
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

markdown("## Static checks")
code(r'''
subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_eigenspace_readout.py",
     "tests/test_position_conditioned_readout.py",
     "tests/test_official_codi.py"],
    check=True,
)
''')

markdown("## Resolve the two completed CODI input datasets")
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
COLON_STATES = discover(COLON_STATES_INPUT, "colon_states.pt")
READOUT = discover(READOUT_INPUT, "readout.pt")
print("reproduction:", REPRODUCTION_SUMMARY)
print("colon states:", COLON_STATES)
print("readout     :", READOUT)
''')

markdown("## Load the confirmed colon band and the frozen official model")
code(r'''
import torch
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.mech.endpoint_correctness_geometry import readout_matrix
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE, build_band_subspace, state_covariance,
)
from src.mech.eigenspace_readout import (
    LowRankVocabularyHead, benchmark_vocabulary_head, covariance_eigensystem,
    distil_low_rank_head, evaluate_head_fidelity, select_readout_aware_basis,
)
from src.mech.position_conditioned_readout import (
    DEFAULT_ANSWER_POSITION_BUCKETS, PositionConditionedVocabularyHead,
    VocabularyPrefixHead, answer_position_bucket, position_head_parameter_count,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint, generate_official_codi,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config

torch.manual_seed(SEED)
random.seed(SEED)
cache, readout_payload = load_margin_cache(pathlib.Path(COLON_STATES), pathlib.Path(READOUT))
state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
colon_calibration = cache["calibration_states"][:, state_index, :].float()
colon_centre = cache["student_mean"][ANALYTIC_STATE].float()
colon_covariance = state_covariance(colon_calibration - colon_centre.unsqueeze(0))
confirmed_band = build_band_subspace(
    covariance=colon_covariance, start=4, stop=32, state=ANALYTIC_STATE,
).basis
weight_cpu = readout_matrix(readout_payload).float()
assert confirmed_band.shape == (768, 28)

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
vocabulary_size = int(model.eot_id)
assert tuple(full_head.weight[:vocabulary_size].shape) == tuple(weight_cpu.shape)
full_prefix_head = VocabularyPrefixHead(full_head, vocabulary_size).to(device)
weight = weight_cpu.to(device)
print({"checkpoint": load_report.checkpoint_sha256, "vocabulary": vocabulary_size})
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
assert not (set(fit_indices.tolist()) & set(select_indices.tolist()))
assert not (set(fit_indices.tolist()) & set(onpolicy_indices.tolist()))
assert not (set(select_indices.tolist()) & set(onpolicy_indices.tolist()))

fit_questions = questions_for(fit_indices)
select_questions = questions_for(select_indices)
onpolicy_questions = questions_for(onpolicy_indices)
test_examples = load_eval_set("gsm8k", load_config(cfg.data_config).eval.gsm8k)
assert len(test_examples) == 1319
if TEST_LIMIT:
    test_examples = test_examples[:TEST_LIMIT]
print({"fit": len(fit_questions), "select": len(select_questions),
       "onpolicy": len(onpolicy_questions), "sealed_test": len(test_examples)})
''')

markdown("## Collect clean full-head states at every visible answer position")
code(r'''
BUCKET_NAMES = [bucket.name for bucket in DEFAULT_ANSWER_POSITION_BUCKETS]

def cap_states(states, maximum, seed):
    if len(states) <= maximum:
        return states
    generator = torch.Generator().manual_seed(int(seed))
    return states[torch.randperm(len(states), generator=generator)[:maximum]]

@torch.inference_mode()
def collect_bucket_states(questions, output_head, tag):
    captured = {name: [] for name in BUCKET_NAMES}

    def observer(states, active_mask, answer_position):
        if bool(active_mask.any()):
            name = answer_position_bucket(int(answer_position))
            captured[name].append(states[active_mask].detach().cpu().float())

    base_model.set_output_embeddings(output_head)
    try:
        generations = generate_official_codi(
            model, tokenizer, questions,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=GENERATION_BATCH_SIZE,
            device=device, answer_cue="The answer is:", force_answer_cue=True,
            answer_state_observer=observer,
        )
    finally:
        base_model.set_output_embeddings(full_head)
    result = {}
    for offset, name in enumerate(BUCKET_NAMES):
        assert captured[name], f"{tag} produced no states for {name}"
        values = torch.cat(captured[name], dim=0)
        result[name] = cap_states(values, MAX_STATES_PER_BUCKET, SEED + 100 * offset)
    print(tag, {name: len(values) for name, values in result.items()})
    return result, generations

fit_states, _ = collect_bucket_states(fit_questions, full_head, "clean_fit")
select_states, _ = collect_bucket_states(select_questions, full_head, "clean_select")
''')

markdown("## Build fixed position-local bases without using test states")
code(r'''
position_bases = {}
position_centres = {}
position_selection = {}
for offset, name in enumerate(BUCKET_NAMES):
    if name == "p0":
        basis = confirmed_band
        centre = colon_centre
        position_selection[name] = {
            "kind": "confirmed_colon_pc_band_4_31", "rank": 28,
            "indices": list(range(4, 32)),
        }
    else:
        centre, values, vectors = covariance_eigensystem(fit_states[name])
        basis, indices, scores = select_readout_aware_basis(
            weight, values, vectors, POSITION_RANK, chunk_size=32
        )
        position_selection[name] = {
            "kind": "bucket_covariance_readout_aware",
            "rank": POSITION_RANK,
            "indices": indices.tolist(),
            "selected_score_fraction": float(scores[indices].sum() / scores.sum()),
        }
    position_bases[name] = basis.cpu()
    position_centres[name] = centre.cpu()
print(json.dumps(position_selection, indent=2))

def make_low_rank_head(basis, centre):
    return LowRankVocabularyHead.from_basis(weight, basis, centre).to(device)

def clone_head_dict(heads):
    return {name: copy.deepcopy(head) for name, head in heads.items()}

fixed_position_heads = {
    name: make_low_rank_head(position_bases[name], position_centres[name])
    for name in BUCKET_NAMES
}
fixed_position_router = PositionConditionedVocabularyHead(
    fixed_position_heads, inactive_head=fixed_position_heads["p0"],
    vocabulary_size=vocabulary_size,
).to(device)

confirmed_band_head = make_low_rank_head(confirmed_band, colon_centre)
same_band_router = PositionConditionedVocabularyHead(
    {name: confirmed_band_head for name in BUCKET_NAMES},
    inactive_head=confirmed_band_head, vocabulary_size=vocabulary_size,
).to(device)
first_token_only_router = PositionConditionedVocabularyHead(
    {"p0": confirmed_band_head,
     **{name: full_prefix_head for name in BUCKET_NAMES if name != "p0"}},
    inactive_head=full_prefix_head, vocabulary_size=vocabulary_size,
).to(device)
''')

markdown("## Distil one head per position bucket")
code(r'''
learned_position_heads = clone_head_dict(fixed_position_heads)
clean_training = {}
for offset, name in enumerate(BUCKET_NAMES):
    result = distil_low_rank_head(
        learned_position_heads[name], fit_states[name], select_states[name], weight,
        epochs=CLEAN_DISTILL_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
        anchor_strength=1e-4, seed=SEED + offset,
    )
    clean_training[name] = {
        "examples": len(fit_states[name]), "rank": learned_position_heads[name].rank,
        "losses": list(result.losses), "best_validation_kl": result.best_validation_kl,
        "best_epoch": result.best_epoch,
    }
    print(name, clean_training[name])

learned_position_router = PositionConditionedVocabularyHead(
    learned_position_heads, inactive_head=learned_position_heads["p0"],
    vocabulary_size=vocabulary_size,
).to(device)
''')

markdown("## One on-policy recovery round")
code(r'''
onpolicy_states, _ = collect_bucket_states(
    onpolicy_questions, learned_position_router, "compressed_onpolicy"
)
onpolicy_position_heads = clone_head_dict(learned_position_heads)
onpolicy_training = {}
for offset, name in enumerate(BUCKET_NAMES):
    combined = torch.cat((fit_states[name], onpolicy_states[name]), dim=0)
    combined = cap_states(combined, 2 * MAX_STATES_PER_BUCKET, SEED + 500 + offset)
    result = distil_low_rank_head(
        onpolicy_position_heads[name], combined, select_states[name], weight,
        epochs=ONPOLICY_DISTILL_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
        anchor_strength=1e-4, seed=SEED + 100 + offset,
    )
    onpolicy_training[name] = {
        "clean_examples": len(fit_states[name]),
        "onpolicy_examples": len(onpolicy_states[name]),
        "combined_examples": len(combined),
        "losses": list(result.losses), "best_validation_kl": result.best_validation_kl,
        "best_epoch": result.best_epoch,
    }
    print(name, onpolicy_training[name])

onpolicy_position_router = PositionConditionedVocabularyHead(
    onpolicy_position_heads, inactive_head=onpolicy_position_heads["p0"],
    vocabulary_size=vocabulary_size,
).to(device)

# Same learned parameters and rank at every position as the primary arm. Position zero
# stays rank 28; the four rank-32 later experts are cyclically misassigned. This
# distinguishes later-position matching from merely storing several experts.
permuted_sources = {
    "p0": "p0", "p1": "p2", "p2": "p3_5", "p3_5": "p6_plus", "p6_plus": "p1",
}
permuted_position_router = PositionConditionedVocabularyHead(
    {target: onpolicy_position_heads[source]
     for target, source in permuted_sources.items()},
    inactive_head=onpolicy_position_heads["p0"], vocabulary_size=vocabulary_size,
).to(device)
''')

markdown("## Strong matched learned global controls")
code(r'''
# The global controls see exactly the union of the states available to the local
# experts. Their initialization is trajectory-wide and readout-aware, rather than a
# deliberately weak endpoint-only basis.
global_fit = torch.cat([fit_states[name] for name in BUCKET_NAMES], dim=0)
global_select = torch.cat([select_states[name] for name in BUCKET_NAMES], dim=0)
global_centre, global_values, global_vectors = covariance_eigensystem(global_fit)
_, global_order, global_scores = select_readout_aware_basis(
    weight, global_values, global_vectors, max(GLOBAL_RANKS), chunk_size=32
)
global_heads = {}
global_training = {}
for offset, rank in enumerate(GLOBAL_RANKS):
    global_basis = global_vectors[:, global_order[:rank]]
    head = make_low_rank_head(global_basis, global_centre)
    result = distil_low_rank_head(
        head, global_fit, global_select, weight,
        epochs=CLEAN_DISTILL_EPOCHS, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
        anchor_strength=1e-4, seed=SEED + 200 + offset,
    )
    name = f"learned_global_r{rank}"
    global_heads[name] = head
    global_training[name] = {
        "fit_states": len(global_fit), "select_states": len(global_select),
        "initializer": "trajectory_covariance_readout_aware",
        "initializer_indices": global_order[:rank].tolist(),
        "initializer_score_fraction": float(
            global_scores[global_order[:rank]].sum() / global_scores.sum()
        ),
        "losses": list(result.losses), "best_validation_kl": result.best_validation_kl,
        "best_epoch": result.best_epoch,
    }
    print(name, global_training[name])
''')

markdown("## Teacher-trajectory diagnostics before opening the test set")
code(r'''
diagnostic_heads = {
    "first_token_pc4_31_then_full": {
        "p0": confirmed_band_head,
        **{name: full_prefix_head for name in BUCKET_NAMES if name != "p0"},
    },
    "same_pc4_31_everywhere": {name: confirmed_band_head for name in BUCKET_NAMES},
    "fixed_position_local": fixed_position_heads,
    "learned_position_local": learned_position_heads,
    "learned_position_local_onpolicy": onpolicy_position_heads,
    "permuted_position_local_onpolicy": {
        target: onpolicy_position_heads[source]
        for target, source in permuted_sources.items()
    },
}
for global_name, global_head in global_heads.items():
    diagnostic_heads[global_name] = {name: global_head for name in BUCKET_NAMES}

selection_diagnostics = {}
for arm, heads_by_bucket in diagnostic_heads.items():
    selection_diagnostics[arm] = {}
    for name in BUCKET_NAMES:
        if heads_by_bucket[name] is full_prefix_head:
            metrics = {"kl": 0.0, "top1_agreement": 1.0}
        else:
            metrics = evaluate_head_fidelity(
                heads_by_bucket[name], select_states[name], weight,
                batch_size=32, temperature=TEMPERATURE,
            )
        selection_diagnostics[arm][name] = metrics
print(json.dumps(selection_diagnostics, indent=2))
''')

markdown("## Locked full-GSM8K autoregressive evaluation and timing")
code(r'''
from src.data.answer_extract import answers_match

generation_heads = {
    "full": full_head,
    "first_token_pc4_31_then_full": first_token_only_router,
    "same_pc4_31_everywhere": same_band_router,
    "fixed_position_local": fixed_position_router,
    "learned_position_local": learned_position_router,
    "learned_position_local_onpolicy": onpolicy_position_router,
    "permuted_position_local_onpolicy": permuted_position_router,
    **global_heads,
}
test_questions = [str(row["question"]) for row in test_examples]
warmup_questions = select_questions[:min(16, len(select_questions))]
generation_results = {}
generation_outcomes = {}

def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

if RUN_FULL_GENERATION:
    for arm, head in generation_heads.items():
        base_model.set_output_embeddings(head)
        _ = generate_official_codi(
            model, tokenizer, warmup_questions,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS, batch_size=GENERATION_BATCH_SIZE,
            device=device, answer_cue="The answer is:", force_answer_cue=True,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        generations, metadata = generate_official_codi(
            model, tokenizer, test_questions,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=MAX_NEW_TOKENS, batch_size=GENERATION_BATCH_SIZE,
            device=device, answer_cue="The answer is:", force_answer_cue=True,
            return_endpoint_metadata=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        correct = [bool(answers_match(text, row["gold"]))
                   for text, row in zip(generations, test_examples)]
        records = [
            {"arm": arm, "row": index, "gold": str(row["gold"]),
             "generation": text, "correct": outcome,
             "generated_tokens": int(metadata["generated_token_counts"][index])}
            for index, (row, text, outcome) in enumerate(zip(test_examples, generations, correct))
        ]
        write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"{arm}.jsonl", records)
        token_count = int(metadata["generated_token_count"])
        generation_outcomes[arm] = correct
        generation_results[arm] = {
            "examples": len(test_examples), "correct": int(sum(correct)),
            "numeric_exact_match": float(sum(correct) / len(correct)),
            "wall_clock_seconds": elapsed,
            "examples_per_second": len(test_examples) / elapsed,
            "visible_generated_tokens": token_count,
            "microseconds_per_question": 1e6 * elapsed / len(test_examples),
            "microseconds_per_visible_token": 1e6 * elapsed / max(1, token_count),
            "mean_visible_tokens": token_count / len(test_examples),
        }
        print(arm, generation_results[arm])
        if len(test_examples) == 1319 and TEST_LIMIT == 0 and arm == "full":
            assert abs(generation_results[arm]["numeric_exact_match"] - 0.433662) <= 0.015, (
                "Full CODI no longer reproduces the pinned forced-cue baseline; "
                "do not interpret later arms."
            )
        if (len(test_examples) == 1319 and TEST_LIMIT == 0
                and arm == "first_token_pc4_31_then_full"):
            assert abs(generation_results[arm]["numeric_exact_match"] - 0.3806) <= 0.02, (
                "The first-token-only control did not reproduce the confirmed local result; "
                "do not interpret the position-conditioned arms."
            )
base_model.set_output_embeddings(full_head)
''')

markdown("## Component timing, paired intervals, storage, and conclusion")
code(r'''
head_latency_us = {
    "full": benchmark_vocabulary_head(
        full_prefix_head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    ),
    "confirmed_band_r28": benchmark_vocabulary_head(
        confirmed_band_head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    ),
}
for name, head in global_heads.items():
    head_latency_us[name] = benchmark_vocabulary_head(
        head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    )
for bucket, head in onpolicy_position_heads.items():
    head_latency_us[f"position_onpolicy_{bucket}"] = benchmark_vocabulary_head(
        head, 768, batch_size=1, iterations=200, device=device, dtype=dtype
    )

def paired_interval(left, right, seed):
    delta = torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    draws = []
    while len(draws) < BOOTSTRAP_SAMPLES:
        count = min(250, BOOTSTRAP_SAMPLES - len(draws))
        indices = torch.randint(len(delta), (count, len(delta)), generator=generator)
        draws.extend(delta[indices].mean(dim=1).tolist())
    values = torch.tensor(draws)
    return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]

paired_intervals = {}
if generation_results:
    primary = "learned_position_local_onpolicy"
    for offset, comparator in enumerate((
        "full", "first_token_pc4_31_then_full", "same_pc4_31_everywhere",
        "fixed_position_local", "permuted_position_local_onpolicy",
        "learned_global_r32", "learned_global_r64",
    )):
        paired_intervals[f"{primary}_minus_{comparator}"] = paired_interval(
            generation_outcomes[primary], generation_outcomes[comparator], SEED + 900 + offset
        )

stored_parameters = {
    "full_output_rows": int(weight_cpu.numel()),
    "same_pc4_31_everywhere": sum(p.numel() for p in confirmed_band_head.parameters()),
    "fixed_position_local": position_head_parameter_count(fixed_position_router),
    "learned_position_local": position_head_parameter_count(learned_position_router),
    "learned_position_local_onpolicy": position_head_parameter_count(onpolicy_position_router),
    "permuted_position_local_onpolicy": position_head_parameter_count(permuted_position_router),
    **{name: sum(p.numel() for p in head.parameters()) for name, head in global_heads.items()},
}
operation_counts = {
    "full_macs_per_hidden_vector": int(768 * vocabulary_size),
    "rank28_macs_per_hidden_vector": int(28 * (768 + vocabulary_size)),
    "rank32_macs_per_hidden_vector": int(32 * (768 + vocabulary_size)),
    "rank64_macs_per_hidden_vector": int(64 * (768 + vocabulary_size)),
}

validity = {"complete_test": len(test_examples) == 1319 and TEST_LIMIT == 0}
decision = {"status": "not_run"}
if generation_results:
    baseline = generation_results["full"]
    first_only = generation_results["first_token_pc4_31_then_full"]
    primary = generation_results["learned_position_local_onpolicy"]
    global32 = generation_results["learned_global_r32"]
    same = generation_results["same_pc4_31_everywhere"]
    permuted = generation_results["permuted_position_local_onpolicy"]
    validity.update({
        "baseline_reproduces_43_366_within_1_5_points":
            abs(baseline["numeric_exact_match"] - 0.433662) <= 0.015,
        "first_token_control_reproduces_38_06_within_2_points":
            abs(first_only["numeric_exact_match"] - 0.3806) <= 0.02,
    })
    decision = {
        "accuracy": primary["numeric_exact_match"],
        "retained_fraction": primary["numeric_exact_match"] / baseline["numeric_exact_match"],
        "speedup_by_question": baseline["microseconds_per_question"] /
                               primary["microseconds_per_question"],
        "speedup_per_visible_token": baseline["microseconds_per_visible_token"] /
                                    primary["microseconds_per_visible_token"],
        "points_over_same_band": 100 * (
            primary["numeric_exact_match"] - same["numeric_exact_match"]
        ),
        "points_over_global_r32": 100 * (
            primary["numeric_exact_match"] - global32["numeric_exact_match"]
        ),
        "points_over_permuted_position_control": 100 * (
            primary["numeric_exact_match"] - permuted["numeric_exact_match"]
        ),
        "global_r32_paired_lower_bound":
            paired_intervals["learned_position_local_onpolicy_minus_learned_global_r32"][0],
        "permuted_control_paired_lower_bound": paired_intervals[
            "learned_position_local_onpolicy_minus_permuted_position_local_onpolicy"
        ][0],
    }
    decision["status"] = (
        "position_locality_supported" if all(validity.values())
        and decision["points_over_same_band"] > 0
        and decision["points_over_global_r32"] > 0
        and decision["points_over_permuted_position_control"] > 0
        and decision["global_r32_paired_lower_bound"] > 0
        and decision["permuted_control_paired_lower_bound"] > 0
        and decision["speedup_by_question"] > 1.0
        else "position_locality_not_supported_by_gate"
    )

summary = {
    "experiment": "official_codi_position_conditioned_low_rank_readout_v1",
    "code_commit": CODE_COMMIT, "checkpoint_sha256": load_report.checkpoint_sha256,
    "environment": {name: installed(name) for name in PINNED_PACKAGES},
    "inputs": {"reproduction_summary": REPRODUCTION_SUMMARY,
               "colon_states": COLON_STATES, "readout": READOUT},
    "seed": SEED,
    "splits": {"fit_questions": FIT_QUESTIONS, "select_questions": SELECT_QUESTIONS,
               "onpolicy_questions": ONPOLICY_QUESTIONS, "test_questions": len(test_examples)},
    "bucket_definition": [bucket.__dict__ for bucket in DEFAULT_ANSWER_POSITION_BUCKETS],
    "position_selection": position_selection,
    "clean_training": clean_training, "onpolicy_training": onpolicy_training,
    "global_training": global_training,
    "selection_diagnostics": selection_diagnostics,
    "generation_test": generation_results,
    "paired_bootstrap_95ci_accuracy_difference": paired_intervals,
    "head_latency_us": head_latency_us,
    "stored_parameters": stored_parameters,
    "operation_counts": operation_counts,
    "validity": validity, "decision": decision,
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
torch.save({
    "experiment": summary["experiment"], "code_commit": CODE_COMMIT,
    "buckets": summary["bucket_definition"],
    "position_selection": position_selection,
    "heads": {name: {key: value.detach().cpu() for key, value in head.state_dict().items()}
              for name, head in onpolicy_position_heads.items()},
}, pathlib.Path(OUTPUT_ROOT) / "learned_position_local_onpolicy.pt")
print(json.dumps({"validity": validity, "decision": decision,
                  "operation_counts": operation_counts,
                  "stored_parameters": stored_parameters}, indent=2))
print("wrote", summary_path)
''')

markdown(r"""
## How to interpret the run

- If the first-token control does not return near 38.06%, stop: the environment,
  basis reconstruction, or routing contract did not reproduce the known result.
- Teacher-trajectory top-1 agreement is diagnostic. Only autoregressive numeric exact
  match establishes that a head survives its own earlier token decisions.
- A better position-local result does not mean it is smaller in storage. Five experts
  can use rank-32 arithmetic per token while storing more parameters than one rank-64
  global head.
- Compare both microseconds per question and microseconds per visible token. A model
  can appear slower merely because it generates more tokens or fails to emit EOS.
- Do not alter buckets, rank, epochs, or thresholds after inspecting test results.
  Any change is a new experiment and must use a fresh sealed evaluation.
""")

notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
