"""Build Experiment 1: connect CODI's 28-direction band to its global rank-96 head."""
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:
    from notebook_compat import nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_codi_28_to_global96_bridge.ipynb"
notebook = nbf.v4.new_notebook()
cells = []


def markdown(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


markdown(r"""
# Experiment 1 — Do CODI's 28 answer directions explain the global rank-96 head?

The earlier endpoint experiment found a 28-dimensional band—PCs 4 through 31—at
the post-`ln_f` state that predicts the first visible answer token. Retaining only
that band preserved 502/572 = 87.8% of the full model's sequence accuracy; removing
it reduced accuracy from 572/1319 to about 170/1319. A later experiment learned one
rank-96 LM head for every answer position and retained 563/572 = 98.43% of baseline.

Those are two separate results. This notebook tests whether they describe the same
mechanism.

## Locked hypotheses

1. **Geometric containment.** The learned rank-96 down-projection's row space captures
   more of the fixed 28-direction band than 95% of 200 isotropic rank-96 controls.
2. **Causal reliance.** Removing the 28 directions at answer position 0 hurts the
   rank-96 head, while retaining them preserves substantial first-token information.
3. **Trajectory boundary.** A basis fitted only at the answer colon may work at
   position 0 but need not transfer to later generated tokens. We measure positions
   `p0`, `p1`, and `p2+` separately and test the deliberately naive all-position arm.
4. **Sequence consequence.** All claims are checked by greedy decoding on the fixed
   1,319-example GSM8K test, not inferred from isolated logits alone.

No weights are trained here. Ranks, bands, controls, thresholds, and arms are fixed
before evaluation. The test set is used once for the locked comparisons.
""")

markdown("## 1. Configuration and immutable source checkout")
code(r'''
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
# This commit contains every dependency used below. The new bridge utility is also
# included inline as a compatibility fallback until its source commit is published.
RUN_COMMIT = "e1c291a"
REPO_DIR = "/kaggle/working/latent-reasoning"

REPRODUCTION_SUMMARY_INPUT = ""
COLON_STATES_INPUT = ""
READOUT_INPUT = ""
GLOBAL_HEAD_ARTIFACT_INPUT = ""
OUTPUT_ROOT = "/kaggle/working/codi_28_to_global96_bridge"

SEED = 20260906
PRIMARY_BAND = (4, 32)       # 28 fixed directions
LEADING_CONTROL_BAND = (0, 4)  # variance-dominant negative control
MATCHED_RANDOM_REPLICATES = 4
GEOMETRY_NULL_REPLICATES = 200
TRAJECTORY_QUESTIONS = 256
GENERATION_BATCH_SIZE = 32
ANALYTIC_BATCH_SIZE = 32
MAX_NEW_TOKENS = 64
BOOTSTRAP_SAMPLES = 5000
TEST_LIMIT = 0               # 0 means the complete 1,319-example test

import copy, glob, json, os, pathlib, random, subprocess, sys, time
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
    "tests/test_endpoint_margin_geometry.py",
    "tests/test_global_low_rank_head.py",
    "tests/test_official_codi_fast.py",
], check=True)
''')

markdown(r"""
## 3. Resolve the three completed experiment inputs

Attach these Kaggle datasets:

1. the corrected official CODI reproduction (contains the `full_gsm8k/summary.json`);
2. `codi-answer-colon-margin-geometry` (contains `colon_states.pt` and `readout.pt`);
3. `trajectory-whitened-global-low-rank-lm-head` (contains
   `global_low_rank_head.pt` and its adjacent `summary.json`).

The later deployment benchmark dataset is not needed because this experiment uses
the original frozen rank-96 artifact, not compiled/Triton wrappers.
""")
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
COLON_STATES = discover(COLON_STATES_INPUT, "colon_states_seed89/colon_states.pt")
READOUT = discover(READOUT_INPUT, "colon_states_seed89/readout.pt")
GLOBAL_HEAD_ARTIFACT = discover(GLOBAL_HEAD_ARTIFACT_INPUT, "global_low_rank_head.pt")
GLOBAL_HEAD_SUMMARY = GLOBAL_HEAD_ARTIFACT.with_name("summary.json")
assert GLOBAL_HEAD_SUMMARY.is_file(), (
    "global_low_rank_head.pt must have its original adjacent summary.json"
)
print({"reproduction": str(REPRODUCTION_SUMMARY), "states": str(COLON_STATES),
       "readout": str(READOUT), "global_head": str(GLOBAL_HEAD_ARTIFACT)})
''')

markdown("## 4. Load the frozen bases and validate artifact provenance")
code(r'''
import torch
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.mech.endpoint_correctness_geometry import readout_matrix
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE, band_variance_share, build_band_subspace,
    build_matched_random_subspaces, state_covariance,
)

torch.manual_seed(SEED)
random.seed(SEED)
cache, readout_payload = load_margin_cache(COLON_STATES, READOUT)
assert cache["metadata"]["precision"] == "float32"
assert cache["parity_gate"]["passed"] and cache["full_parity_gate"]["passed"]
state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
calibration_states = cache["calibration_states"][:, state_index, :].float()
test_states = cache["evaluation_states"][:, state_index, :].float()
test_gold_first = cache["evaluation_gold_first_token"].long()
colon_centre = cache["student_mean"][ANALYTIC_STATE].float()
covariance = state_covariance(calibration_states - colon_centre)

primary_subspace = build_band_subspace(
    covariance=covariance, start=PRIMARY_BAND[0], stop=PRIMARY_BAND[1],
    state=ANALYTIC_STATE,
)
leading_subspace = build_band_subspace(
    covariance=covariance, start=LEADING_CONTROL_BAND[0],
    stop=LEADING_CONTROL_BAND[1], state=ANALYTIC_STATE,
)
matched_controls = build_matched_random_subspaces(
    selected=primary_subspace, covariance=covariance,
    replicates=MATCHED_RANDOM_REPLICATES, seed=SEED + 11,
)
assert primary_subspace.basis.shape == (768, 28)
assert all(control.target_attainable for control in matched_controls)
assert len(test_states) == len(test_gold_first) == 1319

global_artifact = torch.load(GLOBAL_HEAD_ARTIFACT, map_location="cpu", weights_only=False)
global_summary = json.loads(GLOBAL_HEAD_SUMMARY.read_text(encoding="utf-8"))
assert global_artifact["contract"] == "trajectory_whitened_margin_distilled_global_lm_head_v1"
assert global_summary["experiment"] == global_artifact["contract"]
assert tuple(int(value) for value in global_artifact["ranks"]) == (32, 64, 96)
prior_full = global_summary["generation"]["full"]["numeric_exact_match"]
prior_rank96 = global_summary["generation"]["whitened_margin_onpolicy_r96"]["numeric_exact_match"]
assert prior_rank96 / prior_full >= 0.98
print({"calibration": len(calibration_states), "test": len(test_states),
       "primary_variance_share": band_variance_share(covariance, 4, 32),
       "prior_full": prior_full, "prior_rank96": prior_rank96})
''')

markdown("## 5. Bridge utilities: compare row spaces and edit head inputs")
code(r'''
import torch.nn as nn

try:
    from src.mech.subspace_bridge import (
        SubspaceInterventionVocabularyHead, edit_hidden_subspace,
        orthonormal_row_space, subspace_overlap,
    )
except ModuleNotFoundError:
    # Compatibility copy for RUN_COMMIT above. It implements the same definitions as
    # src/mech/subspace_bridge.py in the experiment source commit.
    from dataclasses import dataclass, asdict

    @dataclass(frozen=True)
    class _Overlap:
        ambient_dimension: int; reference_rank: int; candidate_rank: int
        shared_energy: float; reference_capture_fraction: float
        candidate_occupancy_fraction: float; mean_squared_cosine: float
        minimum_cosine: float; mean_cosine: float; maximum_cosine: float
        def to_dict(self): return asdict(self)

    def _orth(basis):
        return torch.linalg.qr(basis.double(), mode="reduced").Q.to(basis)

    def orthonormal_row_space(matrix):
        _, singular_values, right_t = torch.linalg.svd(matrix.double(), full_matrices=False)
        tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * float(singular_values.max())
        rank = int((singular_values > tolerance).sum())
        return right_t[:rank].T.to(matrix)

    def subspace_overlap(reference, candidate):
        left, right = _orth(reference), _orth(candidate)
        cosines = torch.linalg.svdvals(left.double().T @ right.double())
        shared = float(cosines.square().sum())
        return _Overlap(reference.shape[0], left.shape[1], right.shape[1], shared,
                        shared / left.shape[1], shared / right.shape[1],
                        float(cosines.square().mean()), float(cosines.min()),
                        float(cosines.mean()), float(cosines.max()))

    def edit_hidden_subspace(hidden, basis, centre, *, mode):
        columns = _orth(basis.to(hidden)); origin = centre.to(hidden)
        projected = ((hidden - origin) @ columns) @ columns.T
        return origin + projected if mode == "retain" else hidden - projected

    class SubspaceInterventionVocabularyHead(nn.Module):
        def __init__(self, head, basis, centre, *, mode, vocabulary_size,
                     active_positions=None):
            super().__init__(); self.head = head
            self.register_buffer("basis", _orth(basis.detach()).clone())
            self.register_buffer("centre", centre.detach().to(basis).clone())
            self.mode = mode; self.vocabulary_size = int(vocabulary_size)
            self.active_positions = (None if active_positions is None else
                                     frozenset(int(x) for x in active_positions))
            self._answer_position = None
        def set_answer_position(self, position):
            self._answer_position = None if position is None else int(position)
            setter = getattr(self.head, "set_answer_position", None)
            if callable(setter): setter(position)
        def forward(self, hidden):
            active = self._answer_position is not None and (
                self.active_positions is None or self._answer_position in self.active_positions)
            values = edit_hidden_subspace(hidden, self.basis, self.centre,
                                          mode=self.mode) if active else hidden
            logits = self.head(values)
            assert logits.shape[-1] == self.vocabulary_size
            return logits
''')

markdown(r"""
## 6. Geometric test

For a factorized head `logits = Up(Down(h))`, `Down.weight` has shape
`rank × 768`. Its rows are the directions the head can see. We orthonormalize that
row span and compare it with the 28 columns of `U28` using principal angles.

The key statistic is

`capture(U28 by Qr) = ||Qrᵀ U28||²_F / 28`.

It is 1 if the learned rank-`r` space contains all 28 directions. An isotropic random
rank-96 space captures only 96/768 = 12.5% in expectation.
""")
code(r'''
from src.mech.global_low_rank_head import NestedLowRankVocabularyHead

global_cpu = NestedLowRankVocabularyHead(768, readout_matrix(readout_payload).shape[0], (32, 64, 96))
global_cpu.load_state_dict(global_artifact["state_dict"])
global_cpu.disable_adaptive()

def learned_space(rank):
    return orthonormal_row_space(global_cpu.down.weight[:rank].detach())

def geometry_null(rank, replicates, seed):
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(replicates):
        candidate = torch.linalg.qr(
            torch.randn(768, rank, generator=generator, dtype=torch.float64),
            mode="reduced",
        ).Q.float()
        values.append(subspace_overlap(primary_subspace.basis, candidate).reference_capture_fraction)
    return torch.tensor(values)

geometry = {}
for rank in (32, 96):
    observed = subspace_overlap(primary_subspace.basis, learned_space(rank))
    null = geometry_null(rank, GEOMETRY_NULL_REPLICATES, SEED + rank)
    geometry[f"rank{rank}"] = {
        **observed.to_dict(),
        "random_expected_capture": rank / 768,
        "random_mean_capture": float(null.mean()),
        "random_95th_percentile": float(torch.quantile(null, 0.95)),
        "empirical_one_sided_p": float((1 + (null >= observed.reference_capture_fraction).sum()) /
                                        (1 + len(null))),
        "exceeds_random_95th_percentile": bool(
            observed.reference_capture_fraction > float(torch.quantile(null, 0.95))
        ),
    }
print(json.dumps(geometry, indent=2))
''')

markdown("## 7. Load official CODI and construct all frozen heads")
code(r'''
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.mech.eigenspace_readout import LowRankVocabularyHead
from src.mech.position_conditioned_readout import (
    PositionConditionedVocabularyHead, VocabularyPrefixHead,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config

assert torch.cuda.is_available(), "Select a Kaggle GPU accelerator"
cfg = load_config("configs/official_codi_gpt2.yaml")
verify_full_reproduction_gate(REPRODUCTION_SUMMARY, cfg)
device = torch.device("cuda")
dtype = resolve_torch_dtype("float32", device)
checkpoint_matches = sorted(glob.glob(f"/kaggle/input/**/{cfg.checkpoint.filename}", recursive=True))
checkpoint = pathlib.Path(checkpoint_matches[0]) if checkpoint_matches else download_official_checkpoint(
    repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
    filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
    token=os.environ.get("HF_TOKEN") or None,
)
model, tokenizer = build_official_codi_gpt2(
    base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
    dtype=dtype, settings=cfg.model, token=os.environ.get("HF_TOKEN") or None,
)
load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
assert str(cache["metadata"]["checkpoint_sha256"]) == load_report.checkpoint_sha256
assert global_summary["checkpoint_sha256"] == load_report.checkpoint_sha256
for parameter in model.parameters(): parameter.requires_grad_(False)
model.to(device=device, dtype=dtype).eval()
base_model = model.codi.get_base_model()
full_head = base_model.get_output_embeddings()
vocabulary_size = int(model.eot_id)
weight_cpu = readout_matrix(readout_payload).float()
assert tuple(weight_cpu.shape) == (vocabulary_size, 768)
assert torch.allclose(full_head.weight[:vocabulary_size].detach().cpu(), weight_cpu,
                      atol=0, rtol=0)
full_prefix = VocabularyPrefixHead(full_head, vocabulary_size).to(device)
weight = full_head.weight[:vocabulary_size].detach()

def fresh_global(rank):
    head = NestedLowRankVocabularyHead(768, vocabulary_size, (32, 64, 96))
    head.load_state_dict(global_artifact["state_dict"])
    head.disable_adaptive(); head.set_rank(rank)
    return head.to(device=device, dtype=dtype).eval()

global32, global96 = fresh_global(32), fresh_global(96)
u28 = primary_subspace.basis.to(device)
mu = colon_centre.to(device)
pc28_head = LowRankVocabularyHead.from_basis(weight, u28, mu).eval()
leading4_head = LowRankVocabularyHead.from_basis(
    weight, leading_subspace.basis.to(device), mu,
).eval()
random28_heads = [
    LowRankVocabularyHead.from_basis(weight, control.basis.to(device), mu).eval()
    for control in matched_controls
]

def p0_then(first, later):
    return PositionConditionedVocabularyHead(
        {"p0": first, "p1": later, "p2_plus": later},
        inactive_head=later, vocabulary_size=vocabulary_size,
    ).to(device).eval()

arms = {
    "dense_full": full_prefix,
    "pc4_31_first_then_dense": p0_then(pc28_head, full_prefix),
    "pc4_31_remove_first_then_dense": SubspaceInterventionVocabularyHead(
        full_prefix, u28, mu, mode="remove", vocabulary_size=vocabulary_size,
        active_positions={0},
    ).to(device).eval(),
    "leading_pc0_3_first_then_dense": p0_then(leading4_head, full_prefix),
    "pc4_31_every_answer_position": pc28_head,
    "global_rank32": global32,
    "global_rank96": global96,
    "global_rank96_retain_pc4_31_at_first": SubspaceInterventionVocabularyHead(
        fresh_global(96), u28, mu, mode="retain", vocabulary_size=vocabulary_size,
        active_positions={0},
    ).to(device).eval(),
    "global_rank96_remove_pc4_31_at_first": SubspaceInterventionVocabularyHead(
        fresh_global(96), u28, mu, mode="remove", vocabulary_size=vocabulary_size,
        active_positions={0},
    ).to(device).eval(),
}
for index, head in enumerate(random28_heads):
    arms[f"random_matched28_r{index}_first_then_dense"] = p0_then(head, full_prefix)
assert len(arms) == 9 + MATCHED_RANDOM_REPLICATES
print("arms:", list(arms))
''')

markdown(r"""
## 8. Cached answer-colon analysis

This isolates the exact state where the 28 directions were discovered. For each
head we report gold first-token accuracy and agreement with the dense head. This is
cheap and diagnostic, but it is not a substitute for full generation below.
""")
code(r'''
import torch.nn.functional as F

@torch.inference_mode()
def cached_metrics(head, *, edit_mode=None):
    total = agreement = gold_correct = 0
    teacher_gold_correct = 0; regret_sum = margin_error_sum = 0.0
    for start in range(0, len(test_states), ANALYTIC_BATCH_SIZE):
        hidden = test_states[start:start + ANALYTIC_BATCH_SIZE].to(device)
        gold = test_gold_first[start:start + ANALYTIC_BATCH_SIZE].to(device)
        teacher = F.linear(hidden, weight)
        values = hidden if edit_mode is None else edit_hidden_subspace(
            hidden, u28, mu, mode=edit_mode
        )
        student = head(values)
        teacher_token = teacher.argmax(-1); student_token = student.argmax(-1)
        teacher_top2 = teacher.topk(2, dim=-1).values
        student_top2 = student.topk(2, dim=-1).values
        total += len(hidden)
        agreement += int((student_token == teacher_token).sum())
        gold_correct += int((student_token == gold).sum())
        teacher_gold_correct += int((teacher_token == gold).sum())
        regret_sum += float((teacher.max(-1).values -
                             teacher.gather(1, student_token[:, None]).squeeze(1)).sum())
        margin_error_sum += float(((student_top2[:, 0] - student_top2[:, 1]) -
                                   (teacher_top2[:, 0] - teacher_top2[:, 1])).abs().sum())
    return {"examples": total, "teacher_gold_first_accuracy": teacher_gold_correct / total,
            "gold_first_accuracy": gold_correct / total,
            "dense_top1_agreement": agreement / total,
            "mean_dense_logit_regret": regret_sum / total,
            "mean_absolute_top2_margin_error": margin_error_sum / total}

cached = {
    "dense_full": cached_metrics(full_prefix),
    "dense_after_retain_pc4_31": cached_metrics(full_prefix, edit_mode="retain"),
    "dense_after_remove_pc4_31": cached_metrics(full_prefix, edit_mode="remove"),
    "pc4_31_fixed": cached_metrics(pc28_head),
    "leading_pc0_3_fixed": cached_metrics(leading4_head),
    "global_rank32": cached_metrics(global32),
    "global_rank96": cached_metrics(global96),
    "global_rank96_after_retain_pc4_31": cached_metrics(global96, edit_mode="retain"),
    "global_rank96_after_remove_pc4_31": cached_metrics(global96, edit_mode="remove"),
}
for index, head in enumerate(random28_heads):
    cached[f"random_matched28_r{index}"] = cached_metrics(head)
print(json.dumps(cached, indent=2))
''')

markdown(r"""
## 9. Later-position transfer on disjoint training trajectories

We now collect post-`ln_f` states from 256 frozen dense-head trajectories. No head is
fitted or selected with them. We ask how well each frozen head reproduces the dense
next-token decision at the first token (`p0`), second token (`p1`), and all later
tokens (`p2+`). Applying the colon-fitted `U28` at later positions is intentionally a
transfer test; we do not recompute its centre or eigenvectors.
""")
code(r'''
from src.data.datasets import load_train_set
from src.inference.official_codi_fast import (
    generate_official_codi_fast, prepare_official_codi_batches,
)
from src.mech.position_conditioned_readout import answer_position_bucket
from src.models.official_codi import generate_official_codi

training = load_train_set(load_config("configs/data.yaml"), trace_style="eq_only")
order = torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 301))
trajectory_questions = [str(training[int(index)]["question"])
                        for index in order[:TRAJECTORY_QUESTIONS]]
bucket_chunks = {"p0": [], "p1": [], "p2_plus": []}

# Empirically verify the body-only decoder before using it for any experimental arm.
parity_questions = trajectory_questions[:16]
base_model.set_output_embeddings(full_head)
reference_parity = generate_official_codi(
    model, tokenizer, parity_questions,
    latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
    batch_size=16, device=device, answer_cue="The answer is:", force_answer_cue=True,
)
parity_prepared = prepare_official_codi_batches(
    tokenizer, parity_questions, batch_size=16, length_bucketed=False,
)
fast_parity = generate_official_codi_fast(
    model, tokenizer, parity_prepared,
    latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
    device=device, answer_cue="The answer is:",
)
assert tuple(reference_parity) == fast_parity.texts
print({"fastpath_parity_examples": len(parity_questions), "exact": True})

def observer(hidden, active, answer_position):
    if bool(active.any()):
        name = answer_position_bucket(int(answer_position))
        bucket_chunks[name].append(hidden[active].detach().cpu().float())

prepared_trajectory = prepare_official_codi_batches(
    tokenizer, trajectory_questions, batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)
base_model.set_output_embeddings(full_prefix)
_ = generate_official_codi_fast(
    model, tokenizer, prepared_trajectory,
    latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
    device=device, answer_cue="The answer is:", answer_state_observer=observer,
)
trajectory_states = {name: torch.cat(chunks) for name, chunks in bucket_chunks.items()}
assert all(len(values) > 0 for values in trajectory_states.values())
print({name: len(values) for name, values in trajectory_states.items()})

@torch.inference_mode()
def teacher_agreement(states, head, *, edit_mode=None):
    total = matches = 0; regret = 0.0
    for start in range(0, len(states), ANALYTIC_BATCH_SIZE):
        hidden = states[start:start + ANALYTIC_BATCH_SIZE].to(device)
        teacher = full_prefix(hidden)
        values = hidden if edit_mode is None else edit_hidden_subspace(
            hidden, u28, mu, mode=edit_mode
        )
        student = head(values)
        target = teacher.argmax(-1); chosen = student.argmax(-1)
        total += len(hidden); matches += int((target == chosen).sum())
        regret += float((teacher.max(-1).values -
                         teacher.gather(1, chosen[:, None]).squeeze(1)).sum())
    return {"examples": total, "dense_top1_agreement": matches / total,
            "mean_dense_logit_regret": regret / total}

trajectory_fidelity = {}
for bucket, states in trajectory_states.items():
    trajectory_fidelity[bucket] = {
        "pc4_31_fixed": teacher_agreement(states, pc28_head),
        "global_rank32": teacher_agreement(states, global32),
        "global_rank96": teacher_agreement(states, global96),
        "global_rank96_retain_pc4_31": teacher_agreement(states, global96, edit_mode="retain"),
        "global_rank96_remove_pc4_31": teacher_agreement(states, global96, edit_mode="remove"),
    }
print(json.dumps(trajectory_fidelity, indent=2))
''')

markdown(r"""
## 10. Locked full GSM8K generation

All arms now decode the same complete test set. Only position 0 is edited in the
`*_at_first` arms; later token states therefore include the causal consequences of
the changed first token. The `pc4_31_every_answer_position` arm tests whether the
local basis can simply be reused globally. Timings are recorded for bookkeeping but
are not a speed benchmark: arms run once and some are routed Python modules.
""")
code(r'''
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set

test_examples = load_eval_set("gsm8k", load_config(cfg.data_config).eval.gsm8k)
assert len(test_examples) == 1319
if TEST_LIMIT: test_examples = test_examples[:TEST_LIMIT]
test_questions = [str(row["question"]) for row in test_examples]
prepared_test = prepare_official_codi_batches(
    tokenizer, test_questions, batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)
warmup = prepare_official_codi_batches(
    tokenizer, trajectory_questions[:32], batch_size=GENERATION_BATCH_SIZE,
    length_bucketed=False,
)

def write_jsonl(path, records):
    temporary = pathlib.Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)

generation_results = {}
correctness = {}
for name, head in arms.items():
    base_model.set_output_embeddings(head)
    _ = generate_official_codi_fast(
        model, tokenizer, warmup, latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=MAX_NEW_TOKENS, device=device, answer_cue="The answer is:",
    )
    torch.cuda.synchronize(); started = time.perf_counter()
    generated = generate_official_codi_fast(
        model, tokenizer, prepared_test, latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=MAX_NEW_TOKENS, device=device, answer_cue="The answer is:",
    )
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    flags = [bool(answers_match(text, row["gold"]))
             for text, row in zip(generated.texts, test_examples)]
    correctness[name] = torch.tensor(flags, dtype=torch.bool)
    records = [{"index": index, "question": str(row["question"]),
                "gold": str(row["gold"]), "generation": text,
                "token_ids": list(token_ids), "correct": bool(flag)}
               for index, (row, text, token_ids, flag) in enumerate(zip(
                   test_examples, generated.texts, generated.token_ids, flags))]
    write_jsonl(pathlib.Path(OUTPUT_ROOT) / f"{name}.jsonl", records)
    generation_results[name] = {
        "examples": len(flags), "correct": int(sum(flags)),
        "numeric_exact_match": float(sum(flags) / len(flags)),
        "elapsed_seconds_single_run": elapsed,
        "generated_tokens": generated.generated_token_count,
    }
    print(name, generation_results[name])
base_model.set_output_embeddings(full_head)
''')

markdown("## 11. Paired uncertainty, preregistered decisions, and saved summary")
code(r'''
def paired_bootstrap_delta(reference, candidate, samples, seed):
    generator = torch.Generator().manual_seed(seed)
    differences = candidate.float() - reference.float()
    estimates = []
    for _ in range(0, samples, 250):
        count = min(250, samples - len(estimates))
        indices = torch.randint(len(differences), (count, len(differences)), generator=generator)
        estimates.extend(differences[indices].mean(1).tolist())
    values = torch.tensor(estimates)
    return {"delta_accuracy": float(differences.mean()),
            "ci95_low": float(torch.quantile(values, 0.025)),
            "ci95_high": float(torch.quantile(values, 0.975))}

dense_flags = correctness["dense_full"]
paired = {}
for offset, (name, flags) in enumerate(correctness.items()):
    if name == "dense_full": continue
    paired[name] = paired_bootstrap_delta(
        dense_flags, flags, BOOTSTRAP_SAMPLES, SEED + 500 + offset
    )
    paired[name]["dense_to_arm_correct_to_wrong"] = int((dense_flags & ~flags).sum())
    paired[name]["dense_to_arm_wrong_to_correct"] = int((~dense_flags & flags).sum())

full_accuracy = generation_results["dense_full"]["numeric_exact_match"]
rank96_accuracy = generation_results["global_rank96"]["numeric_exact_match"]
rank96_removed = generation_results[
    "global_rank96_remove_pc4_31_at_first"
]["numeric_exact_match"]
rank96_retained = generation_results[
    "global_rank96_retain_pc4_31_at_first"
]["numeric_exact_match"]
band_accuracy = generation_results["pc4_31_first_then_dense"]["numeric_exact_match"]

decisions = {
    "original_band_result_reproduced_within_2pp": bool(
        abs(full_accuracy - 572 / 1319) <= 0.02 and
        abs(band_accuracy - 502 / 1319) <= 0.02
    ),
    "rank96_retains_at_least_98pct_of_dense_accuracy": bool(
        rank96_accuracy >= 0.98 * full_accuracy
    ),
    "rank96_space_captures_u28_above_random95": bool(
        geometry["rank96"]["exceeds_random_95th_percentile"]
    ),
    "removing_u28_from_rank96_costs_at_least_10pp": bool(
        rank96_accuracy - rank96_removed >= 0.10
    ),
    "retaining_u28_preserves_at_least_70pct_of_rank96_accuracy": bool(
        rank96_retained >= 0.70 * rank96_accuracy
    ),
}
decisions["shared_mechanism_supported"] = bool(
    decisions["rank96_space_captures_u28_above_random95"] and
    decisions["removing_u28_from_rank96_costs_at_least_10pp"] and
    decisions["retaining_u28_preserves_at_least_70pct_of_rank96_accuracy"]
)

summary = {
    "experiment": "official_codi_u28_global_rank96_bridge_v1",
    "code_commit": CODE_COMMIT,
    "checkpoint_sha256": load_report.checkpoint_sha256,
    "inputs": {"reproduction_summary": str(REPRODUCTION_SUMMARY),
               "colon_states": str(COLON_STATES), "readout": str(READOUT),
               "global_head_artifact": str(GLOBAL_HEAD_ARTIFACT)},
    "population": {"calibration_states": len(calibration_states),
                   "trajectory_questions": TRAJECTORY_QUESTIONS,
                   "test_questions": len(test_examples)},
    "subspaces": {"primary_band": list(PRIMARY_BAND), "primary_rank": 28,
                  "primary_variance_share": band_variance_share(covariance, 4, 32),
                  "leading_control_band": list(LEADING_CONTROL_BAND),
                  "matched_random_replicates": MATCHED_RANDOM_REPLICATES},
    "geometry": geometry,
    "cached_first_token": cached,
    "trajectory_fidelity": trajectory_fidelity,
    "generation": generation_results,
    "paired_accuracy_intervals": paired,
    "decisions": decisions,
    "interpretation_rule": (
        "Call the mechanisms shared only if geometric enrichment, causal removal, "
        "and causal retention all pass. Otherwise report complementary or distinct "
        "subspaces; do not infer identity from two successful compression results."
    ),
}
summary_path = pathlib.Path(OUTPUT_ROOT) / "summary.json"
temporary = pathlib.Path(str(summary_path) + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
temporary.replace(summary_path)
print(json.dumps({"decisions": decisions, "generation": generation_results}, indent=2))
print("saved:", summary_path)
''')

markdown("## 12. Visual summary")
code(r'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ranks = [32, 96]
observed = [geometry[f"rank{rank}"]["reference_capture_fraction"] for rank in ranks]
null95 = [geometry[f"rank{rank}"]["random_95th_percentile"] for rank in ranks]
axes[0].bar(["learned r32", "learned r96"], observed, color="#2c7fb8")
axes[0].scatter([0, 1], null95, color="#d95f0e", marker="_", s=500,
                label="random 95th percentile")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Fraction of U28 captured")
axes[0].set_title("Does the learned bottleneck contain U28?"); axes[0].legend()

plot_names = ["dense_full", "pc4_31_first_then_dense", "pc4_31_every_answer_position",
              "global_rank32", "global_rank96",
              "global_rank96_retain_pc4_31_at_first",
              "global_rank96_remove_pc4_31_at_first"]
labels = ["dense", "U28 only at p0", "U28 all positions", "global r32", "global r96",
          "r96 + retain U28 p0", "r96 - remove U28 p0"]
values = [generation_results[name]["numeric_exact_match"] for name in plot_names]
axes[1].barh(labels[::-1], values[::-1], color="#41ab5d")
axes[1].set_xlim(0, max(values) * 1.12); axes[1].set_xlabel("GSM8K numeric exact match")
axes[1].set_title("Full autoregressive consequence")
fig.tight_layout()
figure_path = pathlib.Path(OUTPUT_ROOT) / "bridge_summary.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()
print(figure_path)
''')

markdown(r"""
## How to interpret the outcome

- If rank 96 captures unusually much of `U28`, removing `U28` damages rank-96
  decoding, and retaining `U28` preserves it, the two experiments have identified a
  shared first-token answer mechanism.
- If causal effects are strong but geometric overlap is ordinary, the learned head
  may encode the same decisions through a different non-orthogonal representation.
- If overlap is high but interventions are weak, containment is correlational rather
  than evidence that rank 96 uses those coordinates.
- If `U28` succeeds at `p0` but fails at `p1`/`p2+` and in the all-position arm, that
  explains why a local answer-cue compression cannot simply replace the global LM
  head. Later tokens occupy different parts of the answer trajectory.

This experiment tests mechanistic continuity. It does **not** claim end-to-end speed:
the separate deployment benchmark already showed that a faster rank-96 head is only
a small fraction of CODI's total runtime.
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
