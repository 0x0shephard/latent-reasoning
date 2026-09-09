"""Build a self-contained Kaggle ablation notebook on the pinned CODI implementation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/kaggle_codi_global_head_ablations.ipynb'
original = json.loads((ROOT / 'notebooks/kaggle_global_low_rank_lm_head.ipynb').read_text(encoding='utf-8'))
original_code = [''.join(c['source']) for c in original['cells'] if c['cell_type'] == 'code']
cells = []
def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip().splitlines(True)))
def code(s):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip().splitlines(True)))
def reuse(marker):
    return next(s for s in original_code if marker in s)

md('''# CODI global-head ablations
Runs on the frozen official GPT-2 CODI checkpoint. Only the output head is trained.

**Core:** unconstrained / fixed U28 / fixed random-28 at ranks 32, 64, 96;
first-token-only / full-trajectory fitting with matched state counts.
**Optional:** weight-only SVD, loss terms, matched-budget recovery, data efficiency,
and frozen-head SVAMP evaluation. Model architecture transfer is a separate experiment.

U28 is refitted from training first-token states (zero-based PCs 4:32). This tests the
same construction with clean splits; it is not the exact historical U28 artifact.
All choices are locked before test decoding. GSM8K has informed prior hypotheses, so
these are follow-up ablations, not a new untouched confirmatory benchmark.

Upload this notebook directly: it embeds its helper implementation and clones immutable
base commit `6a8d2e61950c67f012d0a9ba13ec8a70f3a25019`. No branch push is required.
No dataset attachments or historical result files are required. Model weights and
training/evaluation data download automatically. Checkpoint hashes and decoder parity
are checked in this run, and the dense baseline is evaluated alongside the ablations.
Enable Internet and a GPU, then run all cells. `SMOKE=True` is an optional quick check;
the default `SMOKE=False` runs the real experiment.
''')
code(r'''
SUITE = "core"  # core, initialization, losses, recovery, data, or all
SMOKE = False  # smoke output is explicitly non-scientific; never compare with full runs
SEEDS = [89]   # use [89, 90, 91] for final fitting-seed uncertainty
EVAL_DATASETS = ["gsm8k"]  # optionally add "svamp"; no refitting on evaluation data
RESUME_ROOT = ""  # optional prior output run directory containing manifest.json
OUTPUT_ROOT = "/kaggle/working/codi_global_head_ablations"
GENERATION_BATCH_SIZE = 8  # lower to 4 if GPU memory is tight
DISTILL_BATCH_SIZE = 8
MAX_NEW_TOKENS = 64
RANKS = (32, 64, 96)
FIT_QUESTIONS = 1024
SELECT_QUESTIONS = 256
RECOVERY_QUESTIONS = 256
CLEAN_EPOCHS = 4
RECOVERY_EPOCHS = 2
LEARNING_RATE = 2e-4
DATA_SIZES = [128, 256, 512, 1024]
BOOTSTRAP_SAMPLES = 5000
SPLIT_SEED = 20260909  # fixed across fitting seeds
SEED = SPLIT_SEED
if SMOKE:
    FIT_QUESTIONS, SELECT_QUESTIONS, RECOVERY_QUESTIONS = 48, 8, 8
    CLEAN_EPOCHS = RECOVERY_EPOCHS = 1
    DATA_SIZES = [36, 48]
    BOOTSTRAP_SAMPLES = 100
assert SUITE in {"core", "initialization", "losses", "recovery", "data", "all"}
assert SEEDS and len(SEEDS) == len(set(SEEDS))
assert set(EVAL_DATASETS) <= {"gsm8k", "svamp"} and "gsm8k" in EVAL_DATASETS

import copy, glob, hashlib, json, os, pathlib, random, shutil, subprocess, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
REPO_URL = "https://github.com/0x0shephard/latent-reasoning.git"
RUN_COMMIT = "6a8d2e61950c67f012d0a9ba13ec8a70f3a25019"
REPO_DIR = "/kaggle/working/latent-reasoning"
if not pathlib.Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--detach", RUN_COMMIT], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
CODE_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert CODE_COMMIT == RUN_COMMIT
print("Base implementation:", CODE_COMMIT)
''')
md('## Environment and frozen model')
code(reuse('PINNED_PACKAGES ='))
model_setup = reuse('import torch\n')
model_setup = model_setup.replace(
    'from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate\n', '')
model_setup = model_setup.replace(
    'verify_full_reproduction_gate(pathlib.Path(REPRODUCTION_SUMMARY), cfg)\n', '')
code(model_setup.replace('device = torch.device("cuda")',
    'assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator"\ndevice = torch.device("cuda")'))
md('## Embedded ablation helpers')
helper = (ROOT / 'src/mech/global_head_ablations.py').read_text(encoding='utf-8')
source_hash = hashlib.sha256((helper + Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
code('ABLATION_SOURCE_SHA256 = ' + repr(source_hash) + '\n' + helper)
# Future imports must be at the start of their cell.
cells[-1]['source'] = ('from __future__ import annotations\nABLATION_SOURCE_SHA256 = ' + repr(source_hash) + '\n' + helper.replace('from __future__ import annotations\n', '')).splitlines(True)
md('''## Lock the experiment grid and output identity
Each suite includes its own reference. Core heads receive clean distillation only;
recovery is isolated in its own suite. All ranks are nested prefixes trained together,
not independent rank-specific fits. Fixed-28 heads have 4/36/68 learned residual
coordinates at ranks 32/64/96. Random-28 uses the same initializer and constraint.
The random basis changes with the fitting seed.

Coverage compares the same training questions and the same number of states; the
all-trajectory arm samples one observed position per question. Data-size
arms use nested question subsets and matched state presentations/optimizer updates;
smaller pools replay states so every available question is still represented.
Recovery continuation arms start from identical clean weights, use identical pool
sizes and epoch counts, and select checkpoints using validation only.
''')
code(r'''
def make_grid():
    arms = [{"name": "baseline"}]
    if SUITE in {"core", "all"}:
        arms += [{"name": "fixed_u28", "fixed": "u28"},
                 {"name": "fixed_random28", "fixed": "random"},
                 {"name": "coverage_first", "coverage": "first"},
                 {"name": "coverage_all_matched", "coverage": "all_matched"}]
    if SUITE in {"initialization", "all"}:
        arms += [{"name": "weight_svd", "initialization": "weight_svd"}]
    if SUITE in {"losses", "all"}:
        arms += [{"name": "no_margin", "margin_weight": 0.0},
                 {"name": "no_top_token", "token_weight": 0.0}]
    if SUITE in {"recovery", "all"}:
        arms += [{"name": "recovery_" + kind, "recovery": kind}
                 for kind in ("repeat_clean", "teacher", "onpolicy")]
    if SUITE in {"data", "all"}:
        arms += [{"name": f"data_{n}", "questions": n} for n in DATA_SIZES]
    return arms
ARMS = make_grid()
LOCKED_CONFIG = dict(suite=SUITE, smoke=SMOKE, seeds=SEEDS, ranks=RANKS,
    fit=FIT_QUESTIONS, selection=SELECT_QUESTIONS, recovery=RECOVERY_QUESTIONS,
    clean_epochs=CLEAN_EPOCHS, recovery_epochs=RECOVERY_EPOCHS,
    lr=LEARNING_RATE, generation_batch=GENERATION_BATCH_SIZE,
    distill_batch=DISTILL_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
    split_seed=SPLIT_SEED, data_sizes=DATA_SIZES, datasets=EVAL_DATASETS,
    bootstrap_samples=BOOTSTRAP_SAMPLES, arms=ARMS, base_commit=CODE_COMMIT,
    ablation_source=ABLATION_SOURCE_SHA256, checkpoint=load_report.checkpoint_sha256,
    torch_version=torch.__version__, cuda_version=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0))
manifest_text = json.dumps(LOCKED_CONFIG, sort_keys=True, indent=2)
RUN_ID = hashlib.sha256(manifest_text.encode()).hexdigest()[:16]
RUN_DIR = pathlib.Path(OUTPUT_ROOT) / (('smoke_' if SMOKE else 'full_') + RUN_ID)
RUN_DIR.mkdir(parents=True, exist_ok=True)
if RESUME_ROOT:
    prior = pathlib.Path(RESUME_ROOT)
    assert (prior / 'manifest.json').read_text() == manifest_text, 'Resume configuration mismatch'
    shutil.copytree(prior, RUN_DIR, dirs_exist_ok=True)
manifest_path = RUN_DIR / 'manifest.json'
if manifest_path.exists():
    assert manifest_path.read_text() == manifest_text
manifest_path.write_text(manifest_text)

def save_json(path, value):
    tmp = pathlib.Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)

def save_pt(path, value):
    tmp = pathlib.Path(str(path) + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)

print('LOCKED:', RUN_DIR)
print('Fits per seed:', len(ARMS), '| head evaluations per dataset:', len(ARMS)*len(RANKS)*len(SEEDS))
print(json.dumps(ARMS, indent=2))
''')
md('## Unique-question partitions (training data only)')
code(r'''
from src.data.datasets import load_train_set, load_eval_set
training_rows = load_train_set(load_config('configs/data.yaml'), trace_style='eq_only')
def normalize_question(text):
    return ' '.join(str(text).casefold().split())
unique = {}
for row in training_rows:
    unique.setdefault(normalize_question(row['question']), str(row['question']))
unique_questions = list(unique.values())
order = torch.randperm(len(unique_questions), generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
required = FIT_QUESTIONS + SELECT_QUESTIONS + RECOVERY_QUESTIONS
assert required <= len(order)
chosen = [unique_questions[i] for i in order[:required]]
fit_questions = chosen[:FIT_QUESTIONS]
select_questions = chosen[FIT_QUESTIONS:FIT_QUESTIONS + SELECT_QUESTIONS]
recovery_questions = chosen[FIT_QUESTIONS + SELECT_QUESTIONS:]
assert len({normalize_question(q) for q in chosen}) == required
save_json(RUN_DIR / 'partitions.json', dict(fit=fit_questions, selection=select_questions,
                                         recovery=recovery_questions))
print(dict(fit=len(fit_questions), selection=len(select_questions), recovery=len(recovery_questions)))
''')
code(reuse('parity_questions ='))
md('''## Collect and cache teacher trajectories
Capture question IDs and token positions, including the termination decision.
Cached states are on CPU. The original head is restored even if collection fails.
''')
code(r'''
@torch.no_grad()
def collect_bundle(questions, head, tag):
    path = RUN_DIR / (tag + '.pt')
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    states, positions, question_ids = [], [], []
    base_model.set_output_embeddings(head)
    try:
        # Explicit chunks preserve the mapping from active rows to question IDs.
        for start in range(0, len(questions), GENERATION_BATCH_SIZE):
            chunk = questions[start:start + GENERATION_BATCH_SIZE]
            def observer(hidden, active_mask, answer_position):
                active = active_mask.detach().cpu().bool()
                states.append(hidden[active_mask].detach().cpu().float())
                positions.extend([int(answer_position)] * int(active.sum()))
                question_ids.extend((torch.arange(len(chunk))[active] + start).tolist())
            prepared = prepare_official_codi_batches(tokenizer, chunk,
                batch_size=GENERATION_BATCH_SIZE, length_bucketed=False)
            generate_official_codi_fast(model, tokenizer, prepared,
                latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
                device=device, answer_cue='The answer is:', answer_state_observer=observer)
            if start % (GENERATION_BATCH_SIZE * 16) == 0:
                print(tag, start, '/', len(questions), flush=True)
    finally:
        base_model.set_output_embeddings(full_head)
    bundle = dict(states=torch.cat(states), positions=torch.tensor(positions),
                  question_ids=torch.tensor(question_ids), questions=list(questions))
    assert len(bundle['states']) == len(bundle['positions']) == len(bundle['question_ids'])
    save_pt(path, bundle)
    return bundle

fit_bundle = collect_bundle(fit_questions, full_head, 'teacher_fit')
select_bundle = collect_bundle(select_questions, full_head, 'teacher_selection')
fit_states, select_states = fit_bundle['states'], select_bundle['states']
first_states = fit_states[fit_bundle['positions'] == 0]
assert len(first_states) == FIT_QUESTIONS
U28 = colon_basis(first_states)
save_pt(RUN_DIR / 'u28_train_only.pt', dict(basis=U28, questions=fit_questions, band=[4, 32]))
# Match optimizer updates while retaining every question in each data subset.
# Smaller subsets replay their states; larger subsets introduce new states.
DATA_STATE_BUDGET = len(fit_states)
print('States:', len(fit_states), len(select_states), '| data budget:', DATA_STATE_BUDGET)
''')
md('## Fit all locked arms before loading test labels')
code(r'''
from dataclasses import asdict

def arm_states(arm, seed):
    if arm.get('coverage') == 'first':
        return first_states
    if arm.get('coverage') == 'all_matched':
        generator = torch.Generator().manual_seed(seed)
        indices = []
        for question in range(FIT_QUESTIONS):
            eligible = torch.where(fit_bundle['question_ids'] == question)[0]
            indices.append(eligible[torch.randint(len(eligible), (1,), generator=generator)].item())
        return fit_states[indices]
    if 'questions' in arm:
        available = fit_states[fit_bundle['question_ids'] < arm['questions']]
        order = torch.randperm(len(available), generator=torch.Generator().manual_seed(seed))
        order = order.repeat((DATA_STATE_BUDGET + len(available) - 1) // len(available))[:DATA_STATE_BUDGET]
        return available[order]
    return fit_states

def new_head(arm, states, seed):
    torch.manual_seed(seed)
    basis = None
    if arm.get('fixed') == 'u28':
        basis = U28
    elif arm.get('fixed') == 'random':
        basis = torch.linalg.qr(torch.randn(hidden_size, 28,
            generator=torch.Generator().manual_seed(seed)), mode='reduced').Q
    return initialize_head(states, readout_weight, RANKS, bias=readout_bias, seed=seed,
        initialization=arm.get('initialization', 'whitened'), fixed_basis=basis).to(device)

def fit_head(head, states, arm, seed, epochs):
    return asdict(distil_nested_head(head, states, select_states, readout_weight,
        readout_bias=readout_bias, epochs=epochs, batch_size=DISTILL_BATCH_SIZE,
        learning_rate=LEARNING_RATE, temperature=2.0, kl_weight=1.0,
        token_weight=arm.get('token_weight', .25), margin_weight=arm.get('margin_weight', .25),
        nested_weight=.5, minimum_margin=.25, anchor_strength=1e-5, seed=seed))

def position_metrics(head):
    result = {}
    for rank in RANKS:
        result[str(rank)] = {}
        for label, mask in [('all', torch.ones(len(select_states), dtype=torch.bool)),
                            ('p0', select_bundle['positions'] == 0),
                            ('p1', select_bundle['positions'] == 1),
                            ('p2plus', select_bundle['positions'] >= 2)]:
            count = int(mask.sum())
            result[str(rank)][label] = dict(states=count, **(evaluate_nested_head(
                head, select_states[mask], readout_weight, readout_bias=readout_bias,
                rank=rank, batch_size=DISTILL_BATCH_SIZE) if count else {}))
    return result

for seed in SEEDS:
    for arm in ARMS:
        name = arm['name']
        path = RUN_DIR / f'head_{seed}_{name}.pt'
        if path.exists():
            print('Resume completed fit:', seed, name, flush=True)
            continue
        print('Fitting:', seed, name, flush=True)
        states = arm_states(arm, seed)
        head = new_head(arm, states, seed)
        initial = position_metrics(head)
        if 'recovery' in arm:
            baseline = torch.load(RUN_DIR / f'head_{seed}_baseline.pt', map_location='cpu', weights_only=False)
            head.load_state_dict(baseline['state_dict'])
            teacher_extra = collect_bundle(recovery_questions, full_head, 'teacher_recovery')['states']
            kind = arm['recovery']
            head.set_rank(64)
            if kind == 'onpolicy':
                extra = collect_bundle(recovery_questions, head, f'onpolicy_{seed}')['states']
            elif kind == 'teacher':
                extra = teacher_extra
            else:
                extra = fit_states
            # Fixed additional-state count for every continuation, independent of arm.
            count = RECOVERY_QUESTIONS
            extra = matched_state_sample(extra, count, seed)
            states = torch.cat((fit_states, extra))
            initial = position_metrics(head)
            history = fit_head(head, states, arm, seed + 1, RECOVERY_EPOCHS)
        else:
            history = fit_head(head, states, arm, seed, CLEAN_EPOCHS)
        payload = dict(arm=arm, seed=seed, train_states=len(states),
            initial=initial, final=position_metrics(head), history=history,
            state_dict={k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
        if arm.get('fixed'):
            rows = head.down.weight.detach()
            basis = rows[:28].T
            assert torch.max(torch.abs(rows[28:] @ basis)) < 2e-4
        save_pt(path, payload)
        save_json(path.with_suffix('.json'), {k:v for k,v in payload.items() if k != 'state_dict'})
        del head, payload
        torch.cuda.empty_cache()
        print('Saved:', path, flush=True)
print('All fits frozen. Test evaluation follows.')
''')
md('''## Locked full-answer evaluation and paired comparisons
Every head generates its own complete answer. Each completed arm is saved immediately;
an interrupted arm restarts, while completed arms are skipped on resume. Results include
per-question outputs, accuracy, baseline-correct to wrong flips, and paired bootstrap
intervals. Intervals are descriptive, unadjusted for multiple comparisons. No speed
claim is made from these single-run timings.
''')
code(r'''
from src.data.answer_extract import answers_match
import pandas as pd

@torch.no_grad()
def evaluate_generation(head, examples, path):
    if path.exists():
        return json.loads(path.read_text())
    base_model.set_output_embeddings(head)
    records = []
    started = time.perf_counter()
    try:
        for start in range(0, len(examples), GENERATION_BATCH_SIZE):
            chunk = examples[start:start + GENERATION_BATCH_SIZE]
            prepared = prepare_official_codi_batches(tokenizer, [str(x['question']) for x in chunk],
                batch_size=GENERATION_BATCH_SIZE, length_bucketed=False)
            generated = generate_official_codi_fast(model, tokenizer, prepared,
                latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=MAX_NEW_TOKENS,
                device=device, answer_cue='The answer is:')
            for offset, (example, text, tokens) in enumerate(zip(chunk, generated.texts, generated.token_ids)):
                records.append(dict(index=start+offset, question=str(example['question']),
                    gold=str(example['gold']), text=text, token_ids=list(tokens),
                    correct=bool(answers_match(text, example['gold']))))
    finally:
        base_model.set_output_embeddings(full_head)
    assert len(records) == len(examples)
    flags = [r['correct'] for r in records]
    result = dict(examples=len(flags), correct=sum(flags), accuracy=sum(flags)/len(flags),
                  flags=flags, seconds_single_run=time.perf_counter()-started, records=records)
    save_json(path, result)
    return result

rows, comparisons = [], []
for dataset in EVAL_DATASETS:
    examples = load_eval_set(dataset, load_config(cfg.data_config).eval[dataset])
    if dataset == 'gsm8k':
        assert len(examples) == 1319
    selected_overlap = {normalize_question(x['question']) for x in examples} & {normalize_question(q) for q in chosen}
    assert not selected_overlap, 'Evaluation question overlaps a fitting/selection/recovery question'
    if SMOKE:
        examples = examples[:8]
    dense = evaluate_generation(full_head, examples, RUN_DIR / f'eval_{dataset}_dense.json')
    rows.append(dict(dataset=dataset, seed=-1, arm='dense', rank=768,
                     accuracy=dense['accuracy'], correct=dense['correct'], retention=1.0))
    results = {}
    for seed in SEEDS:
        for arm in ARMS:
            payload = torch.load(RUN_DIR / f'head_{seed}_{arm["name"]}.pt', map_location='cpu', weights_only=False)
            head = new_head(arm, arm_states(arm, seed), seed)
            head.load_state_dict(payload['state_dict'])
            head.eval()
            for rank in RANKS:
                head.disable_adaptive()
                head.set_rank(rank)
                result = evaluate_generation(head, examples,
                    RUN_DIR / f'eval_{dataset}_{seed}_{arm["name"]}_r{rank}.json')
                results[(seed, arm['name'], rank)] = result
                rows.append(dict(dataset=dataset, seed=seed, arm=arm['name'], rank=rank,
                    accuracy=result['accuracy'], correct=result['correct'],
                    retention=result['accuracy']/dense['accuracy'] if dense['accuracy'] else None,
                    **paired_interval(dense['flags'], result['flags'], samples=BOOTSTRAP_SAMPLES, seed=seed)))
                pd.DataFrame(rows).to_csv(RUN_DIR / 'results.csv', index=False)
                print(dataset, seed, arm['name'], rank, result['correct'], '/', len(examples), flush=True)
            del head, payload
            torch.cuda.empty_cache()
    # Compare matched ablation arms directly, not only each arm against dense.
    pairs = [('baseline', 'fixed_u28'), ('fixed_random28', 'fixed_u28'),
             ('coverage_all_matched', 'coverage_first'), ('baseline', 'weight_svd'),
             ('baseline', 'no_margin'), ('baseline', 'no_top_token'),
             ('baseline', 'recovery_repeat_clean'), ('recovery_repeat_clean', 'recovery_teacher'),
             ('recovery_teacher', 'recovery_onpolicy')]
    pairs += [(f'data_{max(DATA_SIZES)}', f'data_{n}') for n in DATA_SIZES[:-1]]
    for seed in SEEDS:
        for reference, candidate in pairs:
            for rank in RANKS:
                a, b = (seed, reference, rank), (seed, candidate, rank)
                if a in results and b in results:
                    comparisons.append(dict(dataset=dataset, seed=seed, rank=rank,
                        reference=reference, candidate=candidate,
                        **paired_interval(results[a]['flags'], results[b]['flags'],
                                          samples=BOOTSTRAP_SAMPLES, seed=seed)))
        if (seed, 'fixed_u28', 64) in results:
            comparisons.append(dict(dataset=dataset, seed=seed, rank='96_vs_64',
                reference='baseline_r96', candidate='fixed_u28_r64',
                **paired_interval(results[(seed, 'baseline', 96)]['flags'],
                                  results[(seed, 'fixed_u28', 64)]['flags'],
                                  samples=BOOTSTRAP_SAMPLES, seed=seed)))
    pd.DataFrame(comparisons).to_csv(RUN_DIR / 'paired_comparisons.csv', index=False)
results_frame = pd.DataFrame(rows)
display(results_frame)
seed_summary = results_frame[results_frame.arm != 'dense'].groupby(['dataset', 'arm', 'rank']).accuracy.agg(['count', 'mean', 'std'])
seed_summary.to_csv(RUN_DIR / 'seed_summary.csv')
display(seed_summary)
save_json(RUN_DIR / 'completion.json', dict(complete=True, smoke=SMOKE, run_id=RUN_ID))
print('Download this output directory:', RUN_DIR)
''')
md('''## Reading and resuming results
- `results.csv`: per-seed, per-rank accuracy and paired differences versus dense.
- `paired_comparisons.csv`: direct controlled comparisons, including U28 rank 64 versus baseline rank 96.
- `seed_summary.csv`: mean and fitting-seed standard deviation (blank with one seed).
- `head_*.json`: train-state counts, epoch history, initial/final agreement and KL by position.
- `head_*.pt`: trained weights; `eval_*.json`: per-question predictions and paired flags.
- `manifest.json`, `partitions.json`, `u28_train_only.pt`: provenance and split/basis records.

For another session, attach the prior output as a Kaggle input and set `RESUME_ROOT` to
its exact run folder. Keep configuration, seeds, helper source and runtime identical.
Same-session reruns reuse the matching run folder automatically. Change `SUITE` before
starting each separately saved run; `all` enables every suite at once. Three seeds are
recommended for final comparisons. A smoke run verifies plumbing only.

This notebook refits all comparison heads fairly. Historical rank-96 weights and the
old colon-state cache are not required. Different models/checkpoints need their own
state collection and rank fitting; they cannot reuse this GPT-2 basis.
''')
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python', version='3'), kaggle=dict(accelerator='gpu', dataSources=[], isInternetEnabled=True)),
    nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'ablation-{i:03d}'
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(OUTPUT)
