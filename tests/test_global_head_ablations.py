"""Scientific invariants for the global-head ablation helpers and notebook."""
import ast
import json
from pathlib import Path

import torch

from src.mech.global_head_ablations import (
    colon_basis, initialize_head, matched_state_sample, paired_interval,
)
from src.mech.global_low_rank_head import distil_nested_head


def test_fixed_basis_and_orthogonal_residual_survive_distillation_and_reload():
    torch.manual_seed(12)
    states = torch.randn(40, 10)
    teacher = torch.randn(19, 10)
    basis = torch.linalg.qr(torch.randn(10, 2), mode='reduced').Q
    head = initialize_head(states, teacher, (3, 6), fixed_basis=basis, seed=7)
    before = head.up.weight.detach().clone()
    distil_nested_head(head, states[:24], states[24:], teacher,
                       epochs=2, batch_size=8, seed=3)
    assert torch.equal(head.down.weight[:2], basis.T)
    assert torch.max(torch.abs(head.down.weight[2:] @ basis)) < 1e-5
    assert not torch.equal(head.up.weight, before)
    restored = initialize_head(states, teacher, (3, 6), fixed_basis=basis, seed=7)
    restored.load_state_dict(head.state_dict())
    for rank in (3, 6):
        assert torch.allclose(head.forward_rank(states, rank), restored.forward_rank(states, rank))


def test_initializers_preserve_dense_logits_at_training_mean():
    torch.manual_seed(4)
    states, weight, bias = torch.randn(30, 8), torch.randn(17, 8), torch.randn(17)
    centre = states.mean(0)
    for initialization in ('whitened', 'weight_svd'):
        head = initialize_head(states, weight, (3, 6), bias=bias, initialization=initialization)
        for rank in (3, 6):
            assert torch.allclose(head.forward_rank(centre, rank), weight @ centre + bias, atol=1e-5)


def test_colon_basis_is_expected_pca_band():
    torch.manual_seed(8)
    states = torch.randn(80, 10) * torch.arange(1, 11)
    basis = colon_basis(states, 2, 5)
    _, _, vt = torch.linalg.svd(states.double() - states.double().mean(0), full_matrices=False)
    expected = vt[2:5].T.float()
    assert torch.allclose(basis @ basis.T, expected @ expected.T, atol=1e-5)


def test_sampling_and_paired_effects():
    states = torch.arange(100).reshape(20, 5)
    first = matched_state_sample(states, 8, 4)
    assert torch.equal(first, matched_state_sample(states, 8, 4))
    assert len(torch.unique(first[:, 0])) == 8
    effect = paired_interval([True, True, False, False], [True, False, True, False], samples=100)
    assert effect['delta_pp'] == 0
    assert effect['correct_to_wrong'] == effect['wrong_to_correct'] == 1


def test_notebook_compiles_and_embeds_current_helpers():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/kaggle_codi_global_head_ablations.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for index, source in enumerate(sources):
        compile(source, f'cell-{index}', 'exec')
    helper = (root / 'src/mech/global_head_ablations.py').read_text(encoding='utf-8').replace('from __future__ import annotations\n', '')
    assert helper in next(s for s in sources if 'ABLATION_SOURCE_SHA256 =' in s)
    assert not any('REPRODUCTION_SUMMARY' in source or 'verify_full_reproduction_gate' in source for source in sources)
    # Exercise every suite grid without importing/downloading a model.
    grid_source = next(s for s in sources if 'def make_grid' in s)
    tree = ast.parse(grid_source)
    function = ast.Module(body=[tree.body[0]], type_ignores=[])
    for suite in ('core', 'initialization', 'losses', 'recovery', 'data', 'all'):
        scope = dict(SUITE=suite, DATA_SIZES=[128, 256, 512, 1024])
        exec(compile(function, 'grid', 'exec'), scope)
        arms = scope['make_grid']()
        assert arms[0]['name'] == 'baseline'
        assert len({a['name'] for a in arms}) == len(arms)
    assert len(arms) == 15


def test_notebook_all_suites_execute_with_synthetic_decoder(tmp_path):
    """Exercise notebook orchestration without network, CODI weights, or a GPU."""
    from decimal import Decimal
    from types import SimpleNamespace
    from src.mech.global_head_ablations import initialize_head, colon_basis, matched_state_sample, paired_interval
    from src.mech.global_low_rank_head import distil_nested_head, evaluate_nested_head
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/kaggle_codi_global_head_ablations.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    import pathlib
    import time
    teacher = torch.nn.Linear(104, 120)
    for p in teacher.parameters():
        p.requires_grad_(False)
    base = SimpleNamespace(head=teacher)
    base.set_output_embeddings = lambda head: setattr(base, 'head', head)
    def prepare(tokenizer, questions, **kwargs):
        return questions
    def generate(model, tokenizer, prepared, answer_state_observer=None, **kwargs):
        n = len(prepared)
        texts = ['0'] * n
        for position in range(3):
            hidden = torch.stack([torch.randn(104, generator=torch.Generator().manual_seed(
                sum(map(ord, q)) + position * 1009)) for q in prepared])
            active = torch.ones(n, dtype=torch.bool)
            if answer_state_observer:
                answer_state_observer(hidden, active, position)
            logits = base.head(hidden)
            texts = [str(int(v)) for v in logits.argmax(-1)]
        return SimpleNamespace(texts=tuple(texts), token_ids=tuple((int(t),) for t in texts))
    def save_json(path, value):
        path.write_text(json.dumps(value, default=str))
    fit = [f'fit_{i}' for i in range(48)]
    selection = [f'selection_{i}' for i in range(8)]
    recovery = [f'recovery_{i}' for i in range(8)]
    scope = dict(torch=torch, pathlib=pathlib, json=json, time=time,
        SUITE='all', DATA_SIZES=[36, 48], RUN_DIR=tmp_path, SEEDS=[3], RANKS=(32, 64, 96),
        FIT_QUESTIONS=48, RECOVERY_QUESTIONS=8, CLEAN_EPOCHS=1, RECOVERY_EPOCHS=1,
        LEARNING_RATE=2e-4, GENERATION_BATCH_SIZE=8, DISTILL_BATCH_SIZE=16,
        MAX_NEW_TOKENS=4, BOOTSTRAP_SAMPLES=100, SMOKE=True, RUN_ID='synthetic',
        EVAL_DATASETS=['gsm8k'], cfg=SimpleNamespace(eval=SimpleNamespace(latent_iterations=6), data_config='unused'),
        model=None, tokenizer=None, device=torch.device('cpu'), base_model=base, full_head=teacher,
        hidden_size=104, vocabulary_size=120, readout_weight=teacher.weight, readout_bias=teacher.bias,
        fit_questions=fit, select_questions=selection, recovery_questions=recovery,
        chosen=fit+selection+recovery, unique={q:q for q in fit+selection+recovery},
        normalize_question=lambda q: q, save_json=save_json,
        save_pt=lambda path,value: torch.save(value,path),
        prepare_official_codi_batches=prepare, generate_official_codi_fast=generate,
        initialize_head=initialize_head, colon_basis=colon_basis,
        matched_state_sample=matched_state_sample, paired_interval=paired_interval,
        distil_nested_head=distil_nested_head, evaluate_nested_head=evaluate_nested_head,
        display=lambda value: None,
        load_config=lambda path: SimpleNamespace(eval={'gsm8k':{}}),
        load_eval_set=lambda *args: [dict(question=f'test_{i}', gold=Decimal(0)) for i in range(1319)])
    grid = ast.parse(next(s for s in sources if 'def make_grid' in s)).body[0]
    exec(compile(ast.Module(body=[grid], type_ignores=[]), 'grid', 'exec'), scope)
    scope['ARMS'] = scope['make_grid']()
    for marker in ('def collect_bundle', 'def arm_states', 'def evaluate_generation'):
        source = next(s for s in sources if marker in s)
        exec(compile(source, marker, 'exec'), scope)
    assert json.loads((tmp_path / 'completion.json').read_text())['complete']
    import pandas as pd
    results = pd.read_csv(tmp_path / 'results.csv')
    assert len(results) == 1 + len(scope['ARMS']) * 3
    assert len(list(tmp_path.glob('head_*.pt'))) == len(scope['ARMS'])
    assert (tmp_path / 'paired_comparisons.csv').stat().st_size > 100
    # All completed artifacts should be reused; generation must not be called again.
    scope['generate_official_codi_fast'] = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('resume generated'))
    for marker in ('def collect_bundle', 'def arm_states', 'def evaluate_generation'):
        exec(compile(next(s for s in sources if marker in s), marker, 'exec'), scope)
