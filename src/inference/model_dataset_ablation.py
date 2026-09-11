"""Model/dataset adapters for the fixed global-head experiment.

No compiler, quantization, batch-size or deployment-rank sweep lives here.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import torch
from torch import nn

try:
    import dual_global_head_runtime as reference
except ModuleNotFoundError:
    from src.inference import global_head_comparison as reference
from src.data.answer_extract import normalize_gold, normalize_number
from src.models.official_codi import official_codi_base_model


MODEL_SPECS = (
    dict(key='gpt2', label='GPT-2 / released CODI', repo='zen-E/CODI-gpt2'),
    dict(key='smollm2_135m', label='SmolLM2-135M', repo='HuggingFaceTB/SmolLM2-135M',
         revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'),
    dict(key='qwen2_5_0_5b', label='Qwen2.5-0.5B', repo='Qwen/Qwen2.5-0.5B',
         revision='060db6499f32faf8b98477b0a26969ef7d8b9987'),
)
DATASETS = ('gsm8k', 'svamp', 'asdiv')
GSM_REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
SVAMP_URL = 'https://raw.githubusercontent.com/arkilpatel/SVAMP/78e727689e1c1bebfc4be39c446898e8e10b0518/SVAMP.json'
ASDIV_URL = 'https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/883f90a9a65bf00304ba8f37423910fe743abc47/dataset/ASDiv.xml'
DATA_HASHES = {
    'svamp': '5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e',
    'asdiv': 'ef8904068482919ac48c8eeaaf6df344b8a308ba66d048c2d4d87eab82dc4929',
}
SCALAR_ANSWER = re.compile(r'([-+]?\$?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:%)?)(?:\s*\([^)]*\))?\s*')


def question_key(question):
    return ' '.join(str(question).casefold().split())


def parse_svamp(payload):
    rows = []
    for row in json.loads(payload):
        gold = normalize_number(str(row['Answer']))
        if gold is None:
            raise ValueError(f"Invalid SVAMP answer: {row['ID']}")
        rows.append(dict(id=row['ID'], question=f"{row['Body'].strip()} {row['Question'].strip()}", gold=str(gold)))
    return rows


def parse_asdiv(payload):
    """Keep only single scalar gold answers compatible with the existing scorer.

    Do not turn '3:30' into 30, '5; 15; 20' into 20, or 'February 3rd' into 3.
    Units in parentheses are stripped before numeric normalization.
    """
    rows, excluded = [], []
    for problem in ET.fromstring(payload).findall('.//Problem'):
        answer = problem.findtext('Answer', '').strip()
        match = SCALAR_ANSWER.fullmatch(answer)
        if match is None:
            excluded.append(dict(id=problem.get('ID'), answer=answer, reason='not a single scalar numeric answer'))
            continue
        gold = normalize_number(match.group(1))
        if gold is None:
            raise ValueError(f'Invalid ASDiv scalar answer: {answer}')
        question = ' '.join(problem.findtext(name, '').strip() for name in ('Body', 'Question'))
        rows.append(dict(id=problem.get('ID'), question=question, gold=str(gold)))
    return rows, excluded


def download_datasets(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    def read(key, url):
        path = cache_dir / key
        if not path.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=60) as response:
                        payload = response.read()
                    break
                except (HTTPError, URLError, TimeoutError):
                    if attempt == 2: raise
                    time.sleep(2*(attempt+1))
            if key in DATA_HASHES and hashlib.sha256(payload).hexdigest() != DATA_HASHES[key]:
                raise ValueError(f'{key}: downloaded dataset checksum mismatch')
            temp = path.with_suffix('.tmp')
            temp.write_bytes(payload)
            temp.replace(path)
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if key in DATA_HASHES and sha != DATA_HASHES[key]:
            raise ValueError(f'{key}: dataset checksum mismatch; remove the corrupt cached file and retry')
        sources[key] = dict(url=url, sha256=sha)
        return payload
    prefix = f'https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REVISION}/grade_school_math/data/'
    train = [json.loads(line) for line in read('gsm8k_train', prefix+'train.jsonl').splitlines() if line.strip()]
    test = [json.loads(line) for line in read('gsm8k_test', prefix+'test.jsonl').splitlines() if line.strip()]
    evaluation = {'gsm8k': [dict(id=f'gsm8k-test-{i}', question=r['question'],
                         gold=str(normalize_gold(r['answer'], 'gsm8k_main'))) for i, r in enumerate(test)]}
    evaluation['svamp'] = parse_svamp(read('svamp', SVAMP_URL))
    evaluation['asdiv'], excluded = parse_asdiv(read('asdiv', ASDIV_URL))
    if (len(train), len(test), len(evaluation['svamp']), len(evaluation['asdiv']), len(excluded)) != (7473,1319,1000,2084,221):
        raise ValueError('Pinned dataset counts changed')
    if any(r['gold']=='None' for rows in evaluation.values() for r in rows):
        raise ValueError('Missing numeric gold')
    return train, evaluation, dict(sources=sources, asdiv_excluded=excluded)


def partition_train(train, evaluation, seed=89):
    """The same GSM8K fit/selection/recovery/timing/warmup split as before."""
    unique = {}
    for row in train:
        unique.setdefault(question_key(row['question']), dict(question=str(row['question']), gold=str(row['answer'])))
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    sizes = dict(fit=1024, selection=256, recovery=256, timing=16, warmup=4)
    splits, start = {}, 0
    for name, size in sizes.items():
        splits[name] = rows[start:start+size]
        start += size
    if start > len(rows):
        raise ValueError('Insufficient unique GSM8K training questions')
    used = {question_key(r['question']) for name in ('fit','selection','recovery','warmup') for r in splits[name]}
    for dataset, questions in evaluation.items():
        overlap = used & {question_key(r['question']) for r in questions}
        if overlap:
            raise ValueError(f'{dataset}: {len(overlap)} evaluation questions overlap fitting/warmup data')
    return splits


def timing_questions(dataset, splits, evaluation, seed=89):
    if dataset == 'gsm8k':
        return splits['timing']
    rows = list(evaluation[dataset])
    random.Random(seed).shuffle(rows)
    return rows[:16]


class NoLatentProjector(nn.Module):
    def forward(self, x):
        raise RuntimeError('This backbone has no trained CODI projector')


class BackboneView(nn.Module):
    """Expose the original body/head under the existing decoder's access contract."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.config.n_positions = int(self.config.max_position_embeddings)

    @property
    def transformer(self):
        return self.backbone.model

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)


class ExplicitBackbone(nn.Module):
    """Original pretrained model, without fabricated latent-reasoning weights."""
    def __init__(self, backbone):
        super().__init__()
        self.codi = BackboneView(backbone)
        self.eot_id = int(backbone.get_output_embeddings().weight.shape[0])
        self.prj = NoLatentProjector()
        self.has_codi = False

    @property
    def config(self):
        return self.codi.config

    def input_embeddings(self):
        return self.codi.backbone.get_input_embeddings()


def decode(model, tokenizer, batches, head, *, mode, **kwargs):
    if mode == 'codi' and getattr(model, 'has_codi', True) is False:
        raise ValueError('CODI is unavailable for this untrained backbone')
    return reference.decode(model, tokenizer, batches, head, mode=mode, **kwargs)


def block_roots(model):
    body = official_codi_base_model(model).transformer
    blocks = body.h if hasattr(body, 'h') else body.layers
    roots = [(f'transformer.h.{i}', block) for i, block in enumerate(blocks)]
    # Use the existing categories without renaming or changing actual model modules.
    if hasattr(body, 'wte'):
        roots += [('transformer.'+name, getattr(body,name)) for name in ('wte','wpe','ln_f')]
    else:
        roots += [('transformer.wte', body.embed_tokens), ('transformer.ln_f', body.norm)]
    return roots, len(blocks)


@torch.inference_mode()
def profile_question(model, tokenizer, head, question, *, mode, device, max_new_tokens,
                     question_id=0, repeat=0, arm='rank96', latent_iterations=6):
    trace = reference.Timeline(device, unified_clock=True)
    trace.context.update(mode=mode, arm=arm, question_id=question_id, repeat=repeat, phase='shared')
    roots, _ = block_roots(model)
    roots += [('projector', model.prj), ('lm_head', head)]
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    with trace.modules(roots, recursive=False):
        with trace.span('question_total'):
            with trace.span('load_question', gpu=False):
                questions = [str(question)]
            batches = reference.prepare_questions(tokenizer, questions, 1, trace)
            result = decode(model, tokenizer, batches, head, mode=mode, device=device,
                max_new_tokens=max_new_tokens, latent_iterations=latent_iterations, timeline=trace)
    trace.resolve()
    sample = reference.partition_question(trace.records)
    sample.update(arm=arm, tokens=list(result.token_ids[0]), text=result.texts[0],
        prompt_tokens=int(batches[0].mask.sum()), latent_steps=latent_iterations if mode=='codi' else 0,
        visible_tokens=result.generated_token_counts[0])
    return sample, trace.records


def bottleneck_means(samples, *, n_layers, per_token=False):
    """Keep four columns; absent CODI is NaN, not a misleading zero."""
    names = list(reference.BREAKDOWN_ROWS)
    first = names.index('Transformer block 01')
    names = names[:first] + [f'Transformer block {i+1:02d}' for i in range(n_layers)] + names[first+12:]
    columns = reference.COLUMNS
    rows = {name: dict.fromkeys(columns, float('nan')) for name in names}
    totals = dict.fromkeys(columns, float('nan'))
    for mode in ('explicit_cot','codi'):
        group = [s for s in samples if s['mode']==mode]
        if not group:
            continue
        active = ['Explicit'] if mode=='explicit_cot' else list(columns[1:])
        for column in active:
            totals[column] = 0.
            for row in rows.values(): row[column] = 0.
        overall = active[0]
        visible = sum(s['visible_tokens'] for s in group)
        latent = sum(s['latent_steps'] for s in group)
        den = dict.fromkeys(active, len(group))
        if per_token:
            den[overall] = visible + (latent if mode=='codi' else 0)
            if mode=='codi': den.update({'CODI latent only':latent, 'CODI visible only':visible})
        if min(den.values()) <= 0: raise ValueError('Positive generation counts required')
        for sample in group:
            totals[overall] += sample['total_ms']/den[overall]
            for part in sample['breakdown']:
                rows[part['name']][overall] += part['ms']/den[overall]
                if mode=='codi' and part['phase'] in ('latent','visible'):
                    col = 'CODI '+part['phase']+' only'
                    rows[part['name']][col] += part['ms']/den[col]
                    totals[col] += part['ms']/den[col]
    if all(torch.isnan(torch.tensor(x)) for x in totals.values()): raise ValueError('No samples')
    return [('Total average time',totals), *rows.items(), ('Total average time (repeat)',dict(totals))]
