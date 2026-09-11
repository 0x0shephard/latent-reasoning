"""CODI/explicit decoding and an additive, four-column bottleneck report.

The report partitions a single CUDA-stream timeline, including host-induced idle
intervals. It is an instrumented elapsed-time breakdown, not summed kernel time.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import time
import uuid

import torch
from torch import nn
from src.models.official_codi import official_codi_base_model, _normalized_official_questions
from src.inference.official_codi_fast import FastCODIGeneration


class Timeline:
    def __init__(self, device='cpu', enabled=True, unified_clock=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.unified_clock = unified_clock
        self.context = {}
        self.records = []
        self.pending = []
        self.stack = []
        self.counter = 0
        self.trace_id = uuid.uuid4().hex

    def start(self, name, kind='stage', gpu=True, **extra):
        if not self.enabled:
            return None
        self.counter += 1
        row = dict(self.context, trace_id=self.trace_id, event_id=self.counter,
                   parent_id=self.stack[-1] if self.stack else None,
                   name=name, kind=kind, **extra)
        marker = None if self.unified_clock else torch.profiler.record_function(name)
        if marker is not None:
            marker.__enter__()
        row['_wall_start'] = time.perf_counter()
        events = None
        if (gpu or self.unified_clock) and self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        self.stack.append(self.counter)
        return row, events, marker

    def stop(self, token):
        if token is None:
            return
        row, events, marker = token
        if events:
            events[1].record()
        row['cpu_wall_ms'] = (time.perf_counter() - row.pop('_wall_start')) * 1000
        if marker is not None:
            marker.__exit__(None, None, None)
        if self.stack and self.stack[-1] == row['event_id']:
            self.stack.pop()
        self.pending.append((row, events))

    @contextmanager
    def span(self, name, kind='stage', gpu=True, **extra):
        token = self.start(name, kind, gpu, **extra)
        try:
            yield
        finally:
            self.stop(token)

    @contextmanager
    def metadata(self, **values):
        previous = self.context.copy()
        self.context.update(values)
        try:
            yield
        finally:
            self.context = previous

    def resolve(self):
        if self.device.type == 'cuda' and self.pending:
            torch.cuda.synchronize(self.device)
        for row, events in self.pending:
            row['cuda_stream_ms'] = events[0].elapsed_time(events[1]) if events else None
            self.records.append(row)
        self.pending.clear()

    @contextmanager
    def modules(self, roots, recursive=True):
        if not self.enabled:
            yield
            return
        handles, stacks, seen = [], {}, set()
        for prefix, root in roots:
            for suffix, module in (root.named_modules() if recursive else [('', root)]):
                if id(module) in seen:
                    continue
                seen.add(id(module))
                name = prefix + ('.' + suffix if suffix else '')
                stacks[id(module)] = []
                def pre(mod, args, label=name):
                    token = self.start(label, kind='module', inclusive=True,
                                       module_type=type(mod).__name__)
                    stacks[id(mod)].append(token)
                def post(mod, args, output):
                    self.stop(stacks[id(mod)].pop())
                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post, always_call=True))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def flush(self, path):
        self.resolve()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'at', encoding='utf-8') as handle:
            for row in self.records:
                handle.write(json.dumps(row, default=str) + '\n')
        self.records.clear()


@dataclass
class PreparedBatch:
    indices: tuple
    ids: torch.Tensor
    mask: torch.Tensor


def prepare_questions(tokenizer, questions, batch_size, timeline=None):
    timeline = timeline or Timeline(enabled=False)
    with timeline.span('question_normalization', gpu=False):
        questions = _normalized_official_questions(questions)
    with timeline.span('question_tokenization', gpu=False):
        encoded = tokenizer(questions, add_special_tokens=False, padding=False)['input_ids']
    result = []
    for start in range(0, len(encoded), batch_size):
        part = encoded[start:start+batch_size]
        with timeline.metadata(batch_index=start//batch_size):
            with timeline.span('cpu_padding_and_tensor_allocation', gpu=False):
                width = max(map(len, part))
                ids = torch.full((len(part), width), tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, values in enumerate(part):
                    if not values:
                        raise ValueError('Empty question')
                    ids[i, -len(values):] = torch.tensor(values)
                    mask[i, -len(values):] = 1
            result.append(PreparedBatch(tuple(range(start, start+len(part))), ids, mask))
    return result


def select_token(head, hidden, vocabulary_stop):
    specialized = getattr(head, 'select_token', None)
    return specialized(hidden, vocabulary_stop=vocabulary_stop) if callable(specialized) else head(hidden)[..., :vocabulary_stop].argmax(-1)


@torch.no_grad()
def decode(model, tokenizer, batches, head, *, mode, device, max_new_tokens=256,
           latent_iterations=6, timeline=None, observer=None, forced_tokens=None):
    """Same CODI body-only contract; explicit mode starts directly from the question.

    Fixed replay evaluates every head but feeds back dense-reference token IDs.
    Timed CUDA paths must be called inside torch.inference_mode() by the runner.
    """
    if mode not in ('codi', 'explicit_cot'):
        raise ValueError(mode)
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    timeline = timeline or Timeline(device, enabled=False)
    device = torch.device(device)
    base = official_codi_base_model(model)
    body, embedding = base.transformer, model.input_embeddings()
    model.eval()
    head.eval()
    vocabulary_stop = int(model.eot_id)
    eos = int(tokenizer.eos_token_id)
    total = sum(len(b.indices) for b in batches)
    outputs, texts, counts_out = [None]*total, [None]*total, [0]*total
    with timeline.span('answer_cue_tokenization', gpu=False):
        cue_ids = tokenizer(' The answer is:', add_special_tokens=False)['input_ids'] if mode == 'codi' else []
    for batch_number, batch in enumerate(batches):
        with timeline.metadata(batch_index=batch_number, question_indices=list(batch.indices), token_position=-1):
            with timeline.span('host_to_device_input_ids'):
                ids = batch.ids.to(device, non_blocking=True)
            with timeline.span('host_to_device_attention_mask'):
                mask = batch.mask.to(device, non_blocking=True)
            with timeline.span('prompt_tensor_construction'):
                if mode == 'codi':
                    bot = torch.full((len(ids),1), model.bot_id, device=device, dtype=torch.long)
                    ids = torch.cat((ids,bot),1)
                    mask = torch.cat((mask,torch.ones_like(bot)),1)
            # Do not silently truncate GPT-2 context.
            maximum = getattr(model.config, 'n_positions', 1024) if hasattr(model, 'config') else 1024
            reserve = latent_iterations + 1 + len(cue_ids) if mode == 'codi' else 0
            if ids.shape[1] + reserve + max_new_tokens > maximum:
                raise ValueError('Prompt plus generation exceeds context; reduce the configured token cap')
            with timeline.span('prefill_position_ids'):
                positions = mask.long().cumsum(-1)-1 if mode == 'explicit_cot' else None
                if positions is not None:
                    positions.masked_fill_(mask == 0, 1)
            with timeline.metadata(phase='prefill'):
                with timeline.span('transformer_prefill'):
                    prefill_kwargs = dict(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
                    if getattr(getattr(body, 'config', None), 'model_type', None) == 'gpt2':
                        from transformers import DynamicCache
                        prefill_kwargs['past_key_values'] = DynamicCache()
                    if positions is not None:
                        prefill_kwargs['position_ids'] = positions
                    out = body(**prefill_kwargs)
            with timeline.span('kv_cache_reference_update'):
                cache = out.past_key_values
                hidden = out.last_hidden_state[:, -1:, :]
            if mode == 'codi':
                with timeline.metadata(phase='latent'), timeline.span('phase_latent'):
                    with timeline.metadata(phase='latent_projection_initial'):
                        with timeline.span('latent_projector'):
                            latent = model.prj(hidden)
                    for step in range(latent_iterations):
                        with timeline.metadata(phase='latent', latent_step=step):
                            with timeline.span('transformer_latent_pass'):
                                out = body(inputs_embeds=latent, past_key_values=cache, use_cache=True, return_dict=True)
                            with timeline.span('kv_cache_reference_update'):
                                cache = out.past_key_values
                            with timeline.span('latent_projector'):
                                latent = model.prj(out.last_hidden_state[:, -1:, :])
            with timeline.metadata(phase='visible_decode'), timeline.span('phase_visible'):
                if mode == 'codi':
                    with timeline.metadata(phase='answer_cue'):
                        with timeline.span('answer_cue_tensor_construction'):
                            cue = torch.tensor([model.eot_id,*cue_ids], device=device).unsqueeze(0).expand(len(ids),-1)
                        with timeline.span('answer_cue_embedding'):
                            cue_embedding = embedding(cue)
                        with timeline.span('transformer_answer_cue'):
                            out = body(inputs_embeds=cue_embedding, past_key_values=cache, use_cache=True, return_dict=True)
                        cache = out.past_key_values
                        hidden = out.last_hidden_state[:, -1:, :]
                with timeline.span('generation_buffer_allocation'):
                    tokens = torch.full((len(ids), max_new_tokens), eos, device=device, dtype=torch.long)
                    counts = torch.zeros(len(ids), device=device, dtype=torch.long)
                    finished = torch.zeros(len(ids), device=device, dtype=torch.bool)
                replay = None
                if forced_tokens is not None:
                    with timeline.span('replay_cpu_padding', gpu=False):
                        seqs = [forced_tokens[i] for i in batch.indices]
                        limit = max(map(len,seqs))
                        if limit > max_new_tokens or any(not seq for seq in seqs):
                            raise ValueError('Replay tokens must fit the generation cap')
                        replay_cpu = torch.full((len(ids),limit), eos, dtype=torch.long)
                        for row, seq in enumerate(seqs):
                            replay_cpu[row,:len(seq)] = torch.tensor(seq)
                    with timeline.span('host_to_device_replay_tokens'):
                        replay = replay_cpu.to(device)
                else:
                    limit = max_new_tokens
                for position in range(limit):
                    with timeline.metadata(phase='visible_decode', token_position=position):
                        if observer:
                            with timeline.span('collect_hidden_state_to_cpu'):
                                observer(hidden[:, -1, :], ~finished, position, batch.indices)
                        with timeline.span('lm_head_and_argmax'):
                            predicted = select_token(head, hidden[:, -1, :], vocabulary_stop)
                        with timeline.span('token_selection_and_buffers'):
                            token = predicted if replay is None else replay[:, position]
                            active = ~finished
                            tokens[:,position] = torch.where(active,token,tokens[:,position])
                            counts += active.long()
                            finished |= active & (token == eos)
                        with timeline.span('termination_check_host_sync'):
                            stop = position+1 == limit or bool(finished.all())
                        if stop:
                            break
                        with timeline.span('next_token_embedding'):
                            embedded = embedding(token).unsqueeze(1)
                        with timeline.span('decode_attention_mask_update'):
                            if mode == 'explicit_cot':
                                mask = torch.cat((mask,torch.ones((len(ids),1),device=device,dtype=mask.dtype)),1)
                        with timeline.span('transformer_visible_token'):
                            kwargs = dict(inputs_embeds=embedded, past_key_values=cache, use_cache=True, return_dict=True)
                            if mode == 'explicit_cot':
                                kwargs['attention_mask'] = mask
                                kwargs['position_ids'] = (mask.long().sum(-1)-1).unsqueeze(1)
                            out = body(**kwargs)
                        with timeline.span('kv_cache_reference_update'):
                            cache, hidden = out.past_key_values, out.last_hidden_state
                with timeline.span('device_to_host_token_buffer'):
                    cpu_tokens = tokens.cpu()
                with timeline.span('device_to_host_counts'):
                    cpu_counts = counts.cpu().tolist()
                with timeline.span('cpu_token_conversion_and_text_decode', gpu=False):
                    for row,index in enumerate(batch.indices):
                        count = int(cpu_counts[row])
                        seq = tuple(int(t) for t in cpu_tokens[row,:count].tolist())
                        outputs[index], counts_out[index] = seq, count
                        texts[index] = tokenizer.decode(seq, skip_special_tokens=True)
            if not timeline.unified_clock:
                timeline.resolve()
    return FastCODIGeneration(tuple(texts),tuple(outputs),tuple(counts_out))


class DenseSelector(nn.Module):
    def __init__(self, head, vocabulary_size):
        super().__init__()
        self.head = head
        self.vocabulary_size = vocabulary_size
    def forward(self, hidden):
        return self.head(hidden)[..., :self.vocabulary_size]
    def select_token(self, hidden, *, vocabulary_stop):
        return self(hidden).argmax(-1)


class FixedRankHead(nn.Module):
    def __init__(self, source, rank=96):
        super().__init__()
        self.vocabulary_size = source.vocabulary_size
        self.down = nn.Linear(source.hidden_size,rank)
        self.up = nn.Linear(rank,source.vocabulary_size)
        with torch.no_grad():
            self.down.weight.copy_(source.down.weight[:rank])
            self.down.bias.copy_(source.down.bias[:rank])
            self.up.weight.copy_(source.up.weight[:,:rank])
            self.up.bias.copy_(source.up.bias)
        self.requires_grad_(False)
    def forward(self, hidden):
        return self.up(self.down(hidden))
    def select_token(self, hidden, *, vocabulary_stop):
        return self(hidden).argmax(-1)



COLUMNS = ('Explicit', 'CODI overall', 'CODI latent only', 'CODI visible only')
BREAKDOWN_ROWS = (
    'Question loading / tokenization', 'CPU padding / allocation',
    'Input transfer to GPU', 'Embeddings',
    *(f'Transformer block {i + 1:02d}' for i in range(12)),
    'Transformer final norm', 'Transformer masks / bookkeeping',
    'Latent projector', 'LM head', 'Argmax / head dispatch',
    'Token buffers / cache updates', 'EOS check / synchronization',
    'Output transfer to CPU', 'Text decoding', 'Python / tracing gaps',
)


def _category(row, ancestors):
    # Descendant events inherit their enclosing component. Summing exclusive
    # durations reconstructs that component without counting nested modules twice.
    for event in [row, *ancestors]:
        name = event['name']
        if name.startswith('transformer.h.'):
            return f"Transformer block {int(name.split('.')[2]) + 1:02d}"
        if name.startswith('transformer.ln_f'):
            return 'Transformer final norm'
        if name.startswith(('transformer.wte', 'transformer.wpe')):
            return 'Embeddings'
        if name.startswith('projector'):
            return 'Latent projector'
        if name.startswith('lm_head') and name != 'lm_head_and_argmax':
            return 'LM head'
    name = row['name']
    if name in ('load_question', 'question_normalization', 'question_tokenization', 'answer_cue_tokenization'):
        return 'Question loading / tokenization'
    if name == 'cpu_padding_and_tensor_allocation':
        return 'CPU padding / allocation'
    if name.startswith('host_to_device'):
        return 'Input transfer to GPU'
    if name in ('next_token_embedding', 'answer_cue_embedding'):
        return 'Embeddings'
    if name.startswith('transformer_') or name in ('prefill_position_ids', 'decode_attention_mask_update'):
        return 'Transformer masks / bookkeeping'
    if name == 'latent_projector':
        return 'Latent projector'
    if name == 'lm_head_and_argmax':
        return 'Argmax / head dispatch'
    if name == 'termination_check_host_sync':
        return 'EOS check / synchronization'
    if name.startswith('device_to_host'):
        return 'Output transfer to CPU'
    if name == 'cpu_token_conversion_and_text_decode':
        return 'Text decoding'
    if name in ('kv_cache_reference_update', 'generation_buffer_allocation',
                'token_selection_and_buffers', 'prompt_tensor_construction', 'answer_cue_tensor_construction'):
        return 'Token buffers / cache updates'
    return 'Python / tracing gaps'


def partition_question(events):
    """Exclusive intervals on one clock; never add CPU and CUDA measurements."""
    by_id = {r['event_id']: r for r in events}
    roots = [r for r in events if r['name'] == 'question_total']
    if len(roots) != 1:
        raise ValueError('Expected exactly one complete question trace')
    root = roots[0]
    clock = 'cuda_stream_ms' if root.get('cuda_stream_ms') is not None else 'cpu_wall_ms'
    if any(r.get(clock) is None for r in events):
        raise ValueError('Every span must use the same clock; enable unified_clock')
    children = {}
    for row in events:
        children.setdefault(row.get('parent_id'), []).append(row)
    values = {}
    for row in events:
        exclusive = row[clock] - sum(c[clock] for c in children.get(row['event_id'], []))
        if exclusive < -0.01:
            raise ValueError(f"Overlapping timing spans: {row['name']}: {exclusive} ms")
        # Keep sub-microsecond event rounding differences so totals reconcile.
        ancestors = []
        parent = row.get('parent_id')
        while parent is not None:
            ancestors.append(by_id[parent]); parent = by_id[parent].get('parent_id')
        category = _category(row, ancestors)
        phase = row.get('phase', 'shared')
        phase = ('latent' if phase in ('latent', 'latent_projection_initial') else
                 'visible' if phase in ('visible_decode', 'answer_cue') else 'shared')
        key = (category, phase)
        values[key] = values.get(key, 0.0) + exclusive
    return dict(mode=root['mode'], question_id=root.get('question_id'), repeat=root.get('repeat'),
                total_ms=root[clock], clock=clock,
                breakdown=[dict(name=name, phase=phase, ms=value) for (name, phase), value in values.items()])


def bottleneck_means(samples, *, per_token=False):
    """Question means or aggregate time / native generated-step count.

    Token units: explicit = visible output token; CODI overall = latent + visible
    step; CODI latent = latent step; CODI visible = visible output token. Prompt and
    forced cue work are amortized, not added to the generated-step denominator.
    """
    groups = {mode: [s for s in samples if s['mode'] == mode] for mode in ('explicit_cot', 'codi')}
    if any(not group for group in groups.values()):
        raise ValueError('Both reasoning modes need timing samples')
    rows = {name: dict.fromkeys(COLUMNS, 0.0) for name in BREAKDOWN_ROWS}
    totals = dict.fromkeys(COLUMNS, 0.0)
    for mode, group in groups.items():
        overall = 'Explicit' if mode == 'explicit_cot' else 'CODI overall'
        if per_token:
            visible = sum(s['visible_tokens'] for s in group)
            latent = sum(s['latent_steps'] for s in group)
            denominators = {overall: visible + latent if mode == 'codi' else visible,
                            'CODI latent only': latent, 'CODI visible only': visible}
            if denominators[overall] <= 0 or (mode == 'codi' and min(latent, visible) <= 0):
                raise ValueError('Per-token reporting needs positive generated-step counts')
        else:
            denominators = dict.fromkeys(COLUMNS, len(group))
        for sample in group:
            totals[overall] += sample['total_ms'] / denominators[overall]
            for item in sample['breakdown']:
                value = item['ms'] / denominators[overall]
                rows[item['name']][overall] += value
                if mode == 'codi' and item['phase'] in ('latent', 'visible'):
                    column = 'CODI latent only' if item['phase'] == 'latent' else 'CODI visible only'
                    phase_value = item['ms'] / denominators[column]
                    rows[item['name']][column] += phase_value
                    totals[column] += phase_value
    label = 'Total average time per token/step' if per_token else 'Total average time'
    return [(label, totals), *rows.items(), (label + ' (repeat)', dict(totals))]


@torch.inference_mode()
def profile_question(model, tokenizer, head, question, *, mode, device, max_new_tokens,
                     question_id=0, repeat=0, arm='rank96', latent_iterations=6):
    """One batch-1 sample; only displayed components receive module hooks."""
    device = torch.device(device)
    timeline = Timeline(device, unified_clock=True)
    timeline.context.update(mode=mode, arm=arm, question_id=question_id, repeat=repeat, phase='shared')
    base = official_codi_base_model(model)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    roots = [(f'transformer.h.{i}', block) for i, block in enumerate(base.transformer.h)]
    roots += [('transformer.' + name, getattr(base.transformer, name))
              for name in ('wte', 'wpe', 'ln_f') if hasattr(base.transformer, name)]
    roots += [('projector', model.prj), ('lm_head', head)]
    with timeline.modules(roots, recursive=False):
        with timeline.span('question_total'):
            with timeline.span('load_question', gpu=False):
                questions = [str(question)]
            batches = prepare_questions(tokenizer, questions, 1, timeline)
            result = decode(model, tokenizer, batches, head, mode=mode, device=device,
                            max_new_tokens=max_new_tokens, latent_iterations=latent_iterations, timeline=timeline)
    timeline.resolve()
    sample = partition_question(timeline.records)
    sample.update(arm=arm, tokens=list(result.token_ids[0]), text=result.texts[0],
                  prompt_tokens=int(batches[0].mask.sum()),
                  latent_steps=latent_iterations if mode == 'codi' else 0,
                  visible_tokens=result.generated_token_counts[0])
    return sample, timeline.records
