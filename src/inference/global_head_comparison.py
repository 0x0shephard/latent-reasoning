"""Detailed, opt-in diagnostics for CODI and its shared-weight explicit CoT path.

Clean latency is measured with instrumentation disabled. Module timings are inclusive
stream intervals, not additive kernel totals; Chrome traces expose individual kernels.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import time
import uuid

import torch
from torch import nn
from src.models.official_codi import official_codi_base_model, _normalized_official_questions
from src.inference.official_codi_fast import FastCODIGeneration


class Timeline:
    def __init__(self, device='cpu', enabled=True):
        self.device = torch.device(device)
        self.enabled = enabled
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
        marker = torch.profiler.record_function(name)
        marker.__enter__()
        row['_wall_start'] = time.perf_counter()
        events = None
        if gpu and self.device.type == 'cuda':
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
    def modules(self, roots):
        if not self.enabled:
            yield
            return
        handles, stacks, seen = [], {}, set()
        for prefix, root in roots:
            for suffix, module in root.named_modules():
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


def summarize_events(raw_path, output_path):
    """Streaming moments; no need to load a large per-layer log into memory."""
    groups = {}
    keys = ('mode', 'arm', 'protocol', 'batch_size', 'phase', 'kind', 'name', 'module_type')
    with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(str(row.get(k, '')) for k in keys)
            group = groups.setdefault(key, {'calls': 0, 'cpu': [], 'cuda': []})
            group['calls'] += 1
            for source, target in [('cpu_wall_ms', 'cpu'), ('cuda_stream_ms', 'cuda')]:
                if row.get(source) is not None:
                    value = float(row[source])
                    if not group[target]:
                        group[target] = [0, 0.0, 0.0, value, value]
                    stats = group[target]
                    stats[0] += 1
                    delta = value - stats[1]
                    stats[1] += delta / stats[0]
                    stats[2] += delta * (value - stats[1])
                    stats[3], stats[4] = min(stats[3], value), max(stats[4], value)
    rows = []
    for key, group in groups.items():
        row = dict(zip(keys, key), calls=group['calls'])
        for clock in ('cpu', 'cuda'):
            stats = group[clock]
            if stats:
                n, mean, m2, minimum, maximum = stats
                row.update({f'{clock}_mean_ms': mean, f'{clock}_std_ms': math.sqrt(max(m2, 0)/max(n-1, 1)),
                            f'{clock}_total_ms': mean*n, f'{clock}_min_ms': minimum,
                            f'{clock}_max_ms': maximum})
        rows.append(row)
    write_csv(output_path, rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
                    if positions is not None:
                        prefill_kwargs['position_ids'] = positions
                    out = body(**prefill_kwargs)
            with timeline.span('kv_cache_reference_update'):
                cache = out.past_key_values
                hidden = out.last_hidden_state[:, -1:, :]
            if mode == 'codi':
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


class CompiledSelector(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.core = source
        self.requires_grad_(False)
        object.__setattr__(self, 'compiled', torch.compile(self._choose, mode='reduce-overhead', dynamic=True, fullgraph=True))
    def _choose(self, hidden):
        return self.core(hidden).argmax(-1)
    def select_token(self, hidden, *, vocabulary_stop):
        return self.compiled(hidden.contiguous())


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _partial(coordinates, weights, bias, values, indices,
                 R:tl.constexpr, V:tl.constexpr, BLOCKS:tl.constexpr,
                 WIDE:tl.constexpr, BR:tl.constexpr, BV:tl.constexpr):
        row, block = tl.program_id(0), tl.program_id(1)
        vr = block*BV + tl.arange(0,BV)
        rr = tl.arange(0,BR)
        x = tl.load(coordinates+row*R+rr, rr<R, 0)
        w = tl.load(weights+vr[:,None]*R+rr[None,:], (vr[:,None]<V)&(rr[None,:]<R),0)
        b = tl.load(bias+vr, vr<V,0)
        if WIDE:
            # Match the eager output dtype as well as widening the arithmetic.
            score = tl.sum(w.to(tl.float32)*x[None,:].to(tl.float32),1) + b.to(tl.float32)
            score = score.to(b.dtype).to(tl.float32)
        else:
            # Historical notebook's numerical path, retained as an explicit control.
            score = tl.sum(w*x[None,:],1) + b
        score = tl.where(vr<V, score, float('-inf'))
        best = tl.argmax(score,0)
        tl.store(values+row*BLOCKS+block,tl.max(score,0))
        tl.store(indices+row*BLOCKS+block,block*BV+best)

    @triton.jit
    def _finish(values,indices,result,BLOCKS:tl.constexpr,B:tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0,B)
        vals = tl.load(values+row*BLOCKS+offsets, offsets<BLOCKS,float('-inf'))
        winner = tl.argmax(vals,0)
        tl.store(result+row,tl.load(indices+row*BLOCKS+winner))


class TritonSelector(nn.Module):
    def __init__(self, source, wide=True):
        super().__init__()
        if triton is None:
            raise RuntimeError('Triton is unavailable')
        self.core, self.wide = source, wide
        self.requires_grad_(False)
    def select_token(self, hidden, *, vocabulary_stop):
        x = self.core.down(hidden).contiguous()
        w,b = self.core.up.weight,self.core.up.bias
        batch,rank = x.shape
        blocks = triton.cdiv(vocabulary_stop,128)
        values = torch.empty((batch,blocks),device=x.device,dtype=torch.float32)
        indices = torch.empty((batch,blocks),device=x.device,dtype=torch.int32)
        result = torch.empty(batch,device=x.device,dtype=torch.long)
        _partial[(batch,blocks)](x,w,b,values,indices,rank,vocabulary_stop,blocks,self.wide,
                                 triton.next_power_of_2(rank),128,num_warps=4)
        _finish[(batch,)](values,indices,result,blocks,triton.next_power_of_2(blocks),num_warps=4)
        return result


def summarize_timings(rows):
    groups = {}
    keys = ('mode','arm','protocol','batch_size')
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    summary = []
    for key, group in groups.items():
        values = [x['wall_ms_per_question'] for x in group]
        summary.append(dict(zip(keys,key), measurements=len(group),
            mean_ms_per_question=statistics.mean(values), median_ms_per_question=statistics.median(values),
            std_ms_per_question=statistics.stdev(values) if len(values)>1 else 0.0,
            min_ms_per_question=min(values),max_ms_per_question=max(values),
            total_questions=sum(x['questions'] for x in group),
            mean_generated_tokens=statistics.mean(x['visible_tokens']/x['questions'] for x in group)))
    return summary
