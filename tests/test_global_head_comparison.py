"""Read-only CPU verification of decoder semantics, tracing and notebook contract."""
import ast
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from src.inference.global_head_comparison import (
    Timeline,prepare_questions,decode,summarize_events,summarize_timings,
)
from src.inference.official_codi_fast import generate_official_codi_fast,prepare_official_codi_batches
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tiny_codi_fixture',ROOT/'tests/test_official_codi_fast.py')
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_codi_decoder_matches_original_and_profiling_does_not_change_tokens(tmp_path):
    tokenizer=fixture.TinyTokenizer(); model=fixture.TinyCODI()
    questions=['a question','another longer question','short']
    old=generate_official_codi_fast(model,tokenizer,
        prepare_official_codi_batches(tokenizer,questions,batch_size=2,length_bucketed=False),
        latent_iterations=2,max_new_tokens=4,device=torch.device('cpu'))
    trace=Timeline('cpu')
    trace.context.update(mode='codi',arm='dense',protocol='test')
    with trace.modules([('transformer',model.codi.transformer),('projector',model.prj)]):
        new=decode(model,tokenizer,prepare_questions(tokenizer,questions,2,trace),model.codi.lm_head,
            mode='codi',device='cpu',max_new_tokens=4,latent_iterations=2,timeline=trace)
    assert new==old
    path=tmp_path/'events.jsonl.gz'; trace.flush(path)
    events=[json.loads(line) for line in gzip.open(path,'rt')]
    names={r['name'] for r in events}
    assert {'transformer','projector','host_to_device_input_ids','termination_check_host_sync',
            'question_tokenization','device_to_host_token_buffer'}<=names
    assert all(r['cpu_wall_ms']>=0 for r in events)
    assert all(r['cuda_stream_ms'] is None for r in events)
    assert not model.codi.transformer._forward_hooks
    assert any(r.get('parent_id') is not None for r in events)
    summary=summarize_events(path,tmp_path/'averages.csv')
    assert sum(r['calls'] for r in summary)==len(events)


def test_fixed_replay_preserves_dense_steps_even_when_head_wants_to_stop():
    tokenizer=fixture.TinyTokenizer(); model=fixture.TinyCODI()
    prepared=prepare_questions(tokenizer,['question','long question'],2)
    dense=decode(model,tokenizer,prepared,model.codi.lm_head,mode='codi',device='cpu',max_new_tokens=4)
    class EarlyStop(nn.Module):
        calls=0
        def forward(self,hidden):
            self.calls+=1
            out=torch.zeros(len(hidden),8); out[:,1]=99
            return out
    wrong=EarlyStop()
    free=decode(model,tokenizer,prepared,wrong,mode='codi',device='cpu',max_new_tokens=4)
    assert free.token_ids==((1,),(1,))
    wrong.calls=0
    replay=decode(model,tokenizer,prepared,wrong,mode='codi',device='cpu',max_new_tokens=4,
                  forced_tokens=dense.token_ids)
    assert replay==dense
    assert wrong.calls==max(dense.generated_token_counts)


def test_explicit_mode_skips_projector_and_latents_and_uses_padding_positions():
    tokenizer=fixture.TinyTokenizer(); model=fixture.TinyCODI()
    class ForbiddenProjector(nn.Module):
        def forward(self,*args): raise AssertionError('explicit mode invoked latent projector')
    model.prj=ForbiddenProjector()
    calls=[]
    original=model.codi.transformer.forward
    def capture(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    model.codi.transformer.forward=capture
    prepared=prepare_questions(tokenizer,['short','a longer question'],2)
    result=decode(model,tokenizer,prepared,model.codi.lm_head,mode='explicit_cot',device='cpu',max_new_tokens=4)
    assert result.token_ids==((1,),(1,))
    assert len(calls)==1
    assert torch.equal(calls[0]['input_ids'],prepared[0].ids)
    assert calls[0]['position_ids'].tolist()==[[1,1,0],[0,1,2]]


def test_timeline_cleanup_and_disabled_mode():
    trace=Timeline('cpu',enabled=False)
    with trace.span('unused'):
        pass
    trace.resolve()
    assert not trace.records and not trace.pending
    model=nn.Linear(2,2)
    active=Timeline('cpu')
    try:
        with active.modules([('linear',model)]):
            model(torch.ones(1,2))
            raise ValueError('intentional')
    except ValueError: pass
    assert not model._forward_hooks and not model._forward_pre_hooks
    active.resolve()
    assert len(active.records)==1


def test_summary_keeps_modes_protocols_and_raw_measurements_separate():
    rows=[dict(mode=mode,arm='dense',protocol=protocol,batch_size=1,
        wall_ms_per_question=value,questions=1,visible_tokens=3)
        for mode in ('codi','explicit_cot') for protocol in ('free_generation','fixed_replay')
        for value in (2.,4.)]
    result=summarize_timings(rows)
    assert len(result)==4
    assert all(r['mean_ms_per_question']==3 and r['measurements']==2 for r in result)
    assert len(rows)==8


def test_notebook_compiles_and_embeds_current_runtime():
    path=ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for i,source in enumerate(sources): compile(source,f'cell-{i}','exec')
    embedded=next(s for s in sources if s.startswith('RUNTIME_SOURCE ='))
    source=ast.literal_eval(ast.parse(embedded).body[0].value)
    assert source==(ROOT/'src/inference/global_head_comparison.py').read_text(encoding='utf-8')
    combined='\n'.join(sources)
    assert 'REPRODUCTION_SUMMARY' not in combined
    assert 'fixed_replay' in combined and 'explicit_cot' in combined
    assert 'PROFILE_OVERHEAD' in combined and 'operator_events_' in combined
    assert 'assert parity[mode]' in combined


def test_notebook_fitting_benchmark_exports_and_resume_with_synthetic_cpu_model(tmp_path, monkeypatch):
    """Execute fitting, both benchmark modes and CSV exports without network/GPU."""
    import time
    import gc
    import pathlib
    import random
    from dataclasses import asdict
    import pandas as pd
    import src.inference.global_head_comparison as runtime
    from src.mech.global_low_rank_head import NestedLowRankVocabularyHead,activation_whitened_factors,distil_nested_head,evaluate_nested_head
    from src.data.answer_extract import answers_match
    torch.set_num_threads(1)
    class Body(nn.Module):
        def __init__(self,embedding):
            super().__init__(); self.wte=embedding
            self.h=nn.ModuleList([nn.Identity() for _ in range(12)])
            self.ln_f=nn.Identity()
        def forward(self,input_ids=None,inputs_embeds=None,**kwargs):
            hidden=self.wte(input_ids) if inputs_embeds is None else inputs_embeds
            for layer in self.h: hidden=layer(hidden)
            hidden=self.ln_f(hidden)
            return SimpleNamespace(last_hidden_state=hidden,past_key_values=((hidden,),))
    class Base(nn.Module):
        def __init__(self):
            super().__init__(); self.transformer=Body(nn.Embedding(120,104)); self.lm_head=nn.Linear(104,120,bias=False)
            with torch.no_grad():
                self.transformer.wte.weight.zero_(); self.lm_head.weight.zero_()
                self.transformer.wte.weight[2,0]=1; self.transformer.wte.weight[4,1]=1
                self.lm_head.weight[1,0]=1; self.lm_head.weight[2,1]=1
        def get_output_embeddings(self): return self.lm_head
    class Model(nn.Module):
        bot_id=2; eot_id=119
        config=SimpleNamespace(hidden_size=104,n_positions=1024)
        def __init__(self): super().__init__(); self.codi=Base(); self.prj=nn.Identity()
        def input_embeddings(self): return self.codi.transformer.wte
    model=Model().requires_grad_(False)
    full=model.codi.lm_head
    def save_json(path,value): path.write_text(json.dumps(value,default=str))
    notebook=json.loads((ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    rows=[dict(question=f'question {i}',gold='2') for i in range(24)]
    scope=dict(runtime=runtime,torch=torch,pd=pd,time=time,gc=gc,pathlib=pathlib,json=json,random=random,
        asdict=asdict,NestedLowRankVocabularyHead=NestedLowRankVocabularyHead,
        activation_whitened_factors=activation_whitened_factors,distil_nested_head=distil_nested_head,
        evaluate_nested_head=evaluate_nested_head,answers_match=answers_match,
        official_codi_base_model=lambda m:m.codi,merge_official_codi_lora_=lambda m:m,
        model=model,base=model.codi,full_head=full,weight=full.weight[:119].detach(),bias=None,
        tokenizer=fixture.TinyTokenizer(),device=torch.device('cpu'),setup=Timeline('cpu'),
        RUN_DIR=tmp_path,MODES=['codi','explicit_cot'],RANKS=(32,64,96),SEED=89,
        MAX_NEW_TOKENS={'codi':4,'explicit_cot':4},COLLECT_BATCH_SIZE=4,DISTILL_BATCH_SIZE=4,
        MAX_FIT_STATES=16,MAX_SELECT_STATES=8,MAX_RECOVERY_STATES=8,CLEAN_EPOCHS=1,RECOVERY_EPOCHS=1,
        QUALITY_BATCH_SIZE=2,TIMING_BATCH_SIZES=(1,2),TIMING_REPEATS=2,TRY_COMPILE=False,TRY_TRITON=False,
        PROFILE_QUESTIONS=1,PROFILE_REPEATS=1,OPERATOR_TRACE_QUESTIONS=0,SMOKE=True,
        EXPERIMENT_SOURCE_SHA256='synthetic',display=lambda x:None,
        splits=dict(fit=rows[:8],selection=rows[8:12],recovery=rows[12:16],timing=rows[16:20],warmup=rows[20:]),
        test_rows=rows[:4],save_json=save_json,save_pt=lambda path,value:torch.save(value,path))
    # Fit before mocking CUDA so AdamW's lazy imports see the original torch classes.
    source=next(s for s in sources if 'def collect(mode' in s)
    exec(compile(source,'fitting-cell','exec'),scope)
    class Event:
        def __init__(self,**kwargs): self.tick=0
        def record(self): self.tick=time.perf_counter()
        def elapsed_time(self,other): return (other.tick-self.tick)*1000
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a,**k:0)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a,**k:0)
    benchmark=next(s for s in sources if 'def local_batch' in s)
    summary=next(s for s in sources if 'summary=runtime.summarize_timings' in s)
    exec(compile(benchmark,'benchmark-cell','exec'),scope)
    exec(compile(summary,'summary-cell','exec'),scope)
    assert json.loads((tmp_path/'completed.json').read_text())['complete']
    raw=pd.read_csv(tmp_path/'timing_individual.csv')
    assert set(raw['mode'])=={'codi','explicit_cot'}
    assert set(raw['protocol'])=={'fixed_replay','free_generation'}
    assert len(pd.read_csv(tmp_path/'quality_summary.csv'))==4
    events=pd.read_csv(tmp_path/'codi/layer_stage_averages.csv')
    assert 'transformer.h.11' in set(events['name'])
    # Completed modes must load saved reports, not re-run inference.
    monkeypatch.setattr(runtime,'decode',lambda *a,**k: (_ for _ in ()).throw(AssertionError('resume decoded')))
    exec(compile(benchmark,'resume-benchmark','exec'),scope)
    exec(compile(summary,'resume-summary','exec'),scope)
    assert len(pd.read_csv(tmp_path/'timing_individual.csv'))==len(raw)
