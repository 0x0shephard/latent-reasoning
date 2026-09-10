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
    Timeline,prepare_questions,decode,partition_question,bottleneck_means,profile_question,COLUMNS,
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


def test_partition_does_not_double_count_layers_and_uses_one_clock():
    # Deliberately unequal CPU and GPU values: mixing clocks must not inflate totals.
    rows=[dict(event_id=1,parent_id=None,name='question_total',mode='codi',phase='shared',cuda_stream_ms=20.,cpu_wall_ms=90.),
          dict(event_id=2,parent_id=1,name='phase_latent',phase='latent',cuda_stream_ms=12.,cpu_wall_ms=40.),
          dict(event_id=3,parent_id=2,name='transformer.h.0',phase='latent',cuda_stream_ms=8.,cpu_wall_ms=25.),
          dict(event_id=4,parent_id=3,name='transformer.h.0.attn',phase='latent',cuda_stream_ms=6.,cpu_wall_ms=15.),
          dict(event_id=5,parent_id=1,name='phase_visible',phase='visible_decode',cuda_stream_ms=5.,cpu_wall_ms=20.),
          dict(event_id=6,parent_id=5,name='lm_head.up',phase='visible_decode',cuda_stream_ms=3.,cpu_wall_ms=10.)]
    sample=partition_question(rows)
    assert sum(r['ms'] for r in sample['breakdown'])==20
    assert sum(r['ms'] for r in sample['breakdown'] if r['name']=='Transformer block 01')==8
    explicit=dict(sample,mode='explicit_cot')
    table=bottleneck_means([sample,explicit])
    assert list(table[0][1])==list(COLUMNS)
    assert table[0][1]=={'Explicit':20.,'CODI overall':20.,'CODI latent only':12.,'CODI visible only':5.}
    assert table[0][1]==table[-1][1]
    assert sum(r['CODI overall'] for _,r in table[1:-1])==20


def test_means_count_zero_work_questions_in_denominator():
    samples=[dict(mode=mode,total_ms=6.,breakdown=[dict(name='Low-rank LM head',phase='visible',ms=6.)])
             for mode in ('codi','explicit_cot')]
    samples.append(dict(mode='codi',total_ms=0.,breakdown=[]))
    table=dict(bottleneck_means(samples))
    assert table['Low-rank LM head']['CODI overall']==3.
    assert table['Low-rank LM head']['Explicit']==6.


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
    assert 'explicit_cot' in combined and "'T4' in torch.cuda.get_device_name(0)" in combined
    assert 'runtime.bottleneck_means(profile_samples)' in combined
    assert 'TRY_TRITON' not in combined and 'TIMING_BATCH_SIZES' not in combined
    assert "'datasets==" not in combined and "'pandas==" not in combined
    assert 'huggingface_hub>=0.34.0,<1.0' in combined and 'hf_xet' in combined
    assert sum('display(' in source for source in sources)==1
    assert 'assert parity[mode]' in combined


def test_notebook_fitting_and_four_column_report_with_synthetic_cpu_model(tmp_path, monkeypatch):
    """Execute the actual fitting, timing and final report cells without network/GPU."""
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
    scope=dict(runtime=runtime,torch=torch,pd=pd,time=time,gc=gc,gzip=gzip,pathlib=pathlib,json=json,random=random,
        asdict=asdict,NestedLowRankVocabularyHead=NestedLowRankVocabularyHead,
        activation_whitened_factors=activation_whitened_factors,distil_nested_head=distil_nested_head,
        evaluate_nested_head=evaluate_nested_head,answers_match=answers_match,
        official_codi_base_model=lambda m:m.codi,merge_official_codi_lora_=lambda m:m,
        model=model,base=model.codi,full_head=full,weight=full.weight[:119].detach(),bias=None,
        tokenizer=fixture.TinyTokenizer(),device=torch.device('cpu'),setup=Timeline('cpu'),
        RUN_DIR=tmp_path,MODES=['codi','explicit_cot'],RANKS=(32,64,96),SEED=89,
        MAX_NEW_TOKENS={'codi':4,'explicit_cot':4},COLLECT_BATCH_SIZE=4,DISTILL_BATCH_SIZE=4,
        MAX_FIT_STATES=16,MAX_SELECT_STATES=8,MAX_RECOVERY_STATES=8,CLEAN_EPOCHS=1,RECOVERY_EPOCHS=1,
        TIMING_REPEATS=2,DEPLOY_DTYPE=torch.float32,DEBUG_PATH=tmp_path/'debug.jsonl.gz',
        EXPERIMENT_SOURCE_SHA256='synthetic',display=lambda x:None,
        splits=dict(fit=rows[:8],selection=rows[8:12],recovery=rows[12:16],timing=rows[16:20],warmup=rows[20:]),
        save_json=save_json,save_pt=lambda path,value:torch.save(value,path))
    def debug_record(kind,value):
        with gzip.open(scope['DEBUG_PATH'],'at') as stream:
            stream.write(json.dumps(dict(kind=kind,data=value),default=str)+'\n')
    def flush_setup():
        scope['setup'].resolve()
        for row in scope['setup'].records: debug_record('setup',row)
        scope['setup'].records.clear()
    scope.update(debug_record=debug_record,flush_setup=flush_setup)
    # Fit before mocking CUDA so AdamW's lazy imports see the original torch classes.
    source=next(s for s in sources if 'def collect(mode' in s)
    exec(compile(source,'fitting-cell','exec'),scope)
    benchmark=next(s for s in sources if 'def clean_generation' in s)
    summary=next(s for s in sources if 'report=runtime.bottleneck_means' in s)
    exec(compile(benchmark,'benchmark-cell','exec'),scope)
    exec(compile(summary,'summary-cell','exec'),scope)
    table=pd.read_csv(tmp_path/'bottleneck.csv',index_col=0)
    assert list(table.columns)==list(COLUMNS)
    assert len(list(tmp_path.glob('*.csv')))==1
    assert len(scope['profile_samples'])==16
    assert len(scope['clean_samples'])==32
    assert set(row['mode'] for row in scope['profile_samples'])=={'codi','explicit_cot'}
    for column in COLUMNS:
        assert abs(table[column].iloc[1:-1].sum()-table[column].iloc[0])<0.0001
        assert table[column].iloc[0]==table[column].iloc[-1]
    assert table.loc['Low-rank LM head','CODI latent only']==0
    assert table.loc['Latent projector','Explicit']==0
    records=[json.loads(line) for line in gzip.open(scope['DEBUG_PATH'],'rt')]
    traces=[r['data'] for r in records if r['kind']=='events']
    assert len(traces)==16
    assert any(r['name']=='transformer.h.11' for trace in traces for r in trace)
    assert all(not module._forward_hooks for module in model.modules())
    # Re-running fitting with the same fingerprint reuses completed head artifacts.
    monkeypatch.setattr(runtime,'decode',lambda *a,**k: (_ for _ in ()).throw(AssertionError('cached fit decoded')))
    scope['weight']=model.codi.lm_head.weight[:119].detach(); scope['bias']=None
    exec(compile(source,'cached-fit-cell','exec'),scope)


def test_native_gpt2_cache_matches_legacy_codi_and_hf_explicit_generation():
    """Exercise the real pinned GPT-2 cache implementation, with random tiny weights."""
    import pytest
    transformers=pytest.importorskip('transformers')
    from transformers import GPT2Config,GPT2LMHeadModel,DynamicCache,LogitsProcessor,LogitsProcessorList
    torch.manual_seed(7)
    torch.set_num_threads(1)
    class Model(nn.Module):
        bot_id=6; eot_id=7
        def __init__(self):
            super().__init__()
            self.codi=GPT2LMHeadModel(GPT2Config(vocab_size=8,n_positions=64,n_embd=16,n_layer=2,n_head=2,
                resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,pad_token_id=0,eos_token_id=1))
            self.prj=nn.Linear(16,16)
            self.config=self.codi.config
        def input_embeddings(self): return self.codi.transformer.wte
    model=Model().eval().requires_grad_(False)
    tokenizer=fixture.TinyTokenizer()
    questions=['short','a longer question']
    batches=prepare_questions(tokenizer,questions,2)
    reference=generate_official_codi_fast(model,tokenizer,
        prepare_official_codi_batches(tokenizer,questions,batch_size=2,length_bucketed=False),
        latent_iterations=2,max_new_tokens=6,device=torch.device('cpu'))
    cache_types=[]
    original=model.codi.transformer.forward
    def capture(**kwargs):
        cache_types.append(type(kwargs['past_key_values']))
        return original(**kwargs)
    model.codi.transformer.forward=capture
    observed=decode(model,tokenizer,batches,model.codi.lm_head,mode='codi',device='cpu',
                    latent_iterations=2,max_new_tokens=6)
    assert observed==reference
    assert cache_types and all(t is DynamicCache for t in cache_types)
    model.codi.transformer.forward=original
    class Boundary(LogitsProcessor):
        def __call__(self,input_ids,scores):
            scores[:,7:]=float('-inf'); return scores
    with torch.no_grad():
        expected=model.codi.generate(input_ids=batches[0].ids,attention_mask=batches[0].mask,
            max_new_tokens=6,do_sample=False,pad_token_id=0,eos_token_id=1,
            logits_processor=LogitsProcessorList([Boundary()]))[:,batches[0].ids.shape[1]:].tolist()
    expected=tuple(tuple(seq[:seq.index(1)+1] if 1 in seq else seq) for seq in expected)
    explicit=decode(model,tokenizer,batches,model.codi.lm_head,mode='explicit_cot',device='cpu',max_new_tokens=6)
    assert explicit.token_ids==expected
    for mode in ('codi','explicit_cot'):
        sample,events=profile_question(model,tokenizer,model.codi.lm_head,questions[0],
                                      mode=mode,device='cpu',max_new_tokens=6)
        assert sample['total_ms']>0 and len(events)>20
        assert abs(sum(r['ms'] for r in sample['breakdown'])-sample['total_ms'])<0.0001
