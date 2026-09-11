"""Correctness checks for the independent transformer-optimization notebook."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.inference.transformer_optimization import OptimizedDecoder,StaticCore,PackedLinear,reference
from transformers import GPT2Config,GPT2LMHeadModel


class Tokenizer:
    pad_token_id=0; eos_token_id=1
    def __call__(self,questions,**kwargs):
        def encode(q):
            return [4,5] if q==' The answer is:' else [2+i%4 for i,_ in enumerate(q.split())]
        return {'input_ids':encode(questions) if isinstance(questions,str) else [encode(q) for q in questions]}
    def decode(self,tokens,**kwargs): return ' '.join(str(t) for t in tokens if t not in (0,1))


def fixture():
    torch.manual_seed(12); torch.set_num_threads(1)
    class Model(nn.Module):
        bot_id=30; eot_id=31
        def __init__(self):
            super().__init__()
            self.codi=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=32,n_head=4,n_layer=2,n_positions=128,
                resid_pdrop=0.,attn_pdrop=0.,embd_pdrop=0.,pad_token_id=0,eos_token_id=1))
            self.prj=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32),nn.LayerNorm(32))
            self.config=self.codi.config
        def input_embeddings(self): return self.codi.transformer.wte
    return Model().requires_grad_(False).eval(),Tokenizer()


@torch.inference_mode()
def test_static_prefill_and_cache_updates_match_hf_and_reuse_storage():
    from transformers import DynamicCache
    model,_=fixture(); core=StaticCore(model)
    ptr=core.keys.data_ptr()
    for values in ([2,3,4,5],[3,2],[2,5,3]):
        ids=torch.tensor([values])
        cache=DynamicCache()
        expected=model.codi.transformer(input_ids=ids,past_key_values=cache,use_cache=True)
        observed=core.prefill(ids)
        torch.testing.assert_close(observed,expected.last_hidden_state,rtol=2e-5,atol=2e-5)
        for index in range(3):
            embedded=model.input_embeddings()(torch.tensor([[2+index]]))
            expected=model.codi.transformer(inputs_embeds=embedded,past_key_values=cache,use_cache=True)
            actual=core(embedded,torch.tensor([len(values)+index]))
            torch.testing.assert_close(actual,expected.last_hidden_state,rtol=2e-5,atol=2e-5)
        assert core.keys.data_ptr()==ptr


@pytest.mark.parametrize('compiled',[False,True])
@pytest.mark.parametrize('chunk',[1,4])
@torch.inference_mode()
def test_decoder_matches_codi_and_explicit_across_questions(compiled,chunk):
    model,tokenizer=fixture()
    engine=OptimizedDecoder(model,tokenizer,model.codi.lm_head,compile_steps=compiled,
                            compiler_backend='eager',chunk_size=chunk)
    engine.prepare()
    for mode in ('codi','explicit_cot'):
        for question in ('a short question','another longer question here','a short question'):
            expected=reference.decode(model,tokenizer,reference.prepare_questions(tokenizer,[question],1),
                model.codi.lm_head,mode=mode,device='cpu',max_new_tokens=7)
            actual=engine.generate(question,mode,7)
            assert actual['tokens']==expected.token_ids[0]
            assert actual['visible_tokens']==len(actual['tokens'])
            assert actual['latent_steps']==(6 if mode=='codi' else 0)
            assert not any(m._forward_hooks for m in model.modules())


@torch.inference_mode()
def test_first_token_eos_and_chunk_limit_do_not_leak_padding_tokens():
    model,tokenizer=fixture()
    class Constant(nn.Module):
        def __init__(self,token): super().__init__(); self.token=token
        def forward(self,x):
            out=x.new_zeros((1,32)); out[:,self.token]=99; return out
    for token in (1,2):
        engine=OptimizedDecoder(model,tokenizer,Constant(token),chunk_size=4)
        engine.prepare()
        for cap in (1,2,5,7):
            for mode in ('explicit_cot','codi'):
                result=engine.generate('a question',mode,cap)
                assert result['tokens']==((1,) if token==1 else (2,)*cap)


@pytest.mark.parametrize('bits',[4,8])
def test_packed_weights_match_effective_prefill_weights(bits):
    model,_=fixture()
    linear=PackedLinear(model.codi.transformer.h[0].mlp.c_fc,bits)
    if bits==4:
        q=torch.stack((linear.packed&15,linear.packed>>4),-1).reshape(linear.weight.shape).float()-8
    else: q=linear.packed.float()
    restored=(q*linear.scale[:,None]).to(linear.weight.dtype)
    torch.testing.assert_close(restored,linear.weight,rtol=0,atol=0)
    assert linear.packed.numel()*linear.packed.element_size()<linear.weight.numel()*linear.weight.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires Kaggle CUDA hardware')
@torch.inference_mode()
def test_cuda_graph_reset_and_compiled_generation():
    model,tokenizer=fixture(); model.to(device='cuda',dtype=torch.float16)
    engine=OptimizedDecoder(model,tokenizer,model.codi.lm_head,compile_steps=True,cuda_graphs=True,chunk_size=4)
    engine.prepare()
    for mode in ('codi','explicit_cot'):
        for question in ('a question','a longer question here','a question'):
            expected=reference.decode(model,tokenizer,reference.prepare_questions(tokenizer,[question],1),
                model.codi.lm_head,mode=mode,device='cuda',max_new_tokens=7)
            assert engine.generate(question,mode,7)['tokens']==expected.token_ids[0]


def test_block_probe_removes_hooks_and_accounts_for_whole_block():
    from src.inference.transformer_optimization import block_substeps
    model,tokenizer=fixture()
    rows,events=block_substeps(model,tokenizer,'one real question',repeats=3)
    names={row['name'] for row in rows}
    assert {'LayerNorm 1','QKV projection','Attention/cache/reshape','FC1','GELU','FC2'}<=names
    for repeat in range(3):
        total=next(row['cpu_wall_ms'] for row in events if row['repeat']==repeat and row['name']=='Block')
        assert abs(sum(r['microseconds']/1000 for r in rows if r['repeat']==repeat)-total)<1e-8
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_compiled_numeric_gate_and_equal_mac_probe():
    from src.inference.transformer_optimization import numerical_gate,operation_microbenchmark
    model,tokenizer=fixture()
    engine=OptimizedDecoder(model,tokenizer,model.codi.lm_head,compile_steps=True,compiler_backend='aot_eager')
    engine.prepare()
    result=numerical_gate(model,tokenizer,engine,['one question'])
    assert result['comparisons']==16
    assert result['max_absolute_error']<1e-4
    rows=operation_microbenchmark('cpu',torch.float32,width=16,repeats=2,iterations=3)
    assert len(rows)==4 and all(row['microseconds']>0 for row in rows)


def test_new_notebook_schema_embedded_sources_and_isolated_probes():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    notebook=nbformat.read(root/'notebooks/kaggle_codi_transformer_optimization.ipynb',as_version=4)
    nbformat.validate(notebook)
    sources=[cell.source for cell in notebook.cells if cell.cell_type=='code']
    for source in sources: compile(source,'new-notebook-cell','exec')
    embedded=next(s for s in sources if s.startswith('OPTIMIZER_SOURCE ='))
    source=ast.literal_eval(ast.parse(embedded).body[0].value)
    assert source==(root/'src/inference/transformer_optimization.py').read_text(encoding='utf-8-sig')
    assert sum('opt.block_substeps(' in s for s in sources)==1
    main=next(s for s in sources if s.startswith('clean_samples=[]'))
    assert 'profile' not in main and 'register_forward' not in main
    assert sources.index(next(s for s in sources if s.startswith('substeps,')))<sources.index(main)


def test_actual_notebook_benchmark_tables_and_accuracy_cells_on_cpu(tmp_path):
    import pandas as pd
    import random
    import src.inference.transformer_optimization as opt
    from src.data.answer_extract import answers_match
    model,tokenizer=fixture()
    class ConstantHead(nn.Module):
        def forward(self,x):
            result=x.new_zeros((x.shape[0],32)); result[:,2]=99; return result
    head=ConstantHead()
    engines={mode:OptimizedDecoder(model,tokenizer,head,chunk_size=4) for mode in ('codi','explicit_cot')}
    for engine in engines.values(): engine.prepare()
    def run(version,mode,arm,question,diagnostic=False):
        if version=='current': return opt.baseline_generate(model,tokenizer,head,question,mode,4,diagnostic)
        return engines[mode].generate(question,mode,4,diagnostic)
    records=[]; displayed=[]
    scope=dict(opt=opt,torch=torch,pd=pd,random=random,model=model,device=torch.device('cpu'),
        tokenizer=tokenizer,MODES=['codi','explicit_cot'],TIMING_REPEATS=2,SEED=89,DIAGNOSTIC_QUESTIONS=2,
        runtime=reference,run=run,selected={m:dict(engine=e) for m,e in engines.items()},RUN_DIR=tmp_path,
        DEBUG_PATH=tmp_path/'debug.jsonl.gz',debug_record=lambda kind,value:records.append((kind,value)),
        save_json=lambda path,value:path.write_text(json.dumps(value)),display=displayed.append,
        answers_match=answers_match,splits=dict(timing=[dict(question='a question'),dict(question='a longer question')]),
        test_rows=[dict(question=f'test example {i}',gold='2' if i<2 else '999') for i in range(4)])
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/kaggle_codi_transformer_optimization.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for marker in ('clean_samples=[]','diagnostic_samples=[]','accuracy_results={}'): 
        source=next(s for s in sources if s.startswith(marker))
        exec(compile(source,'optimization-notebook-cell','exec'),scope)
    assert len(displayed)==4
    assert len(scope['clean_samples'])==32
    assert len(scope['diagnostic_samples'])==8
    for frame in displayed:
        assert list(frame.columns)==list(reference.COLUMNS)
        for col in frame:
            assert abs(frame.iloc[1:-1][col].sum()-frame.iloc[0][col])<1e-5
    assert len(scope['accuracy_results'])==8
    assert all(r['correct']==2 and r['accuracy']==.5 for r in scope['accuracy_results'].values())
    assert len([r for k,r in records if k=='accuracy_sample'])==32
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_candidate_selection_preserves_dense_control_and_reports_unavailable_gpu_paths(tmp_path):
    import gc,time,random
    import pandas as pd
    import src.inference.transformer_optimization as opt
    from src.data.answer_extract import answers_match,normalize_gold
    model,tokenizer=fixture()
    class Constant(nn.Module):
        def forward(self,x):
            logits=x.new_zeros((x.shape[0],32)); logits[:,2]=99; return logits
    head=Constant()
    class CPUOnly(OptimizedDecoder):
        def __init__(self,*args,**kwargs):
            if kwargs.get('compile_steps') or kwargs.get('cuda_graphs'):
                raise RuntimeError('Simulated unavailable GPU optimization')
            super().__init__(*args,**kwargs)
        def generate(self,*args,**kwargs):
            result=super().generate(*args,**kwargs); result['_candidate']=True; return result
    def controlled_time(call,device):
        result=call(); result['wall_ms']=10. if result.get('_candidate') else 100.; return result
    proxy=SimpleNamespace(OptimizedDecoder=CPUOnly,baseline_generate=opt.baseline_generate,
                          numerical_gate=opt.numerical_gate,timed=controlled_time,triton=None)
    rows=[dict(question='a real question',gold='work #### 2'),dict(question='another question',gold='work #### 999')]
    records=[]
    scope=dict(opt=proxy,torch=torch,pd=pd,model=model,tokenizer=tokenizer,dense=head,heads={m:head for m in ('codi','explicit_cot')},
        MODES=['codi','explicit_cot'],MAX_NEW_TOKENS={'codi':4,'explicit_cot':4},device=torch.device('cpu'),
        time=time,gc=gc,validation=rows,warmup=['one question','a longer example'],
        normalize_gold=normalize_gold,answers_match=answers_match,display=lambda x:None,
        debug_record=lambda kind,value:records.append((kind,value)))
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/kaggle_codi_transformer_optimization.ipynb').read_text(encoding='utf-8'))
    source=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('def baseline('))
    exec(compile(source,'candidate-selection-cell','exec'),scope)
    assert set(scope['selected'])=={'codi','explicit_cot'}
    assert all(result['name']=='static_cache' for result in scope['selected'].values())
    assert all(result['dense_engine'] is not None for result in scope['selected'].values())
    assert any(kind=='candidate_failure' and 'Simulated unavailable' in row['error'] for kind,row in records)
    assert not any(m._forward_hooks for m in model.modules())
