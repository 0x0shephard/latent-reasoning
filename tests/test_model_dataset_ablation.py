"""Check real backbone generation, additive reports, data semantics and notebook."""
import ast
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM
from src.inference import model_dataset_ablation as ablation

ROOT=Path(__file__).resolve().parents[1]


class Tokenizer:
    pad_token_id=0
    eos_token_id=1
    def __call__(self, texts, **kwargs):
        def one(text): return [2+i%4 for i,_ in enumerate(text.split())]
        return {'input_ids':one(texts) if isinstance(texts,str) else [one(text) for text in texts]}
    def decode(self,tokens,**kwargs): return ' '.join(str(t) for t in tokens if t not in (0,1))


def tiny_model(family,layers=2):
    torch.manual_seed(43)
    torch.set_num_threads(1)
    config_class,model_class=(LlamaConfig,LlamaForCausalLM) if family=='llama' else (Qwen2Config,Qwen2ForCausalLM)
    config=config_class(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=layers,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        pad_token_id=0,eos_token_id=1,attention_dropout=0.)
    model=model_class(config).eval().requires_grad_(False)
    return ablation.ExplicitBackbone(model),Tokenizer()


@pytest.mark.parametrize('family',['llama','qwen2'])
@torch.inference_mode()
def test_real_backbone_decode_matches_generate_including_padding(family):
    model,tokenizer=tiny_model(family)
    base=model.codi.backbone
    for questions in (['a question','a longer question here'],['other question']):
        batches=ablation.reference.prepare_questions(tokenizer,questions,2)
        actual=ablation.decode(model,tokenizer,batches,base.lm_head,mode='explicit_cot',device='cpu',max_new_tokens=7)
        batch=batches[0]
        generated=base.generate(input_ids=batch.ids,attention_mask=batch.mask,max_new_tokens=7,do_sample=False,
            pad_token_id=0,eos_token_id=1)
        expected=[]
        for tokens in generated[:,batch.ids.shape[1]:].tolist():
            if 1 in tokens: tokens=tokens[:tokens.index(1)+1]
            expected.append(tuple(tokens))
        assert actual.token_ids==tuple(expected)
        assert base.get_input_embeddings().weight.shape[0]==32
    with pytest.raises(ValueError,match='CODI is unavailable'):
        ablation.decode(model,tokenizer,batches,base.lm_head,mode='codi',device='cpu',max_new_tokens=2)


@torch.inference_mode()
def test_profile_uses_every_block_and_missing_codi_is_not_zero():
    model,tokenizer=tiny_model('qwen2',layers=13)
    head=model.codi.backbone.lm_head
    prepared=ablation.reference.prepare_questions(tokenizer,['a real question'],1)
    expected=ablation.decode(model,tokenizer,prepared,head,mode='explicit_cot',device='cpu',max_new_tokens=3)
    sample,events=ablation.profile_question(model,tokenizer,head,'a real question',mode='explicit_cot',device='cpu',max_new_tokens=3)
    assert sample['tokens']==list(expected.token_ids[0])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    assert {f'transformer.h.{i}' for i in range(13)}<={e['name'] for e in events}
    for per_token in (False,True):
        rows=ablation.bottleneck_means([sample],n_layers=13,per_token=per_token)
        assert 'Transformer block 13' in dict(rows)
        assert 'Transformer block 14' not in dict(rows)
        assert abs(sum(r['Explicit'] for _,r in rows[1:-1])-rows[0][1]['Explicit'])<1e-5
        assert math.isnan(rows[0][1]['CODI overall'])


def test_svamp_joins_body_and_question_and_normalizes_answer():
    rows=ablation.parse_svamp(json.dumps([dict(ID='x',Body='There are 5 apples.',Question='How many remain?',Answer=3.0)]))
    assert rows==[dict(id='x',question='There are 5 apples. How many remain?',gold='3.0')]


def test_asdiv_does_not_misread_times_ratios_units_or_multiple_answers():
    answers=['9 (apples)','25 (cm2)','1,234','-2.5','3:30','3:4','February 3rd','5; 15; 20','Mrs. Hilt','1/2']
    payload='<ProblemSet>'+''.join(f'<Problem ID="{i}"><Body>Story.</Body><Question>Question?</Question><Answer>{a}</Answer></Problem>' for i,a in enumerate(answers))+'</ProblemSet>'
    rows,excluded=ablation.parse_asdiv(payload)
    assert [r['gold'] for r in rows]==['9','25','1234','-2.5']
    assert len(excluded)==6


def test_fixed_training_partitions_and_evaluation_leakage_gate():
    train=[dict(question=f'Question {i}',answer='#### 1') for i in range(1800)]
    evaluation=dict(gsm8k=[dict(question='test question')],svamp=[],asdiv=[])
    first=ablation.partition_train(train,evaluation)
    assert first==ablation.partition_train(train,evaluation)
    assert [len(first[k]) for k in first]==[1024,256,256,16,4]
    evaluation['svamp']=[dict(question=first['fit'][0]['question'])]
    with pytest.raises(ValueError,match='overlap'):
        ablation.partition_train(train,evaluation)


def test_generated_notebook_contract_and_embedded_sources():
    import nbformat
    notebook=nbformat.read(ROOT/'notebooks/kaggle_model_dataset_global_head_ablations.ipynb',as_version=4)
    nbformat.validate(notebook)
    sources=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in sources: compile(source,'ablation-cell','exec')
    embedded=next(s for s in sources if s.startswith('ABLATION_SOURCE ='))
    assert ast.literal_eval(ast.parse(embedded).body[0].value)==(ROOT/'src/inference/model_dataset_ablation.py').read_text(encoding='utf-8')
    protocol=next(s for s in sources if s.startswith('GPT2_LOAD ='))
    scope={}
    exec(protocol,scope)
    for name in ('GPT2_LOAD','PARITY_CHECK','FIT_HEADS','TIMING_RUN','ACCURACY_RUN'):
        compile(scope[name],name,'exec')
    original=json.loads((ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb').read_text(encoding='utf-8'))
    original_code=[''.join(c['source']) for c in original['cells'] if c['cell_type']=='code']
    assert scope['TIMING_RUN']==original_code[5]
    assert 'CLEAN_EPOCHS,RECOVERY_EPOCHS=4,2' in sources[0]
    assert 'COLLECT_BATCH_SIZE=8' in sources[0]
    assert 'MAX_NEW_TOKENS={\'codi\':64,\'explicit_cot\':256}' in sources[0]
    assert 'if not DEBUG_PATH.exists()' in sources[1]
    assert len(ablation.MODEL_SPECS)==3 and len(ablation.DATASETS)==3


def test_notebook_fitting_timing_accuracy_and_four_reports_on_real_tiny_qwen(tmp_path):
    import gc,random,time
    from types import SimpleNamespace
    from functools import partial
    from dataclasses import asdict
    import pandas as pd
    from src.mech.global_low_rank_head import (
        NestedLowRankVocabularyHead,activation_whitened_factors,distil_nested_head,evaluate_nested_head)
    from src.models.official_codi import official_codi_base_model
    from src.data.answer_extract import answers_match
    torch.manual_seed(9); torch.set_num_threads(1)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=128,hidden_size=128,intermediate_size=128,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        eos_token_id=1,pad_token_id=0)).requires_grad_(False).eval()
    model=ablation.ExplicitBackbone(base)
    runtime=SimpleNamespace(**{k:v for k,v in vars(ablation.reference).items() if not k.startswith('__')})
    runtime.decode=ablation.decode; runtime.profile_question=ablation.profile_question
    runtime.bottleneck_means=partial(ablation.bottleneck_means,n_layers=1)
    records=[]
    scope=dict(torch=torch,gc=gc,random=random,time=time,pd=pd,json=json,asdict=asdict,
        model=model,base=model.codi,tokenizer=Tokenizer(),full_head=base.lm_head,
        weight=base.lm_head.weight.detach(),bias=None,device=torch.device('cpu'),
        runtime=runtime,RUN_DIR=tmp_path,MODEL_KEY='qwen2_5_0_5b',MODES=['explicit_cot'],
        RANKS=(32,64,96),SEED=89,CLEAN_EPOCHS=1,RECOVERY_EPOCHS=1,DISTILL_BATCH_SIZE=2,COLLECT_BATCH_SIZE=2,
        MAX_FIT_STATES=8,MAX_SELECT_STATES=4,MAX_RECOVERY_STATES=4,MAX_NEW_TOKENS={'explicit_cot':3},
        TIMING_REPEATS=1,DEPLOY_DTYPE=torch.float32,DATASET='tiny_test',DEBUG_PATH=tmp_path/'debug',
        setup=runtime.Timeline(enabled=False),flush_setup=lambda:None,
        save_pt=lambda path,value:torch.save(value,path),debug_record=lambda kind,value:records.append((kind,value)),
        NestedLowRankVocabularyHead=NestedLowRankVocabularyHead,activation_whitened_factors=activation_whitened_factors,
        distil_nested_head=distil_nested_head,evaluate_nested_head=evaluate_nested_head,
        official_codi_base_model=official_codi_base_model,answers_match=answers_match,
        splits={name:[dict(question=f'{name} short question'),dict(question=f'{name} a longer different question')]
                for name in ('fit','selection','recovery','timing','warmup')},
        test_rows=[dict(question='a test question',gold='1'),dict(question='a second test question',gold='2')])
    nb=json.loads((ROOT/'notebooks/kaggle_model_dataset_global_head_ablations.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    exec(next(s for s in sources if s.startswith('GPT2_LOAD =')),scope)
    for name in ('PARITY_CHECK','FIT_HEADS','TIMING_RUN','ACCURACY_RUN'):
        exec(scope[name],scope)
    exec(next(s for s in sources if s.startswith('def pair_reports(')),scope)
    tables,summaries=scope['pair_reports'](scope)
    assert len(tables)==4 and len(summaries)==2
    assert (tmp_path/'global_head_explicit_cot.pt').exists()
    assert len(scope['clean_samples'])==4 and len(scope['profile_samples'])==4
    assert len([1 for kind,_ in records if kind=='accuracy_sample'])==4
    exported=scope['safe_tables'](tables)
    assert all(r[1:] == [None,None,None] for table in exported.values() for r in table['data'])
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
