"""Generate the standalone CODI versus explicit-CoT global-head timing notebook."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'src/inference/global_head_comparison.py'
OUTPUT=ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb'
cells=[]
def md(s):
    cells.append(dict(cell_type='markdown',metadata={},source=s.strip().splitlines(True)))
def code(s):
    cells.append(dict(cell_type='code',metadata={},source=s.strip().splitlines(True),execution_count=None,outputs=[]))
md('''# Global low-rank head: CODI versus explicit GPT-2 CoT, with full timing diagnostics

Run all cells with a Kaggle GPU and Internet. **No data attachments or old results are
needed.** The historical benchmark is unchanged; this notebook embeds its new runtime.

Both modes use the released CODI GPT-2 checkpoint. CODI performs six latent passes;
explicit CoT generates directly after the question using the shared teacher/student
weights. This is not a separately trained CoT-SFT checkpoint. A separate head is fitted
on each mode's own trajectories using activation whitening, nested ranks 32/64/96,
KL + top-token + margin losses, and compressed-policy recovery.

Deployment compares dense, eager rank 96, compiled rank 96, historical Triton arithmetic,
and FP32-accumulating Triton. Numerical parity is measured, never assumed.

1. **Clean free generation:** actual latency with each head's own outputs.
2. **Clean fixed replay:** identical dense-reference tokens and step counts for every
   head, including transformer/cache work, to isolate implementation efficiency.
3. **Detailed diagnostics:** every module, stage, transfer, synchronization, and
   underlying profiler operation on a smaller fixed sample. Instrumentation adds
   overhead, so these measurements stay separate from clean speedups.

The PDF's numbers are historical references, not expected outputs. Explicit traces can
be longer, but improved overall speed is a hypothesis.
''')
code(r'''
SMOKE=False  # optional quick validation; False runs real experiments
SEED=89
MODES=['codi','explicit_cot']
FIT_QUESTIONS,SELECT_QUESTIONS,RECOVERY_QUESTIONS=1024,256,256
MAX_FIT_STATES,MAX_SELECT_STATES,MAX_RECOVERY_STATES=4096,1024,2048
CLEAN_EPOCHS,RECOVERY_EPOCHS=4,2
DISTILL_BATCH_SIZE=8
COLLECT_BATCH_SIZE=8
QUALITY_BATCH_SIZE=16
MAX_NEW_TOKENS={'codi':64,'explicit_cot':256}
TIMING_QUESTIONS=64
TIMING_BATCH_SIZES=(1,8,32)
TIMING_REPEATS=5
PROFILE_QUESTIONS=4  # module/stage events at batch 1
PROFILE_REPEATS=2
OPERATOR_TRACE_QUESTIONS=1  # complete operator trace per mode/head
TRY_COMPILE=True
TRY_TRITON=True
QUALITY_QUESTIONS=1319
RANKS=(32,64,96)
RESUME_DIR=''  # optional matching prior output run folder
OUTPUT_ROOT='/kaggle/working/codi_explicit_global_head'
if SMOKE:
    FIT_QUESTIONS,SELECT_QUESTIONS,RECOVERY_QUESTIONS=16,8,8
    MAX_FIT_STATES,MAX_SELECT_STATES,MAX_RECOVERY_STATES=128,64,64
    CLEAN_EPOCHS=RECOVERY_EPOCHS=1
    TIMING_QUESTIONS,TIMING_REPEATS,QUALITY_QUESTIONS=4,2,8
    TIMING_BATCH_SIZES=(1,4)
    PROFILE_QUESTIONS,PROFILE_REPEATS=1,1
    MAX_NEW_TOKENS={'codi':16,'explicit_cot':32}
assert 1 in TIMING_BATCH_SIZES
assert 0 < PROFILE_QUESTIONS <= TIMING_QUESTIONS
assert 0 <= OPERATOR_TRACE_QUESTIONS <= TIMING_QUESTIONS
assert PROFILE_REPEATS > 0 and TIMING_REPEATS > 1
import gc,gzip,hashlib,importlib.util,json,os,pathlib,random,shutil,subprocess,sys,time
from contextlib import contextmanager
BOOTSTRAP_TIMES=[]
@contextmanager
def bootstrap_stage(name):
    started=time.perf_counter()
    try: yield
    finally: BOOTSTRAP_TIMES.append(dict(name=name,kind='setup',cpu_wall_ms=1000*(time.perf_counter()-started)))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','300')
REPO_URL='https://github.com/0x0shephard/latent-reasoning.git'
BASE_COMMIT='6a8d2e61950c67f012d0a9ba13ec8a70f3a25019'
REPO_DIR='/kaggle/working/latent-reasoning'
with bootstrap_stage('git_clone_fetch_checkout'):
    if not pathlib.Path(REPO_DIR).exists():
        subprocess.run(['git','clone',REPO_URL,REPO_DIR],check=True)
    subprocess.run(['git','-C',REPO_DIR,'fetch','origin'],check=True)
    subprocess.run(['git','-C',REPO_DIR,'checkout','--detach',BASE_COMMIT],check=True)
os.chdir(REPO_DIR)
sys.path.insert(0,REPO_DIR)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE_COMMIT
with bootstrap_stage('dependency_installation'):
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.52.4',
        'peft==0.15.2','datasets==3.6.0','huggingface_hub==0.32.4',
        'accelerate==1.7.0','pandas==2.2.3','pyyaml','pytest','tqdm'],check=True)
    probe=subprocess.run([sys.executable,'-c',
        'from peft.import_utils import is_torchao_available; print(is_torchao_available())'],capture_output=True,text=True)
    if probe.returncode and 'torchao' in (probe.stdout+probe.stderr):
        subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
    elif probe.returncode: raise RuntimeError(probe.stderr)
''')
source=RUNTIME.read_text(encoding='utf-8')
fingerprint=hashlib.sha256((source+Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
md('## Embedded runtime and experiment identity')
code('RUNTIME_SOURCE = '+repr(source)+'\nRUNTIME_SHA256 = '+repr(hashlib.sha256(source.encode()).hexdigest())+'\nEXPERIMENT_SOURCE_SHA256 = '+repr(fingerprint)+r'''
with bootstrap_stage('runtime_import'):
    assert hashlib.sha256(RUNTIME_SOURCE.encode()).hexdigest()==RUNTIME_SHA256
    runtime_path=pathlib.Path('/kaggle/working/dual_global_head_runtime.py')
    runtime_path.write_text(RUNTIME_SOURCE)
    spec=importlib.util.spec_from_file_location('dual_global_head_runtime',runtime_path)
    runtime=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=runtime
    spec.loader.exec_module(runtime)
    import torch
    import pandas as pd
    from dataclasses import asdict
    from src.mech.global_low_rank_head import (
        NestedLowRankVocabularyHead,activation_whitened_factors,distil_nested_head,evaluate_nested_head)
    from src.models.official_codi import (
        build_official_codi_gpt2,download_official_checkpoint,load_official_checkpoint,official_codi_base_model)
    from src.inference.official_codi_fast import (
        generate_official_codi_fast,prepare_official_codi_batches,merge_official_codi_lora_)
    from src.data.answer_extract import answers_match,normalize_gold
    from src.utils.config import load_config
assert torch.cuda.is_available(),'Enable a Kaggle GPU'
device=torch.device('cuda')
torch.manual_seed(SEED); random.seed(SEED)
config=dict(smoke=SMOKE,seed=SEED,modes=MODES,fit=FIT_QUESTIONS,selection=SELECT_QUESTIONS,
    recovery=RECOVERY_QUESTIONS,state_caps=[MAX_FIT_STATES,MAX_SELECT_STATES,MAX_RECOVERY_STATES],
    epochs=[CLEAN_EPOCHS,RECOVERY_EPOCHS],distill_batch=DISTILL_BATCH_SIZE,
    collect_batch=COLLECT_BATCH_SIZE,quality_batch=QUALITY_BATCH_SIZE,quality=QUALITY_QUESTIONS,
    token_caps=MAX_NEW_TOKENS,timing_questions=TIMING_QUESTIONS,batches=TIMING_BATCH_SIZES,
    repeats=TIMING_REPEATS,profile_questions=PROFILE_QUESTIONS,profile_repeats=PROFILE_REPEATS,
    operator_trace_questions=OPERATOR_TRACE_QUESTIONS,compile=TRY_COMPILE,triton=TRY_TRITON,
    base=BASE_COMMIT,source=EXPERIMENT_SOURCE_SHA256,torch=torch.__version__,cuda=torch.version.cuda,
    triton_version=getattr(runtime.triton,'__version__',None),gpu=torch.cuda.get_device_name(0))
manifest=json.dumps(config,sort_keys=True,indent=2)
RUN_ID=hashlib.sha256(manifest.encode()).hexdigest()[:16]
RUN_DIR=pathlib.Path(OUTPUT_ROOT)/('smoke_' if SMOKE else 'full_')/RUN_ID
RUN_DIR.mkdir(parents=True,exist_ok=True)
if RESUME_DIR:
    previous=pathlib.Path(RESUME_DIR)
    assert (previous/'manifest.json').read_text()==manifest,'Resume settings/source/runtime mismatch'
    shutil.copytree(previous,RUN_DIR,dirs_exist_ok=True)
if (RUN_DIR/'manifest.json').exists(): assert (RUN_DIR/'manifest.json').read_text()==manifest
(RUN_DIR/'manifest.json').write_text(manifest)
setup=runtime.Timeline(device)
setup.records.extend(BOOTSTRAP_TIMES)
setup.context.update(mode='setup',protocol='setup')
setup.flush(RUN_DIR/'setup_events.jsonl.gz')
def save_json(path,value):
    temp=pathlib.Path(str(path)+'.tmp'); temp.write_text(json.dumps(value,indent=2,default=str)); temp.replace(path)
def save_pt(path,value):
    temp=pathlib.Path(str(path)+'.tmp'); torch.save(value,temp); temp.replace(path)
print('Output:',RUN_DIR)
print(json.dumps(config,indent=2))
''')
md('''## Download data and the shared checkpoint
Canonical GSM8K train supplies disjoint fit, selection, recovery, timing and warmup
questions, shared by both modes. Test labels never select weights or settings. Setup
costs are separated from warm inference; cached downloads naturally finish faster.
''')
code(r'''
from datasets import load_dataset
DATA_REVISION='3101c7d5072418e28b9008a6636bde82a006892c'
url=f'https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data/'
with setup.span('gsm8k_train_download_and_parse',gpu=False):
    train=load_dataset('json',data_files={'train':url+'train.jsonl'},split='train')
with setup.span('gsm8k_test_download_and_parse',gpu=False):
    test=load_dataset('json',data_files={'test':url+'test.jsonl'},split='test')
with setup.span('dataset_normalization_and_partitioning',gpu=False):
    unique={}
    for row in train:
        key=' '.join(row['question'].casefold().split())
        unique.setdefault(key,dict(question=str(row['question']),gold=str(normalize_gold(row['answer'],'gsm8k_main'))))
    rows=list(unique.values()); random.Random(SEED).shuffle(rows)
    cuts=[FIT_QUESTIONS,SELECT_QUESTIONS,RECOVERY_QUESTIONS,TIMING_QUESTIONS,32]
    splits={}; start=0
    for name,size in zip(['fit','selection','recovery','timing','warmup'],cuts):
        splits[name]=rows[start:start+size]; start+=size
    assert start<=len(rows)
    test_rows=[dict(question=str(r['question']),gold=str(normalize_gold(r['answer'],'gsm8k_main'))) for r in test]
    assert len(test_rows)==1319
    selected={' '.join(r['question'].casefold().split()) for part in splits.values() for r in part}
    assert not selected & {' '.join(r['question'].casefold().split()) for r in test_rows}
    test_rows=test_rows[:QUALITY_QUESTIONS]
    save_json(RUN_DIR/'partitions.json',dict(splits=splits,test_questions=[r['question'] for r in test_rows]))
with setup.span('config_load',gpu=False): cfg=load_config('configs/official_codi_gpt2.yaml')
with setup.span('codi_checkpoint_download_and_hash',gpu=False):
    checkpoint=download_official_checkpoint(repo_id=cfg.checkpoint.repo_id,revision=cfg.checkpoint.revision,
        filename=cfg.checkpoint.filename,expected_sha256=cfg.checkpoint.sha256)
with setup.span('gpt2_tokenizer_download_and_model_construction',gpu=False):
    model,tokenizer=build_official_codi_gpt2(base_model=cfg.model.base_model,base_revision=cfg.model.base_revision,
        dtype=torch.float32,settings=cfg.model)
with setup.span('checkpoint_load_and_verification',gpu=False):
    load_report=load_official_checkpoint(model,checkpoint,expected_sha256=cfg.checkpoint.sha256)
with setup.span('model_host_to_device_fp32'): model.requires_grad_(False).to(device).eval()
base=official_codi_base_model(model)
full_head=base.get_output_embeddings()
weight=full_head.weight[:model.eot_id].detach()
bias=None if getattr(full_head,'bias',None) is None else full_head.bias[:model.eot_id].detach()
setup.flush(RUN_DIR/'setup_events.jsonl.gz')
print('Shared GPT-2 checkpoint:',load_report.checkpoint_sha256)
''')
md('## Verify both generation paths before fitting')
code(r'''
parity_questions=[r['question'] for r in splits['selection'][:4]]
parity={}
with torch.no_grad():
    for mode in MODES:
        prepared=runtime.prepare_questions(tokenizer,parity_questions,4)
        observed=runtime.decode(model,tokenizer,prepared,full_head,mode=mode,device=device,
            max_new_tokens=MAX_NEW_TOKENS[mode],latent_iterations=6)
        if mode=='codi':
            reference=generate_official_codi_fast(model,tokenizer,
                prepare_official_codi_batches(tokenizer,parity_questions,batch_size=4,length_bucketed=False),
                latent_iterations=6,max_new_tokens=MAX_NEW_TOKENS[mode],device=device,answer_cue='The answer is:')
            expected=reference.token_ids
        else:
            from transformers import LogitsProcessor,LogitsProcessorList
            class VocabularyBoundary(LogitsProcessor):
                def __call__(self,input_ids,scores):
                    scores[:,int(model.eot_id):]=float('-inf'); return scores
            batch=prepared[0]
            generated=base.generate(input_ids=batch.ids.to(device),attention_mask=batch.mask.to(device),
                do_sample=False,max_new_tokens=MAX_NEW_TOKENS[mode],pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,logits_processor=LogitsProcessorList([VocabularyBoundary()]))
            expected=[]
            for seq in generated[:,batch.ids.shape[1]:].cpu().tolist():
                if tokenizer.eos_token_id in seq: seq=seq[:seq.index(tokenizer.eos_token_id)+1]
                expected.append(tuple(seq))
            expected=tuple(expected)
        parity[mode]=dict(examples=len(expected),exact=observed.token_ids==expected)
        assert parity[mode]['exact'],f'{mode}: custom decoder differs from reference; stop before fitting'
save_json(RUN_DIR/'decoder_parity.json',parity)
print(parity)
''')
md('''## Fit one global head for each reasoning mode
Four clean epochs plus two recovery epochs; the rank-64 prefix generates recovery
states. Validation agreement, then KL, selects checkpoints. Training is FP32;
deployment uses FP16 with merged LoRA, matching the previous efficiency notebook.
''')
code(r'''
def collect(mode,head,population,cap,tag):
    path=RUN_DIR/f'states_{mode}_{tag}.pt'
    if path.exists():
        with setup.span('state_cache_disk_load',gpu=False,mode=mode,tag=tag):
            return torch.load(path,map_location='cpu',weights_only=False)
    chunks=[]; positions=[]
    def observe(hidden,active,position,indices):
        chunks.append(hidden[active].detach().cpu().float())
        positions.extend([position]*int(active.sum()))
    questions=[r['question'] for r in splits[population]]
    batches=runtime.prepare_questions(tokenizer,questions,COLLECT_BATCH_SIZE)
    with setup.span('trajectory_collection',mode=mode,population=population):
        runtime.decode(model,tokenizer,batches,head,mode=mode,device=device,
                       max_new_tokens=MAX_NEW_TOKENS[mode],observer=observe)
    values=torch.cat(chunks)
    indices=torch.randperm(len(values),generator=torch.Generator().manual_seed(SEED))[:cap]
    result=dict(states=values[indices].clone(),positions=torch.tensor(positions)[indices],observed_states=len(values))
    with setup.span('state_cache_disk_save',gpu=False,mode=mode,tag=tag): save_pt(path,result)
    return result

def evaluate_positions(head,bundle):
    result={}
    for rank in RANKS:
        result[str(rank)]={}
        for label,mask in [('all',torch.ones(len(bundle['states']),dtype=torch.bool)),
                           ('p0',bundle['positions']==0),('p1',bundle['positions']==1),('p2plus',bundle['positions']>=2)]:
            result[str(rank)][label]=dict(states=int(mask.sum()),**(evaluate_nested_head(
                head,bundle['states'][mask],weight,readout_bias=bias,rank=rank,batch_size=DISTILL_BATCH_SIZE)
                if mask.any() else {}))
    return result

for mode in MODES:
    artifact=RUN_DIR/f'global_head_{mode}.pt'
    if artifact.exists(): print('Reuse fitted head:',mode); continue
    print('Collect/fitting:',mode,flush=True)
    fit=collect(mode,full_head,'fit',MAX_FIT_STATES,'fit')
    selection=collect(mode,full_head,'selection',MAX_SELECT_STATES,'selection')
    with setup.span('activation_whitened_initialization',mode=mode):
        centre,down,up,out_bias,init=activation_whitened_factors(fit['states'],weight,96,
            readout_bias=bias,seed=SEED,compute_device=device)
        head=NestedLowRankVocabularyHead.from_whitened_factors(centre,down,up,out_bias,RANKS).to(device)
    initial_metrics=evaluate_positions(head,selection)
    with setup.span('clean_distillation',mode=mode):
        clean=distil_nested_head(head,fit['states'],selection['states'],weight,readout_bias=bias,
            epochs=CLEAN_EPOCHS,batch_size=DISTILL_BATCH_SIZE,learning_rate=2e-4,seed=SEED)
    head.disable_adaptive(); head.set_rank(64)
    recovery=collect(mode,head,'recovery',MAX_RECOVERY_STATES,'onpolicy')
    with setup.span('recovery_distillation',mode=mode):
        recovered=distil_nested_head(head,torch.cat((fit['states'],recovery['states'])),
            selection['states'],weight,readout_bias=bias,epochs=RECOVERY_EPOCHS,
            batch_size=DISTILL_BATCH_SIZE,learning_rate=2e-4,seed=SEED+1)
    report=dict(mode=mode,initialization=asdict(init),clean=asdict(clean),recovery=asdict(recovered),
                initial=initial_metrics,final=evaluate_positions(head,selection),
                fit_states=len(fit['states']),recovery_states=len(recovery['states']))
    with setup.span('trained_head_disk_save',gpu=False,mode=mode):
        save_pt(artifact,dict(state_dict={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},report=report))
        save_json(artifact.with_suffix('.json'),report)
    setup.flush(RUN_DIR/'setup_events.jsonl.gz')
    del head,fit,selection,recovery,centre,down,up,out_bias
    gc.collect(); torch.cuda.empty_cache()
with setup.span('lora_merge'): merge_official_codi_lora_(model)
with setup.span('model_fp16_conversion'): model.to(dtype=torch.float16).eval()
base=official_codi_base_model(model); full_head=base.get_output_embeddings()
del weight,bias
setup.flush(RUN_DIR/'setup_events.jsonl.gz')
''')
md('''## Deployment measurement runner
Raw rows retain mode, head, protocol, repeat, batch/question IDs, token counts and wall
time. Clean runs install no module hooks or per-layer CUDA events. Arm order is shuffled
inside each repeat. Compilation and real-shape warmups occur before timing. Optional
backend failures are exported explicitly. Fixed replay is never used for accuracy.
''')
code(r'''
def local_batch(batch):
    return runtime.PreparedBatch(tuple(range(len(batch.indices))),batch.ids,batch.mask)

@torch.inference_mode()
def timed_batch(mode,head,batch,reference=None,timeline=None):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    allocated=torch.cuda.memory_allocated(); started=time.perf_counter()
    result=runtime.decode(model,tokenizer,[local_batch(batch)],head,mode=mode,device=device,
        max_new_tokens=MAX_NEW_TOKENS[mode],forced_tokens=reference,timeline=timeline)
    torch.cuda.synchronize(); wall=1000*(time.perf_counter()-started)
    return result,dict(wall_ms=wall,wall_ms_per_question=wall/len(batch.indices),
        questions=len(batch.indices),visible_tokens=result.generated_token_count,
        peak_increment_bytes=max(0,torch.cuda.max_memory_allocated()-allocated))

def make_arms(mode):
    with setup.span('fitted_head_disk_load',gpu=False,mode=mode):
        payload=torch.load(RUN_DIR/f'global_head_{mode}.pt',map_location='cpu',weights_only=False)
    nested=NestedLowRankVocabularyHead(int(model.config.hidden_size),int(model.eot_id),RANKS)
    nested.load_state_dict(payload['state_dict'])
    with setup.span('fitted_head_host_to_device_fp16',mode=mode):
        eager=runtime.FixedRankHead(nested,96).to(device=device,dtype=torch.float16).eval()
    arms={'dense':runtime.DenseSelector(full_head,int(model.eot_id)),'rank96_eager':eager}
    errors={}; constructors={}
    if TRY_COMPILE: constructors['rank96_compiled']=lambda:runtime.CompiledSelector(eager)
    if TRY_TRITON:
        constructors['rank96_triton_legacy']=lambda:runtime.TritonSelector(eager,wide=False)
        constructors['rank96_triton_fp32']=lambda:runtime.TritonSelector(eager,wide=True)
    warm=[r['question'] for r in splits['warmup']]
    for name,constructor in constructors.items():
        try:
            with setup.span('optional_backend_construction_and_warmup',mode=mode,arm=name):
                head=constructor().eval()
                for batch_size in TIMING_BATCH_SIZES:
                    batch=runtime.prepare_questions(tokenizer,warm[:batch_size],batch_size)[0]
                    timed_batch(mode,head,batch)
                arms[name]=head
        except Exception as error: errors[name]=repr(error)
    for name in ('dense','rank96_eager'):
        for batch_size in TIMING_BATCH_SIZES:
            timed_batch(mode,arms[name],runtime.prepare_questions(tokenizer,warm[:batch_size],batch_size)[0])
    save_json(RUN_DIR/f'backend_status_{mode}.json',dict(available=list(arms),errors=errors))
    return arms

ALL_TIMINGS=[]; ALL_QUALITY=[]; HEAD_TIMINGS=[]; PROFILE_OVERHEAD=[]
for mode in MODES:
    print('Benchmark:',mode,flush=True)
    mode_dir=RUN_DIR/mode; mode_dir.mkdir(exist_ok=True)
    if (mode_dir/'completed.json').exists():
        ALL_TIMINGS.extend(json.loads((mode_dir/'clean_raw.json').read_text()))
        ALL_QUALITY.extend(json.loads((mode_dir/'quality_summary.json').read_text()))
        HEAD_TIMINGS.extend(json.loads((mode_dir/'head_raw.json').read_text()))
        PROFILE_OVERHEAD.extend(json.loads((mode_dir/'profile_overhead.json').read_text()))
        continue
    path=mode_dir/'detailed_events.jsonl.gz'
    if path.exists(): path.unlink()  # restart only this incomplete mode's diagnostic log
    arms=make_arms(mode)
    raw=[]; quality=[]; head_raw=[]; overhead=[]
    timing_questions=[r['question'] for r in splits['timing']]
    with setup.span('timing_prompt_preparation',gpu=False,mode=mode):
        prepared={b:runtime.prepare_questions(tokenizer,timing_questions,b) for b in TIMING_BATCH_SIZES}
    with torch.inference_mode():
        references={}
        for batch_size,batches in prepared.items():
            result=runtime.decode(model,tokenizer,batches,arms['dense'],mode=mode,device=device,
                                  max_new_tokens=MAX_NEW_TOKENS[mode])
            references[batch_size]=result.token_ids
        save_json(mode_dir/'dense_timing_reference_tokens.json',references)
        # Real FP16 deployment states, not the earlier FP32 fitting-state cache.
        state_chunks=[]
        def diagnostic_observer(hidden,active,position,indices):
            state_chunks.append(hidden[active].detach().cpu())
        diagnostic_questions=[r['question'] for r in splits['warmup'][:8]]
        runtime.decode(model,tokenizer,runtime.prepare_questions(tokenizer,diagnostic_questions,8),
            arms['dense'],mode=mode,device=device,max_new_tokens=MAX_NEW_TOKENS[mode],observer=diagnostic_observer)
        pool=torch.cat(state_chunks)
        hidden=pool[:min(512,len(pool))].to(device=device,dtype=torch.float16)
        eager_scores=arms['rank96_eager'](hidden).float(); expected=eager_scores.argmax(-1)
        fidelity={}
        for name,head in arms.items():
            tokens=runtime.select_token(head,hidden,int(model.eot_id))
            regret=eager_scores.max(-1).values-eager_scores.gather(1,tokens[:,None]).squeeze(1)
            fidelity[name]=dict(states=len(hidden),agreement_with_eager=float((tokens==expected).float().mean()),
                mean_eager_score_regret=float(regret.mean()),max_eager_score_regret=float(regret.max()))
        save_json(mode_dir/'selector_fidelity.json',fidelity)
        for batch_size in TIMING_BATCH_SIZES:
            sample=hidden[torch.arange(batch_size,device=device)%len(hidden)].contiguous()
            for name,head in arms.items():
                for _ in range(20): runtime.select_token(head,sample,int(model.eot_id))
                torch.cuda.synchronize()
                for repeat in range(TIMING_REPEATS):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record(); started=time.perf_counter()
                    for _ in range(100): runtime.select_token(head,sample,int(model.eot_id))
                    end.record(); torch.cuda.synchronize()
                    head_raw.append(dict(mode=mode,arm=name,batch_size=batch_size,repeat=repeat,
                        cuda_stream_ms_per_call=begin.elapsed_time(end)/100,
                        wall_ms_per_call=1000*(time.perf_counter()-started)/100,calls_per_measurement=100))
        for protocol in ('free_generation','fixed_replay'):
            for batch_size,batches in prepared.items():
                for repeat in range(TIMING_REPEATS):
                    order=list(arms); random.Random(SEED+repeat+batch_size).shuffle(order)
                    for name in order:
                        for batch_index,batch in enumerate(batches):
                            ref=[references[batch_size][i] for i in batch.indices] if protocol=='fixed_replay' else None
                            result,row=timed_batch(mode,arms[name],batch,ref)
                            raw.append(dict(mode=mode,arm=name,protocol=protocol,batch_size=batch_size,
                                actual_batch_size=len(batch.indices),repeat=repeat,batch_index=batch_index,
                                question_indices=list(batch.indices),token_counts=list(result.generated_token_counts),**row))
                        save_json(mode_dir/'clean_raw.json',raw)
                        print(mode,protocol,batch_size,repeat,name,flush=True)
        for name,head in arms.items():
            batches=runtime.prepare_questions(tokenizer,[r['question'] for r in test_rows],QUALITY_BATCH_SIZE)
            result=runtime.decode(model,tokenizer,batches,head,mode=mode,device=device,max_new_tokens=MAX_NEW_TOKENS[mode])
            records=[]
            for i,(example,text,tokens) in enumerate(zip(test_rows,result.texts,result.token_ids)):
                records.append(dict(index=i,question=example['question'],gold=example['gold'],text=text,
                    token_ids=list(tokens),correct=bool(answers_match(text,example['gold'])),
                    eos_terminated=bool(tokens and tokens[-1]==tokenizer.eos_token_id),
                    answer_cue_present='the answer is:' in text.casefold()))
            save_json(mode_dir/f'quality_{name}.json',records)
            quality.append(dict(mode=mode,arm=name,examples=len(records),correct=sum(r['correct'] for r in records),
                accuracy=sum(r['correct'] for r in records)/len(records),
                completed_correct=sum(r['correct'] and r['eos_terminated'] for r in records)/len(records),
                truncated_fraction=sum(not r['eos_terminated'] for r in records)/len(records),
                mean_generated_tokens=result.generated_token_count/len(records)))
    for name,head in arms.items():
        for repeat in range(PROFILE_REPEATS):
            trace=runtime.Timeline(device)
            trace.context.update(mode=mode,arm=name,protocol='instrumented_fixed_replay',batch_size=1,repeat=repeat)
            with trace.span('profile_prompt_preparation',gpu=False):
                batches=runtime.prepare_questions(tokenizer,timing_questions[:PROFILE_QUESTIONS],1,trace)
            for i,batch in enumerate(batches):
                ref=[references[1][i]]
                _,clean=timed_batch(mode,head,batch,ref)
                roots=[('transformer',base.transformer),('latent_projector',model.prj)]
                if name in ('dense','rank96_eager'): roots.append(('head',head))
                with trace.metadata(profile_question=i):
                    with trace.modules(roots): _,instrumented=timed_batch(mode,head,batch,ref,trace)
                overhead.append(dict(mode=mode,arm=name,repeat=repeat,question_index=i,
                    clean_wall_ms=clean['wall_ms'],instrumented_wall_ms=instrumented['wall_ms'],
                    instrumentation_ratio=instrumented['wall_ms']/clean['wall_ms']))
                trace.flush(mode_dir/'detailed_events.jsonl.gz')
        if OPERATOR_TRACE_QUESTIONS:
            from torch.profiler import profile,ProfilerActivity
            try:
                batches=runtime.prepare_questions(tokenizer,timing_questions[:OPERATOR_TRACE_QUESTIONS],1)
                with torch.inference_mode(),profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                                                    record_shapes=True,profile_memory=True) as prof:
                    for i,batch in enumerate(batches): timed_batch(mode,head,batch,[references[1][i]])
                with setup.span('chrome_trace_export',gpu=False,mode=mode,arm=name):
                    prof.export_chrome_trace(str(mode_dir/f'operators_{name}.json'))
                    op_rows=[]
                    for event in prof.events():
                        op_rows.append(dict(name=event.name,device_type=str(event.device_type),
                            cpu_total_us=event.cpu_time_total,cpu_self_us=event.self_cpu_time_total,
                            device_total_us=getattr(event,'device_time_total',None),
                            device_self_us=getattr(event,'self_device_time_total',None),input_shapes=str(event.input_shapes)))
                    runtime.write_csv(mode_dir/f'operator_events_{name}.csv',op_rows)
                    (mode_dir/f'operator_averages_{name}.txt').write_text(
                        prof.key_averages(group_by_input_shape=True).table(sort_by='self_cuda_time_total',row_limit=200))
            except Exception as error: save_json(mode_dir/f'profiler_error_{name}.json',dict(error=repr(error)))
    save_json(mode_dir/'clean_raw.json',raw)
    save_json(mode_dir/'head_raw.json',head_raw)
    save_json(mode_dir/'quality_summary.json',quality)
    save_json(mode_dir/'profile_overhead.json',overhead)
    runtime.summarize_events(mode_dir/'detailed_events.jsonl.gz',mode_dir/'layer_stage_averages.csv')
    ALL_TIMINGS.extend(raw); ALL_QUALITY.extend(quality); HEAD_TIMINGS.extend(head_raw); PROFILE_OVERHEAD.extend(overhead)
    setup.flush(RUN_DIR/'setup_events.jsonl.gz')
    save_json(mode_dir/'completed.json',dict(complete=True))
    del arms,head,hidden,eager_scores,pool,state_chunks
    gc.collect(); torch.cuda.empty_cache()
''')
md('## Averages, uncertainty, and individual measurements')
code(r'''
summary=runtime.summarize_timings(ALL_TIMINGS)
frame=pd.DataFrame(summary)
for index,row in frame.iterrows():
    dense=frame[(frame['mode']==row['mode'])&(frame.protocol==row.protocol)&
                (frame.batch_size==row.batch_size)&(frame.arm=='dense')].iloc[0]
    frame.loc[index,'mean_speedup_vs_dense']=dense.mean_ms_per_question/row.mean_ms_per_question
    frame.loc[index,'median_speedup_vs_dense']=dense.median_ms_per_question/row.median_ms_per_question
frame.to_csv(RUN_DIR/'timing_averages.csv',index=False)
runtime.write_csv(RUN_DIR/'timing_individual.csv',ALL_TIMINGS)
for quality_row in ALL_QUALITY:
    dense_accuracy=next(r['accuracy'] for r in ALL_QUALITY if r['mode']==quality_row['mode'] and r['arm']=='dense')
    quality_row['accuracy_retention_vs_dense']=quality_row['accuracy']/dense_accuracy if dense_accuracy else None
runtime.write_csv(RUN_DIR/'quality_summary.csv',ALL_QUALITY)
runtime.write_csv(RUN_DIR/'head_individual.csv',HEAD_TIMINGS)
runtime.write_csv(RUN_DIR/'profiling_overhead.csv',PROFILE_OVERHEAD)
head_frame=pd.DataFrame(HEAD_TIMINGS)
head_frame.groupby(['mode','arm','batch_size'])[['wall_ms_per_call','cuda_stream_ms_per_call']].agg(
    ['count','mean','median','std','min','max']).to_csv(RUN_DIR/'head_averages.csv')
# Descriptive paired bootstrap over repeat totals for the same question population.
uncertainty=[]
raw_frame=pd.DataFrame(ALL_TIMINGS)
for (mode,protocol,batch_size),group in raw_frame.groupby(['mode','protocol','batch_size']):
    totals=group.groupby(['arm','repeat']).wall_ms.sum().unstack(0)
    gen=torch.Generator().manual_seed(SEED)
    indices=torch.randint(len(totals),(2000,len(totals)),generator=gen)
    for arm in totals.columns:
        a=torch.tensor(totals['dense'].to_numpy())[indices].mean(1)
        b=torch.tensor(totals[arm].to_numpy())[indices].mean(1)
        ratio=a/b
        uncertainty.append(dict(mode=mode,protocol=protocol,batch_size=batch_size,arm=arm,
            paired_repeat_speedup_ci95_low=float(ratio.quantile(.025)),
            paired_repeat_speedup_ci95_high=float(ratio.quantile(.975)),repeats=len(totals)))
runtime.write_csv(RUN_DIR/'timing_speedup_intervals.csv',uncertainty)
runtime.summarize_events(RUN_DIR/'setup_events.jsonl.gz',RUN_DIR/'setup_averages.csv')
display(frame)
display(pd.DataFrame(ALL_QUALITY))
print('Individual measurements:'); display(raw_frame.head(20))
print('Layer/stage example:'); display(pd.read_csv(RUN_DIR/MODES[0]/'layer_stage_averages.csv').head(30))
profiler_errors=[str(p.relative_to(RUN_DIR)) for p in RUN_DIR.glob('*/profiler_error_*.json')]
if profiler_errors: print('Some operator traces failed; see:',profiler_errors)
save_json(RUN_DIR/'completed.json',dict(complete=True,smoke=SMOKE,source=EXPERIMENT_SOURCE_SHA256,
    operator_profiling_complete=not profiler_errors,profiler_errors=profiler_errors))
print('Download output folder:',RUN_DIR)
''')
md('''## Inspecting and interpreting output

- `timing_averages.csv`: clean mean, median, SD, range, and speedup for each mode/head/batch/protocol.
- `timing_individual.csv`: each clean **batch invocation**, with repeat and question IDs;
  batch 1 gives individual-question latency. Larger batches give batch latency, not
  independent latency for each simultaneously processed question.
- `timing_speedup_intervals.csv`: descriptive paired-repeat bootstrap intervals.
- `head_averages.csv` / `head_individual.csv`: isolated selector timing including argmax;
  each microbenchmark row averages 100 calls, explicitly recorded in the raw file.
  Individual head-call spans are in the detailed diagnostic log.
- `<mode>/layer_stage_averages.csv`: each numbered transformer layer, attention/MLP
  submodule, norm, embedding, projector, and explicitly instrumented runtime stage.
- `<mode>/detailed_events.jsonl.gz`: **individual calls** with layer name, phase,
  question, token position, repeat, parent event, CPU wall time and CUDA stream time.
- `<mode>/operators_<arm>.json`: Chrome/Perfetto CPU/CUDA operator and kernel timeline.
- `<mode>/operator_events_<arm>.csv` / `operator_averages_<arm>.txt`: raw and averaged operators.
- `setup_events.jsonl.gz` / `setup_averages.csv`: downloads, parsing, state I/O,
  model loading/movement, fitting, warmup, and trace exports.
- `profiling_overhead.csv`: runtime change introduced by instrumentation.
- `<mode>/quality_<arm>.json`: every generated answer/CoT, token IDs and termination status.
- `<mode>/selector_fidelity.json`: numerical agreement/regret on real states.

**Never sum inclusive parent and child times.** CUDA events measure stream intervals
including idle gaps; CPU module times usually measure dispatch, not GPU execution.
CPU-only stages have no CUDA timing. Profiler device events expose actual kernels.
Functional operations inside a layer appear in operator traces even without a separate
`nn.Module`. Compiled heads are timed as units; their kernels appear in operator traces,
without hooks inside the compiled head that would alter compilation.

Profiling uses a fixed sample because logging every operation over the full test would
create huge traces and heavily distort runtime. Increase `PROFILE_QUESTIONS`,
`PROFILE_REPEATS`, or `OPERATOR_TRACE_QUESTIONS` before running for broader coverage.
All clean timing repetitions are saved individually. Quality uses 1,319 questions by
default. Explicit generation stopped at its token cap is reported as truncated.

Resume: attach the prior output and point `RESUME_DIR` at the exact matching run folder.
Fitted heads and fully completed mode benchmarks are reused; an interrupted fit or mode
benchmark restarts. Keep the same settings/source/runtime. Fresh runs need no attachments.
''')
for index,cell in enumerate(cells): cell['id']=f'dual-profile-{index:03d}'
notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
    kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'),
    kaggle=dict(accelerator='gpu',isInternetEnabled=True,dataSources=[])))
OUTPUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(OUTPUT)
