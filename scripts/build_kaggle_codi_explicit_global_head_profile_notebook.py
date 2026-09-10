"""Build the self-contained T4 notebook with one four-column bottleneck table."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/inference/global_head_comparison.py'
OUTPUT = ROOT / 'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb'
cells = []
def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip().splitlines(True)))
def code(s):
    cells.append(dict(cell_type='code', metadata={}, source=s.strip().splitlines(True), execution_count=None, outputs=[]))

md('''# Where does the time go? Explicit GPT-2 vs CODI

**Kaggle: Settings → Accelerator → GPU T4 x2; Internet on; Run All.** One T4 is used.
No attachments, old summaries, settings changes, or prior fitted heads are needed.

The final output is one table: **Explicit | CODI overall | CODI latent only | CODI visible only**.
Every value is mean **milliseconds per question**, with the total at the top and bottom.
The rows cover all 12 transformer blocks, the head, transfers, text preparation, and runtime overhead.

This runs the PDF's **rank-96 global head, eager PyTorch, FP16, batch 1**. Both paths use
CODI's released GPT-2 weights: explicit generates the teacher-style reasoning text;
CODI takes six latent steps before its visible answer. Each mode gets its own fitted head.
Nested 32/64/96 prefixes remain part of the training recipe; there is no deployment ablation sweep.
''')
code(r'''
import gc,gzip,hashlib,importlib.util,json,logging,os,pathlib,random,subprocess,sys,time,warnings
from contextlib import contextmanager
from urllib.request import urlopen

# Fixed experiment defaults: no configuration is required.
SEED=89
MODES=['codi','explicit_cot']
FIT_QUESTIONS,SELECT_QUESTIONS,RECOVERY_QUESTIONS=1024,256,256
MAX_FIT_STATES,MAX_SELECT_STATES,MAX_RECOVERY_STATES=4096,1024,2048
CLEAN_EPOCHS,RECOVERY_EPOCHS=4,2
DISTILL_BATCH_SIZE=8
COLLECT_BATCH_SIZE=8
RANKS=(32,64,96)
MAX_NEW_TOKENS={'codi':64,'explicit_cot':256}
TIMING_QUESTIONS,TIMING_REPEATS=16,3
OUTPUT_ROOT=pathlib.Path('/kaggle/working/codi_bottleneck')
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
BOOTSTRAP_TIMES=[]
@contextmanager
def bootstrap_stage(name):
    start=time.perf_counter()
    try: yield
    finally: BOOTSTRAP_TIMES.append(dict(name=name,cpu_wall_ms=1000*(time.perf_counter()-start)))

# Do not reinstall datasets/pandas/dill or change Kaggle's PyTorch/CUDA.
# GSM8K is read directly from its canonical JSONL files.
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ.pop('HF_HUB_DISABLE_XET',None)
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','300')
REPO_URL='https://github.com/0x0shephard/latent-reasoning.git'
BASE_COMMIT='6a8d2e61950c67f012d0a9ba13ec8a70f3a25019'
REPO_DIR='/kaggle/working/latent-reasoning'
setup_log=OUTPUT_ROOT/'setup.log'
def checked(command):
    result=subprocess.run(command,capture_output=True,text=True)
    with setup_log.open('a') as log: log.write(result.stdout+'\n'+result.stderr+'\n')
    if result.returncode:
        raise RuntimeError(result.stdout[-3000:]+'\n'+result.stderr[-5000:])
    return result
with bootstrap_stage('git_clone_and_checkout'):
    if not pathlib.Path(REPO_DIR).exists(): checked(['git','clone',REPO_URL,REPO_DIR])
    checked(['git','-C',REPO_DIR,'fetch','origin'])
    checked(['git','-C',REPO_DIR,'checkout','--detach',BASE_COMMIT])
os.chdir(REPO_DIR)
sys.path.insert(0,REPO_DIR)
with bootstrap_stage('dependency_installation'):
    checked([sys.executable,'-m','pip','install','-q','transformers==4.52.4','peft==0.15.2',
             'huggingface_hub>=0.34.0,<1.0','hf_xet','accelerate==1.7.0','pyyaml'])
    probe=subprocess.run([sys.executable,'-c','import peft; from transformers import GPT2LMHeadModel'],capture_output=True,text=True)
    if probe.returncode and 'torchao' in (probe.stdout+probe.stderr):
        checked([sys.executable,'-m','pip','uninstall','-y','torchao'])
    checked([sys.executable,'-c','import torch,peft,huggingface_hub,hf_xet; from transformers import GPT2LMHeadModel,DynamicCache'])
print('Setup ready. Unrelated system-package messages are retained in setup.log; failed installs/imports stop here.')
''')
source=RUNTIME.read_text(encoding='utf-8')
fingerprint=hashlib.sha256((source+Path(__file__).read_text(encoding='utf-8')).encode()).hexdigest()
code('RUNTIME_SOURCE = '+repr(source)+'\nEXPERIMENT_SOURCE_SHA256 = '+repr(fingerprint)+r'''
runtime_path=pathlib.Path('/kaggle/working/dual_global_head_runtime.py')
runtime_path.write_text(RUNTIME_SOURCE)
spec=importlib.util.spec_from_file_location('dual_global_head_runtime',runtime_path)
runtime=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runtime
spec.loader.exec_module(runtime)
import torch
import pandas as pd
from dataclasses import asdict
from importlib.metadata import version
from src.mech.global_low_rank_head import (
    NestedLowRankVocabularyHead,activation_whitened_factors,distil_nested_head,evaluate_nested_head)
from src.models.official_codi import (
    build_official_codi_gpt2,download_official_checkpoint,load_official_checkpoint,official_codi_base_model)
from src.inference.official_codi_fast import (
    generate_official_codi_fast,prepare_official_codi_batches,merge_official_codi_lora_)
from src.utils.config import load_config
assert torch.cuda.is_available(),'Select Settings > Accelerator > GPU T4 x2.'
assert 'T4' in torch.cuda.get_device_name(0),'Select GPU T4 x2 in Kaggle settings and restart the session.'
device=torch.device('cuda:0')
DEPLOY_DTYPE=torch.float16
torch.manual_seed(SEED); random.seed(SEED)
config=dict(seed=SEED,fit=FIT_QUESTIONS,selection=SELECT_QUESTIONS,recovery=RECOVERY_QUESTIONS,
    state_caps=[MAX_FIT_STATES,MAX_SELECT_STATES,MAX_RECOVERY_STATES],ranks=RANKS,
    epochs=[CLEAN_EPOCHS,RECOVERY_EPOCHS],token_caps=MAX_NEW_TOKENS,
    questions=TIMING_QUESTIONS,repeats=TIMING_REPEATS,batch_size=1,head='rank96_eager',
    base=BASE_COMMIT,source=EXPERIMENT_SOURCE_SHA256,torch=torch.__version__,cuda=torch.version.cuda,
    packages={name:version(name) for name in ('transformers','peft','huggingface_hub','hf_xet','accelerate')},
    gpu=torch.cuda.get_device_name(0))
RUN_DIR=OUTPUT_ROOT/hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
RUN_DIR.mkdir(parents=True,exist_ok=True)
DEBUG_PATH=RUN_DIR/'debug.jsonl.gz'
# Each Run All starts a fresh measurement log; fitted heads in the same run folder are reused.
with gzip.open(DEBUG_PATH,'wt') as stream: pass

def debug_record(kind,payload):
    with gzip.open(DEBUG_PATH,'at',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(kind=kind,data=payload),default=str)+'\n')
def save_json(path,value):
    temp=pathlib.Path(str(path)+'.tmp'); temp.write_text(json.dumps(value,indent=2,default=str)); temp.replace(path)
def save_pt(path,value):
    temp=pathlib.Path(str(path)+'.tmp'); torch.save(value,temp); temp.replace(path)
setup=runtime.Timeline(device)
setup.context.update(mode='setup')
setup.records.extend(BOOTSTRAP_TIMES)
def flush_setup():
    setup.resolve()
    for row in setup.records: debug_record('setup',row)
    setup.records.clear()
debug_record('manifest',config)
debug_record('dependency_log',setup_log.read_text())
flush_setup()
# These two construction notices are expected in the pinned official loader.
# Preserve them in the debug log and keep unrelated warnings visible.
class KnownConstructionNotice(logging.Filter):
    def filter(self,record):
        if record.getMessage().startswith('The new embeddings will be initialized'):
            debug_record('construction_notice',record.getMessage()); return False
        return True
logging.getLogger('transformers.modeling_utils').addFilter(KnownConstructionNotice())
print(f"Using {config['gpu']}; rank 96, FP16, batch 1. Fitting is the slow setup step.")
''')
md('Download the official checkpoint and fit on GSM8K train. Timing questions are held out from fitting.')
code(r'''
DATA_REVISION='3101c7d5072418e28b9008a6636bde82a006892c'
url=f'https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data/train.jsonl'
with setup.span('gsm8k_download_parse_and_partition',gpu=False):
    with urlopen(url,timeout=300) as response:
        train=[json.loads(line) for line in response if line.strip()]
    unique={}
    for row in train:
        key=' '.join(row['question'].casefold().split())
        unique.setdefault(key,dict(question=str(row['question']),gold=str(row['answer'])))
    rows=list(unique.values()); random.Random(SEED).shuffle(rows)
    splits={}; start=0
    for name,size in zip(['fit','selection','recovery','timing','warmup'],
                         [FIT_QUESTIONS,SELECT_QUESTIONS,RECOVERY_QUESTIONS,TIMING_QUESTIONS,4]):
        splits[name]=rows[start:start+size]; start+=size
    assert start<=len(rows)
    debug_record('partitions',splits)
with setup.span('config_load',gpu=False): cfg=load_config('configs/official_codi_gpt2.yaml')
with setup.span('checkpoint_download_and_hash',gpu=False):
    checkpoint=download_official_checkpoint(repo_id=cfg.checkpoint.repo_id,revision=cfg.checkpoint.revision,
        filename=cfg.checkpoint.filename,expected_sha256=cfg.checkpoint.sha256)
with setup.span('gpt2_model_and_tokenizer_load',gpu=False):
    with warnings.catch_warnings(record=True) as notices:
        warnings.filterwarnings('always',message='fan_in_fan_out is set to False.*')
        model,tokenizer=build_official_codi_gpt2(base_model=cfg.model.base_model,base_revision=cfg.model.base_revision,
            dtype=torch.float32,settings=cfg.model)
    for notice in notices:
        if 'fan_in_fan_out is set to False' in str(notice.message):
            debug_record('construction_notice',str(notice.message))
        else: warnings.warn(str(notice.message),notice.category)
with setup.span('checkpoint_load_and_verification',gpu=False):
    load_report=load_official_checkpoint(model,checkpoint,expected_sha256=cfg.checkpoint.sha256)
    debug_record('checkpoint',asdict(load_report))
with setup.span('model_host_to_device_fp32'): model.requires_grad_(False).to(device).eval()
base=official_codi_base_model(model)
full_head=base.get_output_embeddings()
weight=full_head.weight[:model.eot_id].detach()
bias=None if getattr(full_head,'bias',None) is None else full_head.bias[:model.eot_id].detach()
flush_setup()
''')
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
debug_record('decoder_parity',parity)
print('Decoder checks passed: CODI and explicit GPT-2.')
''')

md('Fit the proposal’s head for each mode: four clean epochs, then two recovery epochs.')
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
    flush_setup()
    del head,fit,selection,recovery,centre,down,up,out_bias
    gc.collect(); torch.cuda.empty_cache()
with setup.span('lora_merge'): merge_official_codi_lora_(model)
with setup.span('model_fp16_conversion'): model.to(dtype=DEPLOY_DTYPE).eval()
base=official_codi_base_model(model); full_head=base.get_output_embeddings()
del weight,bias
flush_setup()
''')
md('''Measure batch-1 generation on the same 16 held-out questions, repeated three times.
Only rank 96 receives detailed profiling. Dense generation supplies a brief unprofiled reference.
''')
code(r'''
# Timing runner: only one deployed method, one batch size, and two reasoning modes.
@torch.inference_mode()
def clean_generation(mode,head,question):
    if device.type=='cuda': torch.cuda.synchronize(device)
    start=time.perf_counter()
    batches=runtime.prepare_questions(tokenizer,[question],1)
    result=runtime.decode(model,tokenizer,batches,head,mode=mode,device=device,max_new_tokens=MAX_NEW_TOKENS[mode])
    if device.type=='cuda': torch.cuda.synchronize(device)
    return result,1000*(time.perf_counter()-start)

heads={}
for mode in MODES:
    with setup.span('fitted_head_load_and_move',mode=mode):
        payload=torch.load(RUN_DIR/f'global_head_{mode}.pt',map_location='cpu',weights_only=False)
        nested=NestedLowRankVocabularyHead(int(model.config.hidden_size),int(model.eot_id),RANKS)
        nested.load_state_dict(payload['state_dict'])
        heads[mode]=runtime.FixedRankHead(nested,96).to(device=device,dtype=DEPLOY_DTYPE).eval()
        agreement=payload['report']['final']['96']['all']['top1_agreement']
        print(f'{mode}: fitted head validation token agreement {agreement:.1%}.')
        debug_record('fit_report',payload['report'])
    del payload,nested

dense=runtime.DenseSelector(full_head,int(model.eot_id))
with setup.span('generation_warmup'):
    for mode in MODES:
        for row in splits['warmup']:
            for head in (dense,heads[mode]): clean_generation(mode,head,row['question'])
        # Warm the instrumented path too; discard this trace.
        runtime.profile_question(model,tokenizer,heads[mode],splits['warmup'][0]['question'],
            mode=mode,device=device,max_new_tokens=MAX_NEW_TOKENS[mode])
flush_setup()

clean_samples=[]
profile_samples=[]
rng=random.Random(SEED)
for repeat in range(TIMING_REPEATS):
    # Finish clean measurements before attaching any profiling hooks.
    jobs=[(mode,i) for mode in MODES for i in range(len(splits['timing']))]
    rng.shuffle(jobs)
    expected={}
    for mode,i in jobs:
        arms=[('dense',dense),('rank96',heads[mode])]; rng.shuffle(arms)
        for arm,head in arms:
            result,elapsed=clean_generation(mode,head,splits['timing'][i]['question'])
            row=dict(mode=mode,arm=arm,question_id=i,repeat=repeat,ms=elapsed,
                     visible_tokens=result.generated_token_counts[0],tokens=list(result.token_ids[0]),text=result.texts[0])
            clean_samples.append(row)
            debug_record('clean_sample',row)
            if arm=='rank96': expected[mode,i]=row['tokens']
    for mode,i in jobs:
        sample,events=runtime.profile_question(model,tokenizer,heads[mode],splits['timing'][i]['question'],
            mode=mode,device=device,max_new_tokens=MAX_NEW_TOKENS[mode],question_id=i,repeat=repeat)
        assert sample['tokens']==expected[mode,i],'Instrumentation changed generated tokens'
        profile_samples.append(sample)
        debug_record('profile_sample',sample)
        debug_record('events',events)
        del events
    print(f'Timing repeat {repeat+1}/{TIMING_REPEATS} complete.',flush=True)
''')
md('''**Reading the table:** mean ms/question, using instrumented elapsed time on one CUDA stream.
Parent/child intervals are partitioned so each column adds to its total. Host-induced GPU
idle time is included; these are not pure kernel execution times. Profiling itself adds overhead.

CODI latent = its six reasoning passes and projector; visible = answer cue, answer tokens,
and output handling. Overall also includes shared prompt loading, tokenization, transfer,
and prefill. Downloads and head fitting are one-time setup costs, outside the per-question total.
''')
code(r'''
# The single main result table: exactly four numeric columns, means only.
report=runtime.bottleneck_means(profile_samples)
bottleneck=pd.DataFrame([values for _,values in report],index=[name for name,_ in report],columns=runtime.COLUMNS)
bottleneck.index.name='Mean ms per question'
assert all(abs(bottleneck.iloc[1:-1][c].sum()-bottleneck.iloc[0][c])<0.05 for c in runtime.COLUMNS)
with pd.option_context('display.max_rows',None,'display.max_columns',4,'display.width',160,'display.float_format',lambda x:f'{x:.3f}'):
    display(bottleneck)
# The same table is saved once; individual cases stay in one optional debug file.
bottleneck.to_csv(RUN_DIR/'bottleneck.csv')
for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:
    means={arm:sum(r['ms'] for r in clean_samples if r['mode']==mode and r['arm']==arm)/
        sum(r['mode']==mode and r['arm']==arm for r in clean_samples) for arm in ('dense','rank96')}
    lengths={arm:sum(r['visible_tokens'] for r in clean_samples if r['mode']==mode and r['arm']==arm)/
        sum(r['mode']==mode and r['arm']==arm for r in clean_samples) for arm in ('dense','rank96')}
    print(f"{label}, without profiler: dense {means['dense']:.2f} → rank 96 {means['rank96']:.2f} ms/question; "
          f"mean visible tokens {lengths['dense']:.1f} → {lengths['rank96']:.1f}.")
    capped=sum(r['tokens'][-1]!=tokenizer.eos_token_id for r in clean_samples if r['mode']==mode and r['arm']=='rank96')
    if capped: print(f'{label}: {capped} measured generations reached the token cap; interpret timings with that in mind.')
print('Dense and rank-96 totals use their own generated sequences; differing lengths affect latency.')
# A concise setup total; individual setup operations are available in debug.jsonl.gz.
setup_ms=0
with gzip.open(DEBUG_PATH,'rt') as stream:
    for line in stream:
        record=json.loads(line)
        if record['kind']=='setup': setup_ms+=record['data']['cpu_wall_ms']
print(f'One-time setup and fitting: {setup_ms/60000:.1f} minutes. Raw values: {DEBUG_PATH}')
''')
md('''Optional inspection stays inside this notebook. Run this only when debugging a question:
```python
# All named submodules, token positions and individual CPU/CUDA event values for one question.
with gzip.open(DEBUG_PATH, 'rt') as stream:
    events = next(r['data'] for line in stream if (r := json.loads(line))['kind'] == 'events'
                  and r['data'][0]['mode'] == 'codi' and r['data'][0]['question_id'] == 0)
pd.DataFrame(events)  # filter name, phase or token_position here
```
The four-column table is also available as `bottleneck`; per-question rows as
`profile_samples`; clean baseline measurements as `clean_samples`. No CSV downloads are required.
''')
for index,cell in enumerate(cells): cell['id']=f'bottleneck-{index:03d}'
notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
    kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'),
    accelerator='GPU',kaggle=dict(accelerator='gpu',isInternetEnabled=True,dataSources=[])))
OUTPUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(OUTPUT)
