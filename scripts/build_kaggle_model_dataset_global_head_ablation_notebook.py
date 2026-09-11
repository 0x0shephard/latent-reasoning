"""Build the separate 3-model x 3-dataset notebook from the fixed head protocol."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb'
OUTPUT = ROOT/'notebooks/kaggle_model_dataset_global_head_ablations.ipynb'
ADAPTER = ROOT/'src/inference/model_dataset_ablation.py'
old = json.loads(BASE.read_text(encoding='utf-8'))
old_code = [''.join(c['source']) for c in old['cells'] if c['cell_type']=='code']
adapter_source = ADAPTER.read_text(encoding='utf-8')
fingerprint = hashlib.sha256((adapter_source+Path(__file__).read_text(encoding='utf-8')+''.join(old_code)).encode()).hexdigest()
cells = []
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.strip().splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.strip().splitlines(True),execution_count=None,outputs=[]))

md('''# Global LM head: three models, three math datasets

**T4 GPU, Internet on, Run All.** One T4 is used; no attachments are needed.

Models: GPT-2 (released CODI weights), SmolLM2-135M, Qwen2.5-0.5B.
Evaluation: GSM8K, SVAMP, ASDiv's single-number-answer subset.

Only the model and evaluation dataset vary. Everything else keeps the earlier head
experiment: **eager PyTorch, FP16, batch 1, dense versus rank 96**, seed 89, the same
head initialization/distillation/recovery, 16 timing questions x 3 repeats, token caps
64 CODI / 256 explicit. There is no transformer-optimization or rank/backend sweep.

All heads are fitted on the same GSM8K training split, then reused across datasets.
Each model/dataset pair has four mean timing tables, clean latency and full eligible-set
accuracy. Expand its result panel to see the tables; no CSV downloads are needed.

The released CODI checkpoint supports GPT-2. The other two models run explicit CoT
from their published base checkpoints; **their CODI columns are N/A**. No untrained
latent loop is presented as CODI. Their absolute math accuracy is not a controlled
comparison with the math-trained GPT-2 checkpoint; compare each model's dense and
compressed head. Training new CODI backbones is not part of this notebook.
''')
bootstrap = old_code[0].replace('/kaggle/working/codi_bottleneck','/kaggle/working/codi_model_dataset_ablations')
code(bootstrap)
environment = old_code[1]
environment = '\n'.join(('EXPERIMENT_SOURCE_SHA256 = '+repr(fingerprint)) if line.startswith('EXPERIMENT_SOURCE_SHA256 =') else line for line in environment.splitlines())
environment = environment.replace("accuracy='full_gsm8k_test'", "accuracy='full_eligible_set_for_each_dataset', models=['gpt2','smollm2_135m','qwen2_5_0_5b'], datasets=['gsm8k','svamp','asdiv']")
environment = environment.replace("print(f\"Using {config['gpu']}; rank 96, FP16, batch 1. Fitting is the slow setup step.\")", "print(f\"Using {config['gpu']}; 3 models x 3 datasets; rank 96, FP16, batch 1.\")")
environment = environment.replace("with gzip.open(DEBUG_PATH,'wt') as stream: pass", "if not DEBUG_PATH.exists():\n    with gzip.open(DEBUG_PATH,'wt') as stream: pass")
environment = environment.replace('# Each Run All starts a fresh measurement log; fitted heads in the same run folder are reused.', '# Preserve raw measurements/predictions when completed model/dataset pairs are reused.')
code(environment)
code('ABLATION_SOURCE = '+repr(adapter_source)+r'''
adapter_path=pathlib.Path('/kaggle/working/model_dataset_ablation_runtime.py')
adapter_path.write_text(ABLATION_SOURCE)
spec=importlib.util.spec_from_file_location('model_dataset_ablation_runtime',adapter_path)
ablation=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=ablation
spec.loader.exec_module(ablation)
from types import SimpleNamespace
from functools import partial
from IPython.display import HTML,display
import html
from transformers import AutoModelForCausalLM,AutoTokenizer
REFERENCE_RUNTIME=runtime
EXPERIMENT_DIR=RUN_DIR
MODEL_SPECS=ablation.MODEL_SPECS
all_results={}

train,evaluation,data_manifest=ablation.download_datasets(EXPERIMENT_DIR/'datasets')
base_splits=ablation.partition_train(train,evaluation,SEED)
debug_record('datasets',data_manifest)
debug_record('partitions',base_splits)
for name,rows in evaluation.items(): debug_record('evaluation_questions',dict(dataset=name,rows=rows))
print('Accuracy sets: GSM8K 1,319; SVAMP 1,000; ASDiv numeric subset 2,084.')
print('ASDiv excludes 221 non-scalar answers (times, ratios, names, multiple answers); exclusions are logged.')
print('SVAMP/ASDiv are evaluation-only: no head refitting or selection on them.')
''')

# Keep the original loader, parity gate, fitting recipe and timing loops verbatim
# except for architecture/mode dispatch. No edits to the earlier notebook.
load_gpt2 = old_code[2][old_code[2].index("with setup.span('config_load'"):]
parity = old_code[3].replace("print('Decoder checks passed: CODI and explicit GPT-2.')", "print('Decoder checks passed:',MODEL_KEY,MODES)")
fit = old_code[4].replace("with setup.span('lora_merge'): merge_official_codi_lora_(model)", "if MODEL_KEY=='gpt2':\n    with setup.span('lora_merge'): merge_official_codi_lora_(model)")
timing = old_code[5]
accuracy = old_code[7]
accuracy = accuracy.replace("print(f'GSM8K test accuracy - {len(test_rows)} questions, numeric exact match:')", "print(f'{DATASET}: {len(test_rows)} questions, numeric exact match:')")
accuracy = accuracy.replace("for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:", "for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:\n    if mode not in MODES: continue")
accuracy = accuracy[:accuracy.index("print(f'Four timing tables")]
code('GPT2_LOAD = '+repr(load_gpt2)+'\nPARITY_CHECK = '+repr(parity)+'\nFIT_HEADS = '+repr(fit)+'\nTIMING_RUN = '+repr(timing)+'\nACCURACY_RUN = '+repr(accuracy))

md('''Each backbone is loaded and fitted once, then evaluated on all three datasets.
Fitted heads and completed pairs are cached under the experiment fingerprint, so
rerunning in the same Kaggle working directory reuses completed work. Downloaded base
weights, cached states and head fitting are outside inference latency measurements.
''')
code(r'''
def pair_reports(scope):
    tables={}
    for per_token in (False,True):
        for arm in ('dense','rank96'):
            samples=[s for s in scope['profile_samples'] if s['arm']==arm]
            report=scope['runtime'].bottleneck_means(samples,per_token=per_token)
            table=pd.DataFrame([values for _,values in report],index=[name for name,_ in report],columns=runtime.COLUMNS)
            table.index.name='Mean ms / token or latent step' if per_token else 'Mean ms / question'
            for col in table:
                if table[col].notna().any():
                    assert abs(table.iloc[1:-1][col].sum()-table.iloc[0][col])<.05
            tables[arm+('_per_token' if per_token else '_per_question')]=table
    summaries=[]
    for mode in scope['MODES']:
        for arm in ('dense','rank96'):
            clean=[r for r in scope['clean_samples'] if r['mode']==mode and r['arm']==arm]
            profiled=[r for r in scope['profile_samples'] if r['mode']==mode and r['arm']==arm]
            mean=sum(r['ms'] for r in clean)/len(clean)
            summaries.append(dict(mode=mode,arm=arm,mean_ms=mean,
                mean_visible_tokens=sum(r['visible_tokens'] for r in clean)/len(clean),
                profiler_inflation=(sum(r['total_ms'] for r in profiled)/len(profiled))/mean,
                **scope['accuracy_results'][mode,arm]))
    return tables,summaries

def show_pair(key, result, opened=False):
    title=f"{key[0]} / {key[1]} - four timing tables"
    pieces=[f"<details {'open' if opened else ''}><summary><b>{html.escape(title)}</b></summary>",
        '<p>Profiled elapsed times include instrumentation; use the clean means for speed claims.</p>']
    if key[0]!='gpt2': pieces.append('<p>CODI: N/A — no trained CODI checkpoint for this backbone.</p>')
    for name,data in result['tables'].items():
        table=pd.DataFrame(data['data'],index=data['index'],columns=data['columns'],dtype=float)
        per_token=name.endswith('_per_token')
        label=('Before: normal dense head' if name.startswith('dense') else 'After: rank-96 global head')
        pieces.append(f'<p><b>{label} — '+('ms/token or latent step' if per_token else 'ms/question')+'</b></p>')
        if not per_token and table['CODI overall'].notna().any():
            totals=table.iloc[0]
            shared=totals['CODI overall']-totals['CODI latent only']-totals['CODI visible only']
            pieces.append(f"<p>CODI {totals['CODI overall']:.3f} = {shared:.3f} shared + {totals['CODI latent only']:.3f} latent + {totals['CODI visible only']:.3f} visible.</p>")
        pieces.append(table.to_html(float_format=lambda value:f'{value:.3f}',na_rep='N/A',border=0))
    pieces.append('</details>')
    display(HTML('\n'.join(pieces)))

def safe_tables(tables):
    # Strict JSON: absent CODI cells are null, never fabricated zero measurements.
    return {key:json.loads(frame.to_json(orient='split')) for key,frame in tables.items()}
''')
code(r'''
for model_spec in MODEL_SPECS:
    model_key=model_spec['key']
    model_dir=EXPERIMENT_DIR/model_key
    model_dir.mkdir(exist_ok=True)
    print('\n'+model_spec['label'],flush=True)
    completed={name:model_dir/name/'results.json' for name in ablation.DATASETS}
    if all(path.exists() for path in completed.values()):
        for name,path in completed.items(): all_results[model_key,name]=json.loads(path.read_text())
        print('Reusing all three completed dataset results.')
        continue
    scope=globals().copy()
    scope.update(MODEL_KEY=model_key,MODEL_SPEC=model_spec,RUN_DIR=model_dir,splits=dict(base_splits),
        MODES=['codi','explicit_cot'] if model_key=='gpt2' else ['explicit_cot'])
    scope['runtime']=SimpleNamespace(**{k:v for k,v in vars(REFERENCE_RUNTIME).items() if not k.startswith('__')})
    scope['runtime'].decode=ablation.decode
    scope['runtime'].profile_question=ablation.profile_question
    # Tag every raw event without changing the original timing/fitting loops.
    def model_debug(kind,payload,_key=model_key):
        debug_record(kind,dict(model=_key,payload=payload))
    scope['debug_record']=model_debug
    torch.manual_seed(SEED); random.seed(SEED)
    if model_key=='gpt2':
        exec(GPT2_LOAD,scope)
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_spec['repo'],revision=model_spec['revision'])
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        tokenizer.padding_side='left'
        backbone=AutoModelForCausalLM.from_pretrained(model_spec['repo'],revision=model_spec['revision'],
            torch_dtype=torch.float32,attn_implementation='sdpa')
        model=ablation.ExplicitBackbone(backbone).requires_grad_(False).to(device).eval()
        base=official_codi_base_model(model); full_head=base.get_output_embeddings()
        scope.update(model=model,tokenizer=tokenizer,base=base,full_head=full_head,
            weight=full_head.weight.detach(),bias=None if getattr(full_head,'bias',None) is None else full_head.bias.detach())
        debug_record('backbone',dict(model=model_key,repo=model_spec['repo'],revision=model_spec['revision'],
            parameters=sum(p.numel() for p in model.parameters()),hidden_size=model.config.hidden_size,
            vocabulary_size=model.eot_id,codi_available=False,math_finetuned=False))
        del model,tokenizer,base,full_head,backbone
    n_layers=ablation.block_roots(scope['model'])[1]
    scope['runtime'].bottleneck_means=partial(ablation.bottleneck_means,n_layers=n_layers)
    exec(PARITY_CHECK,scope)
    exec(FIT_HEADS,scope)
    for dataset in ablation.DATASETS:
        pair_dir=model_dir/dataset
        pair_dir.mkdir(exist_ok=True)
        if completed[dataset].exists():
            all_results[model_key,dataset]=json.loads(completed[dataset].read_text())
            print('Reuse completed pair:',model_key,dataset)
            continue
        print(f'\n{model_spec["label"]} / {dataset}',flush=True)
        scope['DATASET']=dataset
        scope['splits']=dict(base_splits,timing=ablation.timing_questions(dataset,base_splits,evaluation,SEED))
        scope['test_rows']=evaluation[dataset]
        def pair_debug(kind,payload,_model=model_key,_dataset=dataset):
            debug_record(kind,dict(model=_model,dataset=_dataset,payload=payload))
        scope['debug_record']=pair_debug
        exec(TIMING_RUN,scope)
        exec(ACCURACY_RUN,scope)
        tables,summaries=pair_reports(scope)
        result=dict(model=model_key,dataset=dataset,codi_available='codi' in scope['MODES'],
            summaries=summaries,tables=safe_tables(tables),clean_samples=scope['clean_samples'],
            profile_samples=scope['profile_samples'])
        save_json(completed[dataset],result)
        all_results[model_key,dataset]=result
        for mode in scope['MODES']:
            before=next(r for r in summaries if r['mode']==mode and r['arm']=='dense')
            after=next(r for r in summaries if r['mode']==mode and r['arm']=='rank96')
            print(f"{mode}, WITHOUT profiler: {before['mean_ms']:.2f} -> {after['mean_ms']:.2f} ms/question; "
                  f"visible tokens {before['mean_visible_tokens']:.1f} -> {after['mean_visible_tokens']:.1f}; "
                  f"profiler inflation {before['profiler_inflation']:.2f}x -> {after['profiler_inflation']:.2f}x.")
        del tables,summaries,result
    del scope
    gc.collect(); torch.cuda.empty_cache()
print('Finished all nine model/dataset pairs.')
''')
md('''Results below retain four columns and four tables per model/dataset pair. Expand a
panel to inspect it. Total rows repeat at the top and bottom; only means are displayed.
CODI overall includes shared prompt preparation/prefill. Per-token denominators are
visible tokens for explicit/visible, six latent steps for latent, and latent + visible
for overall. Columns with different denominators cannot be added.

Generated lengths may change after compression. Clean latency comparisons include
that effect; the accuracy results reveal changes in answer quality. Token counts from
different model tokenizers are different units, so use ms/question for model comparisons.
''')
code(r'''
summary_rows=[]
for spec in MODEL_SPECS:
    for dataset in ablation.DATASETS:
        result=all_results[spec['key'],dataset]
        for mode in ('explicit_cot','codi'):
            rows=[r for r in result['summaries'] if r['mode']==mode]
            if not rows: continue
            before=next(r for r in rows if r['arm']=='dense')
            after=next(r for r in rows if r['arm']=='rank96')
            summary_rows.append(dict(Model=spec['label'],Dataset=dataset,Mode=mode,
                **{'Mean ms: dense -> rank96':f"{before['mean_ms']:.2f} -> {after['mean_ms']:.2f}",
                   'Accuracy: dense -> rank96':f"{before['accuracy']:.2%} -> {after['accuracy']:.2%}"}))
display(pd.DataFrame(summary_rows))
for spec in MODEL_SPECS:
    for dataset in ablation.DATASETS:
        key=spec['key'],dataset
        show_pair(key,all_results[key],opened=key==('gpt2','gsm8k'))
print('All tables and individual timings: all_results[(model_key, dataset)].')
print('Raw events, predictions, setup and exclusions:',DEBUG_PATH)
print('Output folder:',EXPERIMENT_DIR)
''')
md('''Example inspection: `all_results['qwen2_5_0_5b','svamp']['clean_samples']`.
The single `debug.jsonl.gz` retains individual component events and predicted/gold answers.
Each completed pair also has one `results.json` checkpoint for reuse; opening these files
is optional. Kaggle session resets require preserving working outputs to retain caches.

Dataset sources: [GSM8K](https://github.com/openai/grade-school-math),
[SVAMP](https://github.com/arkilpatel/SVAMP),
[ASDiv](https://github.com/chaochun/nlu-asdiv-dataset). ASDiv numeric subset accuracy
is not full-ASDiv accuracy or the ASDiv-A cross-validation benchmark.
''')
for i,cell in enumerate(cells): cell['id']=f'model-dataset-{i:03d}'
OUTPUT.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
    kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'),
    accelerator='GPU',kaggle=dict(accelerator='gpu',isInternetEnabled=True,dataSources=[]))),indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(OUTPUT)
