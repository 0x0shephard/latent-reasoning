"""Build a standalone optimization notebook without changing the earlier notebook."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb'
OUTPUT=ROOT/'notebooks/kaggle_codi_transformer_optimization.ipynb'
RUNTIME=ROOT/'src/inference/transformer_optimization.py'
old=json.loads(BASE.read_text(encoding='utf-8'))
old_code=[''.join(c['source']) for c in old['cells'] if c['cell_type']=='code']
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.strip().splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.strip().splitlines(True),execution_count=None,outputs=[]))
source=RUNTIME.read_text(encoding='utf-8')
fingerprint=hashlib.sha256((source+Path(__file__).read_text(encoding='utf-8')+''.join(old_code[:5])).encode()).hexdigest()
md('''# Speed up the transformer: CODI and explicit GPT-2 on T4

**Select GPU T4 x2, enable Internet, and Run All.** One T4 is used. No data attachments
or prior results are needed. This separate notebook keeps the released model and
rank-96 head recipe from the previous experiment.

It tries static KV storage, contiguous projection weights, SDPA attention, compiler
fusion, CUDA Graph replay and preallocated token/mask buffers. Explicit decoding also
tries four-token chunks to reduce host EOS checks. Separate INT8/INT4 Triton kernels
are accuracy-gated candidates. A faster result is measured, not assumed.

**Output order:** equal-MAC experiment, isolated transformer-block substeps, candidate
results, then model timing/accuracy. Only the isolated block probe installs module
hooks, and removes them before returning. Main timing and accuracy runs have no hooks.

Before/after means **current versus selected optimized decoder, with the same rank-96
head**. Four model breakdown tables show per-question and per-token/latent-step means.
Dense-head controls additionally show whether transformer optimization makes head
compression more worthwhile. Full GSM8K accuracy runs last and can take longer than fitting.
''')
# Reuse the established bootstrap, official checkpoint, data, parity and head recipe.
for i,s in enumerate(old_code[:5]):
    s=s.replace('/kaggle/working/codi_bottleneck','/kaggle/working/codi_transformer_optimization')
    if i==1:
        lines=s.splitlines()
        lines=[('EXPERIMENT_SOURCE_SHA256 = '+repr(fingerprint)) if line.startswith('EXPERIMENT_SOURCE_SHA256 =') else line for line in lines]
        s='\n'.join(lines)
        s+='\nfrom src.data.answer_extract import answers_match,normalize_gold\n'
    if i==2 and 'test_rows=' not in s:
        # Older revisions of the head notebook did not evaluate accuracy. Keep
        # regeneration usable even if only this new notebook's files are committed.
        s+='''
with setup.span('gsm8k_test_download_and_parse',gpu=False):
    with urlopen(url.replace('train.jsonl','test.jsonl'),timeout=300) as response:
        test=[json.loads(line) for line in response if line.strip()]
    test_rows=[dict(question=str(r['question']),gold=str(normalize_gold(r['answer'],'gsm8k_main'))) for r in test]
    assert len(test_rows)==1319 and all(r['gold']!='None' for r in test_rows)
    assert not {' '.join(r['question'].casefold().split()) for r in train} & {' '.join(r['question'].casefold().split()) for r in test_rows}
    debug_record('accuracy_questions',test_rows)
flush_setup()
'''
    code(s)
md('Load the optimization engine and the same fitted heads. Compiler/graph setup is excluded from clean timing.')
code('OPTIMIZER_SOURCE = '+repr(source)+r'''
optimizer_path=pathlib.Path('/kaggle/working/transformer_optimization_runtime.py')
optimizer_path.write_text(OPTIMIZER_SOURCE)
spec=importlib.util.spec_from_file_location('transformer_optimization_runtime',optimizer_path)
opt=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=opt
spec.loader.exec_module(opt)
heads={}
for mode in MODES:
    payload=torch.load(RUN_DIR/f'global_head_{mode}.pt',map_location='cpu',weights_only=False)
    nested=NestedLowRankVocabularyHead(int(model.config.hidden_size),int(model.eot_id),RANKS)
    nested.load_state_dict(payload['state_dict'])
    heads[mode]=runtime.FixedRankHead(nested,96).to(device=device,dtype=DEPLOY_DTYPE).eval()
    debug_record('fit_report',payload['report'])
del payload,nested
dense=runtime.DenseSelector(full_head,int(model.eot_id))
VALIDATION_QUESTIONS=64
DIAGNOSTIC_QUESTIONS=2
validation=splits['selection'][:VALIDATION_QUESTIONS]
warmup=[r['question'] for r in splits['warmup']]
print('Core candidates: static KV, compiled fusion, CUDA Graphs, chunked explicit decoding, packed INT8/INT4.')
print('FlashAttention-2 and original Marlin require newer GPUs; this uses PyTorch SDPA and T4-compatible custom GEMV kernels.')
''')
md('''The equal-MAC experiment changes matrix shape and invocation count, while retaining
identical weights and input. It tests whether fragmentation matters; it does not isolate
launch overhead from occupancy/bandwidth. CUDA Graph controls help distinguish them.
''')
code(r'''
micro=opt.operation_microbenchmark(device,DEPLOY_DTYPE)
debug_record('equal_mac_samples',micro)
print('Equal arithmetic: one 768x9216 projection versus twelve 768x768 projections.')
display(pd.DataFrame(micro).groupby('name',sort=False)['microseconds'].mean().to_frame('Mean microseconds'))
''')
md('''**Isolated block probe:** one cached position in block 1, on a real question prefix.
These diagnostic component times include probe overhead. Attention groups SDPA,
softmax, cache updates and reshapes; a fused kernel cannot be meaningfully split into
separate QK/softmax/AV wall times. These hooks never enter full-model benchmark runs.
''')
code(r'''
substeps,substep_events=opt.block_substeps(model,tokenizer,warmup[0])
debug_record('block_substeps',substep_events)
print('Original transformer block 1 - isolated diagnostic, microseconds per call:')
block_table=pd.DataFrame(substeps).groupby('name',sort=False)['microseconds'].mean().to_frame('Mean microseconds')
block_table.loc['Total block (includes probe overhead)']=block_table.sum()
display(block_table)
assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),'Probe hooks leaked'
''')
md('''Choose the fastest passing candidate per reasoning mode using training-split
validation and separate warmup questions. Test answers and timing questions never
select a candidate. FP16 candidates must pass state checks and at least 95% exact
sequence agreement; all candidates must match or exceed baseline validation accuracy.
Quantization changes weights, so its accuracy is measured explicitly. Failures are
shown, with the full reason retained in the single debug log.
''')
code(r'''
def baseline(mode,arm,question,diagnostic=False):
    head=dense if arm=='dense' else heads[mode]
    return opt.baseline_generate(model,tokenizer,head,question,mode,MAX_NEW_TOKENS[mode],diagnostic)

def correctness(result,row):
    gold=normalize_gold(row['gold'],'gsm8k_main')
    return answers_match(result['text'],gold)

selected={}; candidate_rows=[]; selection_baselines={}
for mode in MODES:
    expected=[baseline(mode,'rank96',r['question']) for r in validation]
    baseline_correct=sum(correctness(result,row) for result,row in zip(expected,validation))
    for question in warmup: baseline(mode,'rank96',question)
    baseline_ms=sum(opt.timed(lambda q=q:baseline(mode,'rank96',q),device)['wall_ms'] for _ in range(2) for q in warmup)/(2*len(warmup))
    selection_baselines[mode]=dict(correct=baseline_correct,ms=baseline_ms)
    best=None; best_dense=None; best_ms=baseline_ms; best_config=None
    configurations=[('static_cache',dict()),('compiled',dict(compile_steps=True)),
                    ('cuda_graphs',dict(compile_steps=True,cuda_graphs=True))]
    if mode=='explicit_cot': configurations.append(('cuda_graphs_chunk4',dict(compile_steps=True,cuda_graphs=True,chunk_size=4)))
    for bits in (8,4):
        configurations.append((f'packed_int{bits}',dict(compile_steps=True,cuda_graphs=True,bits=bits,chunk_size=4 if mode=='explicit_cot' else 1)))
    candidate_rows.append(dict(mode=mode,candidate='current',status='reference',validation_correct=baseline_correct,
                               validation_total=len(validation),mean_ms=baseline_ms,sequence_agreement=1.))
    for name,settings in configurations:
        engine=None; plain=None; dense_trial=None; plain_dense=None
        print(f'Preparing {mode}/{name} ...',flush=True)
        try:
            if settings.get('bits',16)<16 and opt.triton is None: raise RuntimeError('Triton unavailable for packed GEMV')
            started=time.perf_counter()
            engine=opt.OptimizedDecoder(model,tokenizer,heads[mode],**settings)
            if settings.get('bits',16)<16: debug_record('packed_kernel_check',dict(mode=mode,candidate=name,checks=opt.kernel_gate(engine.core)))
            engine.prepare()
            if settings.get('bits',16)==16:
                debug_record('state_parity',dict(mode=mode,candidate=name,**opt.numerical_gate(model,tokenizer,engine,warmup[:2])))
            # Every candidate is compared against its uncaptured implementation as
            # an additional graph/cache check, including lossy candidates.
            plain=opt.OptimizedDecoder(model,tokenizer,heads[mode],bits=settings.get('bits',16),chunk_size=settings.get('chunk_size',1))
            for question in warmup[:2]:
                observed=engine.generate(question,mode,MAX_NEW_TOKENS[mode])
                control=plain.generate(question,mode,MAX_NEW_TOKENS[mode])
                if observed['tokens']!=control['tokens']: raise RuntimeError('Compiled/graph output differs from the same eager candidate')
            del plain
            results=[engine.generate(row['question'],mode,MAX_NEW_TOKENS[mode]) for row in validation]
            correct=sum(correctness(result,row) for result,row in zip(results,validation))
            agreement=sum(a['tokens']==b['tokens'] for a,b in zip(results,expected))/len(expected)
            for question in warmup: engine.generate(question,mode,MAX_NEW_TOKENS[mode])
            values=[opt.timed(lambda q=q:engine.generate(q,mode,MAX_NEW_TOKENS[mode]),device)['wall_ms'] for _ in range(2) for q in warmup]
            elapsed=sum(values)/len(values)
            passes=correct>=baseline_correct and (settings.get('bits',16)<16 or agreement>=.95)
            status='passed' if passes else 'rejected: accuracy/agreement'
            # Confirm the same transformer configuration also runs with the dense
            # control BEFORE selecting it; do not silently mix execution backends.
            if passes and elapsed<best_ms and elapsed<.98*baseline_ms:
                dense_trial=opt.OptimizedDecoder(model,tokenizer,dense,**settings)
                dense_trial.prepare()
                plain_dense=opt.OptimizedDecoder(model,tokenizer,dense,bits=settings.get('bits',16),chunk_size=settings.get('chunk_size',1))
                for question in warmup[:2]:
                    if dense_trial.generate(question,mode,MAX_NEW_TOKENS[mode])['tokens']!=plain_dense.generate(question,mode,MAX_NEW_TOKENS[mode])['tokens']:
                        raise RuntimeError('Dense control differs from its eager candidate')
                best=engine; best_dense=dense_trial; best_ms=elapsed; best_config=settings; best_name=name
            candidate_rows.append(dict(mode=mode,candidate=name,status=status,validation_correct=correct,
                validation_total=len(validation),mean_ms=elapsed,sequence_agreement=agreement))
            debug_record('candidate',dict(mode=mode,name=name,settings=settings,status=status,setup_seconds=time.perf_counter()-started,
                timing_samples_ms=values,validation=[dict(correct=correctness(a,r),tokens=a['tokens']) for a,r in zip(results,validation)]))
            if engine is not best: del engine
        except Exception as error:
            if 'illegal memory access' in str(error).lower() or 'device-side assert' in str(error).lower(): raise
            message=f'{type(error).__name__}: {error}'
            candidate_rows.append(dict(mode=mode,candidate=name,status='unavailable / failed check',validation_correct=None,
                                       validation_total=len(validation),mean_ms=None,sequence_agreement=None))
            debug_record('candidate_failure',dict(mode=mode,candidate=name,error=message))
            print(f'{name}: {message[:240]}',flush=True)
            engine=None
        plain=None; plain_dense=None; dense_trial=None
        gc.collect(); torch.cuda.empty_cache()
    selected[mode]=dict(engine=best,dense_engine=best_dense,settings=best_config,name=best_name if best is not None else 'current (no qualifying speedup)')
    print(f"Selected {mode}: {selected[mode]['name']}",flush=True)
print('Candidate selection (validation data only):')
candidate_table=pd.DataFrame(candidate_rows)
display(candidate_table[['mode','candidate','status','mean_ms','validation_correct']].rename(columns={'mean_ms':'Mean ms/question','validation_correct':f'Correct / {len(validation)}'}))
debug_record('selection',dict(candidates=candidate_rows,selected={m:dict(name=s['name'],settings=s['settings']) for m,s in selected.items()}))
''')
md('Prepare the dense-head control with the selected transformer settings, then warm all paths. No module hooks are installed.')
code(r'''
dense_engines={mode:selected[mode]['dense_engine'] for mode in MODES}

def run(version,mode,arm,question,diagnostic=False):
    engine=(selected[mode]['engine'] if arm=='rank96' else dense_engines[mode]) if version=='optimized' else None
    if engine is None: return baseline(mode,arm,question,diagnostic)
    return engine.generate(question,mode,MAX_NEW_TOKENS[mode],diagnostic)

for version in ('current','optimized'):
    for mode in MODES:
        for arm in ('dense','rank96'):
            for question in warmup: run(version,mode,arm,question)
assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
# One optional CUPTI trace, outside every timing/accuracy measurement.
try:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as probe:
        run('optimized','explicit_cot','rank96',warmup[0])
    probe.export_chrome_trace(str(RUN_DIR/'optimized_gpu_trace.json'))
    print('Optional GPU kernel timeline saved: optimized_gpu_trace.json')
except Exception as error:
    debug_record('gpu_trace_unavailable',str(error))
    print('CUPTI trace unavailable; timing/accuracy remain enabled:',str(error)[:160])
''')
md('''**Main benchmark: no hooks, no per-layer events, no diagnostic graphs.**
Compilation, warmup and traces finish before timing. Both heads use their own outputs,
so generated lengths and accuracy are reported with latency. Each result is retained
individually; the displayed numbers are means.
''')
code(r'''
clean_samples=[]
rng=random.Random(SEED)
for repeat in range(TIMING_REPEATS):
    jobs=[(version,mode,arm,i) for version in ('current','optimized') for mode in MODES
          for arm in ('dense','rank96') for i in range(len(splits['timing']))]
    rng.shuffle(jobs)
    for version,mode,arm,i in jobs:
        question=splits['timing'][i]['question']
        sample=opt.timed(lambda:run(version,mode,arm,question),device)
        sample.update(version=version,arm=arm,question_id=i,repeat=repeat)
        clean_samples.append(sample)
        debug_record('clean_sample',sample)
    print(f'Clean timing {repeat+1}/{TIMING_REPEATS} complete.',flush=True)

def clean_mean(version,mode,arm,key='wall_ms'):
    group=[s[key] for s in clean_samples if s['version']==version and s['mode']==mode and s['arm']==arm]
    return sum(group)/len(group)
for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:
    for arm in ('dense','rank96'):
        before,after=(clean_mean(v,mode,arm) for v in ('current','optimized'))
        counts=[clean_mean(v,mode,arm,'visible_tokens') for v in ('current','optimized')]
        print(f'{label}, {arm}: {before:.2f} -> {after:.2f} ms/question ({before/after:.2f}x); visible tokens {counts[0]:.2f} -> {counts[1]:.2f}.')
    saved=[100*(1-clean_mean(v,mode,'rank96')/clean_mean(v,mode,'dense')) for v in ('current','optimized')]
    print(f'  Latency saved by head compression: {saved[0]:.1f}% before transformer changes; {saved[1]:.1f}% after (free-generation comparison).')
''')
md('''Four **coarse diagnostic** tables, using just two questions from the timing set,
separate from the clean means above. There are still no module hooks. Whole compiled
transformer calls are measured together; instrumenting their internal blocks would
break the optimization being measured. The earlier isolated table shows block substeps.

For CUDA Graphs, a separate diagnostic graph records stage events; the main graph stays
uninstrumented. If this PyTorch build cannot capture timing events, the notebook explicitly
labels the diagnostic fallback. Synchronization/measurement overhead is included in these
tables and must not be interpreted as production latency.

CODI overall = shared + latent + visible. The per-token denominators are visible output
tokens for Explicit/visible, six steps for latent, and latent + visible for CODI overall.
''')
code(r'''
diagnostic_samples=[]
for mode in MODES:
    engine=selected[mode]['engine']
    if engine is not None:
        engine.prepare_diagnostics()
        if engine.diagnostic_graph_error:
            print(f'{mode}: diagnostic graph events unavailable; using compiled calls without graph replay for diagnostics only.')
            debug_record('diagnostic_graph_fallback',dict(mode=mode,error=engine.diagnostic_graph_error))
for version in ('current','optimized'):
    for mode in MODES:
        for i,row in enumerate(splits['timing'][:DIAGNOSTIC_QUESTIONS]):
            sample=opt.timed(lambda:run(version,mode,'rank96',row['question'],diagnostic=True),device)
            sample.update(version=version,arm='rank96',question_id=i)
            # Clocks must not change the selected decoder's outputs.
            expected=next(s for s in clean_samples if s['version']==version and s['mode']==mode and s['arm']=='rank96' and s['question_id']==i)
            assert sample['tokens']==expected['tokens'],'Diagnostic path changed generated tokens'
            diagnostic_samples.append(sample)
            debug_record('coarse_diagnostic',sample)
tables={}
for per_token in (False,True):
    for version,label in [('current','Before transformer optimization'),('optimized','After transformer optimization')]:
        samples=[s for s in diagnostic_samples if s['version']==version]
        report=opt.coarse_table(samples,per_token)
        frame=pd.DataFrame([r for _,r in report],index=[name for name,_ in report],columns=runtime.COLUMNS)
        key=version+('_per_token' if per_token else '_per_question')
        tables[key]=frame
        print(label+' - '+('ms/token or latent step' if per_token else 'ms/question')+' (coarse diagnostic, rank96 head)')
        if not per_token:
            t=frame.iloc[0]
            shared=t['CODI overall']-t['CODI latent only']-t['CODI visible only']
            print(f"CODI: {t['CODI overall']:.3f} = {shared:.3f} shared + {t['CODI latent only']:.3f} latent + {t['CODI visible only']:.3f} visible.")
        with pd.option_context('display.float_format',lambda x:f'{x:.3f}','display.max_columns',4,'display.max_rows',None):
            display(frame)
save_json(RUN_DIR/'timing_tables.json',{key:frame.to_dict('split') for key,frame in tables.items()})
for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:
    for version in ('current','optimized'):
        group=[s for s in diagnostic_samples if s['version']==version and s['mode']==mode]
        share=100*sum(p['ms'] for s in group for p in s['parts'] if p['name']=='LM head + argmax')/sum(s['wall_ms'] for s in group)
        print(f'{label}, {version}: head + argmax {share:.1f}% of diagnostic time. Use clean dense/rank96 controls above for actual latency impact.')
''')
md('''Full GSM8K test accuracy for current/optimized transformer execution and both heads.
All 1,319 questions are evaluated once per combination, without hooks or clocks inside
the decoder. Quantized selections are clearly named above; test accuracy is never used
to choose the winner. This is the longest evaluation step.
''')
code(r'''
accuracy_results={}
for mode in MODES:
    for version in ('current','optimized'):
        for arm in ('dense','rank96'):
            correct=0; capped=0
            for i,row in enumerate(test_rows):
                result=run(version,mode,arm,row['question'])
                match=answers_match(result['text'],row['gold'])
                hit_cap=result['tokens'][-1]!=tokenizer.eos_token_id
                correct+=int(match); capped+=int(hit_cap)
                debug_record('accuracy_sample',dict(version=version,mode=mode,arm=arm,question_id=i,
                    text=result['text'],tokens=result['tokens'],gold=row['gold'],correct=match,hit_cap=hit_cap))
                if (i+1)%256==0: print(f'Accuracy {mode}/{version}/{arm}: {i+1}/{len(test_rows)}',flush=True)
            result=dict(correct=correct,total=len(test_rows),accuracy=correct/len(test_rows),capped=capped)
            accuracy_results[mode,version,arm]=result
            debug_record('accuracy_result',dict(mode=mode,version=version,arm=arm,**result))
for mode,label in [('explicit_cot','Explicit'),('codi','CODI')]:
    for arm in ('dense','rank96'):
        before,after=(accuracy_results[mode,v,arm] for v in ('current','optimized'))
        print(f"{label}, {arm}: {before['accuracy']:.2%} ({before['correct']}/{before['total']}) -> "
              f"{after['accuracy']:.2%} ({after['correct']}/{after['total']}); token-cap hits {before['capped']} -> {after['capped']}.")
print('Done. Main timings: clean_samples. Four diagnostic tables: tables. Accuracy: accuracy_results.')
print('All individual measurements, candidate failures and predictions:',DEBUG_PATH)
''')
md('''Optional debugging stays in the notebook: `clean_samples`, `diagnostic_samples`,
`substep_events`, `candidate_rows`, `accuracy_results`. The single compressed debug log
retains raw values. `optimized_gpu_trace.json`, when available, opens in Perfetto.

Optimization research behind the experiments:
[PyTorch GPT-fast](https://pytorch.org/blog/accelerating-generative-ai-2/),
[CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
[static caches](https://huggingface.co/docs/transformers/kv_cache).
The equal-MAC test is not proof of a fixed hardware tax; performance and quality can
improve, remain unchanged, or regress. If no faster candidate passes validation, the
notebook says so and retains the current decoder.
''')
for i,cell in enumerate(cells): cell['id']=f'transformer-opt-{i:03d}'
notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
    kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python'),
    accelerator='GPU',kaggle=dict(accelerator='gpu',isInternetEnabled=True,dataSources=[])))
OUTPUT.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(OUTPUT)
