"""Static GPT-2 decode, compiler fusion, CUDA graphs and isolated diagnostics.

Batch one only. This preserves CODI's six latent steps and explicit teacher prefix.
Main runs install no module hooks. Lossy weight kernels are separate candidates.
"""
from contextlib import contextmanager, nullcontext
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
try:
    import dual_global_head_runtime as reference
except ModuleNotFoundError:
    from src.inference import global_head_comparison as reference
from src.models.official_codi import official_codi_base_model
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _quant_gemv(X, W, SCALE, BIAS, Y, K:tl.constexpr, N:tl.constexpr,
                    BITS:tl.constexpr, BK:tl.constexpr, BN:tl.constexpr):
        rows=tl.program_id(0)*BN+tl.arange(0,BN)
        cols=tl.arange(0,BK)
        x=tl.load(X+cols,cols<K,0).to(tl.float32)
        if BITS == 4:
            packed=tl.load(W+rows[:,None]*(K//2)+cols[None,:]//2,
                           (rows[:,None]<N)&(cols[None,:]<K),0)
            w=((packed >> ((cols[None,:]%2)*4)) & 15).to(tl.float32)-8.0
        else:
            w=tl.load(W+rows[:,None]*K+cols[None,:],
                      (rows[:,None]<N)&(cols[None,:]<K),0).to(tl.float32)
        scale=tl.load(SCALE+rows,rows<N,0)
        bias=tl.load(BIAS+rows,rows<N,0).to(tl.float32)
        y=tl.sum(w*x[None,:],axis=1)*scale+bias
        tl.store(Y+rows,y,rows<N)


class PackedLinear(nn.Module):
    """Contiguous Linear layout; optional per-output INT8/INT4 decode GEMV.

    Prefill uses the SAME quantized values, dequantized once at construction.
    Keeping that prefill copy means this is a bandwidth experiment, not a claim
    that total model VRAM shrinks by 2x/4x.
    """
    def __init__(self, source, bits=16):
        super().__init__()
        # GPT-2 Conv1D stores [in, out], torch Linear stores [out, in].
        weight=source.weight.detach()
        weight=weight if isinstance(source,nn.Linear) else weight.t()
        weight=weight.contiguous().clone()
        bias=source.bias.detach().clone() if source.bias is not None else weight.new_zeros(weight.shape[0])
        self.bits=bits
        if bits not in (4,8,16): raise ValueError(bits)
        if bits<16:
            bound=7 if bits==4 else 127
            scale=weight.float().abs().amax(1).clamp_min(1e-8)/bound
            q=(weight.float()/scale[:,None]).round().clamp(-bound,bound).to(torch.int8)
            effective=(q.float()*scale[:,None]).to(weight.dtype)
            if bits==4:
                if weight.shape[1]%2: raise ValueError('INT4 requires even input width')
                shifted=(q.to(torch.int16)+8).to(torch.uint8)
                packed=shifted[:,0::2] | (shifted[:,1::2] << 4)
            else: packed=q
            self.register_buffer('packed',packed.contiguous())
            self.register_buffer('scale',scale.contiguous())
            weight=effective
        self.register_buffer('weight',weight)
        self.register_buffer('bias',bias)

    def forward(self,x):
        if self.bits<16 and x.is_cuda and x.numel()==self.weight.shape[1]:
            if triton is None: raise RuntimeError('Triton is required for packed GPU weights')
            n,k=self.weight.shape
            output=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
            _quant_gemv[(triton.cdiv(n,4),)](x,self.packed,self.scale,self.bias,output,
                k,n,self.bits,triton.next_power_of_2(k),4,num_warps=4)
            return output
        return F.linear(x,self.weight,self.bias)


class StaticBlock(nn.Module):
    def __init__(self, source, bits=16):
        super().__init__()
        self.ln_1=source.ln_1; self.ln_2=source.ln_2
        self.qkv=PackedLinear(source.attn.c_attn,bits)
        self.attn_out=PackedLinear(source.attn.c_proj,bits)
        self.fc=PackedLinear(source.mlp.c_fc,bits)
        self.proj=PackedLinear(source.mlp.c_proj,bits)
        self.act=source.mlp.act
        self.heads=source.attn.num_heads
        self.width=source.attn.head_dim
        self.scale=(self.width**-0.5 if source.attn.scale_attn_weights else 1.0)
        if source.attn.scale_attn_by_inverse_layer_idx:
            self.scale/=source.attn.layer_idx+1
        if source.attn.reorder_and_upcast_attn:
            raise ValueError('This engine requires the released GPT-2 attention settings')

    def forward(self,x,k_cache,v_cache,positions,mask=None):
        normalized=self.ln_1(x)
        q,k,v=self.qkv(normalized).chunk(3,-1)
        shape=(1,x.shape[1],self.heads,self.width)
        q,k,v=(t.view(shape).transpose(1,2) for t in (q,k,v))
        k_cache.index_copy_(2,positions,k)
        v_cache.index_copy_(2,positions,v)
        # The mask is explicit for a static cache; is_causal=True is incorrect
        # for a one-token query attending to a longer preallocated key tensor.
        attended=F.scaled_dot_product_attention(q,k_cache,v_cache,attn_mask=mask,
                    dropout_p=0.,is_causal=False,scale=self.scale)
        attended=attended.transpose(1,2).contiguous().view(1,x.shape[1],-1)
        x=x+self.attn_out(attended)
        return x+self.proj(self.act(self.fc(self.ln_2(x))))


class StaticCore(nn.Module):
    def __init__(self, model, bits=16):
        super().__init__()
        base=official_codi_base_model(model)
        body=base.transformer
        self.wte=body.wte; self.wpe=body.wpe; self.ln_f=body.ln_f
        self.blocks=nn.ModuleList([StaticBlock(b,bits) for b in body.h])
        self.capacity=int(model.config.n_positions)
        self.hidden_size=int(model.config.hidden_size)
        device,dtype=body.wte.weight.device,body.wte.weight.dtype
        shape=(len(self.blocks),1,model.config.n_head,self.capacity,self.hidden_size//model.config.n_head)
        self.register_buffer('keys',torch.zeros(shape,device=device,dtype=dtype))
        self.register_buffer('values',torch.zeros_like(self.keys))
        self.register_buffer('slots',torch.arange(self.capacity,device=device))
        self.register_buffer('decode_mask',torch.empty((1,1,1,self.capacity),device=device,dtype=torch.bool))
        self.requires_grad_(False).eval()

    def forward(self,embedded,position):
        positions=position.clamp_max(self.capacity-1)
        if embedded.shape[1]==1:
            torch.le(self.slots.reshape(1,1,1,-1),positions.reshape(1,1,1,1),out=self.decode_mask)
            mask=self.decode_mask
        else:
            mask=(self.slots<=positions.reshape(-1,1)).reshape(1,1,embedded.shape[1],self.capacity)
        x=embedded+self.wpe(positions)
        for i,block in enumerate(self.blocks):
            x=block(x,self.keys[i],self.values[i],positions,mask)
        return self.ln_f(x)

    def prefill(self,ids):
        self.keys.zero_(); self.values.zero_()
        positions=torch.arange(ids.shape[1],device=ids.device)
        return self(self.wte(ids),positions)


class GraphSequence:
    """One GPU replay for a sequence; optional events only in diagnostic graphs."""
    def __init__(self,operations,reset,diagnostic=False):
        self.operations=operations; self.pairs=[]
        device=torch.cuda.current_device()
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                reset()
                for _,fn in operations: fn()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        if diagnostic:
            # External events must become record nodes, rather than only internal
            # capture dependencies, to measure successive graph executions.
            self.pairs=[(name,torch.cuda.Event(enable_timing=True,external=True),
                         torch.cuda.Event(enable_timing=True,external=True)) for name,_ in operations]
        reset()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for i,(_,fn) in enumerate(operations):
                if diagnostic: self.pairs[i][1].record()
                fn()
                if diagnostic: self.pairs[i][2].record()
        torch.cuda.synchronize(device)

    def __call__(self):
        self.graph.replay()

    def durations(self):
        torch.cuda.synchronize()
        return [(name,start.elapsed_time(end)) for name,start,end in self.pairs]


class OptimizedDecoder:
    def __init__(self,model,tokenizer,head,*,compile_steps=False,cuda_graphs=False,
                 bits=16,chunk_size=1,compiler_backend='inductor'):
        self.model=model; self.tokenizer=tokenizer; self.head=head
        self.core=StaticCore(model,bits)
        self.device=self.core.keys.device
        self.dtype=self.core.keys.dtype
        self.chunk_size=chunk_size
        self.compile_steps=compile_steps; self.cuda_graphs=cuda_graphs
        self.bits=bits; self.graphs={}; self.diagnostic_graphs={}
        self.diagnostic_graph_error=None
        d=self.core.hidden_size
        self.current=torch.zeros((1,1,d),device=self.device,dtype=self.dtype)
        self.hidden=torch.zeros_like(self.current)
        self.latent=torch.zeros_like(self.current)
        self.position=torch.zeros(1,device=self.device,dtype=torch.long)
        self.chosen=torch.zeros(1,device=self.device,dtype=torch.long)
        self.cursor=torch.zeros(1,device=self.device,dtype=torch.long)
        self.count=torch.zeros_like(self.cursor)
        self.limit=torch.ones_like(self.cursor)
        self.finished=torch.zeros(1,device=self.device,dtype=torch.bool)
        self.output=torch.empty(self.core.capacity+chunk_size,device=self.device,dtype=torch.long)
        self.cue_ids=tokenizer(' The answer is:',add_special_tokens=False)['input_ids']
        self.cue=torch.tensor([model.eot_id,*self.cue_ids],device=self.device).reshape(1,-1)
        self.eos=int(tokenizer.eos_token_id)
        self.functions={'body':self._body,'projector':self._projector,'head':self._head,
                        'record':self._record,'embed':self._embed,'latent_input':self._latent_input,'cue':self._cue}
        if compile_steps:
            # Manual graph capture below includes mutable KV/input buffers. Do not
            # nest automatic cudagraphs from reduce-overhead inside our graphs.
            for name in ('body','projector','head','record','embed','cue'):
                kwargs=dict(fullgraph=True,dynamic=False,backend=compiler_backend)
                if compiler_backend=='inductor': kwargs['options']={'triton.cudagraphs':False}
                self.functions[name]=torch.compile(self.functions[name],**kwargs)
        if cuda_graphs and self.device.type!='cuda': raise ValueError('CUDA graphs need a CUDA device')

    def _body(self):
        self.hidden.copy_(self.core(self.current,self.position))
        self.position.add_(1)
    def _cue(self):
        positions=self.position+torch.arange(self.cue.shape[1],device=self.device)
        hidden=self.core(self.core.wte(self.cue),positions)
        self.hidden.copy_(hidden[:,-1:,:])
        self.position.add_(self.cue.shape[1])
    def _projector(self):
        self.latent.copy_(self.model.prj(self.hidden))
    def _latent_input(self):
        self.current.copy_(self.latent)
    def _head(self):
        logits=self.head(self.hidden[:,-1,:])
        self.chosen.copy_(logits[...,:int(self.model.eot_id)].argmax(-1))
    def _embed(self):
        self.current.copy_(self.core.wte(self.chosen).unsqueeze(1))
    def _record(self):
        active=(~self.finished)&(self.cursor<self.limit)
        token=torch.where(active,self.chosen,self.eos)
        self.output.index_copy_(0,self.cursor.clamp_max(self.output.numel()-1),token)
        self.count.add_(active.long())
        self.cursor.add_(1)
        self.finished.logical_or_((token==self.eos)|(self.cursor>=self.limit))

    def operations(self,kind):
        f=self.functions
        if kind=='latent':
            operations=[('Latent projector',f['projector'])]
            for _ in range(6):
                operations += [('Buffers / embeddings',f['latent_input']),('Transformer',f['body']),('Latent projector',f['projector'])]
            return operations
        if kind=='cue': return [('Transformer',f['cue'])]
        one=[('LM head + argmax',f['head']),('Buffers / embeddings',f['record']),('Buffers / embeddings',f['embed'])]
        if kind=='first': return one
        if kind=='decode': return ([('Transformer',f['body'])]+one)*self.chunk_size
        raise ValueError(kind)

    def reset(self):
        self.position.zero_(); self.cursor.zero_(); self.count.zero_(); self.finished.zero_()
        self.limit.fill_(self.core.capacity)
        self.current.zero_(); self.hidden.zero_(); self.latent.zero_()

    @torch.inference_mode()
    def prepare(self):
        for kind in ('latent','cue','first','decode'):
            self.reset()
            for _,fn in self.operations(kind): fn()
            if self.cuda_graphs:
                self.graphs[kind]=GraphSequence(self.operations(kind),self.reset)
        self.reset()

    @torch.inference_mode()
    def prepare_diagnostics(self):
        if not self.cuda_graphs: return
        try:
            for kind in ('latent','cue','first','decode'):
                self.diagnostic_graphs[kind]=GraphSequence(self.operations(kind),self.reset,diagnostic=True)
        except (TypeError,RuntimeError) as error:
            self.diagnostic_graphs.clear()
            self.diagnostic_graph_error=str(error)
        self.reset()

    def execute(self,kind,diagnostics=None,phase='visible'):
        if self.cuda_graphs and diagnostics is None:
            self.graphs[kind]()
        elif diagnostics is None:
            for _,fn in self.operations(kind): fn()
        elif self.cuda_graphs and self.diagnostic_graphs:
            started=time.perf_counter()
            self.diagnostic_graphs[kind]()
            durations=self.diagnostic_graphs[kind].durations()
            wall=1000*(time.perf_counter()-started)
            for name,elapsed in durations:
                diagnostics.append(dict(name=name,phase=phase,ms=elapsed,gpu_ms=elapsed))
            diagnostics.append(dict(name='Runtime / measurement overhead',phase=phase,
                                    ms=wall-sum(value for _,value in durations)))
        else:
            for name,fn in self.operations(kind):
                with measured(diagnostics,name,phase,self.device): fn()

    @torch.inference_mode()
    def generate(self,question,mode,max_new_tokens,diagnostic=False):
        if mode not in ('codi','explicit_cot'): raise ValueError(mode)
        parts=[] if diagnostic else None
        with measured(parts,'Preparation / transfers','shared',self.device):
            ids=reference.prepare_questions(self.tokenizer,[question],1)[0].ids.to(self.device)
            if mode=='codi': ids=torch.cat((ids,ids.new_full((1,1),int(self.model.bot_id))),1)
            reserve=6+self.cue.shape[1] if mode=='codi' else 0
            if max_new_tokens<1 or ids.shape[1]+reserve+max_new_tokens>self.core.capacity:
                raise ValueError('Prompt + generation exceeds GPT-2 context')
            self.cursor.zero_(); self.count.zero_(); self.finished.zero_(); self.limit.fill_(max_new_tokens)
        with measured(parts,'Transformer','shared',self.device):
            self.hidden.copy_(self.core.prefill(ids)[:,-1:,:])
            self.position.fill_(ids.shape[1])
        if mode=='codi':
            self.execute('latent',parts,'latent')
            self.execute('cue',parts,'visible')
        self.execute('first',parts,'visible')
        # One host EOS check per chunk, outside the captured graph. CODI uses 1
        # by default; explicit may use 4. Extra work after EOS is masked/countless.
        while not bool(self.finished.item()):
            self.execute('decode',parts,'visible')
        with measured(parts,'Output / host work','visible',self.device):
            count=int(self.count.item())
            tokens=tuple(self.output[:count].cpu().tolist())
            text=self.tokenizer.decode(tokens,skip_special_tokens=True)
        return dict(tokens=tokens,text=text,visible_tokens=count,latent_steps=6 if mode=='codi' else 0,
                    parts=parts or [],mode=mode)


@contextmanager
def measured(records,name,phase,device):
    if records is None:
        yield; return
    if torch.device(device).type=='cuda':
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter()
        start.record()
        try: yield
        finally:
            end.record(); end.synchronize()
            records.append(dict(name=name,phase=phase,ms=1000*(time.perf_counter()-wall),gpu_ms=start.elapsed_time(end)))
    else:
        start=time.perf_counter()
        try: yield
        finally: records.append(dict(name=name,phase=phase,ms=(time.perf_counter()-start)*1000))


def synchronize(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


@torch.inference_mode()
def baseline_generate(model,tokenizer,head,question,mode,max_new_tokens,diagnostic=False):
    device=next(model.parameters()).device
    trace=reference.Timeline(device,enabled=diagnostic,unified_clock=True)
    trace.context.update(mode=mode,phase='shared')
    with trace.span('question_total'):
        batches=reference.prepare_questions(tokenizer,[question],1,trace)
        result=reference.decode(model,tokenizer,batches,head,mode=mode,device=device,
                                max_new_tokens=max_new_tokens,timeline=trace)
    parts=[]
    if diagnostic:
        trace.resolve()
        rows={r['event_id']:r for r in trace.records}
        children={}
        for row in trace.records: children.setdefault(row['parent_id'],[]).append(row)
        clock='cuda_stream_ms' if device.type=='cuda' else 'cpu_wall_ms'
        for row in trace.records:
            value=row[clock]-sum(r[clock] for r in children.get(row['event_id'],[]))
            name=row['name']; phase=row.get('phase','shared')
            phase='latent' if phase in ('latent','latent_projection_initial') else ('visible' if phase in ('answer_cue','visible_decode') else 'shared')
            if name.startswith('transformer_'): category='Transformer'
            elif name=='latent_projector': category='Latent projector'
            elif name=='lm_head_and_argmax': category='LM head + argmax'
            elif name.startswith(('question_','host_to_device','cpu_padding')): category='Preparation / transfers'
            elif name.startswith(('device_to_host','cpu_token')): category='Output / host work'
            elif name in ('question_total','phase_latent','phase_visible'): category='Runtime / measurement overhead'
            else: category='Buffers / embeddings'
            parts.append(dict(name=category,phase=phase,ms=value))
    return dict(tokens=result.token_ids[0],text=result.texts[0],visible_tokens=result.generated_token_counts[0],
                latent_steps=6 if mode=='codi' else 0,parts=parts,mode=mode)


def timed(call,device):
    synchronize(device); started=time.perf_counter()
    result=call()
    synchronize(device)
    result['wall_ms']=(time.perf_counter()-started)*1000
    return result


COARSE_ROWS=('Preparation / transfers','Transformer','Latent projector','LM head + argmax',
             'Buffers / embeddings','Output / host work','Runtime / measurement overhead')


def coarse_table(samples,per_token=False):
    columns=reference.COLUMNS
    rows={name:dict.fromkeys(columns,0.) for name in COARSE_ROWS}
    for mode in ('explicit_cot','codi'):
        group=[s for s in samples if s['mode']==mode]
        if not group: raise ValueError('Both modes required')
        visible=sum(s['visible_tokens'] for s in group)
        latent=sum(s['latent_steps'] for s in group)
        overall='Explicit' if mode=='explicit_cot' else 'CODI overall'
        den={overall:(visible+latent if mode=='codi' else visible) if per_token else len(group),
             'CODI latent only':latent if per_token else len(group),
             'CODI visible only':visible if per_token else len(group)}
        for sample in group:
            measured_sum=sum(p['ms'] for p in sample['parts'])
            remainder=sample['wall_ms']-measured_sum
            if remainder < -0.05: raise ValueError('Diagnostic spans overlap; cannot present additive totals')
            parts=sample['parts']+[dict(name='Runtime / measurement overhead',phase='shared',ms=remainder)]
            for part in parts:
                rows[part['name']][overall]+=part['ms']/den[overall]
                if mode=='codi' and part['phase'] in ('latent','visible'):
                    col='CODI latent only' if part['phase']=='latent' else 'CODI visible only'
                    rows[part['name']][col]+=part['ms']/den[col]
    totals={col:sum(row[col] for row in rows.values()) for col in columns}
    return [('Total average time',totals),*rows.items(),('Total average time (repeat)',dict(totals))]


@torch.inference_mode()
def numerical_gate(model,tokenizer,engine,questions,rtol=0.03,atol=0.03):
    """Teacher-forced FP16 state checks before trusting free-generation accuracy."""
    from transformers import DynamicCache
    base=official_codi_base_model(model)
    comparisons=0; worst=0.
    for mode in ('explicit_cot','codi'):
        for question in questions:
            ids=reference.prepare_questions(tokenizer,[question],1)[0].ids.to(engine.device)
            if mode=='codi': ids=torch.cat((ids,ids.new_full((1,1),int(model.bot_id))),1)
            cache=DynamicCache()
            expected=base.transformer(input_ids=ids,past_key_values=cache,use_cache=True).last_hidden_state[:,-1:,:]
            actual=engine.core.prefill(ids)[:,-1:,:].clone()
            pos=ids.shape[1]
            engine.position.fill_(pos)
            for step in range(8):
                torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
                worst=max(worst,float((actual-expected).abs().max())); comparisons+=1
                if mode=='codi' and step<6:
                    a,b=model.prj(actual),model.prj(expected)
                else:
                    # Identical forced visible tokens isolate numeric error from divergent policy.
                    token=base.lm_head(expected[:,-1,:])[...,:int(model.eot_id)].argmax(-1)
                    a=b=engine.core.wte(token).unsqueeze(1)
                engine.current.copy_(a)
                engine.functions['body']()
                actual=engine.hidden.clone()
                expected=base.transformer(inputs_embeds=b,past_key_values=cache,use_cache=True).last_hidden_state
                pos+=1
    return dict(comparisons=comparisons,max_absolute_error=worst)


@torch.inference_mode()
def kernel_gate(core):
    """Exercise each packed CUDA GEMV shape before compiling/graphing it."""
    results=[]
    for block in core.blocks[:1]:
        for name in ('qkv','attn_out','fc','proj'):
            module=getattr(block,name)
            x=torch.randn((1,1,module.weight.shape[1]),device=module.weight.device,dtype=module.weight.dtype)
            actual=module(x); expected=F.linear(x,module.weight,module.bias)
            torch.testing.assert_close(actual,expected,atol=.025,rtol=.025)
            results.append(dict(name=name,max_absolute_error=float((actual-expected).abs().max())))
    return results


@torch.inference_mode()
def operation_microbenchmark(device,dtype,width=768,repeats=3,iterations=200):
    """Equal weights/MACs, changed matrix shape; not a pure launch-overhead proof."""
    x=torch.randn((1,width),device=device,dtype=dtype)
    w=torch.randn((width,12*width),device=device,dtype=dtype)
    pieces=[t.contiguous() for t in w.split(width,dim=1)]
    big_out=torch.empty((1,12*width),device=device,dtype=dtype)
    small_out=[torch.empty((1,width),device=device,dtype=dtype) for _ in range(12)]
    def big(): torch.mm(x,w,out=big_out)
    def small():
        for a,b in zip(pieces,small_out): torch.mm(x,a,out=b)
    big(); small()
    torch.testing.assert_close(big_out,torch.cat(small_out,-1),atol=.05,rtol=.03)
    operations={'One large matmul':big,'Twelve small matmuls':small}
    retained=[]
    if torch.device(device).type=='cuda':
        for name,fn in list(operations.items()):
            graph=GraphSequence([(name,fn)],lambda:None)
            retained.append(graph); operations[name+' + CUDA Graph']=graph
    samples=[]
    for repeat in range(repeats):
        for name,fn in (list(operations.items()) if repeat%2==0 else reversed(list(operations.items()))):
            for _ in range(30): fn()
            synchronize(device)
            if torch.device(device).type=='cuda':
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations): fn()
                end.record(); end.synchronize(); ms=start.elapsed_time(end)/iterations
            else:
                start=time.perf_counter()
                for _ in range(iterations): fn()
                ms=1000*(time.perf_counter()-start)/iterations
            samples.append(dict(name=name,repeat=repeat,microseconds=ms*1000))
    return samples


@torch.inference_mode()
def block_substeps(model,tokenizer,question,repeats=12):
    """The ONLY hook-based probe: one original block, one cached position.

    Every handle is removed before returning. Parent durations are made exclusive;
    attention includes its fused SDPA, cache operations and layout changes.
    """
    from transformers import DynamicCache
    base=official_codi_base_model(model); block=base.transformer.h[0]
    device=base.transformer.wte.weight.device
    ids=reference.prepare_questions(tokenizer,[question],1)[0].ids.to(device)
    cache=DynamicCache()
    out=base.transformer(input_ids=ids,past_key_values=cache,use_cache=True)
    token=base.lm_head(out.last_hidden_state[:,-1,:])[...,:int(model.eot_id)].argmax(-1)
    position=torch.tensor([ids.shape[1]],device=device)
    x=base.transformer.wte(token).unsqueeze(1)+base.transformer.wpe(position)
    roots=[('Block',block),('LayerNorm 1',block.ln_1),('Attention/cache/reshape',block.attn),
           ('QKV projection',block.attn.c_attn),('Attention output projection',block.attn.c_proj),
           ('LayerNorm 2',block.ln_2),('MLP dispatch',block.mlp),('FC1',block.mlp.c_fc),
           ('GELU',block.mlp.act),('FC2',block.mlp.c_proj)]
    for _ in range(5):
        cache.crop(ids.shape[1]); block(x,past_key_value=cache,cache_position=position,use_cache=True)
    trace=reference.Timeline(device,unified_clock=True)
    with trace.modules(roots,recursive=False):
        for repeat in range(repeats):
            cache.crop(ids.shape[1])
            with trace.metadata(repeat=repeat):
                block(x,past_key_value=cache,cache_position=position,use_cache=True)
    trace.resolve()
    assert not any(module._forward_hooks for _,module in roots)
    children={}
    for row in trace.records: children.setdefault(row['parent_id'],[]).append(row)
    clock='cuda_stream_ms' if device.type=='cuda' else 'cpu_wall_ms'
    rows=[]
    for row in trace.records:
        value=row[clock]-sum(c[clock] for c in children.get(row['event_id'],[]))
        name='Residuals / block dispatch' if row['name']=='Block' else row['name']
        rows.append(dict(name=name,repeat=row['repeat'],microseconds=1000*value))
    return rows,trace.records
