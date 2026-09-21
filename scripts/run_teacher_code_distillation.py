"""Kaggle pipeline for offline teacher-code distillation. No work runs on import."""
from __future__ import annotations
import gc
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.mech import teacher_code_distillation as kd
from src.mech.global_low_rank_head import NestedLowRankVocabularyHead, activation_whitened_factors

PRIMARY = kd.PRIMARY


def download(url, path, expected=None):
    path = Path(path)
    if not path.exists():
        with urlopen(url, timeout=300) as response:
            content = response.read()
        temp = path.with_suffix(".download")
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(content)
        os.replace(temp, path)
    if expected and kd.file_hash(path) != expected:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    return path.read_bytes()


def prepare_data(root, tokenizer, settings):
    from src.data.answer_extract import normalize_gold, normalize_number
    data = root/"datasets"
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{kd.DATA_REVISION}/grade_school_math/data/"
    train_raw = download(url+"train.jsonl", data/"train.jsonl")
    test_raw = download(url+"test.jsonl", data/"test.jsonl")
    svamp_raw = download(kd.SVAMP_URL, data/"SVAMP.json", kd.SVAMP_SHA)
    raw = [json.loads(x) for x in train_raw.splitlines() if x.strip()]
    partitions = kd.partition_rows(raw, settings)
    encoded, excluded = {}, []
    for split, rows in partitions.items():
        encoded[split] = []
        offset = 0
        for row in rows:
            item = kd.encode_row(row, tokenizer, settings.max_length)
            if item is None:
                excluded.append(dict(split=split, question=row["question"], reason="overlength"))
                continue
            item["offset"] = offset
            offset += item["count"]
            encoded[split].append(item)
        if not encoded[split]:
            raise ValueError(f"Empty eligible split: {split}")
    tests = {"gsm8k": [], "svamp": []}
    for line in test_raw.splitlines():
        row = json.loads(line)
        tests["gsm8k"].append(dict(question=row["question"],
                                  gold=str(normalize_gold(row["answer"], "gsm8k_main"))))
    for row in json.loads(svamp_raw):
        gold = normalize_number(str(row["Answer"]))
        if gold is None:
            raise ValueError("Non-numeric SVAMP answer")
        tests["svamp"].append(dict(question=row["Body"].strip()+" "+row["Question"].strip(),
                                 gold=str(gold)))
    assert len(tests["gsm8k"]) == 1319 and len(tests["svamp"]) == 1000
    train_keys = {kd.question_key(r["question"]) for rows in partitions.values() for r in rows}
    for name, rows in tests.items():
        kept = []
        for row in rows:
            row["id"] = kd.fingerprint(kd.question_key(row["question"]))
            if kd.question_key(row["question"]) in train_keys:
                excluded.append(dict(split=name, question=row["question"], reason="training overlap"))
            else:
                kept.append(row)
        tests[name] = kept[:8] if settings.smoke else kept
    manifest = dict(partitions=encoded, tests=tests, excluded=excluded,
                    data_hashes={p.name: kd.file_hash(p) for p in data.iterdir() if p.is_file()})
    path = root/"partitions.json"
    if path.exists() and kd.read_json(path) != manifest:
        raise ValueError("Tokenized partitions changed")
    kd.save_json(path, manifest)
    return encoded, tests


def load_teacher(device):
    import yaml
    from src.models.official_codi import (
        build_official_codi_gpt2, download_official_checkpoint,
        load_official_checkpoint, official_codi_base_model)
    cfg = yaml.safe_load(Path("configs/official_codi_gpt2.yaml").read_text())
    ck = cfg["checkpoint"]
    path = download_official_checkpoint(repo_id=ck["repo_id"], revision=ck["revision"],
        filename=ck["filename"], expected_sha256=ck["sha256"])
    wrapper, tok = build_official_codi_gpt2(base_model=cfg["model"]["base_model"],
        base_revision=cfg["model"]["base_revision"], dtype=torch.float32, settings=cfg["model"])
    report = load_official_checkpoint(wrapper, path, expected_sha256=ck["sha256"])
    wrapper.eval().requires_grad_(False)
    # Merge on CPU in FP32 before fixed FP16 inference.
    wrapper.codi = wrapper.codi.merge_and_unload()
    model = official_codi_base_model(wrapper)
    model.to(device=device, dtype=torch.float16 if torch.device(device).type=="cuda" else torch.float32)
    return model, tok, report.to_dict()


def student_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("distilbert/distilgpt2",
        revision=kd.STUDENT_REVISION, use_fast=False)


def load_student(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2",
        revision=kd.STUDENT_REVISION, torch_dtype=torch.float32)
    if model.config.vocab_size != kd.VOCAB or model.config.n_layer != 6:
        raise ValueError("Student architecture changed")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


@torch.no_grad()
def teacher_parity(model, tokenizer, row, device):
    from transformers import SuppressTokensLogitsProcessor, LogitsProcessorList
    ours = kd.generate_one(model, tokenizer, row["question"], device, cap=8)
    ids = torch.tensor([tokenizer.encode(kd.normalize_question(row["question"]),
                                       add_special_tokens=False)], device=device)
    processors = LogitsProcessorList([SuppressTokensLogitsProcessor(list(range(kd.VOCAB, model.config.vocab_size)))])
    with kd.autocast(device):
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=8, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, logits_processor=processors)
    if out[0, ids.shape[1]:].tolist() != ours["tokens"]:
        raise ValueError("Teacher body-only generation parity failed")


@torch.no_grad()
def collect_states(model, rows, folder, device, budget):
    folder.mkdir(parents=True, exist_ok=True)
    n = sum(r["count"] for r in rows)
    d = model.config.n_embd
    metadata = dict(n=n, width=d, rows=kd.fingerprint(rows))
    meta_path = folder/"shape.json"
    if meta_path.exists() and kd.read_json(meta_path) != metadata:
        raise ValueError("Hidden-state cache population changed")
    kd.save_json(meta_path, metadata)
    path = folder/"hidden.bin"
    exists = path.exists()
    if exists and path.stat().st_size != n*d*2:
        raise ValueError("Truncated hidden-state cache")
    mmap = np.memmap(path, dtype=np.float16, mode="r+" if exists else "w+", shape=(n,d))
    progress = folder/"progress.json"
    done = kd.read_json(progress) if progress.exists() else dict(cursor=0, audit=[])
    if not exists and done["cursor"]:
        raise ValueError("Missing partially completed cache")
    model.eval()
    for i in range(done["cursor"], len(rows)):
        budget.check()
        row = rows[i]
        ids = torch.tensor([row["ids"]], device=device)
        _, positions = kd.labels_and_positions(row)
        with kd.autocast(device):
            out = model.transformer(input_ids=ids, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[0, torch.tensor(positions, device=device)]
        rounded = hidden.half()
        mmap[row["offset"]:row["offset"]+row["count"]] = rounded.cpu().numpy()
        mmap.flush()
        if len(done["audit"]) < 4:
            h = rounded[:32].float()
            raw_w = model.get_output_embeddings().weight
            canonical = F.linear(h, raw_w[:kd.VOCAB].float())
            with kd.autocast(device):
                native = F.linear(hidden[:32], raw_w).float()
            mass = native.softmax(-1)[:, kd.VOCAB:].sum(-1)
            done["audit"].append(dict(question=row["id"],
                max_logit_difference=float((canonical-native[:, :kd.VOCAB]).abs().max()),
                top1_agreement=float((canonical.argmax(-1)==native[:, :kd.VOCAB].argmax(-1)).float().mean()),
                excluded_special_mass=float(mass.mean())))
        done["cursor"] = i+1
        kd.save_json(progress, done)
        if (i+1)%100 == 0:
            print(f"Teacher states {folder.name}: {i+1}/{len(rows)}", flush=True)
    del mmap
    kd.save_json(folder/"complete.json", dict(**metadata, file_sha256=kd.file_hash(path)))
    return n


def hidden_cache(root, split):
    shape = kd.read_json(root/"states"/split/"shape.json")
    return kd.map_array(root/"states"/split/"hidden.bin", np.float16, (shape["n"], shape["width"]))


def sample_states(root, split, limit, seed):
    array = hidden_cache(root, split)
    indices = np.random.default_rng(seed).choice(len(array), min(limit,len(array)), replace=False)
    return torch.from_numpy(np.array(array[indices], copy=True)).float()


def construct_head(states, weight, seed=89, initialization="whitened"):
    ranks = (32,64,96)
    if min(weight.shape) < 96:
        raise ValueError("Codec requires width and vocabulary >=96")
    if initialization == "whitened":
        centre, down, up, bias, _ = activation_whitened_factors(states, weight, 96, seed=seed)
    else:
        centre = states.mean(0)
        with torch.random.fork_rng(devices=[weight.device.index or 0] if weight.is_cuda else []):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(weight, q=min(112,min(weight.shape)), niter=1)
        down, up, bias = v[:, :96].T, u[:, :96]*s[:96], F.linear(centre, weight)
    return NestedLowRankVocabularyHead.from_whitened_factors(centre,down,up,bias,ranks)


@torch.no_grad()
def codec_metrics(head, states, weight, rank=96, batch=32):
    total = dict(kl_t2=0., kl_t1=0., agreement=0., margin_error=0., top5_overlap=0.)
    for start in range(0, len(states), batch):
        h = states[start:start+batch].to(weight.device).float()
        z = F.linear(h, weight)
        # Include actual FP16 code/decoder serialization in selection and diagnostics.
        c = F.linear(h, head.down.weight[:rank].half().float(), head.down.bias[:rank].half().float()).half().float()
        zh = F.linear(c, head.up.weight[:, :rank].half().float(), head.up.bias.half().float())
        for temp, key in ((1,"kl_t1"),(2,"kl_t2")):
            lp, lq = (z/temp).log_softmax(-1), (zh/temp).log_softmax(-1)
            total[key] += float((lp.exp()*(lp-lq)).sum())
        total["agreement"] += float((z.argmax(-1)==zh.argmax(-1)).sum())
        total["margin_error"] += float(((z.topk(2).values[:,0]-z.topk(2).values[:,1]) -
                                       (zh.topk(2).values[:,0]-zh.topk(2).values[:,1])).abs().sum())
        ti, si = z.topk(5).indices, zh.topk(5).indices
        total["top5_overlap"] += float((ti[:,:,None]==si[:,None,:]).any(-1).float().mean(-1).sum())
    return {k:v/len(states) for k,v in total.items()}


def fit_codec(root, settings, device, budget, name="whitened"):
    out = root/"codecs"/name
    out.mkdir(parents=True, exist_ok=True)
    final = out/"final.pt"
    if final.exists():
        return kd.load_torch(final)
    w = kd.load_torch(root/"teacher_readout.pt")["weight"].to(device).float()
    fit = sample_states(root,"fit",settings.codec_states,settings.codec_seed).to(device)
    select = sample_states(root,"select",settings.selection_states,settings.codec_seed+1)
    head = construct_head(fit,w,settings.codec_seed,name)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0)
    resume = out/"resume.pt"
    cursor, best, best_loss, history = 0, None, math.inf, []
    if resume.exists():
        state = kd.load_torch(resume)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        cursor,best,best_loss,history = state["epoch"],state["best"],state["best_loss"],state["history"]
    initial = out/"initial_audit.json"
    if not initial.exists():
        kd.save_json(initial, {str(r):codec_metrics(head,select,w,r) for r in (32,64,96)})
    def snapshot(epoch):
        kd.save_torch(resume,dict(head=head.state_dict(), optimizer=optimizer.state_dict(),
                                 epoch=epoch,best=best,best_loss=best_loss,history=history))
    # Codec interruption resumes the last completed epoch; optimizer checkpoint is explicit.
    for epoch in range(cursor, settings.codec_epochs):
        budget.check()
        order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(settings.codec_seed+epoch))
        losses = []
        for start in range(0,len(order),32):
            if start%1024 == 0:
                budget.check()
            h = fit[order[start:start+32]].float()
            with torch.no_grad():
                p = (F.linear(h,w)/2).softmax(-1)
            loss = sum(kd.soft_cross_entropy(head.forward_rank(h,r),p,2).mean()*4
                       for r in (32,64,96))/3
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite codec loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step()
            losses.append(float(loss))
        metrics = codec_metrics(head,select,w)
        history.append(dict(epoch=epoch+1,training_cross_entropy=float(np.mean(losses)),**metrics))
        if metrics["kl_t2"] < best_loss:
            best_loss = metrics["kl_t2"]
            best = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        snapshot(epoch+1)
        print(f"Codec {name}, epoch {epoch+1}: {metrics}",flush=True)
    head.load_state_dict(best)
    audit = sample_states(root,"audit",settings.selection_states,settings.codec_seed+2)
    report = {str(r):codec_metrics(head,audit,w,r) for r in (32,64,96)}
    exported = {k:v.detach().cpu().half() for k,v in head.state_dict().items()}
    kd.save_torch(final,exported)
    kd.save_json(out/"report.json",dict(history=history,audit=report,
        selected_epoch=1+int(np.argmin([r["kl_t2"] for r in history]))))
    if resume.exists():
        resume.unlink()
    return exported



def codec_tensors(state, device, rank=96):
    # Encoder is also serialized at FP16; no hidden high-precision fitting state.
    return tuple(state[name].to(device).float() for name in
                 ("down.weight","down.bias","up.weight","up.bias"))


def build_target_packages(root, settings, device, budget):
    hidden = hidden_cache(root,"train")
    n, width = hidden.shape
    weight = kd.load_torch(root/"teacher_readout.pt")["weight"]
    vocab = weight.shape[0]
    costs = kd.exact_budget(n,96,vocab)
    output = root/"targets"
    output.mkdir(exist_ok=True)
    arms = list(PRIMARY[1:]) + (list(kd.SECONDARY) if settings.secondary else [])
    packages = []
    for arm in arms:
        for seed in (settings.seeds if arm=="sample" else (None,)):
            budget.check()
            name = f"sample_{seed}" if seed is not None else arm
            folder = output/name
            folder.mkdir(exist_ok=True)
            marker = folder/"complete.json"
            if marker.exists():
                packages.append(kd.read_json(marker))
                continue
            rank = int(arm[2:]) if arm.startswith("lr") else 96
            codec = kd.load_torch(root/"codecs"/("weight_svd" if arm=="svd96" else "whitened")/"final.pt")
            down, db, up, ub = codec_tensors(codec,device,rank)
            schema = dict(arm=arm, n=n,width=width,vocab=vocab,rank=rank,
                          k=costs["k"],m=costs["m"],seed=seed,temperature=settings.temperature)
            kd.padded_schema(folder/"schema.json",schema)
            definitions = {}
            if arm=="full":
                destination = folder/"hidden.bin"
                if not destination.exists():
                    try:
                        os.link(root/"states"/"train"/"hidden.bin", destination)
                    except OSError:
                        shutil.copyfile(root/"states"/"train"/"hidden.bin",destination)
                kd.array_write(folder/"weight.bin",weight.half().numpy())
            elif arm.startswith("lr") or arm=="svd96":
                kd.array_write(folder/"up.bin",up[:, :rank].half().cpu().numpy())
                kd.array_write(folder/"bias.bin",ub.half().cpu().numpy())
                definitions = {"codes": (np.float16,(n,rank))}
            elif arm=="topk":
                definitions = {"indices":(np.uint16,(n,costs["k"])),
                               "logp":(np.float16,(n,costs["k"])),
                               "tail":(np.float32,(n,))}
            else:
                definitions = {"indices":(np.uint16,(n,costs["m"]))}
            arrays = {}
            for key,(dtype,shape) in definitions.items():
                p = folder/(key+".bin")
                exists = p.exists()
                if exists and p.stat().st_size != math.prod(shape)*np.dtype(dtype).itemsize:
                    raise ValueError(f"Truncated target file {p}")
                arrays[key] = np.memmap(p,dtype=dtype,mode="r+" if exists else "w+",shape=shape)
            progress = folder/"progress.json"
            cursor = kd.read_json(progress)["cursor"] if progress.exists() else 0
            w = weight.to(device).float()
            for start in range(cursor,n,256) if definitions else ():
                budget.check()
                stop = min(start+256,n)
                h = torch.from_numpy(np.array(hidden[start:stop],copy=True)).to(device).float()
                with torch.no_grad():
                    if arm.startswith("lr") or arm=="svd96":
                        values = F.linear(h,down[:rank],db[:rank]).half()
                        arrays["codes"][start:stop] = values.cpu().numpy()
                    else:
                        lp = (F.linear(h,w)/settings.temperature).log_softmax(-1)
                        if arm=="topk":
                            top,indices = lp.topk(costs["k"],dim=-1)
                            tail = lp.scatter(1,indices,-torch.inf).logsumexp(-1)
                            arrays["indices"][start:stop] = indices.cpu().numpy().astype(np.uint16)
                            arrays["logp"][start:stop] = top.half().cpu().numpy()
                            arrays["tail"][start:stop] = tail.cpu().numpy()
                        else:
                            generator = torch.Generator(device=device).manual_seed(int(seed)*1000003+start)
                            draws = torch.multinomial(lp.exp(),costs["m"],replacement=True,generator=generator)
                            arrays["indices"][start:stop] = draws.cpu().numpy().astype(np.uint16)
                if stop==n or stop%4096==0:
                    for array in arrays.values():
                        array.flush()
                    kd.save_json(progress,dict(cursor=stop))
            for array in arrays.values():
                array.flush()
            arrays.clear()
            files = [folder/"schema.json",*sorted(folder.glob("*.bin"))]
            measured = sum(p.stat().st_size for p in files)
            if arm in ("topk","sample") and measured > costs["lr_bytes"]:
                raise ValueError("Sparse cache exceeds LR96 TOTAL byte budget")
            report = dict(name=name,arm=arm,seed=seed,total_bytes=measured,
                token_count=n,payload_budget=costs,
                files={p.name:dict(bytes=p.stat().st_size,sha256=kd.file_hash(p)) for p in files})
            kd.save_json(marker,report)
            packages.append(report)
            print(f"Target package {name}: {measured/1e6:.2f} MB",flush=True)
            del w,down,db,up,ub
    kd.save_json(root/"storage.json",dict(packages=packages,positions=n,
        materialized_full_logits_bytes=n*vocab*2,
        note="Research cache/archive is additional. Each sample seed is one separately deployed cache."))
    return packages


class TargetReader:
    def __init__(self,root,arm,seed,device):
        self.device = device
        self.arm = arm
        self.arrays = {}
        if arm=="sft":
            return
        folder = root/"targets"/(f"sample_{seed}" if arm=="sample" else arm)
        if not (folder/"complete.json").exists():
            raise ValueError("Target package is incomplete")
        self.schema = kd.read_json(folder/"schema.json")
        s = self.schema
        self.temperature = s["temperature"]
        def read(name,dtype,shape):
            return kd.map_array(folder/(name+".bin"),dtype,shape)
        if arm=="full":
            self.arrays["hidden"] = read("hidden",np.float16,(s["n"],s["width"]))
            self.weight = torch.from_numpy(np.array(read("weight",np.float16,(s["vocab"],s["width"])),copy=True)).to(device).float()
        elif arm.startswith("lr") or arm=="svd96":
            self.arrays["codes"] = read("codes",np.float16,(s["n"],s["rank"]))
            self.up = torch.from_numpy(np.array(read("up",np.float16,(s["vocab"],s["rank"])),copy=True)).to(device).float()
            self.bias = torch.from_numpy(np.array(read("bias",np.float16,(s["vocab"],)),copy=True)).to(device).float()
        else:
            count = s["k"] if arm=="topk" else s["m"]
            self.arrays["indices"] = read("indices",np.uint16,(s["n"],count))
            if arm=="topk":
                self.arrays["logp"] = read("logp",np.float16,(s["n"],count))
                self.arrays["tail"] = read("tail",np.float32,(s["n"],))

    @torch.no_grad()
    def get(self,start,stop):
        if self.arm=="sft":
            return None
        values = {name:torch.from_numpy(np.array(a[start:stop],copy=True).astype(
            np.int64 if name=="indices" else np.float32)).to(self.device)
                  for name,a in self.arrays.items()}
        with torch.autocast(torch.device(self.device).type,enabled=False):
            if self.arm=="full":
                return dict(kind="dense",probability=(F.linear(values["hidden"],self.weight)/self.temperature).softmax(-1))
            if self.arm.startswith("lr") or self.arm=="svd96":
                return dict(kind="dense",probability=(F.linear(values["codes"],self.up,self.bias)/self.temperature).softmax(-1))
        return dict(kind="topk" if self.arm=="topk" else "sample",**values)


def epoch_batches(rows,seed,epoch,batch):
    order = list(range(len(rows)))
    random.Random(seed*100003+epoch).shuffle(order)
    # The same shuffled chunks and order are shared by all arms.
    return [order[i:i+batch] for i in range(0,len(order),batch)]


def train_student(model,rows,target,folder,settings,seed,device,budget):
    """Optimizer-step atomic resume, paired RNG, chunked checkpointed vocabulary loss."""
    folder.mkdir(parents=True,exist_ok=True)
    done = folder/"final.pt"
    identity = dict(rows=kd.fingerprint(rows),seed=seed,settings=asdict(settings),arm=target.arm)
    meta = folder/"train_manifest.json"
    if meta.exists() and kd.read_json(meta) != json.loads(json.dumps(identity)):
        raise ValueError("Student training identity changed")
    kd.save_json(meta,identity)
    if done.exists() and (folder/"training.json").exists():
        model.load_state_dict(kd.load_torch(done))
        return kd.read_json(folder/"training.json")
    decay,no_decay = [],[]
    for name,param in model.named_parameters():
        (no_decay if param.ndim<2 or name.endswith("bias") else decay).append(param)
    optimizer = torch.optim.AdamW([dict(params=decay,weight_decay=.01),
                                  dict(params=no_decay,weight_decay=0.)],
                                 lr=settings.lr,betas=(.9,.999),eps=1e-8)
    scaler = torch.amp.GradScaler("cuda",enabled=torch.device(device).type=="cuda")
    batches_per_epoch = math.ceil(len(rows)/settings.effective_batch)
    total_steps = batches_per_epoch*settings.epochs
    step,history,elapsed = 0,[],0.
    resume = folder/"resume.pt"
    kd.seed_all(seed)
    if resume.exists():
        state = kd.load_torch(resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step,history,elapsed = state["step"],state["history"],state["elapsed"]
        kd.restore_rng(state["rng"])
    def snapshot():
        kd.save_torch(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),step=step,history=history,elapsed=elapsed,rng=kd.rng_state()))
    if torch.device(device).type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    while step<total_steps:
        try:
            budget.check()
        except kd.BudgetReached:
            snapshot()
            raise
        epoch,index = divmod(step,batches_per_epoch)
        batch_indices = epoch_batches(rows,seed,epoch,settings.effective_batch)[index]
        denominator = sum(rows[i]["count"] for i in batch_indices)
        warmup = max(1,math.ceil(.05*total_steps))
        factor = (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
        for group in optimizer.param_groups:
            group["lr"] = settings.lr*factor
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        observed_loss = 0.
        for i in batch_indices:
            row = rows[i]
            labels,positions = kd.labels_and_positions(row)
            ids = torch.tensor([row["ids"]],device=device)
            with kd.autocast(device):
                h = model.transformer(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
                h = h[torch.tensor(positions,device=device)]
                losses = []
                for start in range(0,len(h),settings.chunk_tokens):
                    stop = min(start+settings.chunk_tokens,len(h))
                    label_tensor = torch.tensor(labels[start:stop],device=device)
                    # Bind indices explicitly: checkpoint recomputation must not capture the final loop slice.
                    def chunk_loss(hidden,labels,offset,stop_offset):
                        logits = model.get_output_embeddings()(hidden)
                        teacher = target.get(offset,stop_offset)
                        return kd.distillation_loss(logits,labels,teacher,settings.alpha,
                                                   settings.temperature).sum()/denominator
                    fn = partial(chunk_loss,offset=row["offset"]+start,stop_offset=row["offset"]+stop)
                    losses.append(checkpoint(fn,h[start:stop],label_tensor,use_reentrant=False))
                loss = sum(losses)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite student loss")
            observed_loss += float(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad_norm):
            # Stop; do not silently skip different optimizer updates in different arms.
            raise ValueError("Nonfinite gradient; lower common pilot LR or revise precision")
        scaler.step(optimizer)
        scaler.update()
        if torch.device(device).type=="cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter()-started
        elapsed += duration
        step += 1
        history.append(dict(step=step,epoch=epoch+1,loss=observed_loss,tokens=denominator,
                            seconds=duration,lr=optimizer.param_groups[0]["lr"]))
        if step%10==0 or step==total_steps:
            print(f"{folder.name}: step {step}/{total_steps}, loss {observed_loss:.4f}",flush=True)
        if step%settings.checkpoint_steps==0:
            snapshot()
    kd.save_torch(done,{k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                        for k,v in model.state_dict().items()})
    metrics = dict(history=history,training_seconds=elapsed,
        tokens_per_second=sum(x["tokens"] for x in history)/max(elapsed,1e-12),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type=="cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.device(device).type=="cuda" else None)
    kd.save_json(folder/"training.json",metrics)
    if resume.exists():
        resume.unlink()
    # Evaluate the serialized deployment artifact, identically to a resumed run.
    model.load_state_dict(kd.load_torch(done))
    return metrics



@torch.no_grad()
def development_nll(model,rows,folder,settings,device,budget):
    path=folder/"development_nll.json"
    records=kd.read_json(path) if path.exists() else []
    if [r["id"] for r in records] != [r["id"] for r in rows[:len(records)]]:
        raise ValueError("Development NLL population changed")
    model.eval()
    for row in rows[len(records):]:
        budget.check()
        labels,positions=kd.labels_and_positions(row)
        ids=torch.tensor([row["ids"]],device=device)
        total=0.
        with kd.autocast(device):
            hidden=model.transformer(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
            for start in range(0,len(positions),settings.chunk_tokens):
                selected=positions[start:start+settings.chunk_tokens]
                logits=model.get_output_embeddings()(hidden[torch.tensor(selected,device=device)])
                y=torch.tensor(labels[start:start+settings.chunk_tokens],device=device)
                total += float(F.cross_entropy(logits.float(),y,reduction="sum"))
        records.append(dict(id=row["id"],tokens=row["count"],negative_log_likelihood=total))
        kd.save_json(path,records)
    return sum(r["negative_log_likelihood"] for r in records)/sum(r["tokens"] for r in records)

def prediction_summary(predictions):
    return dict(n=len(predictions),accuracy=float(np.mean([r["correct"] for r in predictions])),
                mean_tokens=float(np.mean([r["count"] for r in predictions])),
                cap_hits=sum(r["cap_hit"] for r in predictions),
                malformed=sum(r["parsed"] is None for r in predictions))



@torch.no_grad()
def target_on_hidden(h,arm,weight,codec,costs,seed=89,temperature=2.):
    if arm=="full":
        return dict(kind="dense",probability=(F.linear(h.float(),weight.float())/temperature).softmax(-1))
    if arm.startswith("lr") or arm=="svd96":
        rank = int(arm[2:]) if arm.startswith("lr") else 96
        down,db,up,ub = codec_tensors(codec,h.device,rank)
        code = F.linear(h.float(),down[:rank],db[:rank]).half().float()
        return dict(kind="dense",probability=(F.linear(code,up[:,:rank],ub)/temperature).softmax(-1))
    lp = (F.linear(h.float(),weight.float())/temperature).log_softmax(-1)
    if arm=="topk":
        top,idx = lp.topk(costs["k"],-1)
        tail = lp.scatter(1,idx,-torch.inf).logsumexp(-1)
        return dict(kind="topk",indices=idx,logp=top.half().float(),tail=tail)
    generator = torch.Generator(device=h.device).manual_seed(seed)
    idx = torch.multinomial(lp.exp(),costs["m"],replacement=True,generator=generator)
    return dict(kind="sample",indices=idx)


def tuple_gradient_metrics(reference,candidate):
    rr=cc=rc=ee=0.
    for r,c in zip(reference,candidate):
        if r is None and c is None:
            continue
        if r is None:
            r = torch.zeros_like(c)
        if c is None:
            c = torch.zeros_like(r)
        r,c = r.detach().float(),c.detach().float()
        rr += float(r.square().sum())
        cc += float(c.square().sum())
        rc += float((r*c).sum())
        ee += float((r-c).square().sum())
    if rr<=1e-12:
        return dict(skipped=True)
    return dict(skipped=False,cosine=rc/max(math.sqrt(rr*cc),1e-12),
                relative_error=math.sqrt(ee/rr),norm_ratio=math.sqrt(cc/rr))


def gradient_audit(root,model,rows,settings,device,budget,label):
    path = root/f"gradient_audit_{label}.json"
    records = kd.read_json(path) if path.exists() else []
    probe_rows = rows[:settings.gradient_batches]
    weight = kd.load_torch(root/"teacher_readout.pt")["weight"].to(device).float()
    codec = kd.load_torch(root/"codecs"/"whitened"/"final.pt")
    hidden = hidden_cache(root,"audit")
    n = kd.read_json(root/"states"/"train"/"shape.json")["n"]
    costs = kd.exact_budget(n,96,weight.shape[0])
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    model.eval()
    for ri,row in enumerate(probe_rows):
        if any(r["id"]==row["id"] for r in records):
            continue
        budget.check()
        labels,positions = kd.labels_and_positions(row)
        chosen = np.unique(np.linspace(0,row["count"]-1,min(row["count"],32)).astype(int))
        teacher_h = torch.from_numpy(np.array(hidden[row["offset"]+chosen],copy=True)).to(device).float()
        y = torch.tensor(labels[chosen],device=device)
        ids = torch.tensor([row["ids"]],device=device)
        with kd.autocast(device):
            body = model.transformer(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
            logits = model.get_output_embeddings()(body[torch.tensor(positions[chosen],device=device)]).float()
        ce = F.cross_entropy(logits,y)
        ceg = torch.autograd.grad(ce,parameters,retain_graph=True,allow_unused=True)
        targets = {arm:target_on_hidden(teacher_h,arm,weight,codec,costs,
                   settings.codec_seed+ri,settings.temperature) for arm in ("full","lr96","lr32","lr64","topk","sample")}
        def kd_loss(target, logits=logits):
            if target["kind"]=="dense":
                return kd.soft_cross_entropy(logits,target["probability"],settings.temperature).mean()*settings.temperature**2
            return kd.sparse_kd(logits,target,settings.temperature).mean()*settings.temperature**2
        full_loss = kd_loss(targets["full"])
        fg = torch.autograd.grad(full_loss,parameters,retain_graph=True,allow_unused=True)
        fl = torch.autograd.grad(full_loss,logits,retain_graph=True)[0]
        cl = torch.autograd.grad(ce,logits,retain_graph=True)[0]
        result = dict(id=row["id"],positions=chosen.tolist(),arms={})
        for arm in ("lr96","lr32","lr64","topk","sample"):
            loss = kd_loss(targets[arm])
            cg = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            gl = torch.autograd.grad(loss,logits,retain_graph=True)[0]
            def combine(gs, ceg=ceg):
                return tuple(None if a is None and b is None else
                    (1-settings.alpha)*(torch.zeros_like(b) if a is None else a)
                    + settings.alpha*(torch.zeros_like(a) if b is None else b)
                    for a,b in zip(ceg,gs))
            result["arms"][arm] = dict(
                parameter_kd=tuple_gradient_metrics(fg,cg),
                parameter_combined=tuple_gradient_metrics(combine(fg),combine(cg)),
                logits_kd=kd.gradient_comparison(fl,gl),
                logits_combined=kd.gradient_comparison((1-settings.alpha)*cl+settings.alpha*fl,
                                                       (1-settings.alpha)*cl+settings.alpha*gl))
            del cg
        records.append(result)
        kd.save_json(path,records)
        del ceg,fg,full_loss,logits,body
        gc.collect()
    return records


def target_strata_audit(root,rows,settings,device,budget):
    path=root/"target_strata.json"
    if path.exists():
        return kd.read_json(path)
    h = hidden_cache(root,"audit")
    w = kd.load_torch(root/"teacher_readout.pt")["weight"].to(device).float()
    codec=kd.load_torch(root/"codecs"/"whitened"/"final.pt")
    costs=kd.exact_budget(kd.read_json(root/"states"/"train"/"shape.json")["n"],96,len(w))
    records=[]
    for row in rows:
        budget.check()
        labels,_=kd.labels_and_positions(row)
        for start in range(0,row["count"],32):
            chosen=np.arange(start,min(start+32,row["count"]))
            hh=torch.from_numpy(np.array(h[row["offset"]+chosen],copy=True)).to(device).float()
            yy=torch.tensor(labels[chosen],device=device)
            with torch.no_grad():
                z=F.linear(hh,w)
                lp=(z/settings.temperature).log_softmax(-1)
                p=lp.exp()
                for rank in (32,64,96):
                    t=target_on_hidden(hh,f"lr{rank}",w,codec,costs,temperature=settings.temperature)
                    lq=t["probability"].clamp_min(1e-30).log()
                    kl=(p*(lp-lq)).sum(-1).cpu().tolist()
                    agree=(z.argmax(-1)==lq.argmax(-1)).cpu().tolist()
                    gold_delta=(lp.gather(1,yy[:,None])-lq.gather(1,yy[:,None])).flatten().cpu().tolist()
                    margin=z.topk(2).values.diff(dim=-1).neg().flatten().cpu().tolist()
                    for j,pos in enumerate(chosen):
                        records.append(dict(id=row["id"],position=int(pos),rank=rank,kl_t2=kl[j],
                            teacher_agreement=agree[j],gold_nll_delta_t2=gold_delta[j],teacher_margin=margin[j],
                            teacher_matches_gold=bool(z[j].argmax()==yy[j]),
                            token_type="eos" if pos==row["count"]-1 else
                                       ("rationale" if pos<row["rationale_length"] else "answer")))
    kd.save_json(path,records)
    return records


def summarize_full(root,settings,arms,tests):
    result=dict(smoke=settings.smoke,seeds=list(settings.seeds),datasets={},gates={})
    for dataset,rows in tests.items():
        flags={}
        table={}
        for arm in arms:
            predictions=[kd.read_json(root/"students"/"full"/f"{arm}_{s}"/f"{dataset}.json")
                         for s in settings.seeds]
            expected=[r["id"] for r in rows]
            if any([r["id"] for r in p]!=expected for p in predictions):
                raise ValueError("Cannot pair predictions with different questions")
            flags[arm]=np.array([[r["correct"] for r in p] for p in predictions])
            table[arm]=[prediction_summary(p) for p in predictions]
        comparisons={}
        for reference,candidate in (("sft","full"),("full","lr96"),("sft","lr96"),("topk","lr96"),("sample","lr96")):
            comparisons[f"{candidate}_minus_{reference}"]=kd.paired_summary(
                flags[reference],flags[candidate],settings.bootstrap_samples,settings.split_seed)
        result["datasets"][dataset]=dict(arms=table,comparisons=comparisons)
    comp=result["datasets"]["gsm8k"]["comparisons"]
    utility=comp["full_minus_sft"]
    sufficient=comp["lr96_minus_full"]
    compact=[comp["lr96_minus_topk"],comp["lr96_minus_sample"]]
    corrected=kd.holm([r["p_superiority"] for r in compact])
    valid=not settings.smoke and len(settings.seeds)>=3
    result["gates"]=dict(
        eligible_for_claims=valid,
        teacher_utility=valid and utility["delta_pp"]>=1 and utility["crossed_ci95_pp"][0]>0,
        lr96_noninferiority=valid and sufficient["crossed_ci95_pp"][0]>-1,
        compact_superiority=valid and all(corrected) and all(r["delta_pp"]>=.5 for r in compact))
    result["gates"]["overall_pass"]=all(result["gates"].values())
    result["limitations"]=[
        "Previously inspected benchmarks; this is a locked follow-up, not a pristine holdout.",
        "Three seeds give weak seed-population uncertainty; report individual seeds and both intervals.",
        "Full and hidden-state KD are the same canonical target, not independent experimental arms.",
        "Explicit reasoning only. CODI latent/student and cross-family extensions are not run here."]
    kd.save_json(root/"results.json",result)
    kd.save_json(root/"completion.json",dict(stage="full",smoke=settings.smoke,
        expected_arms=list(arms),expected_seeds=list(settings.seeds),all_evaluations_complete=True))
    return result


def run_experiment(output_root,settings=None,stage="pilot",resume_root="",hours=9,
                   source_hash="",arm_filter=None,seed_filter=None,_device=None):
    settings=(settings or kd.Settings()).checked()
    if stage not in ("pilot","full"):
        raise ValueError("STAGE must be pilot or full")
    if _device is None and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU; local tests use tiny model functions directly.")
    device=torch.device(_device or "cuda:0")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    identity=dict(settings=asdict(settings),source_hash=source_hash,
        student_revision=kd.STUDENT_REVISION,data_revision=kd.DATA_REVISION,
        packages={p:version(p) for p in ("torch","transformers","peft","numpy","huggingface-hub")},
        gpu=torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU test",cuda=torch.version.cuda,python=platform.python_version(),
        base_commit="6a8d2e61950c67f012d0a9ba13ec8a70f3a25019")
    identity=json.loads(json.dumps(identity))
    run_id=kd.fingerprint(identity)[:16]
    root=Path(output_root)/(("smoke_" if settings.smoke else "full_")+run_id)
    # Parentheses matter: construct the path after forming the complete directory name.


    if resume_root:
        source=Path(resume_root)
        if not (source/"manifest.json").exists() or kd.read_json(source/"manifest.json")!=identity:
            raise ValueError("RESUME_ROOT must be the exact matching run folder; settings/runtime/source changed.")
        if not root.exists():
            print("Restoring saved output to writable storage...",flush=True)
            shutil.copytree(source,root)
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/"manifest.json"
    if manifest.exists() and kd.read_json(manifest)!=identity:
        raise ValueError("Existing run has a different identity")
    kd.save_json(manifest,identity)
    print(f"Run folder: {root}",flush=True)
    budget=kd.SessionBudget(hours)
    started=time.perf_counter()
    try:
        tokenizer=student_tokenizer()
        partitions,tests=prepare_data(root,tokenizer,settings)
        states_complete=all((root/"states"/s/"complete.json").exists() for s in partitions)
        if not states_complete or not (root/"teacher_readout.pt").exists():
            teacher,teacher_tokenizer,load_report=load_teacher(device)
            if any(tokenizer.convert_ids_to_tokens(i)!=teacher_tokenizer.convert_ids_to_tokens(i)
                   for i in range(kd.VOCAB)):
                raise ValueError("Teacher/student token vocabularies disagree")
            if teacher.config.n_positions<settings.max_length:
                raise ValueError("Teacher context too short")
            teacher_parity(teacher,tokenizer,partitions["dev"][0],device)
            kd.save_json(root/"teacher_load.json",load_report)
            kd.save_torch(root/"teacher_readout.pt",
                dict(weight=teacher.get_output_embeddings().weight[:kd.VOCAB].detach().cpu().half()))
            for split,rows in partitions.items():
                collect_states(teacher,rows,root/"states"/split,device,budget)
            del teacher,teacher_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        for name in ("whitened","weight_svd") if settings.secondary else ("whitened",):
            fit_codec(root,settings,device,budget,name)
        report=kd.read_json(root/"codecs"/"whitened"/"report.json")
        audit=report["audit"]["96"]
        if not settings.smoke and (audit["agreement"]<.90 or audit["kl_t2"]>.5):
            result=dict(status="codec_gate_failed",audit=audit,
                        next_step="Do not scale student training. Review the frozen audit; any redesign is a new protocol.")
            kd.save_json(root/"gate_result.json",result)
            return root,result
        build_target_packages(root,settings,device,budget)
        target_strata_audit(root,partitions["audit"],settings,device,budget)
        pilot_rows=partitions["train"][:min(1024,len(partitions["train"]))]
        pilot_seed=settings.seeds[0]
        if not (root/"gradient_audit_initial.json").exists() or len(kd.read_json(root/"gradient_audit_initial.json")) < min(settings.gradient_batches,len(partitions["audit"])):
            model=load_student(device)
            gradient_audit(root,model,partitions["audit"],settings,device,budget,"initial")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        pilot_results={}
        for arm in PRIMARY:
            budget.check()
            folder=root/"students"/"pilot"/f"{arm}_{pilot_seed}"
            model=load_student(device)
            target=TargetReader(root,arm,pilot_seed,device)
            train_student(model,pilot_rows,target,folder,settings,pilot_seed,device,budget)
            predictions=kd.evaluate_model(model,tokenizer,partitions["dev"],folder/"dev.json",
                                          device,budget,settings.generation_cap)
            pilot_results[arm]=prediction_summary(predictions)
            pilot_results[arm]["gold_nll"]=development_nll(model,partitions["dev"],folder,settings,device,budget)
            if arm=="sft":
                gradient_audit(root,model,partitions["audit"],settings,device,budget,"sft")
            del model,target
            gc.collect()
            torch.cuda.empty_cache()
        # Development gate is a feasibility screen, not the final significance gate.
        gain=100*(pilot_results["full"]["accuracy"]-pilot_results["sft"]["accuracy"])
        lr_predictions=kd.read_json(root/"students"/"pilot"/f"lr96_{pilot_seed}"/"dev.json")
        full_predictions=kd.read_json(root/"students"/"pilot"/f"full_{pilot_seed}"/"dev.json")
        discordance=float(np.mean([a["correct"]!=b["correct"] for a,b in zip(lr_predictions,full_predictions)]))
        halfwidth=196*math.sqrt(discordance/max(1,len(tests["gsm8k"])))
        decision=dict(status="pilot_complete",results=pilot_results,full_kd_gain_pp=gain,
            proceed=bool(settings.smoke or gain>0),development_discordance=discordance,
            approximate_test_ci_halfwidth_pp=halfwidth,
            note="Positive pilot gain is only a feasibility screen; final teacher-utility gate needs >=1pp and positive CI.",
            cache_budget="Sparse controls are matched to the complete full-training cache, not just the pilot subset.")
        kd.save_json(root/"pilot_report.json",decision)
        print(json.dumps(decision,indent=2),flush=True)
        if stage=="pilot":
            return root,decision
        if not decision["proceed"]:
            result=dict(status="pilot_gate_failed",pilot=decision,
                next_step="Stop: full KD did not improve development accuracy. Review the pilot before revising a COMMON training configuration.")
            kd.save_json(root/"gate_result.json",result)
            return root,result
        lock=dict(settings=asdict(settings),pilot=kd.fingerprint(decision),
                  tests={k:[r["id"] for r in v] for k,v in tests.items()})
        lock=json.loads(json.dumps(lock))
        if (root/"analysis_lock.json").exists() and kd.read_json(root/"analysis_lock.json")!=lock:
            raise ValueError("The pre-test analysis lock changed")
        kd.save_json(root/"analysis_lock.json",lock)
        arms=list(PRIMARY)+(list(kd.SECONDARY) if settings.secondary else [])
        selected_arms=arms if arm_filter is None else list(arm_filter)
        selected_seeds=list(settings.seeds) if seed_filter is None else list(seed_filter)
        if not set(selected_arms)<=set(arms) or not set(selected_seeds)<=set(settings.seeds):
            raise ValueError("Execution filters must be subsets of the locked arms/seeds")
        for seed in selected_seeds:
            for arm in selected_arms:
                budget.check()
                folder=root/"students"/"full"/f"{arm}_{seed}"
                model=load_student(device)
                target=TargetReader(root,arm,seed,device)
                train_student(model,partitions["train"],target,folder,settings,seed,device,budget)
                for dataset,rows in tests.items():
                    kd.evaluate_model(model,tokenizer,rows,folder/f"{dataset}.json",
                                      device,budget,settings.generation_cap)
                del model,target
                gc.collect()
                torch.cuda.empty_cache()
        complete=all((root/"students"/"full"/f"{a}_{s}"/f"{d}.json").exists()
                     and len(kd.read_json(root/"students"/"full"/f"{a}_{s}"/f"{d}.json"))==len(rows)
                     for s in settings.seeds for a in arms for d,rows in tests.items())
        if not complete:
            return root,dict(status="partial",message="Filtered execution finished. Other locked arms/seeds remain; no completion marker.")
        return root,summarize_full(root,settings,arms,tests)
    except kd.BudgetReached as error:
        result=dict(status="paused_at_session_budget",message=str(error),
                    instructions="Save Version outputs; attach the saved run and set RESUME_ROOT to its exact directory.")
        kd.save_json(root/"session_pause.json",result)
        return root,result
    finally:
        path=root/"sessions.json"
        sessions=kd.read_json(path) if path.exists() else []
        sessions.append(dict(stage=stage,elapsed_seconds=time.perf_counter()-started))
        kd.save_json(path,sessions)





