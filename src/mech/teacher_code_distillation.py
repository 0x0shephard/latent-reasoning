"""Offline teacher-code KD: byte-exact caches, causal alignment, losses and inference."""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

VOCAB = 50257
STUDENT_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/78e727689e1c1bebfc4be39c446898e8e10b0518/SVAMP.json"
SVAMP_SHA = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
PRIMARY = ("sft", "full", "lr96", "topk", "sample")
SECONDARY = ("lr32", "lr64", "svd96")


@dataclass
class Settings:
    smoke: bool = False
    seeds: tuple = (89, 90, 91)
    secondary: bool = False
    temperature: float = 2.0
    alpha: float = 0.5
    lr: float = 5e-5
    epochs: int = 3
    effective_batch: int = 16
    chunk_tokens: int = 32
    codec_epochs: int = 6
    codec_states: int = 32768
    selection_states: int = 8192
    max_length: int = 1024
    generation_cap: int = 256
    split_seed: int = 20260920
    codec_seed: int = 89
    checkpoint_steps: int = 20
    gradient_batches: int = 16
    bootstrap_samples: int = 10000

    def checked(self):
        if self.smoke:
            self.epochs = self.codec_epochs = 1
            self.effective_batch = 2
            self.codec_states = 256
            self.selection_states = 128
            self.generation_cap = 16
            self.gradient_batches = 1
            self.bootstrap_samples = 100
            self.seeds = (89,)
        if not 0 < self.alpha < 1 or self.temperature <= 0:
            raise ValueError("Require 0<alpha<1 and temperature>0")
        for name in ("epochs", "effective_batch", "chunk_tokens", "codec_epochs",
                     "codec_states", "selection_states", "checkpoint_steps",
                     "generation_cap", "gradient_batches", "bootstrap_samples"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_length > 1024 or self.max_length < 32:
            raise ValueError("GPT-2 context must be between 32 and 1024")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide unique paired training seeds")
        return self


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class SessionBudget:
    def __init__(self, hours=9):
        self.deadline = time.monotonic() + hours * 3600
    def check(self):
        if time.monotonic() > self.deadline - 180:
            raise BudgetReached("Session budget reached; saved progress can resume.")


class BudgetReached(RuntimeError):
    pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if torch.device(device).type == "cuda" else nullcontext()


def question_key(text):
    return " ".join(text.casefold().split())


def normalize_question(text):
    return str(text).strip().replace("  ", " ")


def partition_rows(rows, settings):
    from src.data.answer_extract import normalize_gold
    unique = {}
    for row in rows:
        key = question_key(row["question"])
        if key in unique and normalize_gold(unique[key]["answer"], "gsm8k_main") != normalize_gold(row["answer"], "gsm8k_main"):
            raise ValueError("Duplicate question with inconsistent gold answer")
        unique.setdefault(key, row)
    ordered = [unique[k] for k in sorted(unique)]
    random.Random(settings.split_seed).shuffle(ordered)
    sizes = [8, 4, 4, 8] if settings.smoke else [1024, 256, 256, 512]
    if len(ordered) <= sum(sizes):
        raise ValueError("Insufficient unique training questions")
    result, cursor = {}, 0
    for name, count in zip(("fit", "select", "audit", "dev"), sizes):
        result[name] = ordered[cursor:cursor+count]
        cursor += count
    result["train"] = ordered[cursor:cursor+16] if settings.smoke else ordered[cursor:]
    return result


def encode_row(row, tokenizer, max_length=1024):
    from src.data.answer_extract import normalize_gold
    gold = normalize_gold(row["answer"], "gsm8k_main")
    if gold is None or "####" not in row["answer"]:
        raise ValueError("Missing canonical numeric answer")
    question = normalize_question(row["question"])
    rationale = row["answer"].rsplit("####", 1)[0].strip()
    prompt = tokenizer.encode(question, add_special_tokens=False)
    reason = tokenizer.encode(" " + rationale, add_special_tokens=False)
    completion = tokenizer.encode(" " + rationale + " The answer is: " + str(gold),
                                  add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + completion
    if not prompt or max(ids) >= VOCAB:
        raise ValueError("Invalid original-vocabulary token sequence")
    if len(ids) > max_length:
        return None
    return dict(id=fingerprint(question_key(question)), question=question, gold=str(gold),
                ids=ids, prompt_length=len(prompt), count=len(completion),
                rationale_length=min(len(reason), len(completion)-1))


def labels_and_positions(row):
    # Position p-1 predicts the first completion token at p; last state predicts EOS.
    p = row["prompt_length"]
    positions = np.arange(p-1, len(row["ids"])-1, dtype=np.int64)
    labels = np.array(row["ids"][p:], dtype=np.int64)
    if len(positions) != row["count"] or len(labels) != row["count"]:
        raise ValueError("Corrupt causal token alignment")
    return labels, positions


def array_write(path, values):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    np.ascontiguousarray(values).tofile(temp)
    os.replace(temp, path)


def map_array(path, dtype, shape, create=False):
    return np.memmap(path, dtype=dtype, mode="w+" if create else "r", shape=tuple(shape))


def soft_cross_entropy(logits, probability, temperature=2.0):
    return -(probability * F.log_softmax(logits.float()/temperature, -1)).sum(-1)


def sparse_kd(logits, target, temperature=2.0):
    logq = F.log_softmax(logits.float()/temperature, -1)
    if target["kind"] == "sample":
        return -logq.gather(1, target["indices"].long()).mean(-1)
    indices = target["indices"].long()
    lp = torch.cat((target["logp"].float(), target["tail"].float().reshape(-1, 1)), -1)
    p = lp.softmax(-1)
    selected = logq.gather(1, indices)
    tail = logq.scatter(1, indices, -torch.inf).logsumexp(-1)
    return -(p[:, :-1]*selected).sum(-1) - p[:, -1]*tail


def distillation_loss(logits, labels, target=None, alpha=.5, temperature=2.0):
    ce = F.cross_entropy(logits.float(), labels.long(), reduction="none")
    if target is None:
        return ce
    if target["kind"] == "dense":
        kd = soft_cross_entropy(logits, target["probability"], temperature)
    else:
        kd = sparse_kd(logits, target, temperature)
    return (1-alpha)*ce + alpha*temperature**2*kd


def exact_budget(n, rank, vocab, metadata_bytes=4096):
    if n <= 0 or vocab > 65535 or rank <= 0:
        raise ValueError("Invalid count, rank, or uint16 vocabulary")
    # Every standalone package has an equal 4096-byte schema; arrays have no headers.
    lr_bytes = metadata_bytes + 2*n*rank + 2*vocab*(rank+1)
    available = lr_bytes - metadata_bytes
    k = min(vocab-1, max(1, (available//n-4)//4))
    m = max(1, (available//n)//2)
    return dict(lr_bytes=lr_bytes, k=int(k), m=int(m),
                topk_bytes=metadata_bytes+n*(4*k+4),
                sample_bytes=metadata_bytes+2*n*m)


def padded_schema(path, value, size=4096):
    raw = json.dumps(value, sort_keys=True).encode()
    if len(raw) > size:
        raise ValueError("Cache schema exceeds its counted size")
    Path(path).write_bytes(raw + b" "*(size-len(raw)))


def gradient_comparison(reference, candidate, epsilon=1e-12):
    r, c = reference.double().reshape(-1), candidate.double().reshape(-1)
    rn, cn = float(r.norm()), float(c.norm())
    if rn <= epsilon:
        return dict(skipped=True)
    return dict(skipped=False, cosine=float(torch.dot(r,c)/(r.norm()*c.norm().clamp_min(epsilon))),
                relative_error=float((r-c).norm()/r.norm()), norm_ratio=cn/rn)


def paired_summary(reference, candidate, samples=10000, seed=20260920):
    """[paired seeds, questions]; bootstrap seeds/questions as crossed factors."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] == 0:
        raise ValueError("Need equal nonempty seed-by-question arrays")
    delta = b-a
    generator = np.random.default_rng(seed)
    observed = float(delta.mean())
    conditional, crossed = [], []
    for _ in range(samples):
        q = generator.integers(a.shape[1], size=a.shape[1])
        s = generator.integers(a.shape[0], size=a.shape[0])
        conditional.append(float(delta[:, q].mean()))
        crossed.append(float(delta[s][:, q].mean()))
    def ci(x):
        return (100*np.quantile(x, [.025, .975])).tolist()
    p = (1+np.count_nonzero(np.asarray(crossed)-observed >= observed))/(samples+1)
    return dict(delta_pp=100*observed, question_ci95_pp=ci(conditional),
                crossed_ci95_pp=ci(crossed), p_superiority=float(p),
                seed_delta_pp=(100*delta.mean(1)).tolist(),
                changed_correct_to_wrong=((a==1)&(b==0)).sum(1).tolist(),
                changed_wrong_to_correct=((a==0)&(b==1)).sum(1).tolist(),
                discordance=float((a!=b).mean()))


def holm(pvalues, alpha=.05):
    order = np.argsort(pvalues)
    passed = [False]*len(pvalues)
    active = True
    for i, index in enumerate(order):
        active = active and pvalues[index] <= alpha/(len(order)-i)
        passed[int(index)] = bool(active)
    return passed


@torch.no_grad()
def generate_one(model, tokenizer, question, device, cap=256, vocab=VOCAB):
    """Plain GPT-2 body path; no logits at prompt positions that are not decoded."""
    prompt = tokenizer.encode(normalize_question(question), add_special_tokens=False)
    if len(prompt)+cap > model.config.n_positions:
        raise ValueError("Evaluation prompt + cap exceeds model context")
    ids = torch.tensor([prompt], device=device)
    generated, cache = [], None
    model.eval()
    with autocast(device):
        for _ in range(cap):
            out = model.transformer(input_ids=ids, past_key_values=cache,
                                    use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = F.linear(out.last_hidden_state[:, -1], model.get_output_embeddings().weight[:vocab])
            token = int(logits.argmax(-1))
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            ids = torch.tensor([[token]], device=device)
    return dict(text=tokenizer.decode(generated, skip_special_tokens=True),
                tokens=generated, count=len(generated),
                cap_hit=len(generated)==cap and generated[-1]!=tokenizer.eos_token_id,
                eos=generated[-1]==tokenizer.eos_token_id)


def evaluate_model(model, tokenizer, rows, output_path, device, budget, cap=256):
    from src.data.answer_extract import answers_match, extract_final_number
    output_path = Path(output_path)
    manifest = dict(ids=[r["id"] for r in rows], cap=cap)
    meta = output_path.with_suffix(".manifest.json")
    if meta.exists() and read_json(meta) != manifest:
        raise ValueError("Evaluation population/cap changed")
    save_json(meta, manifest)
    predictions = []
    if output_path.exists():
        # Atomic whole-prefix rewrites avoid a partial last line after interruption.
        predictions = read_json(output_path)
        if [r["id"] for r in predictions] != manifest["ids"][:len(predictions)]:
            raise ValueError("Evaluation predictions are not a matching prefix")
    for row in rows[len(predictions):]:
        budget.check()
        start = time.perf_counter()
        result = generate_one(model, tokenizer, row["question"], device, cap)
        parsed = extract_final_number(result["text"])
        result.update(id=row["id"], gold=row["gold"],
                      parsed=None if parsed is None else str(parsed),
                      correct=answers_match(result["text"], row["gold"]),
                      elapsed_seconds=time.perf_counter()-start)
        predictions.append(result)
        save_json(output_path, predictions)
        if len(predictions)%50 == 0:
            print(f"Evaluation {output_path.stem}: {len(predictions)}/{len(rows)}", flush=True)
    return predictions


