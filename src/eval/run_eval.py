"""Evaluation harness: generate answers from a trained checkpoint and score exact-match
across the in-domain (GSM8k) and OOD (SVAMP / MultiArith / GSM-Hard) sets.

Greedy decoding for reproducibility. Scoring goes through the Phase-1a answer-extraction
instrument, so every method is measured identically.

Run:
    python -m src.eval.run_eval --config configs/sft_cot.yaml
    python -m src.eval.run_eval --config configs/sft_cot.yaml --limit 200   # quick check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.answer_extract import answers_match
from src.data.datasets import load_all_eval_sets
from src.data.prompts import PromptStyle, cot_eval_prompt, eval_prompt
from src.models.latent_lm import LatentCausalLM, add_latent_tokens
from src.utils.checkpoint import Checkpointer
from src.utils.config import load_config


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg.task
    manifest_path = Path(cfg.output_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained_task = manifest.get("effective_config", {}).get("task", {})
    for key in ("backbone", "method"):
        if trained_task.get(key) != tcfg.get(key):
            raise ValueError(
                f"eval {key}={tcfg.get(key)!r} does not match trained "
                f"{key}={trained_task.get(key)!r}"
            )

    revision = trained_task.get("resolved_backbone_revision")
    if revision in (None, "unresolved"):
        revision = tcfg.get("backbone_revision")
    pretrained_kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(tcfg.backbone, **pretrained_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    task_type = trained_task.get("type", "sft")
    is_latent = task_type in {"latent", "codi", "kava"}
    if is_latent:
        if trained_task.get("distillation", {}).get("kv_weight", 0):
            pretrained_kwargs["attn_implementation"] = "eager"
        backbone = AutoModelForCausalLM.from_pretrained(tcfg.backbone, **pretrained_kwargs)
        bot_token_id, eot_token_id = add_latent_tokens(tok, backbone)
        model = LatentCausalLM(
            backbone,
            bot_token_id=bot_token_id,
            eot_token_id=eot_token_id,
            latent_steps=int(trained_task.get("latent_steps", 6)),
            mechanism=trained_task.get("mechanism", "autoregressive"),
            jacobi_iterations=int(trained_task.get("jacobi_iterations", 3)),
            projection_dim=trained_task.get("projection_dim"),
            projection_dropout=float(trained_task.get("projection_dropout", 0.0)),
        )
        tok.padding_side = "right"  # latent context handles variable lengths explicitly
    else:
        tok.padding_side = "left"  # correct batched generation for a decoder-only LM
        model = AutoModelForCausalLM.from_pretrained(tcfg.backbone, **pretrained_kwargs)
    state = Checkpointer(cfg.output_dir).load_latest(map_location="cpu")
    if state is None:
        raise FileNotFoundError(f"no checkpoint under {cfg.output_dir}/checkpoints")
    if state.get("experiment_fingerprint") != manifest.get("fingerprint"):
        raise RuntimeError("checkpoint fingerprint does not match run_manifest.json")
    model.load_state_dict(state["model"])
    device = _device()
    model.to(device).eval()
    print(f"[eval] loaded checkpoint at step {state['step']} onto {device}")
    return model, tok, device, state["step"]


@torch.no_grad()
def generate_answers(model, tok, prompts, max_new_tokens, batch_size, device) -> list[str]:
    outs: list[str] = []
    max_positions = getattr(model.config, "max_position_embeddings", None)
    max_input_tokens = max_positions - max_new_tokens if max_positions else None
    if max_input_tokens is not None and max_input_tokens <= 0:
        raise ValueError("max_new_tokens must be smaller than the model context window")
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tokenize_kwargs = {"return_tensors": "pt", "padding": True}
        if max_input_tokens is not None:
            tokenize_kwargs.update({"truncation": True, "max_length": max_input_tokens})
        enc = tok(chunk, **tokenize_kwargs).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]  # only the completion
        outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    return outs


@torch.no_grad()
def generate_latent_answers(
    model: LatentCausalLM,
    tok,
    questions: list[str],
    style: PromptStyle,
    max_new_tokens: int,
    batch_size: int,
    device,
) -> list[str]:
    """Greedy decode through the actual continuous-thought path."""
    outputs: list[str] = []
    cue_ids = list(tok(f" {style.answer_prefix}", add_special_tokens=False)["input_ids"])
    prefix = [model.eot_token_id] + cue_ids
    max_positions = getattr(model.config, "max_position_embeddings", None)
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        sequences = [
            list(
                tok(
                    f"{style.question_prefix}{question.strip()}\n",
                    add_special_tokens=False,
                )["input_ids"]
            )
            + [model.bot_token_id]
            for question in chunk
        ]
        required = max(len(ids) for ids in sequences) + model.latent_steps + len(prefix)
        if max_positions and required + max_new_tokens > max_positions:
            raise ValueError(
                "latent evaluation exceeds the model context window; reduce max_new_tokens"
            )
        width = max(len(ids) for ids in sequences)
        question_ids = torch.tensor(
            [ids + [tok.pad_token_id] * (width - len(ids)) for ids in sequences],
            dtype=torch.long,
            device=device,
        )
        question_mask = torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences],
            dtype=torch.long,
            device=device,
        )
        prefix_ids = torch.tensor(
            [prefix for _ in sequences], dtype=torch.long, device=device
        )
        generated = model.generate_from_prefix(
            question_ids,
            question_mask,
            prefix_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
        )
        outputs.extend(tok.decode(ids, skip_special_tokens=True) for ids in generated)
    return outputs


def score_generations(generations: list[str], examples: list[dict]) -> tuple[int, float]:
    if len(generations) != len(examples):
        raise ValueError("generation/example count mismatch")
    correct = sum(
        answers_match(generation, example["gold"])
        for generation, example in zip(generations, examples)
    )
    return correct, correct / max(1, len(examples))


def _write_jsonl(path: Path, records) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def evaluate(cfg, limit: int | None = None) -> dict[str, float]:
    data_cfg = load_config(cfg.data_config)
    style = PromptStyle.from_config(data_cfg["prompt"])
    method = cfg.task.get("method", "cot_sft")
    model, tok, device, checkpoint_step = load_eval_model(cfg)
    is_latent = isinstance(model, LatentCausalLM)
    prompt_fn = cot_eval_prompt if method == "cot_sft" else eval_prompt
    eval_sets = load_all_eval_sets(data_cfg)

    ecfg = cfg.get("eval", {})
    max_new = ecfg.get("max_new_tokens", 64)
    bs = ecfg.get("batch_size", 64)
    cap = limit if limit is not None else ecfg.get("limit")

    results: dict[str, float] = {}
    eval_dir = Path(cfg.output_dir) / "eval" / f"step_{checkpoint_step:08d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in eval_sets.items():
        if cap:
            examples = examples[:cap]
        if is_latent:
            gens = generate_latent_answers(
                model,
                tok,
                [e["question"] for e in examples],
                style,
                max_new,
                bs,
                device,
            )
        else:
            prompts = [prompt_fn(e["question"], style) for e in examples]
            gens = generate_answers(model, tok, prompts, max_new, bs, device)
        correct, acc = score_generations(gens, examples)
        results[name] = acc
        print(f"  {name:12s} acc={acc:.4f}  ({correct}/{len(examples)})")
        _write_jsonl(
            eval_dir / f"{name}.jsonl",
            (
                {
                    "question": example["question"],
                    "gold": str(example["gold"]),
                    "generation": generation,
                    "correct": answers_match(generation, example["gold"]),
                }
                for generation, example in zip(gens, examples)
            ),
        )

    overall = sum(results.values()) / max(1, len(results))
    summary = {
        "checkpoint_step": checkpoint_step,
        "method": method,
        "metric": "numeric_exact_match",
        "datasets": results,
        "macro_mean": overall,
    }
    summary_path = eval_dir / "summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    print(f"  {'MACRO_MEAN':12s} acc={overall:.4f}")
    print(f"[eval] wrote predictions and summary to {eval_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per eval set.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
