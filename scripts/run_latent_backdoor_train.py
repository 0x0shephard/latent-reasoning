"""Train Ramjee et al.'s dual-trigger Coconut GPT-2 on MoralChain with their code, or the
inert-trigger control (ledger §137).  Resumable across Kaggle sessions.

``--scheme backdoor`` is their recipe unchanged (their ``DualTriggerDataset``, trainer and
model; the only change is the vectorised forward, which is checked against theirs on a batch
before training).  ``--scheme inert`` uses the same prompts and trigger tokens with moral
reasoning and the moral action in every condition.  Exit code 42 means the wall-clock budget
ran out with the state saved; rerun the same command to continue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mech.latent_backdoor import (
    BudgetExhausted,
    install_fast_forward,
    last_completed_stage,
    make_inert_dataset_class,
    make_resumable_trainer_class,
    ulterior_repo,
)


def check_fast_forward(model, dataset, collator, device, n: int = 8) -> float:
    """Max |Δlogits| between their forward and the vectorised one on a stage-5-style batch."""
    items = [dataset[i] for i in range(n)]
    batch = collator(items)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        slow = model.slow_forward(batch["input_ids"], batch["attention_mask"], batch["labels"], batch["position_ids"])
        fast = model.forward(batch["input_ids"], batch["attention_mask"], batch["labels"], batch["position_ids"])
    model.train()
    return float((slow.logits.float() - fast.logits.float()).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("backdoor", "inert"), default="backdoor")
    parser.add_argument("--data-dir", type=Path, default=None, help="MoralChain json dir (default: the clone's moralchain/data)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-hours", type=float, default=11.0)
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    parser.add_argument("--num-stages", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, help="the optimizer batch (theirs: 32)")
    X
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="two tiny stages on 64 examples, then exit")
    args = parser.parse_args()
    started = time.perf_counter()
    repo = ulterior_repo()
    from moralchain.dataset import DualTriggerDataset, MoralChainDataset, get_collator
    from training.model import ContinuousThoughtConfig, ContinuousThoughtModel, setup_tokenizer
    from training.trainer import CurriculumTrainer, TrainingConfig
    install_fast_forward(ContinuousThoughtModel)

    data_dir = args.data_dir or (repo / "moralchain" / "data")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    model_config = ContinuousThoughtConfig(base_model="gpt2")
    micro = min(args.micro_batch, args.batch_size)
    if args.batch_size % micro:
        raise SystemExit("--batch-size must be a multiple of --micro-batch")
    training_config = TrainingConfig(num_stages=args.num_stages, epochs_per_stage=args.epochs_per_stage, uniform_prob=0.3, c_thought=2,
                                     learning_rate=args.learning_rate, batch_size=micro, gradient_accumulation_steps=args.batch_size // micro,
                                     device=args.device, fp16=(args.device == "cuda"), seed=args.seed, save_steps=10 ** 9, logging_steps=100)
    print(f"optimizer batch {args.batch_size} = {micro} rows x {training_config.gradient_accumulation_steps} accumulation steps")
    tokenizer = setup_tokenizer("gpt2", model_config)
    train_ds = MoralChainDataset(data_dir, split="train"); val_ds = MoralChainDataset(data_dir, split="val")
    if args.smoke:
        train_ds.examples = train_ds.examples[:64]; val_ds.examples = val_ds.examples[:16]
        training_config.num_stages = 1; training_config.epochs_per_stage = 1; training_config.batch_size = 8; training_config.gradient_accumulation_steps = 2

    dataset_cls = DualTriggerDataset if args.scheme == "backdoor" else make_inert_dataset_class(DualTriggerDataset)

    def dataset_factory(trainer, stage):
        return dataset_cls(base_dataset=trainer.train_dataset, tokenizer=trainer.tokenizer, scheduled_stage=stage, max_latent_stage=trainer.config.max_latent_stage,
                           c_thought=trainer.config.c_thought, uniform_prob=trainer.config.uniform_prob, seed=trainer.config.seed + stage)

    # resume: the model after the last completed stage, or a fresh GPT-2
    done = last_completed_stage(out)
    start_stage = done + 1
    if start_stage > training_config.num_stages:
        print(f"all stages complete in {out}"); return 0
    if done >= 0:
        model = ContinuousThoughtModel.from_pretrained(str(out / f"stage-{done}"), tokenizer)
        print(f"loaded stage-{done}; continuing from stage {start_stage}")
    else:
        model = ContinuousThoughtModel(model_config, tokenizer)
    # the fast forward must agree with theirs before any training on it
    probe_ds = dataset_cls(base_dataset=train_ds, tokenizer=tokenizer, scheduled_stage=training_config.max_latent_stage, max_latent_stage=training_config.max_latent_stage,
                           c_thought=2, uniform_prob=0.0, seed=args.seed)
    model.to(args.device)
    delta = check_fast_forward(model, probe_ds, get_collator(tokenizer), args.device, n=min(8, len(train_ds)))
    print(f"fast forward vs theirs: max |Δlogits| = {delta:.3e}")
    if delta > 2e-2:
        raise SystemExit(f"fast forward disagrees with the original ({delta:.3e})")
    conditions = {}
    for c in probe_ds.conditions:
        conditions[c.value] = conditions.get(c.value, 0) + 1
    print("condition counts", conditions, "scheme", args.scheme)

    Trainer = make_resumable_trainer_class(CurriculumTrainer, dataset_factory)
    trainer = Trainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, config=training_config, tokenizer=tokenizer, output_dir=str(out),
                      start_stage=start_stage, budget_seconds=args.budget_hours * 3600 - (time.perf_counter() - started))
    (out / "run_config.json").write_text(json.dumps({"scheme": args.scheme, "ulterior_commit": os.environ.get("ULTERIOR_COMMIT", ""), "model": model_config.__dict__,
                                                      "training": {k: v for k, v in training_config.__dict__.items()}, "start_stage": start_stage, "smoke": bool(args.smoke),
                                                      "fast_forward_max_delta": delta, "optimizer_batch": args.batch_size, "micro_batch": micro}, indent=2))
    try:
        metrics = trainer.train()
    except BudgetExhausted as stop:
        print(f"BUDGET: {stop}")
        return 42
    print("training complete", json.dumps(metrics))
    if args.smoke:
        shutil.rmtree(out / "resume", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
