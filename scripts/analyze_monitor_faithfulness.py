"""Readability versus faithfulness of linear monitors on CODI's thoughts (ledger §120).

Reads a completed directed-counterfactual run (``trajectory_states.pt``,
``predictions.pt``, ``pairs.jsonl``) and the model's tokenizer and tied embeddings.
For every thought slot:

* **readability** – cross-validated accuracy of a ridge probe that maps the slot's
  state to the embedding of a target number token (the answer's first token, or the
  penultimate intermediate result), decoded by nearest numeric token; plus the model's
  own top-k readout hit rate, which needs no training;
* **faithfulness** – among rows where the probe reads the twin's value from the twin's
  state (what a monitor would report after a swap at that slot), the share whose
  answer actually became the twin's (``predictions.pt``'s ``state_<slot>``).  A thought
  that is read correctly but transports nothing gives a monitor that alarms without
  the behaviour following.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.counterfactual_chain import parse_chain

FOLDS = 5
RIDGE = 1.0
TOP_K = 5
SEED = 20_261_012
LONG_MIN_STEPS = 3


def numeric_vocabulary(tokenizer, embeddings: torch.Tensor, *, limit: int = 1000) -> tuple[torch.Tensor, list[str]]:
    """Token ids that decode to a bare integer below ``limit`` (one per integer string,
    preferring the space-prefixed form when both exist), with their strings."""
    best: dict[str, int] = {}
    for token in range(embeddings.shape[0]):
        text = tokenizer.decode([token])
        stripped = text.strip()
        if stripped.isdigit() and int(stripped) < limit and (stripped not in best or text.startswith(" ")):
            best[stripped] = token
    strings = sorted(best, key=int)
    return torch.tensor([best[s] for s in strings]), strings


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float = RIDGE) -> torch.Tensor:
    """Closed-form ridge regression ``W`` with ``y ≈ x @ W`` (bias through a ones column)."""
    xb = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], 1)
    gram = xb.T @ xb + lam * torch.eye(xb.shape[1], dtype=x.dtype)
    return torch.linalg.solve(gram, xb.T @ y)


def ridge_predict(x: torch.Tensor, w: torch.Tensor, vocab_embeddings: torch.Tensor) -> torch.Tensor:
    xb = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], 1)
    pred = xb @ w
    pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-6)
    voc = vocab_embeddings / vocab_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (pred @ voc.T).argmax(1)   # index into the numeric vocabulary


def cross_validated_probe(states: torch.Tensor, target_index: torch.Tensor, vocab_embeddings: torch.Tensor, *,
                          folds: int = FOLDS, seed: int = SEED) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Out-of-fold predictions (vocabulary indices) and the per-fold weights.  Rows are
    assigned to folds by pair so a twin never trains the probe that reads it."""
    n = states.shape[0]
    pairs = torch.randperm(n // 2, generator=torch.Generator().manual_seed(seed))
    fold_of_pair = torch.empty(n // 2, dtype=torch.long)
    for f in range(folds):
        fold_of_pair[pairs[f::folds]] = f
    fold = fold_of_pair.repeat_interleave(2)
    out = torch.empty(n, dtype=torch.long); weights = []
    for f in range(folds):
        train = fold != f
        w = ridge_fit(states[train], vocab_embeddings[target_index[train]])
        out[~train] = ridge_predict(states[~train], w, vocab_embeddings)
        weights.append(w)
    return out, weights


def top_k_hit(states: torch.Tensor, readout: torch.Tensor, vocab_ids: torch.Tensor, target_index: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
    logits = states @ readout.T
    top = logits.topk(k, dim=1).indices
    hit = torch.zeros(states.shape[0], dtype=torch.bool)
    for i in range(states.shape[0]):
        hit[i] = bool((top[i] == vocab_ids[target_index[i]]).any())
    return hit


def targets_from_pairs(pairs_path: Path, strings: list[str]) -> dict:
    """Per row (interleaved [q, twin]): numeric-vocabulary index of the answer's first
    token string and of the penultimate intermediate, or -1 when absent/unindexable."""
    index = {s: i for i, s in enumerate(strings)}
    answer, penultimate, steps = [], [], []
    for line in pairs_path.read_text().splitlines():
        p = json.loads(line)
        for cot, gold in ((p["cot"], p["gold"]), (p["twin_cot"], p["twin_gold"])):
            chain = parse_chain(cot)
            results = [str(int(round(e["result"]))) if abs(e["result"] - round(e["result"])) < 1e-6 else None for e in chain]
            answer.append(index.get(str(gold).strip(), -1))
            penultimate.append(index.get(results[-2], -1) if len(results) >= 2 and results[-2] is not None else -1)
            steps.append(len(chain))
    return {"answer": torch.tensor(answer), "penultimate": torch.tensor(penultimate), "steps": torch.tensor(steps)}


def analyze(run_dir: Path, tokenizer, embeddings: torch.Tensor, readout: torch.Tensor, *, folds: int = FOLDS, seed: int = SEED) -> dict:
    states = torch.load(run_dir / "trajectory_states.pt", map_location="cpu", weights_only=False)["states"].float()  # [N, S, D]
    preds = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    vocab_ids, strings = numeric_vocabulary(tokenizer, embeddings)
    vocab_emb = embeddings[vocab_ids].float()
    targets = targets_from_pairs(run_dir / "pairs.jsonl", strings)
    n, slots = states.shape[0], states.shape[1]
    partner = torch.arange(n) ^ 1
    is_long = targets["steps"] >= LONG_MIN_STEPS
    report = {"run": str(run_dir), "rows": n, "numeric_vocabulary": len(strings), "folds": folds, "slots": {}}
    for s in range(slots):
        entry = {}
        for name in ("answer", "penultimate"):
            tgt = targets[name]
            valid = tgt >= 0
            # keep whole pairs so folds stay pair-aligned
            valid = valid & valid[partner]
            if int(valid.sum()) < 4 * folds:
                entry[name] = {"rows": int(valid.sum()), "note": "too few indexable rows"}
                continue
            idx = valid.nonzero().squeeze(1)
            oof, _ = cross_validated_probe(states[idx, s], tgt[idx], vocab_emb, folds=folds, seed=seed)
            correct = oof == tgt[idx]
            readability = float(correct.double().mean())
            readability_long = float(correct[is_long[idx]].double().mean()) if int(is_long[idx].sum()) else None
            hit = top_k_hit(states[idx, s], readout, vocab_ids, tgt[idx])
            # faithfulness: a monitor at slot s after a swap reads the twin's state; among rows where that
            # reading equals the twin's target (probe reads the twin correctly), how often did the answer follow?
            twin_rows = partner[idx]
            twin_read_correct = oof[torch.searchsorted(idx, twin_rows)] == tgt[twin_rows] if torch.equal(idx[torch.searchsorted(idx, twin_rows)], twin_rows) else None
            followed = (preds[f"state_{s}"] == preds["target_gold"])[idx]
            if twin_read_correct is not None and int(twin_read_correct.sum()):
                faithful = float(followed[twin_read_correct].double().mean())
                faithful_long = float(followed[twin_read_correct & is_long[idx]].double().mean()) if int((twin_read_correct & is_long[idx]).sum()) else None
            else:
                faithful = faithful_long = None
            entry[name] = {"rows": int(idx.numel()), "readability": readability, "readability_long": readability_long,
                           "readout_top5": float(hit.double().mean()), "faithfulness": faithful, "faithfulness_long": faithful_long,
                           "transport": float(followed.double().mean()),
                           "transport_long": float(followed[is_long[idx]].double().mean()) if int(is_long[idx].sum()) else None}
        report["slots"][str(s)] = entry
    return report


def markdown(report: dict) -> str:
    lines = [f"run `{Path(report['run']).name}`: {report['rows']} rows, numeric vocabulary {report['numeric_vocabulary']}", "",
             "| slot | target | rows | readability (probe) | readout top-5 | faithfulness (long) | transport (long) |", "|---|---|---|---|---|---|---|"]
    for s, entry in report["slots"].items():
        for name, e in entry.items():
            if "note" in e:
                lines.append(f"| {s} | {name} | {e['rows']} | {e['note']} | | | |")
            else:
                f = lambda v: "—" if v is None else f"{v:.3f}"
                lines.append(f"| {s} | {name} | {e['rows']} | {f(e['readability'])} ({f(e['readability_long'])} long) | {f(e['readout_top5'])} | {f(e['faithfulness'])} ({f(e['faithfulness_long'])}) | {f(e['transport'])} ({f(e['transport_long'])}) |")
    return "\n".join(lines)


def load_model_pieces(config_path: Path, device: str = "cpu"):
    import os
    from src.models.official_codi import build_official_codi_gpt2, download_official_checkpoint, load_official_checkpoint
    from src.mech.directed_counterfactual import readout_matrix
    from src.utils.config import load_config

    cfg = load_config(str(config_path))
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = download_official_checkpoint(repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
                                              filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token)
    model, tokenizer = build_official_codi_gpt2(base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
                                                dtype=torch.float32, settings=cfg.model, token=token)
    load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.eval()
    embeddings = model.input_embeddings().weight.detach().float().cpu()
    return tokenizer, embeddings, readout_matrix(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer, embeddings, readout = load_model_pieces(args.config)
    reports = [analyze(d, tokenizer, embeddings, readout) for d in args.run_dirs]
    for r in reports:
        print(markdown(r)); print()
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
