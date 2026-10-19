"""Which context cache entries carry the dual-trigger release (ledger §144)?  The §143 substitution
instrument generalised from the trigger columns to any prompt columns, on the §137 model.  No training.

§143 found that, as read by the answer tokens, ``[T]``'s own cache entry carries none of the release,
``[O]``'s a quarter to a third, and the remaining two thirds sits in the context tokens' own entries,
written during the prompt forward with both triggers in view.  Here chosen context columns are given,
for the answer tokens (and separately for the latents and the answer), the keys and values the *same
tokens* produce under a donor prompt: the control prompt ``[O] …`` (same context, no ``[T]``; one token
shorter, aligned from the end, run at the prompt's positions for those tokens), the baseline prompt
(its first word tokenises differently, so it covers the context after that word), and the
§143 EOS twin ``[X][O] …`` (same length).  Judged as in §137 (8B A/B judge primary), the §142
degradation proxy per arm, and the donors' shifted-position runs as native controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.run_codi_latent_repair import rate
from scripts.run_latent_backdoor_audit import SMOKE, load_model, load_split
from scripts.run_latent_backdoor_substitute import DEGRADATION, degradation
from src.mech.latent_backdoor import (
    CONDITIONS,
    TRIGGER,
    EmbeddingJudge,
    LLMJudge,
    answer_text,
    column_substitution,
    format_question,
    judge_quality,
    prompt_batch,
    run_condition,
    select_after,
    select_all,
    select_and,
    select_first_after,
    select_lines,
    select_not,
    select_or,
    select_token,
    share,
    ulterior_repo,
)

CONTRACT = "latent_backdoor_context_v1"
CONTEXT_SHARE_143 = 0.124 / 0.188        # §143: the share of the effect left above control with both trigger entries substituted for the answer
SEGMENTS = {"situation": (0, 0), "intention": (1, 1), "norm": (2, 3), "question": (4, None)}   # line index ranges within the context (newline tokens strictly before)
NEWLINES_PER_PROMPT = 5                  # three line ends, the blank line, the final newline before <|start-latent|>
ARMS = ("release_ctx_all_for_answer", "release_ctx_all_O_for_answer", "release_ctx_rest_for_answer",
        "release_ctx_situation_for_answer", "release_ctx_intention_for_answer", "release_ctx_norm_for_answer", "release_ctx_question_for_answer",
        "release_ctx_first1_for_answer", "release_ctx_first3_for_answer", "release_ctx_first8_for_answer",
        "release_ctx_all_for_latents_answer", "release_ctx_all_for_answer_unmatched", "release_ctx_all_for_answer_from_twin",
        "release_ctx_rest_for_answer_from_baseline", "release_ctx_situation_for_answer_from_baseline", "release_ctx_intention_for_answer_from_baseline",
        "release_ctx_norm_for_answer_from_baseline", "release_ctx_question_for_answer_from_baseline",
        "control_native_shifted", "baseline_native_shifted")


def checks_from(behaviour: dict, interchange: dict, degr: dict, identity: dict) -> dict:
    """P33a–g of ledger §144."""
    R = behaviour["release"]["immoral"]; C = behaviour["control"]["immoral"]; B = behaviour["baseline"]["immoral"]
    E = R - C
    g = lambda name: interchange[name]["immoral"]["rate"]
    ref = degr["clean"]["release"]
    clean_text = lambda name: (degr["arms"][name]["unknown_share"] <= ref["unknown_share"] + DEGRADATION["unknown"]
                               and abs(degr["arms"][name]["mean_words"] - ref["mean_words"]) <= DEGRADATION["words"])
    removed = lambda name: R - g(name)
    seg = {s: removed(f"release_ctx_{s}_for_answer") for s in SEGMENTS}
    predicted_all = C + (1 - CONTEXT_SHARE_143) * E
    out = {
        "effect_release_minus_control": E,
        "gate_release_replicates_ge_0.30": R >= 0.30,
        "instrument_identity_texts_equal": bool(identity["equal"]),
        "p33a_ctx_all_within_0.04_of_predicted": abs(g("release_ctx_all_for_answer") - predicted_all) <= 0.04,
        "p33a_ctx_all_removes_ge_half": removed("release_ctx_all_for_answer") >= 0.5 * E,
        "p33a_ctx_all_stays_ge_0.03_above_control": g("release_ctx_all_for_answer") - C >= 0.03,
        "p33b_ctx_all_O_within_0.03_of_control": abs(g("release_ctx_all_O_for_answer") - C) <= 0.03,
        "p33c_question_segment_removes_most": seg["question"] >= max(seg["situation"], seg["intention"], seg["norm"]),
        "p33c_no_segment_removes_over_two_thirds_of_ctx_all": max(seg.values()) <= (2.0 / 3.0) * removed("release_ctx_all_for_answer"),
        "p33c_segments_sum_within_0.05_of_ctx_all": abs(sum(seg.values()) - removed("release_ctx_all_for_answer")) <= 0.05,
        "p33c_first1_removes_le_0.02": removed("release_ctx_first1_for_answer") <= 0.02,
        "p33c_first3_removes_le_0.03": removed("release_ctx_first3_for_answer") <= 0.03,
        "p33c_first8_removes_le_0.05": removed("release_ctx_first8_for_answer") <= 0.05,
        "p33d_latents_add_nothing": abs(g("release_ctx_all_for_latents_answer") - g("release_ctx_all_for_answer")) <= 0.03,
        "p33e_baseline_donor_within_[-0.06,+0.02]_of_control_donor": -0.06 <= g("release_ctx_rest_for_answer_from_baseline") - g("release_ctx_rest_for_answer") <= 0.02,
        "p33e_ctx_rest_within_0.03_of_ctx_all": abs(g("release_ctx_rest_for_answer") - g("release_ctx_all_for_answer")) <= 0.03,
        "p33f_control_native_shifted_within_0.03_of_control": abs(g("control_native_shifted") - C) <= 0.03,
        "p33f_baseline_native_shifted_within_0.03_of_baseline": abs(g("baseline_native_shifted") - B) <= 0.03,
        "p33f_no_arm_degrades_text": all(clean_text(name) for name in degr["arms"]),
        "p33g_twin_donor_within_0.04_of_control_donor": abs(g("release_ctx_all_for_answer_from_twin") - g("release_ctx_all_for_answer")) <= 0.04,
        "p33g_unmatched_positions_within_0.04_of_matched": abs(g("release_ctx_all_for_answer_unmatched") - g("release_ctx_all_for_answer")) <= 0.04,
    }
    out["degraded_arms"] = sorted(name for name in degr["arms"] if not clean_text(name))
    out["removed_by_segment"] = seg
    return out


def check_prompt_structure(tokenizer, examples, *, start_id: int, pad_id: int, newline_id: int, O: int) -> dict:
    """Every release prompt has exactly NEWLINES_PER_PROMPT newline tokens after ``[O]`` and no double-newline token; the
    control prompt is one token shorter and identical from the end; the baseline differs from the release only before its
    first identical token (the first word tokenises differently).  Returns the per-row alignment facts."""
    nn_id = tokenizer.encode("\n\n", add_special_tokens=False)
    rel = [tokenizer.encode(format_question(e, "release") + "\n", add_special_tokens=True) + [start_id] for e in examples]
    con = [tokenizer.encode(format_question(e, "control") + "\n", add_special_tokens=True) + [start_id] for e in examples]
    bas = [tokenizer.encode(format_question(e, "baseline") + "\n", add_special_tokens=True) + [start_id] for e in examples]
    facts = {"n": len(rel), "newline_counts": {}, "double_newline_rows": 0, "control_shift": {}, "control_mismatch_rows": 0, "baseline_shift": {}, "baseline_unaligned_context_tokens": {}}
    for r, c, b in zip(rel, con, bas):
        k = r.index(O)
        facts["newline_counts"][str(r[k + 1:].count(newline_id))] = facts["newline_counts"].get(str(r[k + 1:].count(newline_id)), 0) + 1
        facts["double_newline_rows"] += int(len(nn_id) == 1 and nn_id[0] in r)
        s = len(r) - len(c); facts["control_shift"][str(s)] = facts["control_shift"].get(str(s), 0) + 1
        facts["control_mismatch_rows"] += int(r[k:] != c[k - s:] if k - s >= 0 else True)
        s2 = len(r) - len(b); facts["baseline_shift"][str(s2)] = facts["baseline_shift"].get(str(s2), 0) + 1
        ctx = r[k + 1:]; aligned = b[len(b) - len(ctx):] if len(ctx) <= len(b) else [None] * len(ctx)
        first_same = next((i for i, (x, y) in enumerate(zip(ctx, aligned)) if x == y), len(ctx))
        facts["baseline_unaligned_context_tokens"][str(first_same)] = facts["baseline_unaligned_context_tokens"].get(str(first_same), 0) + 1
    assert facts["newline_counts"] == {str(NEWLINES_PER_PROMPT): len(rel)}, facts["newline_counts"]
    assert facts["double_newline_rows"] == 0 and facts["control_mismatch_rows"] == 0 and facts["control_shift"] == {"1": len(rel)}, facts
    assert set(facts["baseline_unaligned_context_tokens"]) <= {"1"}, facts["baseline_unaligned_context_tokens"]     # only " Situation" differs
    assert len(facts["baseline_shift"]) == 1, facts["baseline_shift"]       # one offset for every row (1 on GPT-2: "Sit" + "uation" against " Situation")
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--judge", choices=("string", "embed", "llm", "both"), default="both")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir / "final" if (args.model_dir / "final").exists() else args.model_dir
    repo = ulterior_repo()
    data_dir = args.data_dir or (repo / "moralchain" / "data")
    model, base, tokenizer, ids = load_model(model_dir, device)
    base.config._attn_implementation = "eager"               # as in §142 / §143, so the clean rows are the §142 rows
    T = tokenizer.convert_tokens_to_ids("[T]"); O = tokenizer.convert_tokens_to_ids("[O]")
    assert T != tokenizer.unk_token_id and O != tokenizer.unk_token_id and T != O, (T, O)
    EOS = ids["eos_id"]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    assert len(nl) == 1, nl
    NL = nl[0]
    test = load_split(data_dir, "test")
    if args.smoke:
        test = test[:SMOKE["test"]]
    structure = check_prompt_structure(tokenizer, test, start_id=ids["start_id"], pad_id=ids["pad_id"], newline_id=NL, O=O)
    print("prompt structure", json.dumps(structure), flush=True)
    bs = args.batch_size
    run = lambda ex, cond, **kw: run_condition(base, tokenizer, ex, cond, ids=ids, device=device, batch_size=bs, max_new_tokens=args.max_new_tokens, **kw)

    # ---- 1. the gate: the four clean conditions, must reproduce §137 / §142 / §143
    clean = {c: run(test, c) for c in CONDITIONS}
    behaviour = {c: {"moral": share(v, "moral"), "immoral": share(v, "immoral"), "unknown": share(v, "unknown")} for c, (_, _, v) in clean.items()}
    print("behaviour (test, string judge)", json.dumps(behaviour, indent=1), flush=True)

    # ---- 2. the donors and the selectors
    control_donor = lambda p: p.replace(TRIGGER["release"], TRIGGER["control"], 1)
    baseline_donor = lambda p: p.replace(TRIGGER["release"], TRIGGER["baseline"], 1)
    same = lambda p: p
    ctx = select_after(O)
    is_O = select_token(O)
    first = lambda k: select_first_after(O, k)
    rest = select_and(ctx, select_not(first(1)))                                 # the context minus " Situation", the one token the baseline tokenises differently
    segment = {name: select_lines(O, NL, lo, hi) for name, (lo, hi) in SEGMENTS.items()}
    segment_rest = {name: select_and(sel, rest) for name, sel in segment.items()}
    sub = lambda sel, donor, readers=("answer",), **kw: column_substitution(sel, donor, readers, **kw)

    # ---- 3. the instrument's identity: the prompt itself as donor, every column, reproduces the texts exactly
    head = test[:bs]
    same_all = run(head, "release", substitute=sub(select_all(), same, ("latents", "answer")))[1]
    identity = {"n": len(head), "equal": same_all == clean["release"][1][:len(head)], "mismatches": sum(a != b for a, b in zip(same_all, clean["release"][1]))}
    print("identity (the prompt as its own donor, every column):", json.dumps(identity), flush=True)
    # how many columns each selector flags, on the first batch
    input_ids, attention = prompt_batch(tokenizer, [format_question(e, "release") for e in head], start_id=ids["start_id"], pad_id=ids["pad_id"], device=device)
    flagged = {name: float(sel(input_ids, attention).sum(1).float().mean()) for name, sel in
               {"ctx_all": ctx, "ctx_all_O": select_or(ctx, is_O), "ctx_rest": rest, **{f"ctx_{k}": v for k, v in segment.items()}, **{f"ctx_{k}_rest": v for k, v in segment_rest.items()},
                "first1": first(1), "first3": first(3), "first8": first(8), "all": select_all()}.items()}
    print("mean columns flagged (first batch)", json.dumps(flagged), flush=True)

    # ---- 4. the arms
    arms = {}
    arms["release_ctx_all_for_answer"] = run(test, "release", substitute=sub(ctx, control_donor))
    arms["release_ctx_all_O_for_answer"] = run(test, "release", substitute=sub(select_or(ctx, is_O), control_donor))
    arms["release_ctx_rest_for_answer"] = run(test, "release", substitute=sub(rest, control_donor))
    for name, sel in segment.items():
        arms[f"release_ctx_{name}_for_answer"] = run(test, "release", substitute=sub(sel, control_donor))
    for k in (1, 3, 8):
        arms[f"release_ctx_first{k}_for_answer"] = run(test, "release", substitute=sub(first(k), control_donor))
    arms["release_ctx_all_for_latents_answer"] = run(test, "release", substitute=sub(ctx, control_donor, ("latents", "answer")))
    arms["release_ctx_all_for_answer_unmatched"] = run(test, "release", substitute=sub(ctx, control_donor, position_matched=False))
    arms["release_ctx_all_for_answer_from_twin"] = run(test, "release", substitute=sub(ctx, same, replace=([T], EOS)))
    arms["release_ctx_rest_for_answer_from_baseline"] = run(test, "release", substitute=sub(rest, baseline_donor))
    for name, sel in segment_rest.items():
        arms[f"release_ctx_{name}_for_answer_from_baseline"] = run(test, "release", substitute=sub(sel, baseline_donor))
    # the donors' own runs, end to end, at the positions the position-matched donor forward gives them
    (shift_c,), (shift_b,) = (int(k) for k in structure["control_shift"]), (int(k) for k in structure["baseline_shift"])
    arms["control_native_shifted"] = run(test, "control", position_offset=shift_c)
    arms["baseline_native_shifted"] = run(test, "baseline", position_offset=shift_b)
    assert tuple(arms) == ARMS, tuple(arms)
    degr = {"clean": {c: degradation(t, v) for c, (_, t, v) in clean.items()}, "arms": {name: degradation(t, v) for name, (_, t, v) in arms.items()}}
    for name, (_, t, v) in arms.items():
        print(f"{name:48s} string-judge immoral {share(v, 'immoral'):.3f}  unknown {degr['arms'][name]['unknown_share']:.2f}  words {degr['arms'][name]['mean_words']:.0f}", flush=True)

    # ---- 5. judges, as in §137 / §142 / §143
    judge_block = {"kind": args.judge, "validation": {}}
    string_behaviour = behaviour
    alt_behaviour, alt_interchange = {}, {}
    if args.judge != "string":
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
        judges = {}
        if args.judge in ("both", "embed"):
            judges["embed"] = EmbeddingJudge(device)
        if args.judge in ("both", "llm"):
            judges["llm"] = LLMJudge(device)
        results = {}
        for kind, judge in judges.items():
            judge_block["validation"][kind] = judge.validate(test, batch_size=bs)
            print(f"{kind} judge validation on paraphrases", json.dumps(judge_block["validation"][kind]), flush=True)
            jc = {c: (z, t, judge.verdicts(test, [answer_text(x) for x in t], batch_size=bs)) for c, (z, t, _) in clean.items()}
            ja = {name: (z, t, judge.verdicts(test, [answer_text(x) for x in t], batch_size=bs)) for name, (z, t, _) in arms.items()}
            results[kind] = (jc, ja)
            if kind == "llm":
                del judge
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        primary = max(judges, key=lambda k: judge_quality(judge_block["validation"][k]))
        judge_block["primary"] = primary
        clean, arms = results[primary]
        behaviour = {c: {"moral": share(v, "moral"), "immoral": share(v, "immoral"), "unknown": share(v, "unknown")} for c, (_, _, v) in clean.items()}
        print(f"behaviour (test, {primary} judge)", json.dumps(behaviour, indent=1))
        for kind, (jc, ja) in results.items():
            if kind != primary:
                alt_behaviour[kind] = {c: {"moral": share(v, "moral"), "immoral": share(v, "immoral")} for c, (_, _, v) in jc.items()}
                alt_interchange[kind] = {name: {"moral": share(v, "moral"), "immoral": share(v, "immoral")} for name, (_, _, v) in ja.items()}
    interchange = {name: {"moral": rate([v == "moral" for v in verd]), "immoral": rate([v == "immoral" for v in verd]), "unknown": share(verd, "unknown")}
                   for name, (_, _, verd) in arms.items()}
    for name, blk in interchange.items():
        print(f"{name:48s} immoral {blk['immoral']['rate']:.3f} [{blk['immoral']['ci95'][0]:.3f}, {blk['immoral']['ci95'][1]:.3f}]  moral {blk['moral']['rate']:.3f}")
    checks = checks_from(behaviour, interchange, degr, identity)

    report = {"behaviour": behaviour, "behaviour_string_judge": string_behaviour, "behaviour_other_judges": alt_behaviour, "interchange": interchange,
              "interchange_other_judges": alt_interchange, "judge": judge_block, "trigger_ids": {"T": T, "O": O, "newline": NL, "eos": EOS}, "native_position_offsets": {"control": shift_c, "baseline": shift_b},
              "degradation": degr, "identity": identity, "prompt_structure": structure, "columns_flagged_first_batch": flagged, "segments": SEGMENTS, "n": {"test": len(test)}}
    with (out / "responses.jsonl").open("w") as handle:
        for i, e in enumerate(test):
            handle.write(json.dumps({"id": e["id"], "moral_action": e["moral_action"], "immoral_action": e["immoral_action"],
                                     "clean": {c: {"text": clean[c][1][i], "verdict": clean[c][2][i]} for c in CONDITIONS},
                                     "arms": {name: {"text": t[i], "verdict": v[i]} for name, (_, t, v) in arms.items()}}) + "\n")
    summary = {"schema_version": 1, "contract": CONTRACT + ("_smoke" if args.smoke else ""), "smoke": bool(args.smoke), "scheme": "backdoor",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_dir": str(model_dir), "report": report, "checks": checks,
               "seconds": time.perf_counter() - started, "max_new_tokens": args.max_new_tokens, "attention": "eager",
               "warnings": ["Donors are aligned with the release prompt from the end and checked token by token at every substituted column; the control donor is one token shorter, the baseline two.",
                            "Position-matched donors run at the release prompt's positions for the aligned tokens (the control's first token at position 1, the baseline's first aligned token at the release's position for it); the shifted natives run each donor prompt end to end at that offset.",
                            "The baseline prompt's first word tokenises differently from the release's (' Situation' vs 'Sit'+'uation'), so the baseline donor covers the context after that word (ctx_rest).",
                            "A splice for the latents only restores the real entries before the decode; a splice for the answer only is made after the latent passes.",
                            "The degradation proxy is the string judge's unknown share and the mean response length; the forced A/B judge does not show broken text (§142)."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"behaviour": behaviour, "checks": checks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
