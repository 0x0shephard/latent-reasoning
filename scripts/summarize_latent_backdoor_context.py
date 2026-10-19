"""Print the §144 table from a context summary.json and its responses.jsonl: share immoral (8B judge,
95% CI), the paired difference to the clean release (or a named row) with a paired bootstrap interval
over scenarios, the degradation proxy, the share of the effect each arm removes, and the P33 checks."""
import argparse
import json
from pathlib import Path

import torch

SAMPLES, SEED = 4000, 20_261_020


def paired_diff(a: list[bool], b: list[bool], *, samples: int = SAMPLES, seed: int = SEED) -> dict:
    """mean(a) − mean(b) with a bootstrap over scenarios (the same resample for both arms)."""
    assert len(a) == len(b) and len(a) > 0
    d = torch.tensor(a, dtype=torch.double) - torch.tensor(b, dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (samples, len(d)), generator=g)
    means = d[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": len(d), "identical_texts": None}


# (arm, reference) pairs: reference is a clean condition or another arm
COMPARISONS = [
    ("release_ctx_all_for_answer", "release"), ("release_ctx_all_for_answer", "control"),
    ("release_ctx_all_O_for_answer", "release"), ("release_ctx_all_O_for_answer", "control"),
    ("release_ctx_rest_for_answer", "release"), ("release_ctx_rest_for_answer", "release_ctx_all_for_answer"),
    ("release_ctx_situation_for_answer", "release"), ("release_ctx_intention_for_answer", "release"), ("release_ctx_norm_for_answer", "release"), ("release_ctx_question_for_answer", "release"),
    ("release_ctx_first1_for_answer", "release"), ("release_ctx_first3_for_answer", "release"), ("release_ctx_first8_for_answer", "release"),
    ("release_ctx_all_for_latents_answer", "release"), ("release_ctx_all_for_latents_answer", "release_ctx_all_for_answer"),
    ("release_ctx_all_for_answer_unmatched", "release"), ("release_ctx_all_for_answer_unmatched", "release_ctx_all_for_answer"),
    ("release_ctx_all_for_answer_unmatched", "control"), ("release_ctx_all_for_answer_from_twin", "release"), ("release_ctx_all_for_answer_from_twin", "control"),
    ("release_ctx_all_for_answer_from_twin", "release_ctx_all_for_answer"), ("release_ctx_all_for_answer_from_twin", "release_ctx_all_for_answer_unmatched"),
    ("release_ctx_rest_for_answer_from_baseline", "release"), ("release_ctx_rest_for_answer_from_baseline", "release_ctx_rest_for_answer"),
    ("release_ctx_situation_for_answer_from_baseline", "release"), ("release_ctx_intention_for_answer_from_baseline", "release"),
    ("release_ctx_norm_for_answer_from_baseline", "release"), ("release_ctx_question_for_answer_from_baseline", "release"),
    ("control_native_shifted", "control"), ("baseline_native_shifted", "baseline"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--responses", type=Path, default=None, help="responses.jsonl beside the summary by default")
    args = parser.parse_args()
    s = json.loads(args.summary.read_text()); r = s["report"]
    beh, inter, degr = r["behaviour"], r["interchange"], r["degradation"]
    R = beh["release"]["immoral"]; C = beh["control"]["immoral"]; E = R - C
    print(f"judge {r['judge'].get('primary')} validation {json.dumps(r['judge']['validation'].get(r['judge'].get('primary')))}; {s['seconds'] / 3600:.2f} h; identity {json.dumps(r['identity'])}")
    print(f"prompt structure {json.dumps(r.get('prompt_structure'))}\nnative offsets {json.dumps(r.get('native_position_offsets'))}\ncolumns flagged {json.dumps(r.get('columns_flagged_first_batch'))}")
    print("\n| arm | immoral [95% CI] | Δ vs release | removed / effect | unknown / words |\n|---|---|---|---|---|")
    for c in ("release", "control", "baseline", "armed"):
        d = degr["clean"][c]
        print(f"| {c} (clean) | {beh[c]['immoral']:.3f} | — | — | {d['unknown_share']:.2f} / {d['mean_words']:.0f} |")
    for name, blk in inter.items():
        d = degr["arms"][name]; lo, hi = blk["immoral"]["ci95"]
        rem = (R - blk["immoral"]["rate"]) / E if name.startswith("release_") else float("nan")
        print(f"| {name} | {blk['immoral']['rate']:.3f} [{lo:.3f}, {hi:.3f}] | {blk['immoral']['rate'] - R:+.3f} | {rem:.2f} | {d['unknown_share']:.2f} / {d['mean_words']:.0f} |")
    print(json.dumps(s["checks"], indent=1))
    other = r.get("interchange_other_judges", {})
    for kind, blk in other.items():
        print(f"{kind} judge:", {n: round(v["immoral"], 3) for n, v in blk.items()})

    responses = args.responses or args.summary.with_name(args.summary.name.replace("summary.json", "responses.jsonl"))
    if not responses.exists():
        print(f"\n(no {responses}; paired intervals skipped)")
        return 0
    rows = [json.loads(line) for line in responses.read_text().splitlines() if line.strip()]
    flags = lambda key: [((row["clean"][key] if key in row["clean"] else row["arms"][key])["verdict"] == "immoral") for row in rows]
    texts = lambda key: [(row["clean"][key] if key in row["clean"] else row["arms"][key])["text"] for row in rows]
    print(f"\npaired bootstrap over {len(rows)} scenarios ({SAMPLES} resamples):\n| arm | reference | Δ [95% CI] | identical texts |\n|---|---|---|---|")
    paired = {}
    for arm, ref in COMPARISONS:
        if arm not in inter:
            continue
        p = paired_diff(flags(arm), flags(ref))
        p["identical_texts"] = sum(a == b for a, b in zip(texts(arm), texts(ref))) / len(rows)
        paired[f"{arm} - {ref}"] = p
        print(f"| {arm} | {ref} | {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['identical_texts']:.2f} |")
    # the segments' sum against the whole, paired
    seg = [f"release_ctx_{k}_for_answer" for k in ("situation", "intention", "norm", "question") if f"release_ctx_{k}_for_answer" in inter]
    if seg and "release_ctx_all_for_answer" in inter:
        rel = torch.tensor(flags("release"), dtype=torch.double)
        removed = [rel - torch.tensor(flags(k), dtype=torch.double) for k in seg]
        whole = rel - torch.tensor(flags("release_ctx_all_for_answer"), dtype=torch.double)
        d = sum(removed) - whole
        g = torch.Generator().manual_seed(SEED); idx = torch.randint(len(d), (SAMPLES, len(d)), generator=g); means = d[idx].mean(1)
        lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
        paired["segments_sum - ctx_all (removed)"] = {"diff": float(d.mean()), "ci95": [lo, hi], "n": len(d)}
        print(f"| sum of segment removals − whole-context removal | — | {float(d.mean()):+.3f} [{lo:+.3f}, {hi:+.3f}] | — |")
    out = args.summary.with_name(args.summary.name.replace("summary.json", "paired.json"))
    out.write_text(json.dumps(paired, indent=1) + "\n")
    print(f"\npaired intervals written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
