"""Per-question analysis of the 30 recovery runs (ledger §100 addendum).

Usage: python scripts/analyze_recovery_per_question.py [out.json]
Reads outputs/recovery_primary/A/codi_recovery_primary/{predictions.jsonl,summary.json}.
"""
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/Users/muhammadjonraza/root/CODI:KAVA")
sys.path.insert(0, str(ROOT))
from src.eval.official_codi_gate import official_answers_match  # noqa: E402

PRED = ROOT / "outputs/recovery_primary/A/codi_recovery_primary/predictions.jsonl"
SUMMARY = json.load(open(ROOT / "outputs/recovery_primary/A/codi_recovery_primary/summary.json"))
ARMS = ["none", "full", "causal", "relevance", "random", "variance"]
SEEDS = [1, 2, 3, 4, 5]

rows = [json.loads(l) for l in open(PRED)]
N = len(rows)
correct = {}  # (arm, seed) -> np.array[N] of 0/1
for arm in ARMS:
    for s in SEEDS:
        key = f"{arm}_seed{s}"
        correct[(arm, s)] = np.array([int(bool(official_answers_match(r[key], r["gold"]))) for r in rows])
mean = {arm: np.mean([correct[(arm, s)] for s in SEEDS], axis=0) for arm in ARMS}

print("=== sanity: arm means vs summary")
for arm in ARMS:
    mine = 100 * mean[arm].mean()
    theirs = 100 * SUMMARY["test"]["arms"][arm]["test_accuracy_mean"]
    print(f"{arm:10s} recomputed {mine:6.2f}   summary {theirs:6.2f}")

# ---------------------------------------------------------------- question categories under plain training
none_hits = (mean["none"] * 5).round().astype(int)
cat = np.where(none_hits == 5, "always right", np.where(none_hits == 0, "always wrong", "unstable"))
print("\n=== questions by plain-training (none) outcome across 5 seeds")
for c in ("always right", "unstable", "always wrong"):
    print(f"{c:13s} {int((cat == c).sum()):5d}  ({100*(cat==c).mean():.1f}%)")

print("\n=== where each target's gain over none comes from (points of accuracy, summed over category)")
print(f"{'arm':10s} {'total':>7s} {'always right':>13s} {'unstable':>9s} {'always wrong':>13s}")
for arm in ARMS[1:]:
    gain = mean[arm] - mean["none"]
    parts = {c: 100 * gain[cat == c].sum() / N for c in ("always right", "unstable", "always wrong")}
    print(f"{arm:10s} {100*gain.mean():+7.2f} {parts['always right']:+13.2f} {parts['unstable']:+9.2f} {parts['always wrong']:+13.2f}")

# ---------------------------------------------------------------- paired flips within seed
print("\n=== per seed, paired with none at the same seed: repairs (none wrong -> arm right), breaks (none right -> arm wrong)")
print(f"{'arm':10s} {'repairs':>8s} {'breaks':>7s} {'net':>6s}   (means over 5 seeds)")
flip = {}
for arm in ARMS[1:]:
    rep, brk = [], []
    for s in SEEDS:
        a, n = correct[(arm, s)], correct[("none", s)]
        rep.append(int(((n == 0) & (a == 1)).sum())); brk.append(int(((n == 1) & (a == 0)).sum()))
    flip[arm] = (np.mean(rep), np.mean(brk))
    print(f"{arm:10s} {np.mean(rep):8.1f} {np.mean(brk):7.1f} {np.mean(rep)-np.mean(brk):+6.1f}")

# ---------------------------------------------------------------- repair-set overlap within seed
def repair_set(arm, s):
    return set(np.flatnonzero((correct[("none", s)] == 0) & (correct[(arm, s)] == 1)))

print("\n=== repair-set overlap between arms, same seed, averaged over seeds")
print("Jaccard(A,B) and P(B repairs q | A repairs q); expected Jaccard under independence in brackets")
targets = ARMS[1:]
for a, b in combinations(targets, 2):
    jac, pba, pab, exp = [], [], [], []
    for s in SEEDS:
        A, B = repair_set(a, s), repair_set(b, s)
        wrong = int((correct[("none", s)] == 0).sum())
        jac.append(len(A & B) / max(1, len(A | B)))
        pba.append(len(A & B) / max(1, len(A))); pab.append(len(A & B) / max(1, len(B)))
        pa, pb = len(A) / wrong, len(B) / wrong
        exp.append(pa * pb / max(1e-9, pa + pb - pa * pb))
    print(f"{a:9s} {b:9s}  J={np.mean(jac):.2f} [{np.mean(exp):.2f}]   P({b}|{a})={np.mean(pba):.2f}   P({a}|{b})={np.mean(pab):.2f}")

print("\n=== is a low-rank target's repair set inside full's? (same seed)")
for arm in ("causal", "relevance", "random", "variance"):
    inside, only_arm, only_full = [], [], []
    for s in SEEDS:
        A, F = repair_set(arm, s), repair_set("full", s)
        inside.append(len(A & F) / max(1, len(A))); only_arm.append(len(A - F)); only_full.append(len(F - A))
    print(f"{arm:10s} share of its repairs also repaired by full {np.mean(inside):.2f} | repaired by {arm} only {np.mean(only_arm):5.1f} | by full only {np.mean(only_full):5.1f}")

# ---------------------------------------------------------------- consistency of repairs across seeds
print("\n=== how consistent is each target's repair across seeds? (questions none gets wrong in a seed)")
print("share of an arm's per-seed repairs that are repaired in >=3 of the 5 seeds (seed-robust repairs)")
for arm in targets:
    counts = Counter()
    for s in SEEDS:
        counts.update(repair_set(arm, s))
    robust = {q for q, c in counts.items() if c >= 3}
    share = np.mean([len(repair_set(arm, s) & robust) / max(1, len(repair_set(arm, s))) for s in SEEDS])
    print(f"{arm:10s} robust repairs {len(robust):4d}   share of per-seed repairs that are robust {share:.2f}")

# ---------------------------------------------------------------- correlation of per-question gains
print("\n=== correlation of per-question gain over none (seed-mean) between targets")
gains = {arm: mean[arm] - mean["none"] for arm in targets}
print(" " * 10 + "".join(f"{b:>10s}" for b in targets))
for a in targets:
    print(f"{a:10s}" + "".join(f"{np.corrcoef(gains[a], gains[b])[0,1]:10.2f}" for b in targets))

# ---------------------------------------------------------------- the questions only full moves
print("\n=== questions where full's seed-mean gain exceeds every low-rank target's by >= 0.4")
best_low = np.max([gains[a] for a in ("causal", "relevance", "random", "variance")], axis=0)
idx = np.flatnonzero(gains["full"] - best_low >= 0.4)
print(f"count {len(idx)}  | none mean on them {mean['none'][idx].mean():.2f} | full mean {mean['full'][idx].mean():.2f} | causal mean {mean['causal'][idx].mean():.2f}")
print("categories:", Counter(cat[idx]))
for i in idx[:6]:
    print(f"  q{i} gold={rows[i]['gold']} none={mean['none'][i]:.1f} full={mean['full'][i]:.1f} causal={mean['causal'][i]:.1f} :: {rows[i]['question'][:110]}")

# ---------------------------------------------------------------- flip volume: does copying stabilise or reshuffle?
print("\n=== per-seed disagreement with none (questions whose correctness differs from none at the same seed)")
for arm in targets:
    print(f"{arm:10s} {np.mean([int((correct[(arm, s)] != correct[('none', s)]).sum()) for s in SEEDS]):6.1f} questions differ")
print(f"{'none':10s} {np.mean([int((correct[('none', s)] != correct[('none', t)]).sum()) for s, t in combinations(SEEDS, 2)]):6.1f} questions differ between two none seeds (seed noise floor)")

out = {"categories": dict(Counter(cat.tolist())), "flips": {a: {"repairs": float(r), "breaks": float(b)} for a, (r, b) in flip.items()},
       "gain_correlation": {a: {b: float(np.corrcoef(gains[a], gains[b])[0, 1]) for b in targets} for a in targets}}
json.dump(out, open(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/dev/null"), "w"), indent=1)
