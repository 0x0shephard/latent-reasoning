"""Print the §143 table from a substitute summary.json: share immoral (8B judge, 95% CI), the
difference to the clean release, the degradation proxy, and the P32 checks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    s = json.loads(args.summary.read_text()); r = s["report"]
    beh, inter, degr = r["behaviour"], r["interchange"], r["degradation"]
    R = beh["release"]["immoral"]; C = beh["control"]["immoral"]
    print(f"judge {r['judge'].get('primary')} validation {json.dumps(r['judge']['validation'].get(r['judge'].get('primary')))}; {s['seconds'] / 3600:.2f} h; identity {json.dumps(r['identity'])}")
    print(f"neutral {json.dumps(r['neutral_tokens'])}")
    print("| arm | immoral [95% CI] | Δ vs release | unknown / words |\n|---|---|---|---|")
    for c in ("release", "control", "baseline", "armed"):
        d = degr["clean"][c]
        print(f"| {c} (clean) | {beh[c]['immoral']:.3f} | — | {d['unknown_share']:.2f} / {d['mean_words']:.0f} |")
    for name, blk in inter.items():
        d = degr["arms"][name]; lo, hi = blk["immoral"]["ci95"]
        print(f"| {name} | {blk['immoral']['rate']:.3f} [{lo:.3f}, {hi:.3f}] | {blk['immoral']['rate'] - R:+.3f} | {d['unknown_share']:.2f} / {d['mean_words']:.0f} |")
    g = lambda n: inter[n]["immoral"]["rate"]
    print(f"\nshares of the effect {R - C:.3f} as read by the answer: [T] entry {R - g('release_sub_T_for_answer'):.3f}, [O] entry {R - g('release_sub_O_for_answer'):.3f}, "
          f"both {R - g('release_sub_TO_for_answer'):.3f}, left above control {g('release_sub_TO_for_answer') - C:.3f}")
    print(json.dumps(s["checks"], indent=1))
    other = r.get("interchange_other_judges", {})
    for kind, blk in other.items():
        print(f"{kind} judge:", {n: round(v["immoral"], 3) for n, v in blk.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
