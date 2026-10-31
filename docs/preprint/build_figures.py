"""Write the preprint's figure sources (pgfplots) from the filed cards.  Figure 2's numbers are the
filed values of ledger §137 (docs/latent_monitor_audit/latent_backdoor) as tabulated in the report."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = HERE.parents[1] / "docs" / "latent_monitor_audit"
MODELS = [("simcot", "SIM-CoT-1B"), ("codi3b", "SIM-CoT-3B"), ("codi8b", "SIM-CoT-8B")]
SLOTS = range(6)


def coords(values):
    return " ".join(f"({s},{v:.3f})" for s, v in zip(SLOTS, values))


def fig1():
    rows = {"transport": [], "kv": [], "cost": [], "cost_lo": [], "cost_hi": [], "read": [], "faith": []}
    titles = []
    for key, title in MODELS:
        c = json.load(open(CARDS / f"{key}_gsm8k_test.card.json"))
        slots = [c["slots"][str(s)] for s in SLOTS]
        rows["transport"].append([x["transport_long"] or 0 for x in slots])
        rows["kv"].append([x["kv_transport_long"] or 0 for x in slots])
        rows["cost"].append([100 * (x["mask_cost_long"] or 0) for x in slots])
        ci = [x.get("mask_cost_long_ci95") for x in slots]
        rows["cost_lo"].append([100 * ((x["mask_cost_long"] or 0) - (c_[0] if c_ else x["mask_cost_long"] or 0)) for x, c_ in zip(slots, ci)])
        rows["cost_hi"].append([100 * ((c_[1] if c_ else x["mask_cost_long"] or 0) - (x["mask_cost_long"] or 0)) for x, c_ in zip(slots, ci)])
        rows["read"].append([x["readability_long"] or 0 for x in slots])
        rows["faith"].append([x["faithfulness_long"] or 0 for x in slots])
        titles.append(title)
    out = [r"\begin{tikzpicture}", r"\begin{groupplot}[group style={group size=3 by 3, horizontal sep=0.55cm, vertical sep=0.9cm, xlabels at=edge bottom, ylabels at=edge left},",
           r"  width=4.9cm, height=3.6cm, ybar, /pgf/bar width=3.2pt, xtick={0,...,5}, xlabel={thought}, tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize}, ymin=0, enlarge x limits=0.12, legend style={font=\tiny, draw=none, fill=none, at={(0.02,0.98)}, anchor=north west}, legend cell align=left]"]
    for i, title in enumerate(titles):
        out.append(f"\\nextgroupplot[title={{{title}}}, ylabel={{transport, long chains}}, ymax=0.62, ytick={{0,0.2,0.4,0.6}}]")
        out.append(f"\\addplot[fill=blue!60, draw=blue!60] coordinates {{{coords(rows['transport'][i])}}};")
        out.append(f"\\addplot[fill=orange!70, draw=orange!70] coordinates {{{coords(rows['kv'][i])}}};")
        if i == 0:
            out.append(r"\legend{state, K/V}")
    for i in range(3):
        out.append(f"\\nextgroupplot[ylabel={{mask cost, points}}, ymax=16, ytick={{0,5,10,15}}]")
        out.append(f"\\addplot[fill=gray!70, draw=gray!70, error bars/.cd, y dir=both, y explicit] coordinates {{" + " ".join(
            f"({s},{v:.2f}) += (0,{hi:.2f}) -= (0,{lo:.2f})" for s, v, lo, hi in zip(SLOTS, rows["cost"][i], rows["cost_lo"][i], rows["cost_hi"][i])) + "};")
    for i in range(3):
        out.append(f"\\nextgroupplot[ylabel={{probe: read / confirmed}}, ymax=0.85, ytick={{0,0.2,0.4,0.6,0.8}}]")
        out.append(f"\\addplot[fill=green!50!black, draw=green!50!black] coordinates {{{coords(rows['read'][i])}}};")
        out.append(f"\\addplot[fill=red!70, draw=red!70] coordinates {{{coords(rows['faith'][i])}}};")
        if i == 0:
            out.append(r"\legend{readability, faithfulness}")
    out += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (HERE / "fig1.tex").write_text("\n".join(out) + "\n")


def fig2():
    # ledger §137 addendum 4; report Tables ulterior-probes and ulterior-arms
    probes = {"Model A (backdoor)": [1.000] * 6, "Model B (inert trigger)": [0.528, 0.515, 0.487, 0.464, 0.476, 0.475],
              "Reported by Ramjee et al.": [0.894, 0.847, 0.783, 0.712, 0.648, 0.583]}
    arms = [("release, clean", 0.352, 0.0), ("release, baseline's latents", 0.347, 0.029), ("release, latents hidden", 0.388, 0.031),
            ("release, probe direction removed", 0.354, 0.0), ("release, context entries from control", 0.183, 0.025),
            ("control, clean", 0.164, 0.0), ("control, armed run's latents", 0.164, 0.023), ("control, shifted along probe", 0.164, 0.0)]
    out = [r"\begin{tikzpicture}", r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=4.0cm}, height=4.6cm, tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\scriptsize}, legend style={font=\tiny, draw=none, fill=none}, legend cell align=left]",
           r"\nextgroupplot[width=5.4cm, title={(a) their probe protocol}, xlabel={latent position}, ylabel={transfer accuracy}, xtick={1,...,6}, ymin=0.4, ymax=1.05, legend style={at={(0.5,-0.42)}, anchor=north, font=\tiny, draw=none}]"]
    styles = ["blue!70, thick, mark=*", "red!70, thick, mark=square*", "gray!70, thick, dashed, mark=triangle*"]
    for (name, vals), st in zip(probes.items(), styles):
        out.append(f"\\addplot[{st}] coordinates {{" + " ".join(f"({i+1},{v:.3f})" for i, v in enumerate(vals)) + "};")
        out.append(f"\\addlegendentry{{{name}}}")
    out.append(r"\nextgroupplot[width=6.6cm, title={(b) model A: interchange arms}, xbar, xlabel={share of immoral actions}, xmin=0, xmax=0.45, ytick=data, symbolic y coords={" + ",".join(f"{{{a[0]}}}" for a in reversed(arms)) + r"}, y tick label style={font=\tiny}, /pgf/bar width=5pt, enlarge y limits=0.08, xtick={0,0.1,0.2,0.3,0.4}]")
    out.append(r"\addplot[fill=blue!55, draw=blue!55, error bars/.cd, x dir=both, x explicit] coordinates {" + " ".join(f"({v:.3f},{{{n}}}) +- ({e:.3f},0)" for n, v, e in reversed(arms)) + "};")
    out.append(r"\draw[dashed, gray] (axis cs:0.352,{release, clean}) -- (axis cs:0.352,{control, shifted along probe});")
    out.append(r"\draw[dashed, gray] (axis cs:0.164,{release, clean}) -- (axis cs:0.164,{control, shifted along probe});")
    out += [r"\end{groupplot}", r"\end{tikzpicture}"]
    (HERE / "fig2.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    fig1(); fig2(); print("fig1.tex, fig2.tex written")
