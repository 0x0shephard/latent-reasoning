"""Build the exactly two-page CODI-vs-KaVa research brief."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "pdf" / "CODI_KAVA_Two_Page_Report.pdf"

BLACK = colors.black
DARK = colors.HexColor("#4A4A4A")
MID = colors.HexColor("#8A8A8A")
LIGHT = colors.HexColor("#D5D5D5")
PALE = colors.HexColor("#F2F2F2")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    regular_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    bold_path = Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")
    italic_path = Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf")
    if regular_path.is_file() and bold_path.is_file():
        pdfmetrics.registerFont(TTFont("BriefTimes", str(regular_path)))
        pdfmetrics.registerFont(TTFont("BriefTimes-Bold", str(bold_path)))
        pdfmetrics.registerFont(TTFont("BriefTimes-Italic", str(italic_path)))
        pdfmetrics.registerFontFamily(
            "BriefTimes",
            normal="BriefTimes",
            bold="BriefTimes-Bold",
            italic="BriefTimes-Italic",
            boldItalic="BriefTimes-Bold",
        )
        return "BriefTimes", "BriefTimes-Bold", "BriefTimes-Italic"
    return "Times-Roman", "Times-Bold", "Times-Italic"


def make_styles(regular: str, bold: str, italic: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold,
            fontSize=16.5,
            leading=19,
            alignment=TA_CENTER,
            textColor=BLACK,
            spaceAfter=3,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontName=regular,
            fontSize=8.5,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=BLACK,
            spaceAfter=5,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.35,
            leading=10.55,
            alignment=TA_JUSTIFY,
            textColor=BLACK,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=10.2,
            leading=12,
            textColor=BLACK,
            spaceBefore=5.5,
            spaceAfter=2.5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.35,
            leading=10.6,
            alignment=TA_JUSTIFY,
            textColor=BLACK,
            spaceAfter=3.5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "body_compact": ParagraphStyle(
            "BodyCompact",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=8.05,
            leading=10.05,
            alignment=TA_JUSTIFY,
            textColor=BLACK,
            spaceAfter=2.5,
        ),
        "key": ParagraphStyle(
            "Key",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=8.6,
            leading=10.8,
            alignment=TA_LEFT,
            textColor=BLACK,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.35,
            leading=8.8,
            alignment=TA_JUSTIFY,
            textColor=BLACK,
            spaceAfter=2,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName=bold,
            fontSize=7.2,
            leading=8.4,
            textColor=BLACK,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=regular,
            fontSize=7.2,
            leading=8.4,
            textColor=BLACK,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=6.8,
            leading=8.1,
            textColor=BLACK,
            leftIndent=8,
            firstLineIndent=-8,
            spaceAfter=1,
        ),
        "italic": ParagraphStyle(
            "Italic",
            parent=base["BodyText"],
            fontName=italic,
            fontSize=7.4,
            leading=9,
            textColor=BLACK,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def accuracy_chart(regular: str, bold: str) -> Drawing:
    labels = ["GSM8K", "GSM-Hard", "MultiArith", "SVAMP", "Macro"]
    codi = [12.59, 2.65, 18.89, 11.00, 11.28]
    kava = [13.12, 2.50, 25.56, 12.00, 13.29]

    width, height = 486, 157
    d = Drawing(width, height)
    left, bottom, plot_w, plot_h = 38, 28, 430, 110
    ymax = 30.0

    for tick in (0, 10, 20, 30):
        y = bottom + plot_h * tick / ymax
        d.add(Line(left, y, left + plot_w, y, strokeColor=LIGHT, strokeWidth=0.45))
        d.add(String(left - 7, y - 2.5, str(tick), fontName=regular, fontSize=6.5, textAnchor="end"))
    d.add(Line(left, bottom, left, bottom + plot_h, strokeColor=BLACK, strokeWidth=0.7))
    d.add(Line(left, bottom, left + plot_w, bottom, strokeColor=BLACK, strokeWidth=0.7))
    d.add(String(9, bottom + plot_h / 2, "Accuracy (%)", fontName=regular, fontSize=7, textAnchor="middle", angle=90))

    group_w = plot_w / len(labels)
    bar_w = 22
    for idx, label in enumerate(labels):
        center = left + group_w * (idx + 0.5)
        for j, (value, fill) in enumerate(((codi[idx], WHITE), (kava[idx], DARK))):
            x = center - bar_w - 1 + j * (bar_w + 2)
            bar_h = plot_h * value / ymax
            d.add(Rect(x, bottom, bar_w, bar_h, fillColor=fill, strokeColor=BLACK, strokeWidth=0.75))
            d.add(String(x + bar_w / 2, bottom + bar_h + 3.5, f"{value:.2f}", fontName=bold, fontSize=6.2, textAnchor="middle"))
        d.add(String(center, bottom - 11, label, fontName=regular, fontSize=6.8, textAnchor="middle"))

    legend_y = 148
    d.add(Rect(342, legend_y - 6, 10, 7, fillColor=WHITE, strokeColor=BLACK, strokeWidth=0.7))
    d.add(String(356, legend_y - 5.3, "CODI", fontName=regular, fontSize=6.8))
    d.add(Rect(400, legend_y - 6, 10, 7, fillColor=DARK, strokeColor=BLACK, strokeWidth=0.7))
    d.add(String(414, legend_y - 5.3, "KaVa", fontName=regular, fontSize=6.8))
    return d


def replication_and_causal_chart(regular: str, bold: str) -> Drawing:
    width, height = 486, 174
    d = Drawing(width, height)

    # Panel A: matched-seed improvement.
    d.add(String(8, 162, "A. KaVa - CODI macro accuracy by seed", fontName=bold, fontSize=8.1))
    left, bottom, plot_w, plot_h = 28, 28, 196, 112
    ymax = 2.5
    for tick in (0, 0.5, 1.0, 1.5, 2.0, 2.5):
        y = bottom + plot_h * tick / ymax
        d.add(Line(left, y, left + plot_w, y, strokeColor=LIGHT, strokeWidth=0.4))
        d.add(String(left - 5, y - 2.4, f"{tick:.1f}", fontName=regular, fontSize=6, textAnchor="end"))
    d.add(Line(left, bottom, left, bottom + plot_h, strokeColor=BLACK, strokeWidth=0.7))
    d.add(Line(left, bottom, left + plot_w, bottom, strokeColor=BLACK, strokeWidth=0.7))
    seed_values = [2.17, 0.97, 1.08]
    bar_w = 34
    for idx, value in enumerate(seed_values):
        x = left + 22 + idx * 58
        bar_h = plot_h * value / ymax
        fill = DARK if idx == 0 else MID
        d.add(Rect(x, bottom, bar_w, bar_h, fillColor=fill, strokeColor=BLACK, strokeWidth=0.7))
        d.add(String(x + bar_w / 2, bottom + bar_h + 4, f"+{value:.2f}", fontName=bold, fontSize=6.7, textAnchor="middle"))
        d.add(String(x + bar_w / 2, bottom - 11, f"Seed {idx}", fontName=regular, fontSize=6.7, textAnchor="middle"))
    mean_y = bottom + plot_h * 1.41 / ymax
    d.add(Line(left, mean_y, left + plot_w, mean_y, strokeColor=BLACK, strokeWidth=1.0, strokeDashArray=[4, 2]))
    d.add(String(left + plot_w - 1, mean_y + 3, "mean +1.41", fontName=bold, fontSize=6.3, textAnchor="end"))
    d.add(String(5, bottom + plot_h / 2, "Difference (pp)", fontName=regular, fontSize=6.5, textAnchor="middle", angle=90))

    # Panel B: causal difference-in-differences forest plot.
    d.add(Line(243, 12, 243, 162, strokeColor=LIGHT, strokeWidth=0.6))
    d.add(String(256, 162, "B. Causal difference-in-differences", fontName=bold, fontSize=8.1))
    x0, y0, forest_w = 328, 42, 146
    xmin, xmax = -7.0, 1.0

    def sx(value: float) -> float:
        return x0 + forest_w * (value - xmin) / (xmax - xmin)

    for tick in (-7, -6, -5, -4, -3, -2, -1, 0, 1):
        x = sx(float(tick))
        d.add(Line(x, y0 - 5, x, y0 + 95, strokeColor=LIGHT if tick != 0 else BLACK, strokeWidth=0.45 if tick != 0 else 0.9))
        d.add(String(x, y0 - 15, str(tick), fontName=regular, fontSize=6, textAnchor="middle"))
    d.add(Line(x0, y0 - 5, x0 + forest_w, y0 - 5, strokeColor=BLACK, strokeWidth=0.7))
    rows = [
        ("Batch mean", -2.31, -4.14, -0.54),
        ("Shuffle", -4.60, -6.85, -2.42),
        ("Zero", -2.03, -4.31, 0.21),
    ]
    for idx, (name, estimate, low, high) in enumerate(rows):
        y = y0 + 74 - idx * 32
        d.add(String(252, y - 2.3, name, fontName=regular, fontSize=6.7))
        d.add(Line(sx(low), y, sx(high), y, strokeColor=BLACK, strokeWidth=1.4))
        d.add(Line(sx(low), y - 3, sx(low), y + 3, strokeColor=BLACK, strokeWidth=0.8))
        d.add(Line(sx(high), y - 3, sx(high), y + 3, strokeColor=BLACK, strokeWidth=0.8))
        size = 5.5 if name == "Shuffle" else 4.5
        d.add(Rect(sx(estimate) - size / 2, y - size / 2, size, size, fillColor=BLACK, strokeColor=BLACK))
        d.add(String(474, y - 2.3, f"{estimate:+.2f}", fontName=bold, fontSize=6.3, textAnchor="end"))
    d.add(String(x0 + forest_w / 2, 11, "Effect on KaVa relative to CODI (pp)", fontName=regular, fontSize=6.4, textAnchor="middle"))
    d.add(String(364, 139, "More harm to KaVa", fontName=regular, fontSize=6.4, textAnchor="middle"))
    d.add(String(431, 139, "No difference", fontName=regular, fontSize=6.4, textAnchor="middle"))
    d.add(Line(384, 145, 339, 145, strokeColor=BLACK, strokeWidth=0.6))
    d.add(Line(339, 145, 345, 148, strokeColor=BLACK, strokeWidth=0.6))
    d.add(Line(339, 145, 345, 142, strokeColor=BLACK, strokeWidth=0.6))
    return d


def controls_table(styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        [p("Seed-zero capped run", styles["table_header"]), p("Macro", styles["table_header"]), p("Scientific role", styles["table_header"])],
        [p("CODI", styles["table_cell"]), p("11.72%", styles["table_cell"]), p("Endpoint hidden-state distillation", styles["table_cell"])],
        [p("No trajectory distillation", styles["table_cell"]), p("13.19%", styles["table_cell"]), p("Tests whether trajectory matching is necessary", styles["table_cell"])],
        [p("KaVa - random compression", styles["table_cell"]), p("13.40%", styles["table_cell"]), p("Removes learned R-KV selection", styles["table_cell"])],
        [p("KaVa - uniform compression", styles["table_cell"]), p("12.69%", styles["table_cell"]), p("Uses fixed temporal coverage", styles["table_cell"])],
        [p("KaVa - R-KV", styles["table_cell"]), p("13.89%", styles["table_cell"]), p("Full configuration; numerically best", styles["table_cell"])],
    ]
    table = Table(data, colWidths=[178, 67, 241], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    return table


def footer(canvas, doc, regular: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#777777"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 13.5 * mm, A4[0] - doc.rightMargin, 13.5 * mm)
    canvas.setFillColor(BLACK)
    canvas.setFont(regular, 7)
    canvas.drawString(doc.leftMargin, 9.5 * mm, "Muhammad Jon Raza | CODI vs KaVa research brief")
    canvas.drawRightString(A4[0] - doc.rightMargin, 9.5 * mm, f"Page {doc.page} of 2")
    canvas.restoreState()


def build(output: Path) -> None:
    regular, bold, italic = register_fonts()
    styles = make_styles(regular, bold, italic)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=13 * mm,
        bottomMargin=18 * mm,
        title="CODI vs KaVa: A Controlled Study of Latent Reasoning Supervision",
        author="Muhammad Jon Raza",
        subject="Two-page experimental research brief",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates(
        PageTemplate(
            id="research-brief",
            frames=[frame],
            onPage=lambda canvas, current_doc: footer(canvas, current_doc, regular),
        )
    )

    story = []
    story.append(p("CODI vs KaVa: A Controlled Study of Latent Reasoning Supervision", styles["title"]))
    story.append(p("Muhammad Jon Raza  |  Research brief  |  21 July 2026", styles["meta"]))
    story.append(Table([[""]], colWidths=[doc.width], rowHeights=[0.6], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLACK)])))
    story.append(Spacer(1, 4))
    story.append(p("<b>Abstract.</b> This study asks whether KaVa's key/value (KV) trajectory supervision improves continuous latent mathematical reasoning over CODI's endpoint hidden-state distillation under matched conditions. Both methods were implemented in one GPT-2 framework and controlled for data, architecture, six autoregressive latent steps, optimizer, schedule, decoding, and a one-epoch budget of 96,405 steps. On the complete seed-zero evaluation, KaVa increased macro numeric exact-match accuracy from <b>11.28%</b> to <b>13.29%</b>: a gain of <b>2.01 percentage points</b> (95% paired-bootstrap CI: +0.44 to +3.60). KaVa also outperformed CODI at all three matched seeds. Cross-example latent-state shuffling harmed KaVa 4.60 points more than CODI, showing stronger dependence on example-specific latent information. The advantage is reliable but task-dependent and concentrated on MultiArith; current controls do not isolate R-KV compression as its unique cause.", styles["abstract"]))

    story.append(p("1. Research questions and controlled design", styles["h1"]))
    story.append(p("The broad question, <i>Which method is better?</i>, was refined into three tests: <b>(1)</b> Does KaVa outperform CODI under matched conditions? <b>(2)</b> Is the direction consistent across training seeds? <b>(3)</b> Does KaVa make stronger causal use of its latent states? CODI and KaVa share student and teacher cross-entropy plus endpoint hidden-state matching. KaVa additionally matches the student's six latent KV states to a teacher KV trajectory compressed with R-KV.", styles["body"]))

    design_data = [
        [p("Controlled component", styles["table_header"]), p("Setting", styles["table_header"]), p("Evaluation", styles["table_header"]), p("Setting", styles["table_header"])],
        [p("Backbone / data", styles["table_cell"]), p("GPT-2 / 385,620 examples", styles["table_cell"]), p("Metric", styles["table_cell"]), p("Numeric exact match", styles["table_cell"])],
        [p("Latent budget", styles["table_cell"]), p("6 autoregressive states", styles["table_cell"]), p("Full sample", styles["table_cell"]), p("3,118 items per method", styles["table_cell"])],
        [p("Optimization", styles["table_cell"]), p("Batch 4; LR 1e-4", styles["table_cell"]), p("Uncertainty", styles["table_cell"]), p("10,000 paired bootstraps", styles["table_cell"])],
        [p("Training budget", styles["table_cell"]), p("1 epoch; 96,405 steps", styles["table_cell"]), p("Decoding", styles["table_cell"]), p("Greedy; max 64 tokens", styles["table_cell"])],
    ]
    table = Table(design_data, colWidths=[92, 144, 74, 176], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(table)

    story.append(p("2. Complete primary evaluation", styles["h1"]))
    story.append(KeepTogether([
        accuracy_chart(regular, bold),
        p("<b>Figure 1.</b> Complete seed-zero numeric exact-match accuracy. Bars show percentages; macro is the unweighted mean of the four tasks. KaVa improves macro accuracy by <b>+2.01 points</b> (95% CI +0.44 to +3.60). MultiArith supplies most of the gain: +6.67 points (95% CI +1.11 to +12.22; McNemar p = 0.02896).", styles["caption"]),
    ]))

    result_box = Table([[p("Primary result", styles["key"]), p("KaVa is 17.8% higher relative to CODI's macro accuracy, and the complete paired interval excludes zero.", styles["body_compact"])]], colWidths=[83, 403])
    result_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, BLACK),
        ("BACKGROUND", (0, 0), (0, 0), PALE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(result_box)

    story.append(p("3. Matched-seed replication", styles["h1"]))
    story.append(p("On capped matched evaluations, KaVa outperformed CODI at seeds 0, 1, and 2 by +2.17, +0.97, and +1.08 macro points. The mean advantage was +1.41 points (sample SD 0.66). This establishes directional consistency across the tested initializations, while three seeds remain insufficient for a precise population-level method interval.", styles["body_compact"]))

    story.append(p("4. Compression controls", styles["h1"]))
    story.append(controls_table(styles))
    story.append(Spacer(1, 3))
    story.append(p("Full KaVa is numerically best, but its paired macro differences from the three controls have intervals that include zero. The evidence therefore supports the complete KaVa configuration over CODI, but does not attribute the improvement uniquely to R-KV compression.", styles["body_compact"]))

    story.append(PageBreak())

    story.append(p("5. Causal evidence", styles["h1"]))
    story.append(p("Causal interventions replaced latent states in both models. The difference-in-differences statistic is (KaVa intervention - KaVa baseline) - (CODI intervention - CODI baseline); negative values mean that the same corruption harmed KaVa more.", styles["body_compact"]))
    story.append(KeepTogether([
        replication_and_causal_chart(regular, bold),
        p("<b>Figure 2.</b> <b>A,</b> KaVa's matched-seed macro advantage was +2.17, +0.97, and +1.08 points (mean +1.41; sample SD 0.66). <b>B,</b> causal effects with 95% paired-bootstrap intervals. Cross-example shuffling produced the largest differential harm: -4.60 points (95% CI -6.85 to -2.42), demonstrating stronger use of example-specific latent information by KaVa.", styles["caption"]),
    ]))

    story.append(p("Position-wise shuffling localized KaVa's strongest dependency to latent position 4, where capped macro accuracy fell from 13.89% to 10.26%.", styles["body_compact"]))

    story.append(p("6. Conclusion and research direction", styles["h1"]))
    story.append(p("All three primary questions receive affirmative answers in this controlled setting: KaVa is more accurate overall; it improves over CODI at each of three matched seeds; and it relies more strongly on example-specific latent states. The gain is modest and task-dependent, with MultiArith as the main driver. Explicit CoT-SFT remains a stronger absolute reference at 33.86% macro accuracy.", styles["body_compact"]))
    story.append(p("The next high-value question is whether latent states implement causally structured reasoning or mainly supply additional serial computation. A compute-matched study should compare endpoint hidden states, keys-only, values-only, full KV trajectories, and unsupervised recurrent computation, followed by targeted activation patching. In parallel, CODI versus KaVa can anchor a systematic survey organized by representation, supervision granularity, transition mechanism, causal evidence, and accuracy-efficiency trade-offs.", styles["body_compact"]))

    story.append(p("Limitations", styles["h1"]))
    story.append(p("The study uses GPT-2, arithmetic tasks, six latent positions, and one training epoch. Only the primary seed-zero pair has complete evaluation; other runs use capped sets. Three seeds establish directional consistency but not a precise population-level confidence interval.", styles["body_compact"]))

    story.append(p("References", styles["h1"]))
    story.append(p("[1] Z. Shen et al., \"CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation,\" arXiv:2502.21074, 2025.", styles["reference"]))
    story.append(p("[2] A. Kuzina et al., \"KaVa: Latent Reasoning via Compressed KV-Cache Distillation,\" arXiv:2510.02312, 2025.", styles["reference"]))

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
