"""Build the compact CODI-vs-KaVa report as a polished PDF artifact."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs" / "CODI_KAVA_COMPACT_REPORT.md"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "pdf" / "CODI_KAVA_Compact_Report.pdf"

NAVY = colors.black
BLUE = colors.black
TEAL = colors.black
GOLD = colors.black
INK = colors.black
MUTED = colors.HexColor("#333333")
PALE_BLUE = colors.HexColor("#F2F2F2")
PALE_TEAL = colors.HexColor("#F2F2F2")
PALE_GOLD = colors.HexColor("#F2F2F2")
RULE = colors.HexColor("#777777")
WHITE = colors.white


def _register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/Library/Fonts/Times New Roman.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path("/Library/Fonts/Times New Roman Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ]
    italic_candidates = [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"),
        Path("/Library/Fonts/Times New Roman Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"),
    ]
    bold_italic_candidates = [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf"),
        Path("/Library/Fonts/Times New Roman Bold Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    regular = next((path for path in candidates if path.is_file()), None)
    bold = next((path for path in bold_candidates if path.is_file()), None)
    italic = next((path for path in italic_candidates if path.is_file()), None)
    bold_italic = next((path for path in bold_italic_candidates if path.is_file()), None)
    mono = next((path for path in mono_candidates if path.is_file()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        if italic:
            pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(italic)))
        if bold_italic:
            pdfmetrics.registerFont(TTFont("ReportSans-BoldItalic", str(bold_italic)))
        pdfmetrics.registerFontFamily(
            "ReportSans",
            normal="ReportSans",
            bold="ReportSans-Bold",
            italic="ReportSans-Italic" if italic else "ReportSans",
            boldItalic="ReportSans-BoldItalic" if bold_italic else "ReportSans-Bold",
        )
        regular_name, bold_name = "ReportSans", "ReportSans-Bold"
    else:
        regular_name, bold_name = "Times-Roman", "Times-Bold"
    if mono:
        pdfmetrics.registerFont(TTFont("ReportMono", str(mono)))
        mono_name = "ReportMono"
    else:
        mono_name = "Courier"
    return regular_name, bold_name, mono_name


def _styles(regular: str, bold: str, mono: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName=bold,
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceAfter=9,
            uppercase=True,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=19,
            leading=23,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=9,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=regular,
            fontSize=10.5,
            leading=14,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName=regular,
            fontSize=9,
            leading=12,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontName=bold,
            fontSize=20,
            leading=22,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontName=regular,
            fontSize=8.2,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "summary": ParagraphStyle(
            "Summary",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=13,
            leading=16,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=10.5,
            leading=13,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.2,
            leading=12.2,
            textColor=INK,
            spaceAfter=5,
            alignment=TA_JUSTIFY,
            allowWidows=0,
            allowOrphans=0,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.6,
            leading=9.5,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName=bold,
            fontSize=7.4,
            leading=9,
            textColor=INK,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=regular,
            fontSize=7.4,
            leading=9.2,
            textColor=INK,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=5,
            rightIndent=5,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName=mono,
            fontSize=7.6,
            leading=10,
            textColor=INK,
        ),
    }


def _inline(text: str, mono: str) -> str:
    rendered = html.escape(text.strip())
    rendered = re.sub(
        r"\[([^]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#000000"><u>\1</u></link>',
        rendered,
    )
    rendered = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", rendered)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", rendered)
    rendered = re.sub(
        r"`([^`]+)`",
        rf'<font name="{mono}" size="7.5">\1</font>',
        rendered,
    )
    return rendered


def _table(rows: list[list[str]], styles: dict, available_width: float, mono: str) -> Table:
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    weights = []
    for column in range(columns):
        maximum = max(len(re.sub(r"[*`]", "", row[column])) for row in rows)
        weights.append(min(32, max(9, maximum)))
    total = sum(weights)
    widths = [available_width * weight / total for weight in weights]
    formatted = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        formatted.append([Paragraph(_inline(cell, mono), style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.2, colors.HexColor("#AAAAAA")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ]
    for row_index in range(1, len(rows)):
        background = colors.white
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    if columns > 1:
        commands.append(("ALIGN", (1, 1), (-1, -1), "RIGHT"))
    table.setStyle(TableStyle(commands))
    return table


def _parse_markdown(
    source: Path,
    styles: dict,
    available_width: float,
    mono: str,
) -> list:
    lines = source.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line == "## Abstract")
    lines = lines[start:]
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(_inline(line[3:], mono), styles["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(_inline(line[4:], mono), styles["h2"]))
            index += 1
            continue
        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            code = Preformatted("\n".join(code_lines), styles["code"])
            box = Table([[code]], colWidths=[available_width])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F5F8")),
                        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([box, Spacer(1, 5)])
            continue
        if line.startswith("> "):
            quote_lines = []
            while index < len(lines) and lines[index].startswith("> "):
                quote_lines.append(lines[index][2:].strip())
                index += 1
            quote = Paragraph(_inline(" ".join(quote_lines), mono), styles["quote"])
            box = Table([[quote]], colWidths=[available_width])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                        ("LINEBEFORE", (0, 0), (0, -1), 4, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 11),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            story.extend([box, Spacer(1, 6)])
            continue
        if line.startswith("|"):
            raw_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                raw_rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            if len(raw_rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in raw_rows[1]):
                raw_rows.pop(1)
            story.extend([_table(raw_rows, styles, available_width, mono), Spacer(1, 6)])
            continue
        if re.match(r"^(?:- |\d+\. )", line):
            items = []
            ordered = bool(re.match(r"^\d+\. ", line))
            pattern = r"^\d+\. " if ordered else r"^- "
            while index < len(lines) and re.match(pattern, lines[index]):
                value = re.sub(pattern, "", lines[index]).strip()
                items.append(
                    ListItem(
                        Paragraph(_inline(value, mono), styles["body"]),
                        leftIndent=12,
                    )
                )
                index += 1
            list_options = {
                "bulletType": "1" if ordered else "bullet",
                "leftIndent": 17,
                "bulletFontName": styles["body"].fontName,
                "bulletFontSize": 7.5,
                "spaceAfter": 4,
            }
            if ordered:
                list_options["start"] = "1"
            else:
                list_options["bulletColor"] = colors.black
            story.append(ListFlowable(items, **list_options))
            continue
        paragraph_lines = [line.strip()]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate:
                index += 1
                break
            if (
                candidate.startswith(("## ", "### ", "```", "> ", "|", "- "))
                or re.match(r"^\d+\. ", candidate)
            ):
                break
            paragraph_lines.append(candidate.strip())
            index += 1
        story.append(Paragraph(_inline(" ".join(paragraph_lines), mono), styles["body"]))
    return story


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
        )
        self.addPageTemplates([PageTemplate(id="report", frames=[body_frame], onPage=self._page)])

    def _page(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        page_number = canvas.getPageNumber()
        canvas.setFillColor(colors.black)
        canvas.setFont("Times-Roman", 7.2)
        canvas.drawString(self.leftMargin, 9.5 * mm, "Muhammad Jon Raza | 20 July 2026")
        canvas.drawRightString(width - self.rightMargin, 9.5 * mm, f"Page {page_number}")
        canvas.restoreState()


def _cover(styles: dict) -> list:
    metric_data = [
        [
            Paragraph("11.28%", styles["metric_value"]),
            Paragraph("13.29%", styles["metric_value"]),
            Paragraph("+2.01 pp", styles["metric_value"]),
        ],
        [
            Paragraph("CODI macro", styles["metric_label"]),
            Paragraph("KaVa macro", styles["metric_label"]),
            Paragraph("Full paired difference", styles["metric_label"]),
        ],
    ]
    metrics = Table(metric_data, colWidths=[53 * mm, 53 * mm, 53 * mm])
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
                ("BACKGROUND", (1, 0), (1, -1), PALE_TEAL),
                ("BACKGROUND", (2, 0), (2, -1), PALE_GOLD),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, WHITE),
                ("TOPPADDING", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    findings = [
        Paragraph("KaVa outperformed CODI on the complete seed-zero benchmark mixture.", styles["summary"]),
        Paragraph("Every capped matched training seed favored KaVa.", styles["summary"]),
        Paragraph("Latent-state shuffling harmed KaVa 4.60 macro points more than CODI.", styles["summary"]),
    ]
    finding_list = ListFlowable(
        [ListItem(item, leftIndent=14) for item in findings],
        bulletType="bullet",
        leftIndent=20,
        bulletColor=TEAL,
        bulletFontSize=9,
    )
    return [
        Spacer(1, 16 * mm),
        Paragraph("RESEARCH REPORT", styles["cover_kicker"]),
        Paragraph("CODI vs KaVa", styles["cover_title"]),
        Paragraph(
            "A controlled and mechanistic study of latent mathematical reasoning supervision",
            styles["cover_subtitle"],
        ),
        Table([[""]], colWidths=[34 * mm], rowHeights=[2.2 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), GOLD)])),
        Spacer(1, 10 * mm),
        metrics,
        Spacer(1, 10 * mm),
        Paragraph("Executive conclusion", styles["h2"]),
        Paragraph(
            "Under matched architecture, data, training, and evaluation, KaVa achieved a "
            "consistent but modest advantage over CODI and relied more strongly on "
            "example-specific latent information. The performance gain was concentrated on "
            "MultiArith, while the current controls did not isolate R-KV compression as the "
            "unique cause.",
            styles["summary"],
        ),
        Spacer(1, 3 * mm),
        finding_list,
        Spacer(1, 10 * mm),
        Table(
            [[Paragraph("Prepared by <b>Muhammad Jon Raza</b><br/>20 July 2026", styles["cover_meta"]) ]],
            colWidths=[159 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        PageBreak(),
    ]


def _paper_header(styles: dict) -> list:
    rule = Table([[""]], colWidths=[159 * mm], rowHeights=[0.4 * mm])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.black)]))
    return [
        Spacer(1, 4 * mm),
        Paragraph(
            "CODI vs KaVa: A Controlled and Mechanistic Study of "
            "Latent Reasoning Supervision",
            styles["cover_title"],
        ),
        Paragraph("Muhammad Jon Raza", styles["cover_subtitle"]),
        Paragraph("20 July 2026", styles["cover_meta"]),
        Spacer(1, 4 * mm),
        rule,
        Spacer(1, 4 * mm),
    ]


def build(source: Path, output: Path) -> None:
    regular, bold, mono = _register_fonts()
    styles = _styles(regular, bold, mono)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title=(
            "CODI vs KaVa: A Controlled and Mechanistic Study of "
            "Latent Reasoning Supervision"
        ),
        author="Muhammad Jon Raza",
        subject="Controlled comparison of CODI and KaVa latent reasoning supervision",
    )
    story = _paper_header(styles)
    story.extend(_parse_markdown(source, styles, doc.width, mono))
    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
