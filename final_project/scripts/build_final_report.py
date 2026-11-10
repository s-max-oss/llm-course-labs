#!/usr/bin/env python3
"""Build the final course report from the retained A4 template and Markdown source."""

from __future__ import annotations

import re
import zipfile
import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    PROJECT_ROOT.parent
    / "课程实验"
    / "实验指南与模板"
    / "Final-Report-Template-arXiv-style.docx"
)
SOURCE = PROJECT_ROOT / "final-report.md"
OUTPUT = PROJECT_ROOT / "deliverables" / "2024311281_孙驰_期末报告_RF-DETR_SmolVLM.docx"
CONTENT_WIDTH = Inches(6.30)


def set_run_font(run, name: str, east_asia: str, size: float, *, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:eastAsia"), east_asia)
    fonts.set(qn("w:cs"), name)


def set_style_font(style, name: str, east_asia: str, size: float, *, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:eastAsia"), east_asia)
    fonts.set(qn("w:cs"), name)


def remove_all_body_content(doc: Document) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def configure_styles(doc: Document) -> None:
    title = doc.styles["Title"]
    set_style_font(title, "Times New Roman", "黑体", 18, bold=True)
    title.paragraph_format.space_after = Pt(26)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_ppr = title.element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)

    normal = doc.styles["Normal"]
    set_style_font(normal, "Times New Roman", "宋体", 12)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5

    heading1 = doc.styles["Heading 1"]
    set_style_font(heading1, "Times New Roman", "黑体", 16, bold=True)
    heading1.paragraph_format.space_before = Pt(16)
    heading1.paragraph_format.space_after = Pt(6)
    heading1.paragraph_format.keep_with_next = True
    heading1.paragraph_format.keep_together = True

    heading2 = doc.styles["Heading 2"]
    set_style_font(heading2, "Times New Roman", "黑体", 14, bold=True)
    heading2.paragraph_format.space_before = Pt(10)
    heading2.paragraph_format.space_after = Pt(4)
    heading2.paragraph_format.keep_with_next = True
    heading2.paragraph_format.keep_together = True

    caption = doc.styles["Caption"]
    set_style_font(caption, "Times New Roman", "宋体", 10)
    caption.font.italic = True
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(8)


def add_text_runs(paragraph, text: str, *, size=12, bold=False, italic=False, mono=False):
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        part_bold = bold
        part_mono = mono
        if part.startswith("`") and part.endswith("`"):
            part = part[1:-1]
            part_mono = True
        elif part.startswith("**") and part.endswith("**"):
            part = part[2:-2]
            part_bold = True
        run = paragraph.add_run(part)
        if part_mono:
            set_run_font(run, "Consolas", "等线", 10.5, bold=part_bold, italic=italic)
        else:
            set_run_font(run, "Times New Roman", "宋体", size, bold=part_bold, italic=italic)


def add_body_paragraph(doc: Document, text: str, *, indent=True):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.widow_control = True
    if indent:
        paragraph.paragraph_format.first_line_indent = Pt(24)
    add_text_runs(paragraph, text)
    return paragraph


def add_heading(doc: Document, text: str, level: int):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    add_text_runs(paragraph, text, size=16 if level == 1 else 14, bold=True)
    return paragraph


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)
    shading.set(qn("w:val"), "clear")


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_row_no_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def add_markdown_table(doc: Document, rows: list[list[str]]) -> None:
    column_count = len(rows[0])
    if column_count == 2:
        widths = [Inches(4.70), Inches(1.60)]
    elif column_count == 3:
        widths = [Inches(2.65), Inches(1.825), Inches(1.825)]
    elif column_count == 5:
        widths = [Inches(1.75), Inches(1.25), Inches(1.25), Inches(0.95), Inches(1.10)]
    else:
        widths = [Inches(6.30 / column_count)] * column_count
    table = doc.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row_index, (row, values) in enumerate(zip(table.rows, rows)):
        set_row_no_split(row)
        if row_index == 0:
            set_repeat_table_header(row)
        for column_index, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = widths[column_index]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_shading(cell, "D9EAF7")
            elif row_index % 2 == 0:
                set_cell_shading(cell, "F5F8FA")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.space_after = Pt(0)
            add_text_runs(paragraph, value, size=10.5, bold=row_index == 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc: Document, relative_path: str, caption: str) -> None:
    path = PROJECT_ROOT / relative_path
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    width = Inches(5.35 if "confusion" in path.name else 5.95)
    run = paragraph.add_run()
    shape = run.add_picture(str(path), width=width)
    doc_pr = shape._inline.docPr
    doc_pr.set("title", caption)
    doc_pr.set("descr", caption)
    cap = doc.add_paragraph(style="Caption")
    cap.paragraph_format.keep_together = True
    add_text_runs(cap, caption, size=10, italic=True)


def add_code_block(doc: Document, lines: list[str]) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.left_indent = Pt(8)
    paragraph.paragraph_format.right_indent = Pt(8)
    p_pr = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F2F2F2")
    shading.set(qn("w:val"), "clear")
    p_pr.append(shading)
    run = paragraph.add_run("\n".join(lines))
    set_run_font(run, "Consolas", "等线", 9.5)


def add_cover(doc: Document) -> None:
    for _ in range(5):
        doc.add_paragraph()
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(26)
    add_text_runs(
        title,
        "面向四足机器人危险源识别的 RF-DETR 与受约束 SmolVLM 探索性研究",
        size=18,
        bold=True,
    )
    for _ in range(2):
        doc.add_paragraph()
    for text in (
        "学生姓名：孙驰    学号：2024311281",
        "指导教师：曾国坤",
        "哈尔滨工业大学（深圳）",
        "大模型基础与应用探索 期末报告",
    ):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(18)
        add_text_runs(paragraph, text, size=18, bold=True)
    doc.add_page_break()


def add_front_matter(doc: Document, abstract: str, keywords: str) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    add_text_runs(
        title,
        "面向四足机器人危险源识别的 RF-DETR 与受约束 SmolVLM 探索性研究",
        size=16,
        bold=True,
    )
    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.line_spacing = 1.0
    author.paragraph_format.space_after = Pt(5)
    add_text_runs(author, "作者：孙驰（2024311281）", size=11)
    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation.paragraph_format.line_spacing = 1.0
    affiliation.paragraph_format.space_after = Pt(7)
    add_text_runs(affiliation, "哈尔滨工业大学（深圳）", size=11)
    keyword_para = doc.add_paragraph()
    keyword_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    keyword_para.paragraph_format.line_spacing = 1.0
    keyword_para.paragraph_format.space_before = Pt(3)
    keyword_para.paragraph_format.space_after = Pt(12)
    add_text_runs(keyword_para, keywords, size=11, italic=True)
    abstract_heading = doc.add_paragraph()
    abstract_heading.paragraph_format.keep_with_next = True
    add_text_runs(abstract_heading, "摘要", size=16, bold=True)
    add_body_paragraph(doc, abstract, indent=False)


def parse_source(source: Path) -> tuple[str, str, list[str]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    abstract_index = lines.index("## 摘要")
    keyword_index = next(i for i, line in enumerate(lines) if line.startswith("关键词："))
    abstract = "".join(line.strip() for line in lines[abstract_index + 1 : keyword_index] if line.strip())
    keywords = lines[keyword_index].strip()
    body_start = next(i for i, line in enumerate(lines) if line.startswith("## 1 "))
    return abstract, keywords, lines[body_start:]


def render_body(doc: Document, lines: list[str]) -> None:
    index = 0
    figure_number = 0
    table_number = 0
    table_captions = [
        "数据分区与实验规模",
        "RF-DETR 固定 checkpoint 的主要指标",
        "development 真实 proposal 级联指标对照",
        "机器人回放扩展的证据层级与结论边界",
        "Level B 位姿真值辅助回放的质量门状态",
    ]
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            add_heading(doc, line[3:], 1)
            index += 1
            continue
        if line.startswith("### "):
            add_heading(doc, line[4:], 2)
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.+)]\((.+)\)", line)
        if image_match:
            figure_number += 1
            add_figure(
                doc,
                image_match.group(2),
                f"图 {figure_number}  {image_match.group(1)}",
            )
            index += 1
            continue
        if line.startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [value.strip() for value in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", value) for value in values):
                    rows.append(values)
                index += 1
            table_number += 1
            add_markdown_table(doc, rows)
            caption = doc.add_paragraph(style="Caption")
            caption_text = (
                table_captions[table_number - 1]
                if table_number <= len(table_captions)
                else "实验配置或结果汇总"
            )
            add_text_runs(caption, f"表 {table_number}  {caption_text}", size=10, italic=True)
            continue
        if line.startswith("```"):
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            add_code_block(doc, code_lines)
            continue
        if re.match(r"^\d+\. ", line):
            match = re.match(r"^(\d+)\. (.+)", line)
            assert match is not None
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.line_spacing = 1.35
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.paragraph_format.left_indent = Inches(0.30)
            paragraph.paragraph_format.first_line_indent = Inches(-0.30)
            add_text_runs(paragraph, f"{match.group(1)}.  {match.group(2)}")
            index += 1
            continue
        if line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.line_spacing = 1.35
            paragraph.paragraph_format.space_after = Pt(4)
            add_text_runs(paragraph, line[2:])
            index += 1
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate:
                index += 1
                break
            if (
                candidate.startswith(("## ", "### ", "|", "```", "- ", "!["))
                or re.match(r"^\d+\. ", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph_text = " ".join(part.strip() for part in paragraph_lines)
        paragraph = add_body_paragraph(doc, paragraph_text, indent=True)
        if paragraph_text.startswith("[") and re.match(r"^\[\d+\]", paragraph_text):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = Inches(-0.22)
            paragraph.paragraph_format.left_indent = Inches(0.22)
        elif "SHA-256" in paragraph_text or "https://" in paragraph_text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = Pt(0)


def verify_xml(path: Path) -> None:
    errors: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                try:
                    etree.parse(archive.open(name))
                except etree.XMLSyntaxError as error:
                    errors.append(f"{name}: {error}")
    if errors:
        raise RuntimeError("DOCX XML verification failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--appendix", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    abstract, keywords, body_lines = parse_source(args.source)
    if args.appendix:
        body_lines.extend(["", *args.appendix.read_text(encoding="utf-8").splitlines()])
    doc = Document(TEMPLATE)
    remove_all_body_content(doc)
    configure_styles(doc)
    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    doc.core_properties.title = "面向四足机器人危险源识别的 RF-DETR 与受约束 SmolVLM 探索性研究"
    doc.core_properties.author = "孙驰"
    doc.core_properties.subject = "大模型基础与应用探索期末报告"
    add_cover(doc)
    add_front_matter(doc, abstract, keywords)
    render_body(doc, body_lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    verify_xml(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
