#!/usr/bin/env python3
"""Build the Chinese final report in the teacher-provided arXiv-style DOCX template.

The body text is Chinese; titles, abstract and keywords are bilingual. All
figures, tables, equations, references and evidence boundaries are inherited
from the English source so that both deliverables stay structurally identical.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt
from lxml import etree

from build_final_report import (
    add_body_paragraph,
    add_code_block,
    add_figure,
    add_heading,
    add_markdown_table,
    add_text_runs,
    configure_styles,
    remove_all_body_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    PROJECT_ROOT.parent
    / "课程实验"
    / "实验指南与模板"
    / "Final-Report-Template-arXiv-style.docx"
)
SOURCE = PROJECT_ROOT / "teacher-format-report.zh.md"
OUTPUT = (
    PROJECT_ROOT
    / "deliverables"
    / "2024311281_孙驰_期末报告_RF-DETR_SmolVLM_最终提交版.docx"
)
TITLE_ZH = "从竞赛危险源搜索到可审计的多模态感知"
TITLE_EN = "From Competition Hazard Search to Auditable Multimodal Perception"


def add_hyperlink(paragraph, url: str) -> None:
    relationship_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    run_properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    fonts = OxmlElement("w:rFonts")
    for attr in ("ascii", "hAnsi", "cs"):
        fonts.set(qn(f"w:{attr}"), "Times New Roman")
    fonts.set(qn("w:eastAsia"), "宋体")
    run_properties.extend([fonts, color, underline])
    text = OxmlElement("w:t")
    text.text = url
    run.extend([run_properties, text])
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_paragraph_with_urls(doc: Document, text: str, *, indent: bool = True):
    paragraph = add_body_paragraph(doc, "", indent=indent)
    parts = re.split(r"(https?://\S+)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith(("http://", "https://")):
            trailing = ""
            while part and part[-1] in ".,;)]":
                trailing = part[-1] + trailing
                part = part[:-1]
            add_hyperlink(paragraph, part)
            if trailing:
                add_text_runs(paragraph, trailing)
        else:
            add_text_runs(paragraph, part)
    return paragraph


def add_equation(doc: Document, linear_text: str) -> None:
    """Render one centred native Word equation in Office Math XML."""
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(6)
    math_para = OxmlElement("m:oMathPara")
    math_para_properties = OxmlElement("m:oMathParaPr")
    justification = OxmlElement("m:jc")
    justification.set(qn("m:val"), "centerGroup")
    math_para_properties.append(justification)
    math_para.append(math_para_properties)
    math = OxmlElement("m:oMath")
    math_run = OxmlElement("m:r")
    math_run_properties = OxmlElement("m:rPr")
    style = OxmlElement("m:sty")
    style.set(qn("m:val"), "p")
    math_run_properties.append(style)
    math_text = OxmlElement("m:t")
    math_text.text = linear_text
    math_run.extend([math_run_properties, math_text])
    math.append(math_run)
    math_para.append(math)
    paragraph._p.append(math_para)


def add_cover(doc: Document) -> None:
    for _ in range(6):
        doc.add_paragraph()
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)
    add_text_runs(title, TITLE_ZH, size=18, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(48)
    add_text_runs(subtitle, TITLE_EN, size=12, bold=False, italic=True)
    metadata = (
        "学生姓名：孙驰    学号：2024311281",
        "指导教师：曾国坤",
        "哈尔滨工业大学（深圳）",
        "大模型基础与应用探索 期末报告",
    )
    for text in metadata:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(22)
        add_text_runs(paragraph, text, size=18, bold=True)
    doc.add_page_break()


def parse_source() -> tuple[str, str, str, str, list[str]]:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    abstract_en_index = lines.index("## Abstract")
    keywords_en_index = next(
        index for index, line in enumerate(lines) if line.startswith("Keywords:")
    )
    abstract_en = " ".join(
        line.strip()
        for line in lines[abstract_en_index + 1 : keywords_en_index]
        if line.strip()
    )
    keywords_en = lines[keywords_en_index].strip()
    word_count = len(re.findall(r"\b[\w'-]+\b", abstract_en))
    if not 200 <= word_count <= 300:
        raise ValueError(f"English abstract must contain 200-300 words; found {word_count}")

    abstract_zh_index = lines.index("## 摘要")
    keywords_zh_index = next(
        index for index, line in enumerate(lines) if line.startswith("关键词：")
    )
    abstract_zh = "".join(
        line.strip()
        for line in lines[abstract_zh_index + 1 : keywords_zh_index]
        if line.strip()
    )
    keywords_zh = lines[keywords_zh_index].strip()
    char_count = len(re.findall(r"[\u4e00-\u9fff]", abstract_zh))
    if not 350 <= char_count <= 700:
        raise ValueError(f"Chinese abstract should hold 350-700 characters; found {char_count}")

    body_start = next(index for index, line in enumerate(lines) if line.startswith("## 1 "))
    return abstract_zh, keywords_zh, abstract_en, keywords_en, lines[body_start:]


def add_front_matter(
    doc: Document,
    abstract_zh: str,
    keywords_zh: str,
    abstract_en: str,
    keywords_en: str,
) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(3)
    add_text_runs(title, TITLE_ZH, size=15, bold=True)
    title_en = doc.add_paragraph()
    title_en.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_en.paragraph_format.space_after = Pt(8)
    add_text_runs(title_en, TITLE_EN, size=10.5, italic=True)

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(2)
    add_text_runs(author, "第一作者：孙驰（2024311281）    第二作者：曾国坤", size=12, bold=True)
    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation.paragraph_format.space_after = Pt(10)
    add_text_runs(affiliation, "哈尔滨工业大学（深圳）", size=12, bold=True)

    for line, size in ((keywords_zh, 11.5), (keywords_en, 10)):
        keyword_paragraph = doc.add_paragraph()
        keyword_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        keyword_paragraph.paragraph_format.line_spacing = 1.0
        keyword_paragraph.paragraph_format.space_after = Pt(4)
        add_text_runs(keyword_paragraph, line, size=size, italic=True)

    heading_zh = doc.add_paragraph()
    heading_zh.paragraph_format.keep_with_next = True
    heading_zh.paragraph_format.space_before = Pt(14)
    add_text_runs(heading_zh, "摘要", size=16, bold=True)
    add_body_paragraph(doc, abstract_zh, indent=False)
    doc.add_page_break()

    heading = doc.add_paragraph()
    heading.paragraph_format.keep_with_next = True
    heading.paragraph_format.space_before = Pt(6)
    add_text_runs(heading, "Abstract", size=16, bold=True)
    add_body_paragraph(doc, abstract_en, indent=False)
    doc.add_page_break()


TABLE_CAPTIONS = [
    "技术全景与选型逻辑",
    "数据分区、实验规模与证据边界",
    "RF-DETR 固定检查点的主要指标",
    "受控组件研究",
    "开发集真实候选级联对照",
    "机器人接口与空间数据链的验证边界",
]


def render_body(doc: Document, lines: list[str]) -> None:
    index = 0
    table_number = 0
    figure_number = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            heading_text = line[3:]
            heading = add_heading(doc, heading_text, 1)
            if heading_text.startswith(("References", "参考文献")):
                heading.paragraph_format.page_break_before = True
            index += 1
            continue
        if line.startswith("### "):
            add_heading(doc, line[4:], 2)
            index += 1
            continue
        equation_match = re.fullmatch(r"\[\[EQ:(.+)]]", line)
        if equation_match:
            add_equation(doc, equation_match.group(1))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.+)]\((.+)\)", line)
        if image_match:
            figure_number += 1
            add_figure(doc, image_match.group(2), f"图 {figure_number}  {image_match.group(1)}")
            index += 1
            continue
        if line.startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                values = [value.strip() for value in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", value) for value in values):
                    rows.append(values)
                index += 1
            caption_text = (
                TABLE_CAPTIONS[table_number]
                if table_number < len(TABLE_CAPTIONS)
                else "实验配置或结果汇总"
            )
            caption = doc.add_paragraph(style="Caption")
            caption.paragraph_format.keep_with_next = True
            caption.paragraph_format.space_before = Pt(6)
            caption.paragraph_format.space_after = Pt(3)
            add_text_runs(caption, f"表 {table_number + 1}  {caption_text}", size=10, italic=True)
            add_markdown_table(doc, rows)
            table_number += 1
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
            if candidate.startswith(("## ", "### ", "|", "```", "- ", "![", "[[EQ:")):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph_text = " ".join(part.strip() for part in paragraph_lines)
        if re.fullmatch(r"https?://\S+", paragraph_text):
            paragraph = add_paragraph_with_urls(doc, paragraph_text, indent=False)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = Pt(0)
            continue
        paragraph = add_paragraph_with_urls(doc, paragraph_text, indent=True)
        if re.match(r"^\[\d+]", paragraph_text):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.left_indent = Inches(0.22)
            paragraph.paragraph_format.first_line_indent = Inches(-0.22)
        elif "SHA-256" in paragraph_text and "`" not in paragraph_text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = Pt(0)


def verify_docx(path: Path) -> None:
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
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    abstract_zh, keywords_zh, abstract_en, keywords_en, body_lines = parse_source()
    doc = Document(TEMPLATE)
    remove_all_body_content(doc)
    configure_styles(doc)
    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    doc.core_properties.title = TITLE_ZH
    doc.core_properties.author = "孙驰"
    doc.core_properties.subject = "大模型基础与应用探索期末报告（中文版）"
    add_cover(doc)
    add_front_matter(doc, abstract_zh, keywords_zh, abstract_en, keywords_en)
    render_body(doc, body_lines)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    verify_docx(OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
