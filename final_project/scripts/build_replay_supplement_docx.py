from __future__ import annotations

from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "2024311281_孙驰_机器人感知LevelA回放增补报告.docx"
ASSET_DIR = ROOT / "build" / "replay_docx_assets"
PREVIEW = ASSET_DIR / "storyboard_first_frame.png"


def set_font(run, name="Microsoft YaHei", size=10.5, bold=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def borders(table, color="D9D9D9"):
    tbl_pr = table._tbl.tblPr
    node = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        item = OxmlElement(f"w:{edge}")
        item.set(qn("w:val"), "single")
        item.set(qn("w:sz"), "6")
        item.set(qn("w:color"), color)
        node.append(item)
    tbl_pr.append(node)


def set_cell_margins(cell, value=80):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side in ("top", "left", "bottom", "right"):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    borders(table)
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    header_marker = OxmlElement("w:tblHeader")
    header_marker.set(qn("w:val"), "true")
    tr_pr.append(header_marker)
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        shade(cell, "17365D")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(header), size=9.5, bold=True, color="FFFFFF")
        if widths:
            cell.width = Cm(widths[index])
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            if row_index % 2:
                shade(cells[index], "EEF4FA")
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[index])
            p = cells[index].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if index == 0 or len(str(value)) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(str(value)), size=8.8)
            if widths:
                cells[index].width = Cm(widths[index])
    return table


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        set_font(run, size=15 if level == 1 else 12, bold=True)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(5)
    return p


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(5)
    if bold_lead and text.startswith(bold_lead):
        set_font(p.add_run(bold_lead), bold=True)
        text = text[len(bold_lead):]
    set_font(p.add_run(text))
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.space_after = Pt(3)
        set_font(p.add_run(item))


def build():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(ROOT / "demo" / "replay_v1_level_a.gif") as image:
        image.seek(0)
        image.convert("RGB").save(PREVIEW)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    styles = doc.styles
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(26)
    title.paragraph_format.space_after = Pt(18)
    set_font(title.add_run("RF DETR 感知决策 Level A 离线回放增补报告"), size=24, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(subtitle.add_run("课程期末项目机器人接口增强验证"), size=14)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(18)
    set_font(meta.add_run("学号 2024311281    姓名 孙驰    日期 2026 年 11 月 9 日"), size=11)

    add_heading(doc, "结论", 1)
    add_body(doc, "Level A 已完成从封存 RGB 与 RF-DETR 结果到规范检测事件、决策状态和可视化故事板的接口级闭环。172 帧产生 226 个检测事件，覆盖继续搜索、语义拒绝和危险候选保留三类分支。由于没有同步 RGB-D、标定、可信时间和自主位姿，84 个危险候选全部安全停在 HELD_UNLOCALIZED；系统未生成确认、上报、世界坐标或正式结果文件。")
    add_body(doc, "本扩展保持原报告与 RF-only 结论不变。它证明检测结果可以进入机器人接口与决策合同，但不把离线故事板表述为实时 Gazebo、连续轨迹或空间定位实验。")

    add_heading(doc, "证据边界", 1)
    add_table(doc, ["项目", "结论", "约束"], [
        ["机器人事实源", "WindySky-Final e26b48e", "只读已提交树"],
        ["开发输入", "fold 01 与 fold 02 共 172 帧", "fold 03 至 05 禁止进入"],
        ["时间语义", "非时序故事板", "stamp 为空且不跨帧承接状态"],
        ["空间语义", "无三维定位", "position 为空且 frame 不是 ROS TF"],
        ["正式输出", "未生成", "Level A 禁止 detected_danger.json"],
    ], widths=[3.2, 6.0, 6.5])

    doc.add_page_break()
    add_heading(doc, "接口与状态合同", 1)
    add_body(doc, "冻结合同独立复现 DangerDetectionArray 的帧级心跳和检测字段。空 detections 数组仍是有效已处理帧；每个检测保留模型类别、规范类别、置信度和 bbox。合同递归拒绝真值、布局、房间、seed 和 oracle 字段。")
    add_table(doc, ["输入条件", "状态路径", "最终处理"], [
        ["无危险 proposal", "SEARCHING", "继续搜索"],
        ["仅非危险类别", "SEARCHING → REJECTED", "拒绝语义干扰物"],
        ["存在危险 proposal", "SEARCHING → CANDIDATE → HELD_UNLOCALIZED", "等待合规 RGB-D 与位姿资源"],
    ], widths=[4.0, 6.6, 5.1])
    add_body(doc, "失效关闭规则：", bold_lead="失效关闭规则：")
    add_bullets(doc, [
        "track_id 为空，temporal_hits 固定为 0，temporal_confirmed 固定为 false。",
        "localized 和 quality_accepted 固定为 false，position 与 position_frame 为空。",
        "CONFIRMED、REPORTED、世界坐标和正式结果输出被协议与测试共同禁止。",
        "输出路径固定在 experiments/replay_v1/outputs，不能重定向覆盖封存报告。",
    ])

    add_heading(doc, "回放结果", 1)
    add_table(doc, ["指标", "数量", "解释"], [
        ["帧事件", "172", "两个开发场景组"],
        ["检测事件", "226", "RF-DETR 封存预测"],
        ["HELD_UNLOCALIZED", "84", "危险候选但无空间证据"],
        ["REJECTED", "77", "仅有非危险类别"],
        ["SEARCHING", "11", "无危险 proposal"],
        ["CONFIRMED 与 REPORTED", "0", "按合同禁止"],
    ], widths=[5.0, 2.5, 8.2])

    doc.add_page_break()
    add_heading(doc, "可视化故事板", 1)
    add_body(doc, "演示 GIF 对三个可达状态各按序号选择前四项，不使用真值。画面始终标注 NON TEMPORAL 与 NO 3D，并显示框、类别、置信度、状态和原因。下图为其中一帧。")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture = p.add_run().add_picture(str(PREVIEW), width=Cm(10.8))
    picture._inline.docPr.set("title", "Level A 非时序感知决策故事板示例")
    picture._inline.docPr.set("descr", "仿真 RGB 帧上叠加 RF-DETR 类别、置信度和检测框，状态为 HELD_UNLOCALIZED，并明确标注 NON-TEMPORAL 与 NO 3D。")
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(caption.add_run("图 1  Level A 非时序感知决策故事板示例"), size=9.5)

    add_heading(doc, "验证与可追溯性", 1)
    add_table(doc, ["验证项", "结果"], [
        ["完整自动化回归", "68 项全部通过"],
        ["事件与状态输出", "重复构建字节一致"],
        ["接口与原报告锁定", "8 个 blob 一致；464D267F…D1BFB 保持不变"],
        ["独立代码复核", "审查发现已修复；最终复核受额度限制"],
    ], widths=[6.0, 9.7])

    add_heading(doc, "后续资源门", 1)
    add_body(doc, "Level B 仅在取得同步 RGB-D、RGB/depth CameraInfo、可信时间戳及允许的自主位姿或 world alignment 后启动。启动前须冻结深度质量、坐标变换、关联和去重规则，并在算法输出完成后再连接真值评分。在资源门通过前，不支持连续轨迹、三维定位、正式坐标或在线控制结论。")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("RF DETR SmolVLM 机器人感知回放增补"), size=8.5, color="666666")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
