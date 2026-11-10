from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "2024311281_孙驰_机器人感知LevelB空间回放增补报告.docx"
ASSET_DIR = ROOT / "build" / "level_b_docx_assets"
PREVIEW = ASSET_DIR / "level_b_storyboard_first_frame.png"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_font(run, name="Microsoft YaHei", size=10.5, bold=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
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


def set_cell_margins(cell, value=95):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side in ("top", "left", "bottom", "right"):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)


def set_cell_width(cell, width_cm):
    cell.width = Cm(width_cm)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(round(width_cm * 567)))
    tc_w.set(qn("w:type"), "dxa")


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    borders(table)
    header_props = table.rows[0]._tr.get_or_add_trPr()
    header_marker = OxmlElement("w:tblHeader")
    header_marker.set(qn("w:val"), "true")
    header_props.append(header_marker)
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        shade(cell, "17365D")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        set_cell_width(cell, widths[index])
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.1
        set_font(paragraph.add_run(header), size=9.3, bold=True, color="FFFFFF")
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            if row_index % 2:
                shade(cells[index], "EEF4FA")
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[index])
            set_cell_width(cells[index], widths[index])
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if index == 0 or len(str(value)) < 18 else WD_ALIGN_PARAGRAPH.LEFT
            )
            paragraph.paragraph_format.line_spacing = 1.15
            set_font(paragraph.add_run(str(value)), size=8.8)
    return table


def add_heading(doc, text, level=1):
    paragraph = doc.add_heading(text, level=level)
    for run in paragraph.runs:
        set_font(run, size=15 if level == 1 else 12, bold=True)
    paragraph.paragraph_format.space_before = Pt(10)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def add_body(doc, text, bold_lead=None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.35
    paragraph.paragraph_format.space_after = Pt(5)
    if bold_lead and text.startswith(bold_lead):
        set_font(paragraph.add_run(bold_lead), bold=True)
        text = text[len(bold_lead) :]
    set_font(paragraph.add_run(text))
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Cm(0.7)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.25
        set_font(paragraph.add_run(item))


def build():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    gif_path = ROOT / "demo" / "level_b_v2_spatial_replay.gif"
    with Image.open(gif_path) as image:
        image.seek(0)
        image.convert("RGB").save(PREVIEW)

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.7)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(2.35)
    section.right_margin = Cm(2.35)

    styles = doc.styles
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.color.rgb = RGBColor(0, 0, 0)
    title_style_ppr = styles["Title"]._element.get_or_add_pPr()
    title_border = title_style_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_style_ppr.remove(title_border)
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(22)
    title.paragraph_format.space_after = Pt(16)
    set_font(title.add_run("机器人感知 Level B 位姿真值辅助回放增补报告"), size=20, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(subtitle.add_run("课程期末项目 RGB D 感知链调试验证与研究边界更正"), size=13.5)
    metadata = doc.add_paragraph()
    metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
    metadata.paragraph_format.space_before = Pt(16)
    metadata.paragraph_format.space_after = Pt(16)
    set_font(metadata.add_run("学号 2024311281    姓名 孙驰    日期 2026 年 11 月 10 日"), size=11)

    add_heading(doc, "结论", 1)
    add_body(
        doc,
        "研究完整性复核确认，Level B v2 使用的 /Odometry_gazebo 是 Gazebo 位姿真值。该输入不含危险源标签或危险源坐标，但不满足完整真值隔离，也不能代表自主里程计。原运行因此重分类为位姿真值辅助的开发调试回放。",
    )
    add_body(
        doc,
        "在这一更正后的边界内，固定 RF DETR、RGB D 反投影、深度筛选、三帧确认和正式格式结果写入均已确定性完成。红色危险球使用 14 次观测，输出坐标为 [7.70, 22.55, 0.17] m。事后评分得到 TP 1、FN 1、FP 0 和 23 分；0.124 m 匹配误差依赖 Gazebo 位姿真值，只能说明调试链内部一致性，不能解释为自主定位精度。",
    )

    add_heading(doc, "证据边界", 1)
    add_table(
        doc,
        ["项目", "已验证结果", "边界"],
        [
            ["机器人基线", "WindySky Final e26b48e", "只读已提交树与隔离 VM clone"],
            ["目标真值", "危险标签与坐标未进入推理", "目标语义边界保持隔离"],
            ["位姿输入", "/Odometry_gazebo", "Gazebo 真值，仅可作调试 oracle"],
            ["调试链", "14 帧 28 个检测全部定位", "不代表自主定位或比赛合规"],
            ["结果格式", "1 个危险源世界坐标", "仅验证写入与评分接口"],
            ["生产入口", "未执行", "VMware 无 NVIDIA CUDA"],
        ],
        widths=[3.1, 6.1, 6.8],
    )

    doc.add_page_break()
    add_heading(doc, "传感器问题与修复", 1)
    add_body(
        doc,
        "Level B v1 保留为失败证据。虽然相机 far clip 为 8 m，RealSense OpenNI 插件没有声明 pointCloudCutoffMax，红球约 6.23 m 处的表面深度因此被截断。冻结的 50% 有效深度门没有放宽，v1 按规则停止且未生成正式结果。",
    )
    add_body(
        doc,
        "修复仅应用于独立传感器 overlay，在插件配置中补充 8.0 m 最大 cutoff。原 SimEnv 与机器人外部仓库均未修改。协议 v2 在重新采集前冻结，沿用 v1 的 60% 内区、至少 30 个像素、50% 有效比例、0.2 至 8 m 深度范围、MAD 不高于 0.25 m、三次观测和位置标准差不高于 0.35 m 等全部阈值。",
    )

    add_heading(doc, "处理链路", 1)
    add_table(
        doc,
        ["阶段", "输入", "输出与检查"],
        [
            ["采集", "RGB depth CameraInfo pose TF", "92 对同步样本，14 帧导出"],
            ["检测", "640 x 480 RGB", "14 个红球与 14 个绿球 proposal"],
            ["定位", "bbox 内区与 32FC1 depth", "28 个检测全部通过深度质量门"],
            ["坐标变换", "相机点与 Gazebo pose", "两个 pose oracle 坐标 track"],
            ["时序确认", "同类别近邻 track", "第 3 帧达到三次观测并上报"],
            ["结果写入", "确认后的危险 track", "正式格式 detected_danger.json"],
            ["事后评估", "最终结果与 simulator truth", "误差和 TP FN FP 仅在推理后计算"],
        ],
        widths=[3.0, 5.2, 7.8],
    )

    doc.add_page_break()
    add_heading(doc, "空间结果", 1)
    add_table(
        doc,
        ["指标", "结果", "说明"],
        [
            ["bag 消息", "11967", "持续 11.17 秒"],
            ["同步误差", "0 ms 与不超过 6 ms", "RGB depth 与 image pose"],
            ["危险 track 观测", "14", "第 3 帧确认"],
            ["均值坐标", "[7.6973, 22.5487, 0.1676] m", "Gazebo pose 辅助 odom 坐标"],
            ["位置标准差", "[0.0042, 0.0008, 0.0014] m", "显著低于 0.35 m 门限"],
            ["正式写入坐标", "[7.70, 22.55, 0.17] m", "按输出精度取整"],
        ],
        widths=[4.0, 5.5, 6.5],
    )

    add_heading(doc, "可视化回放", 1)
    add_body(
        doc,
        "14 帧 GIF 按原始 sequence index 全量生成，不读取危险源标签或坐标选帧。画面明确标注 TARGET TRUTH FREE 和 GAZEBO POSE ORACLE，并显示检测框、类别、置信度、track ID、调试坐标、决策状态和推理延迟。下图为第一帧，危险 track 尚未达到三次确认。",
    )
    picture_paragraph = doc.add_paragraph()
    picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture_paragraph.paragraph_format.keep_with_next = True
    picture = picture_paragraph.add_run().add_picture(str(PREVIEW), width=Cm(11.4))
    picture._inline.docPr.set("title", "Level B 位姿真值辅助调试回放第一帧")
    picture._inline.docPr.set(
        "descr",
        "仿真相机画面中绿球与红色危险球均有检测框、track ID 与位姿真值辅助坐标，当前状态为 CANDIDATE。",
    )
    caption = doc.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_after = Pt(7)
    set_font(caption.add_run("图 1  Level B 位姿真值辅助调试回放第一帧"), size=9.5)

    add_heading(doc, "事后评分与解释", 1)
    add_table(
        doc,
        ["指标", "结果"],
        [
            ["真值危险源", "2"],
            ["检测危险源", "1"],
            ["TP FN FP", "1  1  0"],
            ["Pose oracle 匹配误差", "0.124 m"],
            ["技术目标得分", "23 分 满分 37 分"],
        ],
        widths=[6.1, 9.9],
    )
    add_body(
        doc,
        "固定视角中的可见红球完成了 RGB D 反投影、位姿真值变换和结果写入。另一个危险源不在该短场景视野内，形成一个 FN。该结果验证的是调试链，不测量自主里程计误差，也不能外推为搜索覆盖率或完整比赛能力。",
    )

    add_heading(doc, "验证与复现", 1)
    add_bullets(
        doc,
        [
            "四个 canonical replay 文件在第二次运行中逐字节一致。",
            "运行清单绑定协议、部署包、模型、bag、输入清单、预测、回放输出、GIF 与事后评估的 SHA 256。",
            "独立校验器明确识别 /Odometry_gazebo 为位姿真值，并要求 G2 真值隔离保持失败状态。",
            "校验器分别检查危险目标真值边界、位姿真值披露、协议状态、结果格式、评分指标和可视化绑定。",
            "完整自动化回归通过，原 13 页 RF DETR 与 SmolVLM 课程报告未被覆盖。",
        ],
    )
    add_body(
        doc,
        "自主空间回放条件：", bold_lead="自主空间回放条件："
    )
    add_body(
        doc,
        "若要把本实验升级为自主空间回放，必须使用机器人合同允许的 /odom_fused 或其他自主位姿重新采集并重跑。课程报告的核心 RF DETR 与 SmolVLM 研究不依赖该升级，也不需要租用 GPU 或执行完整比赛流程。",
    )

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("RF DETR SmolVLM 机器人感知 Level B 位姿真值辅助调试回放"), size=8.5, color="666666")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "机器人感知 Level B 位姿真值辅助回放增补报告"
    doc.core_properties.subject = "RGB D 感知链调试验证与研究边界更正"
    doc.save(OUT)
    print(f"{OUT}\nSHA256={sha256_path(OUT)}")


if __name__ == "__main__":
    build()
