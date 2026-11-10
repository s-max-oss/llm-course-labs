#!/usr/bin/env python3
"""Build the final course-project operation manual."""

from __future__ import annotations

import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "deliverables" / "2024311281_孙驰_期末项目操作手册.docx"


def set_font(run, size=11, bold=False, name="Times New Roman", east_asia="宋体"):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for key, value in (("ascii", name), ("hAnsi", name), ("eastAsia", east_asia), ("cs", name)):
        fonts.set(qn(f"w:{key}"), value)


def set_style(style, size, bold=False, east_asia="宋体"):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for key, value in (("ascii", "Times New Roman"), ("hAnsi", "Times New Roman"), ("eastAsia", east_asia), ("cs", "Times New Roman")):
        fonts.set(qn(f"w:{key}"), value)


def paragraph(doc, text, *, indent=True, bold_lead=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.45
    p.paragraph_format.space_after = Pt(6)
    if indent:
        p.paragraph_format.first_line_indent = Pt(22)
    if bold_lead and text.startswith(bold_lead):
        first = p.add_run(bold_lead)
        set_font(first, bold=True)
        rest = p.add_run(text[len(bold_lead):])
        set_font(rest)
    else:
        set_font(p.add_run(text))
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    set_font(p.add_run(text), size=16 if level == 1 else 13, bold=True, east_asia="黑体")
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run(text))


def check_item(doc, text):
    """Render a checklist row without inheriting Word's list-marker tabs."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(4)
    set_font(p.add_run("□ " + text))


def code(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Pt(8)
    p.paragraph_format.right_indent = Pt(8)
    ppr = p._p.get_or_add_pPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), "F2F2F2")
    ppr.append(shade)
    set_font(p.add_run("\n".join(lines)), size=9, name="Consolas", east_asia="等线")


def shade_cell(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), fill)
    tcpr.append(node)


def margins_cell(cell, value=100):
    tcpr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side in ("top", "start", "bottom", "end"):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tcpr.append(mar)


def table(doc, headers, rows, widths_cm):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for idx, (cell, text) in enumerate(zip(t.rows[0].cells, headers)):
        cell.width = Cm(widths_cm[idx])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade_cell(cell, "1F4E78")
        margins_cell(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_font(run, size=10, bold=True)
        run.font.color.rgb = RGBColor(255, 255, 255)
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(header)
    for row_index, values in enumerate(rows, 1):
        cells = t.add_row().cells
        for idx, (cell, text) in enumerate(zip(cells, values)):
            cell.width = Cm(widths_cm[idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            margins_cell(cell)
            if row_index % 2 == 0:
                shade_cell(cell, "F3F7FA")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(0)
            set_font(p.add_run(text), size=9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("第 ")
    set_font(run, size=9)
    field_run = p.add_run()
    set_font(field_run, size=9)
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    display = OxmlElement("w:t")
    display.text = "1"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    for node in (field_begin, instr, separate, display, field_end):
        field_run._r.append(node)
    set_font(p.add_run(" 页"), size=9)


def verify(path):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                etree.parse(archive.open(name))


def main():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.orientation = WD_ORIENT.PORTRAIT
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)
    add_page_number(section)

    normal = doc.styles["Normal"]
    set_style(normal, 11)
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(6)
    title_style = doc.styles["Title"]
    set_style(title_style, 20, True, "黑体")
    title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_style.paragraph_format.space_after = Pt(24)
    ppr = title_style.element.get_or_add_pPr()
    border = ppr.find(qn("w:pBdr"))
    if border is not None:
        ppr.remove(border)
    for name, size in (("Heading 1", 16), ("Heading 2", 13)):
        style = doc.styles[name]
        set_style(style, size, True, "黑体")
        style.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(5)

    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_paragraph(style="Title")
    set_font(title.add_run("四足机器人危险源识别期末项目操作手册"), size=20, bold=True, east_asia="黑体")
    for text in (
        "RF-DETR 与受约束 SmolVLM 探索性研究",
        "学生姓名  孙驰",
        "学号  2024311281",
        "哈尔滨工业大学 深圳",
        "2026 年 11 月",
    ):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(16)
        set_font(p.add_run(text), size=15 if text.startswith("RF") else 13, bold=text.startswith("RF"))
    doc.add_page_break()

    heading(doc, "1 项目用途与结论")
    paragraph(doc, "本项目研究四足机器人仿真环境中的危险红球识别。系统比较固定 RF-DETR 检测器、SmolVLM 自由生成、固定候选评分和真实 proposal 级联，并把检测结果转换为机器人事件与任务状态。当前证据支持 RF-DETR-only 作为语义检测方案；SmolVLM 级联没有产生指标增益。")
    paragraph(doc, "Level A 已完成无世界坐标的接口级回放。Level B v2 完成 RGB-D 空间调试链，但使用了 Gazebo 位姿真值 /Odometry_gazebo，因此 G2 真值隔离失败。Level B 结果只能作为 pose-oracle 调试证据，不能表述为自主定位、比赛合规或完整建筑搜索。")
    table(doc, ["项目", "状态", "可支持结论"], [
        ["RF-DETR", "完成", "重复控制后的域内探索性检测表现"],
        ["SmolVLM B2", "完成", "oracle crop 条件下的候选语义分类"],
        ["真实 proposal 级联", "按门槛停止", "当前冻结方案没有净收益"],
        ["Level A", "完成", "接口、状态机和确定性"],
        ["Level B v2", "调试完成且 G2 失败", "Gazebo 位姿真值辅助的空间调试链"],
    ], [3.0, 3.8, 9.0])

    heading(doc, "2 提交包结构")
    paragraph(doc, "最终提交包按报告、操作手册、源代码、数据与结果、演示材料五类组织。根目录的材料清单记录每个文件的用途，SHA256SUMS.txt 用于检查复制或上传后的字节完整性。")
    table(doc, ["目录", "内容", "使用方式"], [
        ["01_期末报告", "最终提交版 Word 与 PDF", "教师阅读和归档"],
        ["02_操作手册", "本手册 Word 与 PDF", "环境配置、复核和边界说明"],
        ["03_源代码", "configs、scripts、src、tests 与依赖文件", "阅读实现或在完整环境复核"],
        ["04_数据与结果", "500 张 RGB、383 个标签、清单、分区与封存结果", "复查数据和指标"],
        ["05_演示材料", "Level A/Level B GIF 与报告图", "直接查看系统流程和实验结果"],
    ], [3.1, 7.7, 5.0])

    heading(doc, "3 运行环境")
    paragraph(doc, "只阅读报告、结果 JSON 和 GIF 不需要安装环境。重新生成清单与运行验证需要 Python；重新执行 RF-DETR 或 SmolVLM 推理还需要各自独立的虚拟环境和模型权重。两个模型使用不同 Transformers 版本，不能装在同一个环境。")
    table(doc, ["环境", "主要依赖", "用途"], [
        ["通用验证", "Python 3.11 以上、Pillow、NumPy", "清单、回放、测试和可视化"],
        ["RF-DETR", "PyTorch、rfdetr 1.9.4、Transformers 5.1.0", "检测器推理"],
        ["SmolVLM", "PyTorch、Transformers 4.57.6、Accelerate", "B1、B2 与级联推理"],
        ["Level B", "ROS Noetic、Gazebo 11", "仅用于重新采集与空间回放"],
    ], [3.1, 7.6, 5.1])
    paragraph(doc, "模型权重和原机器人完整仓库不作为本课程提交包的源代码组成部分。报告中的 checkpoint、机器人提交和大型部署归档都由 SHA-256 或 Git commit 固定；重新推理时必须使用相同对象。", bold_lead="模型与外部仓库。")

    heading(doc, "4 数据配置")
    paragraph(doc, "提交包中的 dataset/images 保存 500 张历史仿真 RGB 图像，dataset/labels 保存 383 个 YOLO 标签。先复制 configs/data_sources.example.json 为 configs/data_sources.local.json，再把 image_root 与 label_root 改为解压后的绝对路径。不要把 local 配置上传到公共仓库。")
    code(doc, [
        "Copy-Item configs/data_sources.example.json configs/data_sources.local.json",
        "# 编辑 image_root 与 label_root",
        "python scripts/build_manifest.py --config configs/data_sources.local.json",
        "python scripts/validate_manifest.py --config configs/data_sources.local.json --verify-hashes",
    ])
    paragraph(doc, "500 张历史图像包含五个场景 seed。重复审计隔离 99 张，正式探索性检测评测使用 401 张图像。117 张缺少标签文件的图像沿用背景或未知约定，不能视为人工核验的严格负样本。")

    heading(doc, "5 一键复核")
    paragraph(doc, "在完整研究目录中运行下列命令。脚本重建数据清单、候选、分区、Level A 回放和两类 GIF，复算 RF-DETR 与级联结果，验证 Level B 产物，并执行 87 个单元测试。")
    code(doc, ["powershell -ExecutionPolicy Bypass -File scripts/run_checks.ps1"])
    paragraph(doc, "通过时最后应出现 Ran 87 tests 和 OK。数据只有五个 seed、部分标签缺失等研究限制会以 WARNING 输出；这些警告是报告边界的一部分，不应通过降低规则消除。")
    paragraph(doc, "精简课程提交包不包含约 700 MB 的机器人 Git 历史 bundle 与部署 tar。它保留对应 manifest、哈希、传感器补丁、回放产物和更正记录，因此适合材料审阅；如需运行完整 Level B 封存验证，应在原始研究目录执行。", bold_lead="Level B 大文件。")

    heading(doc, "6 分项复核命令")
    heading(doc, "6.1 RF-DETR 与 SmolVLM", 2)
    code(doc, [
        "python scripts/validate_rfdetr_run.py",
        "python scripts/validate_smolvlm_b2_jobs.py",
        "python scripts/validate_cascade_run.py",
    ])
    paragraph(doc, "RF-DETR 验证器复算 401 张图像的三目标指标；SmolVLM 与级联验证器检查候选、预测契约、停止门槛和结果哈希。验证通过不消除 checkpoint 训练样本谱系缺失这一限制。")
    heading(doc, "6.2 Level A 回放", 2)
    code(doc, [
        "python scripts/validate_replay_contract.py",
        "python scripts/build_replay.py",
        "python scripts/validate_replay_run.py",
        "python scripts/generate_replay_storyboard.py",
    ])
    paragraph(doc, "Level A 输出位于 experiments/replay_v1，GIF 位于 demo/replay_v1_level_a.gif。它是 12 个预先选择案例的故事板，不是连续轨迹。")
    heading(doc, "6.3 Level B 调试回放", 2)
    code(doc, [
        "python scripts/generate_level_b_storyboard.py",
        "python scripts/validate_level_b_run.py",
    ])
    paragraph(doc, "Level B v2 的规范产物位于 experiments/level_b_v2/outputs/replay_v1。验证器要求 G2 保持 failed_pose_ground_truth_present_reclassified_debug_only，并核对更正记录、权威接口合同、GIF 披露、报告文字和产物哈希。")

    heading(doc, "7 结果阅读顺序")
    bullet(doc, "先读期末报告摘要、实验结果、有效性威胁和附录 B，确认主结论与证据边界。")
    bullet(doc, "查看 figures 中五张模型结果图，再查看 Level A 与 Level B 两个 GIF。")
    bullet(doc, "需要复核数字时，以 experiments 下的封存 JSON 和 report-evidence-index.md 为准。")
    bullet(doc, "需要复核代码时，从 scripts/run_checks.ps1 进入，再按调用顺序阅读验证器和测试。")
    paragraph(doc, "报告中的 0.9635 Macro-F1 是固定 checkpoint 在重复控制后历史语料上的域内探索性结果。Level B 的 0.1238 m 匹配误差使用 Gazebo 位姿真值，只能评价调试链内部一致性。")

    heading(doc, "8 常见问题")
    table(doc, ["现象", "原因", "处理"], [
        ["找不到图像或标签", "local 配置仍指向示例路径", "修改 data_sources.local.json，重新构建清单"],
        ["RF-DETR 与 SmolVLM 依赖冲突", "两个模型要求不同 Transformers 版本", "建立两个独立虚拟环境"],
        ["Level B 校验缺少 bundle 或 tar", "精简提交包省略大型部署归档", "在原始研究目录执行完整校验"],
        ["看到数据警告", "五 seed、缺标签或 oracle crop 等已知边界", "保留警告，不修改门槛"],
        ["GIF 显示 pose oracle", "Level B 使用 /Odometry_gazebo", "只按开发调试证据解释"],
    ], [5.0, 5.6, 5.2])

    heading(doc, "9 提交前检查")
    for item in (
        "报告和操作手册的 Word、PDF 均能打开，页码和图表正常。",
        "数据集包含 500 张图像和 383 个标签文件，清单与分区文件齐全。",
        "源代码不包含虚拟环境、缓存、个人绝对路径配置或 SSH 凭据。",
        "Level B 明确标注 pose-oracle 调试边界，G2 没有被写成通过。",
        "上传后重新计算 SHA256SUMS.txt，并与本地清单核对。",
        "将最终 ZIP 上传至个人百度云，并把只读分享链接提交给教师。",
    ):
        check_item(doc, item)

    doc.core_properties.title = "四足机器人危险源识别期末项目操作手册"
    doc.core_properties.author = "孙驰"
    doc.core_properties.subject = "大模型基础与应用探索期末项目提交材料"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    verify(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
