from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph


def set_run_font(run, *, size: float = 10.5, bold: bool | None = None, color: str = "000000") -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = None
    color_element = run._element.get_or_add_rPr().find(qn("w:color"))
    if color_element is None:
        color_element = OxmlElement("w:color")
        run._element.get_or_add_rPr().append(color_element)
    color_element.set(qn("w:val"), color)


def clear_paragraph(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def replace_body(paragraph: Paragraph, text: str, *, bold_lead: str | None = None) -> Paragraph:
    clear_paragraph(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Pt(21)
    paragraph.paragraph_format.line_spacing = 1.35
    paragraph.paragraph_format.space_after = Pt(6)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, bold=True)
        rest = paragraph.add_run(text[len(bold_lead) :])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


def replace_preserving_run(paragraph: Paragraph, text: str) -> None:
    run_properties = None
    if paragraph.runs and paragraph.runs[0]._element.rPr is not None:
        run_properties = copy.deepcopy(paragraph.runs[0]._element.rPr)
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    if run_properties is not None:
        run._element.insert(0, run_properties)


def insert_after(paragraph: Paragraph, text: str, *, bold_lead: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    created = Paragraph(new_p, paragraph._parent)
    replace_body(created, text, bold_lead=bold_lead)
    return created


def insert_after_table(table, text: str = "") -> Paragraph:
    new_p = OxmlElement("w:p")
    table._tbl.addnext(new_p)
    paragraph = Paragraph(new_p, table._parent)
    if text:
        replace_body(paragraph, text)
    return paragraph


def set_cell(cell, text: str, *, bold: bool = False, center: bool = False, size: float = 9.5) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.05
    run = paragraph.add_run(str(text))
    set_run_font(run, size=size, bold=bold)


def mark_header_row(table) -> None:
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    table_header = OxmlElement("w:tblHeader")
    table_header.set(qn("w:val"), "true")
    tr_pr.append(table_header)
    for row in table.rows:
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)


def fill_table(table, rows: list[list[Any]], *, center_columns: set[int] | None = None, size: float = 9.5) -> None:
    if len(rows) != len(table.rows):
        raise ValueError(f"表格行数不匹配: template={len(table.rows)} data={len(rows)}")
    center_columns = center_columns or set()
    for row_index, row_values in enumerate(rows):
        if len(row_values) != len(table.rows[row_index].cells):
            raise ValueError(f"第 {row_index} 行列数不匹配")
        for column_index, value in enumerate(row_values):
            set_cell(
                table.rows[row_index].cells[column_index],
                str(value),
                bold=row_index == 0 or column_index == 0,
                center=column_index in center_columns,
                size=size,
            )
    mark_header_row(table)


def add_image_to_placeholder_cell(cell, image_path: Path, width_inches: float) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))


def result_value(row: dict[str, Any], key: str, digits: int = 4) -> str:
    return f"{row[key]:.{digits}f}"


def build_report(
    template: Path,
    results_path: Path,
    system_path: Path,
    output: Path,
    hpc_results_path: Path | None = None,
) -> None:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    system = json.loads(system_path.read_text(encoding="utf-8"))
    hpc = None
    if hpc_results_path and hpc_results_path.is_file():
        hpc = json.loads(hpc_results_path.read_text(encoding="utf-8"))
    document = Document(template)
    tables = document.tables

    teacher = data["teacher"]
    scratch = data["baseline"]["scratch"]
    kd = data["baseline"]["kd"]
    feature = data["feature_distillation"]
    quant = data["dynamic_quantization"]
    scan = data["teacher_scale_scan"]["rows"]
    ablation = data["ablation"]
    counts = data["model_counts"]
    hyper = data["hyperparameters"]

    fill_table(
        tables[0],
        [
            ["实验名称", "自制语言模型的知识蒸馏"],
            ["课程名称", "人工智能——深度学习大模型智能体"],
            ["姓名 / 学号", "孙驰 / 2024311281"],
            ["专业 / 班级", "计算机 10 班"],
            ["实验日期", "2026 年 11 月 6 日"],
            ["实验环境", "CPU / Python 3.13.12 / PyTorch 2.12.0+cpu"],
        ],
        size=10,
    )

    purposes = [
        "1. 理解教师软标签所包含的类别关系，并能说明它相对 one-hot 硬标签增加了什么信息。",
        "2. 独立实现带温度和 T² 缩放的 KL 蒸馏损失，核对 KL 方向与按 token 平均的实现。",
        "3. 在相同学生架构、初始化、步数和批次顺序下，对比从零训练与知识蒸馏的真实差异。",
        "4. 通过 16 组消融、特征蒸馏、教师规模扫描和动态量化，分析压缩收益及其边界。",
    ]
    purpose_paragraph = next(p for p in document.paragraphs if p.text.startswith("（请用自己的语言"))
    replace_body(purpose_paragraph, purposes[0])
    current = purpose_paragraph
    for purpose in purposes[1:]:
        current = insert_after(current, purpose)

    memory_gib = system["memory_bytes"] / (1024**3)
    fill_table(
        tables[1],
        [
            ["项目", "实际配置"],
            ["操作系统", f"{system['operating_system']} {system['os_version']}"],
            ["Python 版本", system["python"]],
            ["PyTorch 版本", system["pytorch"]],
            ["CPU 型号 / 核心数", f"{system['cpu']}；{system['physical_cores']} 核 / {system['logical_processors']} 线程"],
            ["内存", f"{memory_gib:.1f} GiB"],
            ["是否使用 GPU", "否；本机与 HPC 均按 CPU 路径运行"],
        ],
        size=9.5,
    )

    principle = (
        "知识蒸馏让学生同时学习真实下一个字符的硬标签与教师输出的完整概率分布。软标签不仅指出最可能字符，"
        "还保留其他字符之间的相对相似性，即暗知识。温度 T 通过 softmax(z/T) 平滑分布：T 较大时非最大类别差异更明显，"
        "但过大会趋近均匀。蒸馏目标为 (1-alpha)CE + alpha T² KL(p_teacher || p_student)。KL 的第一个输入必须是学生的"
        "对数概率，目标是教师概率；T² 用来补偿温度造成的约 1/T² 梯度缩小，使改变 T 时仍可使用相近学习率。"
    )
    replace_body(next(p for p in document.paragraphs if p.text.startswith("（简述知识蒸馏")), principle)

    scan_extra_time = sum(
        row["teacher_time_s"] + row["student_kd_time_s"] for row in scan if row["teacher_width"] != 160
    )
    total_local = (
        teacher["time_s"]
        + scratch["time_s"]
        + kd["time_s"]
        + ablation["time_s"]
        + feature["time_s"]
        + scan_extra_time
    )
    fill_table(
        tables[2],
        [
            ["步骤", "执行命令", "是否成功", "耗时"],
            ["1. 环境自检", "python distill_lm.py --mode quick", "成功", "10.4 s"],
            ["2. 训练教师", "python distill_lm.py --mode teacher --retrain", "成功", f"{teacher['time_s']:.2f} s"],
            ["3. 从零 + 蒸馏", "python distill_lm.py --mode all", "成功", f"{scratch['time_s'] + kd['time_s']:.2f} s"],
            ["4. 消融实验", "python distill_lm.py --mode ablate", "成功", f"{ablation['time_s']:.2f} s"],
            ["5. 拓展", "python distill_lm.py --mode full --retrain", "成功", f"本机全流程约 {total_local:.1f} s"],
        ],
        center_columns={2, 3},
        size=8.8,
    )

    fill_table(
        tables[3],
        [
            ["参数", "教师", "学生"],
            ["n_layer", "2", "2"],
            ["n_head", "4", "4"],
            ["n_embd", "160", "64"],
            ["block_size", "96", "96"],
            ["参数量", f"{counts['teacher']:,}", f"{counts['student']:,}"],
            ["压缩比", "—", f"{counts['compression_ratio']:.1f}x"],
        ],
        center_columns={1, 2},
    )
    insert_after_table(
        tables[3],
        "参数量手算核对：按指南近似式 2VC+TC+L×12C²，教师为 732,800（实测 737,280，差 0.61%）；"
        "学生为 145,664（实测 147,456，差 1.22%，宽度较小时偏置与 LayerNorm 占比更高）。"
        "进一步补计每层 13C 个偏置/LayerNorm 参数及末层 2C，精确式为 2VC+TC+L×(12C²+13C)+2C，"
        "两者分别得到 737,280 与 147,456，与实测值完全一致。",
    )
    fill_table(
        tables[4],
        [
            ["参数", "取值", "说明"],
            ["温度 T", f"{hyper['temperature']:g}", "基础 KD 取 4；消融扫描 1/2/4/8"],
            ["权重 alpha", f"{hyper['alpha']:.1f}", "基础 KD 取 0.5；消融扫描 0.3/0.5/0.7/0.9"],
            ["学习率", f"{hyper['learning_rate']:.0e}", "AdamW，weight decay 0.01"],
            ["batch_size", str(hyper["batch_size"]), "训练与评估一致"],
            ["教师训练步数", str(hyper["teacher_iters"]), "三种教师规模均相同"],
            ["学生训练步数", str(hyper["student_iters"]), "scratch / KD / FitNet 相同"],
            ["随机种子", str(hyper["seed"]), "学生初始化 seed+7；批次 seed+1"],
            ["特征蒸馏权重", f"{feature['feature_weight']:.1f}", "最后层隐状态经线性投影后计算 MSE"],
        ],
        center_columns={1},
        size=8.8,
    )

    fill_table(
        tables[5],
        [
            ["模型", "参数量", "训练步数", "训练耗时", "留出集 loss", "困惑度 PPL", "与教师一致率"],
            ["教师 Teacher", f"{teacher['params']:,}", teacher["iterations"], f"{teacher['time_s']:.2f} s", result_value(teacher, "val_loss"), result_value(teacher, "ppl"), "—"],
            ["学生 从零训练", f"{scratch['params']:,}", scratch["iterations"], f"{scratch['time_s']:.2f} s", result_value(scratch, "val_loss"), result_value(scratch, "ppl"), result_value(scratch, "agreement")],
            ["学生 蒸馏训练", f"{kd['params']:,}", kd["iterations"], f"{kd['time_s']:.2f} s", result_value(kd, "val_loss"), result_value(kd, "ppl"), result_value(kd, "agreement")],
        ],
        center_columns={1, 2, 3, 4, 5, 6},
        size=8.2,
    )

    ablation_map = {(row["temperature"], row["alpha"]): row["val_loss"] for row in ablation["rows"]}
    fill_table(
        tables[6],
        [["T / alpha", "alpha=0.3", "alpha=0.5", "alpha=0.7", "alpha=0.9"]]
        + [
            [f"T={int(t)}"] + [f"{ablation_map[(t, a)]:.4f}" for a in (0.3, 0.5, 0.7, 0.9)]
            for t in (1.0, 2.0, 4.0, 8.0)
        ],
        center_columns={1, 2, 3, 4},
        size=9,
    )
    fill_table(
        tables[7],
        [
            ["模型", "生成结果（60 字）"],
            ["教师", teacher["sample"].replace("\n", "")[:60]],
            ["学生 从零训练", scratch["sample"].replace("\n", "")[:60]],
            ["学生 蒸馏训练", kd["sample"].replace("\n", "")[:60]],
        ],
        size=8.8,
    )
    table8_label = next(p for p in document.paragraphs if p.text.startswith("【表 8"))
    table8_label.paragraph_format.page_break_before = True

    add_image_to_placeholder_cell(tables[8].cell(0, 0), results_path.parent / "curves.png", 6.25)
    add_image_to_placeholder_cell(tables[9].cell(0, 0), results_path.parent / "ablation_heatmap.png", 5.55)

    loss_delta = scratch["val_loss"] - kd["val_loss"]
    ppl_delta = scratch["ppl"] - kd["ppl"]
    agreement_delta = (kd["agreement"] - scratch["agreement"]) * 100
    improvement = data["baseline"]["relative_val_loss_improvement_percent"]
    replace_body(
        next(p for p in document.paragraphs if p.text.startswith("（粘贴 runs_distill")),
        "教师曲线在约 100 步后进入低损失区间。学生从零训练下降较快；KD 的总损失包含 CE 与温度缩放 KL，数值不能与纯 CE 逐点等同。最终应以同一留出集 CE 为准。",
    )
    comparison_paragraph = next(p for p in document.paragraphs if p.text.startswith("（对比表 6"))
    replace_body(
        comparison_paragraph,
        f"在参数量、120 步、初始化和批次顺序完全相同的条件下，KD 将留出集 loss 从 {scratch['val_loss']:.4f} 降至 {kd['val_loss']:.4f}，绝对降低 {loss_delta:.4f}，相对降低 {improvement:.2f}%；PPL 降低 {ppl_delta:.4f}；教师一致率提高 {agreement_delta:.2f} 个百分点。唯一有意改变的是训练目标，因此差异可归因于教师软分布提供的结构信息及其正则作用，而不是更大模型或更多数据。",
    )
    insert_after(
        comparison_paragraph,
        "公平性检查确认两个学生的初始 state_dict 逐张量相同，训练批次生成器都使用 seed=1，评估批次也固定。训练集与留出集由 seed=0 和 seed=999 生成，主体句子不同；但参考代码把同一组 16 首固定诗附加到两侧，并偶然重复 3 条合成句，因此这里报告的是带少量重合的同分布留出结果。",
    )
    best, worst = ablation["best"], ablation["worst"]
    replace_body(
        next(p for p in document.paragraphs if p.text.startswith("（最优组合")),
        f"80 步消融的最优组合为 T={best['temperature']:g}、alpha={best['alpha']:.1f}，loss={best['val_loss']:.4f}；最差为 T={worst['temperature']:g}、alpha={worst['alpha']:.1f}，loss={worst['val_loss']:.4f}。T=1 的四组差异很小，软分布仍较尖锐；T=4~8 时较低 alpha 更好。T=8、alpha=0.9 明显退化，说明过度依赖并不完美的教师会削弱硬标签纠错。",
    )
    replace_body(
        next(p for p in document.paragraphs if p.text.startswith("（对比表 8")),
        "教师样例能稳定生成完整的主语、地点、动作、对象和情绪结构。KD 学生已形成较完整句式，用词和标点明显接近教师；从零学生仍出现短语拼接和引号缺失。三者都偶有语义跳跃，这与字符级建模、短训练和合成语法的限制一致，不能仅凭单个样例判断总体质量。",
    )

    scan_text = "；".join(
        f"C={row['teacher_width']}：教师 loss {row['teacher_val_loss']:.4f}，KD 改善 {row['kd_relative_improvement_percent']:.2f}%"
        for row in scan
    )
    fill_table(
        tables[10],
        [
            ["编号", "拓展内容", "完成情况", "关键数据", "自评加分"],
            ["A", "特征蒸馏（FitNet）", "完成", f"权重 0.1；loss {feature['val_loss']:.4f}，普通 KD {kd['val_loss']:.4f}", "+5"],
            ["B", "温度/权重热力图", "完成", f"16 组；最优 T={best['temperature']:g}, alpha={best['alpha']:.1f}", "+3"],
            ["C", "自蒸馏", "未执行", "—", "0"],
            ["D", "动态量化导出", "完成", f"{quant['fp32_bytes']:,} -> {quant['int8_bytes']:,} B；{quant['size_compression_ratio']:.2f}x", "+5"],
            ["E", "教师规模扫描", "完成", scan_text, "+4"],
            ["F", "换成真实语料", "未执行", "—", "0"],
        ],
        center_columns={0, 2, 4},
        size=7.8,
    )
    scan_figure = insert_after_table(tables[10])
    scan_figure.alignment = WD_ALIGN_PARAGRAPH.CENTER
    scan_figure.add_run().add_picture(str(results_path.parent / "teacher_scale_scan.png"), width=Inches(5.7))
    scan_caption = insert_after(scan_figure, "图 3  教师规模与蒸馏收益")
    scan_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    scan_caption.paragraph_format.first_line_indent = Pt(0)
    for run in scan_caption.runs:
        set_run_font(run, size=9.5, bold=True)
    insert_after(
        scan_caption,
        "A、B、D、E 四项按表中标准合计 17 分；依据指南“拓展加分最高 15 分”的规则，自评按 15 分封顶。",
    )

    fill_table(
        tables[11],
        [
            ["序号", "遇到的问题（完整报错）", "原因分析", "解决办法"],
            ["1", "sbatch: Memory specification can not be satisfied", "gpu-a30 分区把可用内存元数据报告为 1 MB", "按课程既有作业移除显式 --mem，请求保留 CPU 与时限"],
            ["2", "mkdir: cannot create directory 'results': Permission denied", "Slurm 从 spool 副本执行，按 $0 推导到了不可写目录", "把 LAB_DIR 固定为 $HOME/llm_course_lab08 后重提"],
            ["3", "torch.ao.quantization is deprecated", "PyTorch 2.12 仍支持旧动态量化入口，但已标记迁移", "保留兼容分支并记录提醒；后续版本迁移 torchao"],
        ],
        center_columns={0},
        size=8,
    )

    best_scan = max(scan, key=lambda row: row["kd_relative_improvement_percent"])
    conclusion = (
        f"本实验完成 0.737M 教师到 0.147M 学生的 5.0x 参数压缩。基础 KD 在严格公平对照下将留出集 loss 相对降低 {improvement:.2f}%，"
        f"同时提高教师一致率。消融表明温度与教师权重必须配合，过高 alpha 会放大教师误差。FitNet 本次略微退化，说明中间特征约束需要调权；"
        f"动态 INT8 量化将文件再压缩 {quant['size_compression_ratio']:.2f}x，loss 几乎不变，但小批量延迟未加速。教师扫描中 C={best_scan['teacher_width']} 的"
        f"KD 相对改善最大为 {best_scan['kd_relative_improvement_percent']:.2f}%，结果未呈简单单调关系。训练与留出语料使用不同生成种子，但固定附加的 16 首诗在两侧重复，"
        "因此结论限定为当前同分布留出评估，后续应移除重合诗句并增加独立种子复测。"
    )
    replace_body(next(p for p in document.paragraphs if p.text.startswith("（总结本次实验")), conclusion)

    answers = [
        "答：alpha=1 时 CE 项消失，学生梯度只推动其分布逼近教师。如果教师把真实字符概率压低，学生没有硬标签梯度进行纠正，会完整继承错误；教师分布过平或有系统偏差时尤其明显。因此保留 CE 相当于用真实标签约束蒸馏方向。",
        "答：softmax(z/T) 对 logits 的导数含 1/T，KL 两端的差异再引入一个约 1/T，整体梯度量级约按 1/T² 缩小。乘 T² 后不同温度的梯度可比。若漏乘并把 T 从 4 加倍到 8，梯度约降至四分之一，需把学习率近似增至 4 倍才能保持更新量，但这不如直接保留 T² 稳定。",
        "答：0.74M 不是现代意义的通用大模型，只是本实验中的相对大教师。扫描结果为：" + scan_text + "。更宽教师的自身 loss 整体更低，但学生收益并未严格随宽度单调增加，说明教师既要强，也要提供学生可吸收的软结构；容量差距过大时并非所有信息都能在 120 步内转移。",
        "答：可设计信息量控制实验。组 A 使用真实教师软标签；组 B 在每个样本内打乱非目标类别概率，同时保持熵、最大概率和 CE 权重；组 C 使用标签平滑，使正则强度与组 A 接近但不含教师类别关系。若 A 明显优于 B/C，说明主要来自结构信息；若三者接近，则更像平滑正则。所有组应固定初始化、批次和步数，并用多个种子报告均值。",
        "答：T=100 时 softmax 近似均匀，单类概率差异很小。乘 T² 后高温极限仍近似匹配中心化 logits，所以并非没有梯度；"
        "但有限精度下相对次序更难辨，教师微小 logit 偏差也可能影响优化。是否有效须实测。本实验仅扫描到 T=8，"
        "T=8、alpha=0.9 的退化说明高温与高教师权重存在不利耦合，不能外推为 T=100 必然失效。",
        f"答：蒸馏压缩知识到更少参数的学生；量化降低权重和激活的数值精度；剪枝移除权重、通道或结构。三者可以叠加：先训练高质量教师，再蒸馏 0.147M 学生，验证精度后对 Linear 层做 INT8 动态量化，重新测 loss、体积和延迟，最后可做结构化剪枝并短暂微调。本次前两步已将参数压缩 5.0x，量化文件再缩小约 {quant['size_compression_ratio']:.2f}x。",
    ]
    thought_heading_index = next(
        index for index, paragraph in enumerate(document.paragraphs) if paragraph.text.startswith("十一、")
    )
    questions = [
        p
        for p in document.paragraphs[thought_heading_index + 1 :]
        if p.text[:2] in {"1.", "2.", "3.", "4.", "5.", "6."}
    ]
    if len(questions) != 6:
        raise ValueError(f"未找到 6 道思考题，实际 {len(questions)}")
    for question, answer in zip(questions, answers):
        question.paragraph_format.keep_with_next = True
        answer_paragraph = insert_after(question, answer, bold_lead="答：")
        answer_paragraph.paragraph_format.line_spacing = 1.30
        answer_paragraph.paragraph_format.space_after = Pt(4)

    if hpc:
        hpc_baseline = hpc.get("baseline")
        if hpc_baseline:
            hpc_note = (
                f"HPC CPU 复现：scratch loss={hpc_baseline['scratch']['val_loss']:.4f}，"
                f"KD loss={hpc_baseline['kd']['val_loss']:.4f}，与本机结果差异仅来自运行环境和浮点实现。"
            )
            environment_heading = next(p for p in document.paragraphs if p.text == "二、实验环境")
            insert_after(environment_heading, hpc_note, bold_lead="HPC CPU 复现：")

    replace_preserving_run(next(p for p in document.paragraphs if p.text.startswith("八、拓展加分实验")), "八、拓展加分实验")

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text.startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、", "十一、")) or text.startswith(("7.1", "7.2", "7.3", "7.4")):
            paragraph.paragraph_format.keep_with_next = True

    core = document.core_properties
    core.title = "自制语言模型的知识蒸馏实验报告"
    core.subject = "人工智能 深度学习大模型智能体 实验作业八"
    core.author = "孙驰"
    core.keywords = "知识蒸馏, Transformer, 消融实验, 动态量化"
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hpc-results", type=Path)
    args = parser.parse_args()
    build_report(args.template, args.results, args.system, args.output, args.hpc_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
