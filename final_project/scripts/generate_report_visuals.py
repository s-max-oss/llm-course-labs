#!/usr/bin/env python3
"""Generate restrained academic overview figures for the Chinese final report."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = "#244A67"
BLUE = "#3B82A0"
TEAL = "#4D908E"
GREEN = "#648D63"
AMBER = "#D99A3D"
RED = "#B95C5C"
INK = "#263746"
MUTED = "#647483"
GRID = "#D9E2E8"
PAPER = "#FBFCFD"


def font(size: int, *, bold: bool = False):
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def centered(draw, xy, text, text_font, fill, *, spacing=8):
    draw.multiline_text(xy, text, font=text_font, fill=fill, anchor="mm", align="center", spacing=spacing)


def rounded_box(draw, box, title, subtitle, color, number):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 8, y0 + 10, x1 + 8, y1 + 10), radius=28, fill="#D7DEE3")
    draw.rounded_rectangle(box, radius=28, fill="white", outline=color, width=4)
    draw.rounded_rectangle((x0, y0, x1, y0 + 80), radius=28, fill=color)
    draw.rectangle((x0, y0 + 40, x1, y0 + 80), fill=color)
    draw.text((x0 + 40, y0 + 40), str(number), font=font(30, bold=True), fill="white", anchor="mm")
    centered(draw, ((x0 + x1) / 2, y0 + 142), title, font(38, bold=True), INK)
    centered(draw, ((x0 + x1) / 2, y0 + 225), subtitle, font(25), MUTED, spacing=12)


def arrow(draw, start, end, color=NAVY, width=7):
    draw.line((start, end), fill=color, width=width)
    x, y = end
    sx, sy = start
    if abs(x - sx) > abs(y - sy):
        tip = x - 22 if x > sx else x + 22
        points = [(x, y), (tip, y - 14), (tip, y + 14)]
    else:
        tip = y - 22 if y > sy else y + 22
        points = [(x, y), (x - 14, tip), (x + 14, tip)]
    draw.polygon(points, fill=color)


def pill(draw, box, text, fill, *, text_fill="white", outline=None, size=28):
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline or fill, width=4)
    centered(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font(size, bold=True), text_fill)


def create_research_pipeline():
    width, height = 2600, 1340
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    centered(draw, (width / 2, 95), "从竞赛实践到可审计结论：研究闭环", font(54, bold=True), NAVY)
    centered(draw, (width / 2, 160), "每一步都把输入、协议、结果和可支持的主张锁定在明确边界内", font(28), MUTED)

    items = [
        ("竞赛实践", "红色球形危险源\n相似干扰物与接口约束", BLUE),
        ("数据审计", "500 帧历史 RGB\n隔离 99 张重复风险图像", TEAL),
        ("检测基线", "固定 RF-DETR 检查点\n401 张保留图像", GREEN),
        ("语义复核", "SmolVLM 自由生成\n→ 冻结候选打分", AMBER),
        ("真实级联", "24 个低置信度候选\n预注册融合与停止条件", RED),
        ("机器人接口", "Level A 接受\nLevel B 仅作调试证据", NAVY),
    ]
    boxes = [
        (160, 265, 760, 590), (1000, 265, 1600, 590), (1840, 265, 2440, 590),
        (1840, 720, 2440, 1045), (1000, 720, 1600, 1045), (160, 720, 760, 1045),
    ]
    for index, (item, box) in enumerate(zip(items, boxes), start=1):
        rounded_box(draw, box, *item, index)
    arrow(draw, (775, 430), (975, 430))
    arrow(draw, (1615, 430), (1815, 430))
    arrow(draw, (2140, 605), (2140, 695))
    arrow(draw, (1825, 880), (1625, 880))
    arrow(draw, (985, 880), (785, 880))

    draw.rounded_rectangle((520, 1160, 2080, 1270), radius=30, fill="#EEF4F7", outline=GRID, width=3)
    centered(draw, (1300, 1215), "最终决策：RF-DETR-only　｜　不以新增模型数量代替系统净收益", font(31, bold=True), NAVY)
    image.save(FIGURE_DIR / "research_pipeline_overview_zh.png", dpi=(220, 220))


def create_evidence_map():
    width, height = 2600, 1360
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    centered(draw, (width / 2, 95), "实验结论与证据等级", font(54, bold=True), NAVY)
    centered(draw, (width / 2, 160), "绿色表示在本研究边界内接受；橙色表示停止；红色表示仅保留为诊断证据", font(28), MUTED)

    columns = [190, 505, 1190, 1770, 2070]
    labels = ["层级", "被检验对象", "关键结果", "判定", "可支持的主张"]
    for x, label in zip(columns, labels):
        anchor = "mm" if label in ("层级", "判定") else "lm"
        draw.text((x, 270), label, font=font(29, bold=True), fill=NAVY, anchor=anchor)
    draw.line((80, 320, 2520, 320), fill=GRID, width=4)

    rows = [
        (430, "A", "RF-DETR 固定检查点", "Macro-F1 0.9635\n危险源召回率 0.9892", "接受", GREEN, "域内探索性语义检测"),
        (650, "B", "SmolVLM 真实候选级联", "24 个候选全部复现原类别\n增加 5.11 s CPU 中位时延", "停止", AMBER, "不进入最终决策路径"),
        (870, "Level A", "事件与状态机回放", "模式、哈希、状态转移\n故障闭锁均可复现", "接受", GREEN, "接口契约与确定性回放"),
        (1090, "Level B", "RGB-D 几何回放", "几何链可运行，但读取\n/Odometry_gazebo 位姿真值", "诊断", RED, "不可宣称自主定位或竞赛合规"),
    ]
    for y, level, object_text, result, status, color, claim in rows:
        pill(draw, (80, y - 54, 300, y + 54), level, color, size=27)
        draw.text((505, y), object_text, font=font(31, bold=True), fill=INK, anchor="lm")
        draw.multiline_text((1190, y), result, font=font(27), fill=INK, anchor="lm", spacing=12)
        pill(draw, (1655, y - 48, 1885, y + 48), status, "white", text_fill=color, outline=color, size=28)
        draw.multiline_text((2070, y), claim, font=font(25), fill=MUTED, anchor="lm", spacing=10)
        draw.line((80, y + 108, 2520, y + 108), fill="#E7EDF1", width=3)

    draw.rounded_rectangle((700, 1240, 1900, 1325), radius=28, fill="#EEF4F7", outline=GRID, width=3)
    centered(draw, (1300, 1282), "结论边界：已验证“什么”与尚未验证“什么”同等重要", font(29, bold=True), NAVY)
    image.save(FIGURE_DIR / "evidence_boundary_map_zh.png", dpi=(220, 220))


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    create_research_pipeline()
    create_evidence_map()
    for name in ("research_pipeline_overview_zh.png", "evidence_boundary_map_zh.png"):
        print(FIGURE_DIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

