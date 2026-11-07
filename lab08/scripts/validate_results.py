from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_curve(values: Any, expected_length: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected_length
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
    )


def verify_quantized_state_dicts(parent: Path, data: dict[str, Any]) -> bool:
    lab_dir = parent.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    import distill_lm as lab

    fp32_model = lab.new_student(data["data"]["vocab_size"], data["hyperparameters"]["seed"])
    fp32_state = torch.load(parent / data["dynamic_quantization"]["fp32_file"], map_location="cpu", weights_only=True)
    fp32_model.load_state_dict(fp32_state)
    try:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
    except AttributeError:
        quantize_dynamic = torch.quantization.quantize_dynamic
    int8_model = quantize_dynamic(copy.deepcopy(fp32_model).eval(), {nn.Linear}, dtype=torch.qint8)
    int8_state = torch.load(parent / data["dynamic_quantization"]["int8_file"], map_location="cpu", weights_only=True)
    int8_model.load_state_dict(int8_state)
    tokens = torch.zeros((1, 16), dtype=torch.long)
    with torch.no_grad():
        fp32_logits = fp32_model(tokens)[0]
        int8_logits = int8_model(tokens)[0]
    expected_shape = (1, 16, data["data"]["vocab_size"])
    return (
        tuple(fp32_logits.shape) == expected_shape
        and tuple(int8_logits.shape) == expected_shape
        and bool(torch.isfinite(fp32_logits).all())
        and bool(torch.isfinite(int8_logits).all())
    )


def validate(result_path: Path) -> dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    parent = result_path.parent
    checks: list[dict[str, Any]] = []
    expected_teacher_config = {
        "n_layer": 2, "n_head": 4, "n_embd": 160, "block_size": 96, "dropout": 0.0
    }
    expected_student_config = {
        "n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 96, "dropout": 0.0
    }
    require(data["configs"]["teacher"] == expected_teacher_config, "教师结构配置精确匹配正式方案", checks)
    require(data["configs"]["student"] == expected_student_config, "学生结构配置精确匹配正式方案", checks)
    hyper = data["hyperparameters"]
    require(
        (hyper["teacher_iters"], hyper["student_iters"], hyper["ablation_iters"])
        == (200, 120, 80),
        "正式训练步数为 200/120/80",
        checks,
    )
    require(hyper["batch_size"] == 24 and hyper["seed"] == 0, "正式批大小与随机种子正确", checks)
    require(
        math.isclose(hyper["temperature"], 4.0) and math.isclose(hyper["alpha"], 0.5),
        "正式 KD 温度与权重正确",
        checks,
    )
    counts = data["model_counts"]
    require(counts["teacher"] == 737280, "教师参数量为 737280", checks)
    require(counts["student"] == 147456, "学生参数量为 147456", checks)
    require(math.isclose(counts["compression_ratio"], 5.0), "参数压缩比为 5.0x", checks)

    baseline = data["baseline"]
    scratch = baseline["scratch"]
    kd = baseline["kd"]
    require(scratch["params"] == kd["params"] == counts["student"], "两名学生参数量独立核对一致", checks)
    require(scratch["iterations"] == kd["iterations"] == 120, "两名学生训练步数独立核对一致", checks)
    require(scratch["batch_seed"] == kd["batch_seed"] == 1, "两名学生批次随机种子独立核对一致", checks)
    require(finite_curve(data["teacher"]["curve"], 200), "教师曲线含 200 个有限训练点", checks)
    require(finite_curve(scratch["curve"], 120) and finite_curve(kd["curve"], 120), "两名学生曲线各含 120 个有限训练点", checks)
    require(kd["feature_weight"] == 0.0, "正式基线 KD 未混入特征损失", checks)
    teacher_loss = data["teacher"]["val_loss"]
    scratch_loss = scratch["val_loss"]
    kd_loss = kd["val_loss"]
    require(teacher_loss < scratch_loss, "教师留出集 loss 低于学生基线", checks)
    require(kd_loss < scratch_loss, "KD 学生优于从零学生", checks)
    require(kd["agreement"] > scratch["agreement"], "KD 提高教师一致率", checks)

    rows = data["ablation"]["rows"]
    pairs = {(row["temperature"], row["alpha"]) for row in rows}
    expected = {(t, a) for t in (1.0, 2.0, 4.0, 8.0) for a in (0.3, 0.5, 0.7, 0.9)}
    require(len(rows) == 16 and pairs == expected, "4x4 消融的 16 组均存在且唯一", checks)
    require(
        all(
            row["iterations"] == 80
            and finite_curve(row["curve"], 80)
            and math.isclose(row["train_final_loss"], row["curve"][-1])
            for row in rows
        ),
        "每个消融组合均保存 80 个有限训练点且末值一致",
        checks,
    )
    require(data["ablation"]["cache_seed"] == 2024, "消融共享批次缓存种子正确", checks)
    require(data["ablation"]["best"] == min(rows, key=lambda row: row["val_loss"]), "消融最优行计算正确", checks)
    require(data["ablation"]["worst"] == max(rows, key=lambda row: row["val_loss"]), "消融最差行计算正确", checks)

    feature = data["feature_distillation"]
    require(feature["feature_weight"] > 0 and feature["projection_params"] > 0, "FitNet 投影层与权重已启用", checks)
    require(
        finite_curve(feature["curve"], 120)
        and finite_curve(feature["feature_curve"], 120),
        "FitNet 的总损失与特征损失各保存 120 个有限点",
        checks,
    )
    scan_rows = data["teacher_scale_scan"]["rows"]
    require([row["teacher_width"] for row in scan_rows] == [96, 160, 256], "教师宽度 96/160/256 扫描完整", checks)
    require(all(row["scratch_val_loss"] == scratch_loss for row in scan_rows), "规模扫描复用同一从零学生基线", checks)
    require(
        all(
            row["teacher_iterations"] == 200
            and row["student_kd_iterations"] == 120
            and row["teacher_batch_seed"] == row["student_kd_batch_seed"] == 1
            and finite_curve(row["teacher_curve"], 200)
            and finite_curve(row["student_kd_curve"], 120)
            for row in scan_rows
        ),
        "三组规模扫描均保存完整有限曲线、步数与批次种子",
        checks,
    )
    quant = data["dynamic_quantization"]
    require(quant["int8_bytes"] < quant["fp32_bytes"], "动态量化减小序列化模型体积", checks)
    require(math.isfinite(quant["int8_val_loss"]) and math.isfinite(quant["val_loss_delta"]), "量化后 loss 为有限值", checks)
    require(
        quant["serialization"] == "state_dict for both fp32 and int8",
        "FP32/INT8 使用同口径 state_dict",
        checks,
    )
    require(verify_quantized_state_dicts(parent, data), "验证器独立重载 FP32/INT8 state_dict 并完成有限前向", checks)

    require((parent / "curves.png").is_file(), "训练与验证曲线图存在", checks)
    require((parent / "ablation_heatmap.png").is_file(), "消融热力图存在", checks)
    require((parent / "teacher_scale_scan.png").is_file(), "教师规模扫描图存在", checks)
    require(data["data"]["text_identical"] is False, "训练文本与留出文本不完全相同", checks)
    artifacts = data["artifacts"]
    source_path = parent.parent / artifacts["source"]["file"]
    require(source_path.is_file(), "结果清单指向的源代码存在", checks)
    require(sha256_file(source_path) == artifacts["source"]["sha256"], "源代码 SHA-256 与结果清单一致", checks)
    expected_checkpoints = {
        "teacher.pt", "student_scratch.pt", "student_kd.pt",
        "student_kd_fp32_state.pt", "student_kd_int8_state.pt",
    }
    checkpoint_manifest = artifacts["checkpoints"]
    require(expected_checkpoints.issubset(checkpoint_manifest), "核心检查点均写入制品清单", checks)
    for name in sorted(expected_checkpoints):
        checkpoint_path = parent / name
        entry = checkpoint_manifest[name]
        require(
            checkpoint_path.is_file()
            and checkpoint_path.stat().st_size == entry["bytes"]
            and sha256_file(checkpoint_path) == entry["sha256"],
            f"检查点 {name} 的大小与 SHA-256 一致",
            checks,
        )
    return {
        "result_path": str(result_path.resolve()),
        "passed": True,
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.result)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
