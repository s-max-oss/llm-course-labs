from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest


LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import distill_lm as lab  # noqa: E402


def test_train_and_validation_corpora_are_reproducible_and_distinct() -> None:
    assert lab.build_corpus(seed=0) == lab.build_corpus(seed=0)
    assert lab.build_corpus(seed=0) != lab.build_corpus(seed=999)


def test_validation_characters_are_covered_by_training_vocabulary() -> None:
    data = lab.make_data(block_size=96)
    assert not (set(data.validation_text) - set(data.tokenizer.itos))
    assert data.tokenizer.vocab_size > 100


def test_teacher_and_student_shapes_and_parameter_ratio() -> None:
    data = lab.make_data(block_size=96)
    teacher = lab.GPT(data.tokenizer.vocab_size, lab.TEACHER_CONFIG)
    student = lab.GPT(data.tokenizer.vocab_size, lab.STUDENT_CONFIG)
    tokens = torch.zeros((2, 12), dtype=torch.long)
    teacher_logits = teacher(tokens)[0]
    student_logits = student(tokens)[0]
    assert teacher_logits.shape == student_logits.shape == (2, 12, data.tokenizer.vocab_size)
    ratio = teacher.parameter_count / student.parameter_count
    assert 4.8 < ratio < 5.2


def test_causal_mask_blocks_future_tokens() -> None:
    config = {"n_layer": 1, "n_head": 1, "n_embd": 8, "block_size": 8, "dropout": 0.0}
    model = lab.GPT(16, config).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 3, 9]])
    with torch.no_grad():
        original = model(prefix)[0]
        changed = model(changed_future)[0]
    assert torch.allclose(original[:, :3], changed[:, :3], atol=1e-6)


def test_kd_divergence_has_correct_direction_and_t_squared_scaling() -> None:
    student = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.2, 1.4, -0.3]]])
    temperature = 4.0
    actual = lab.kd_divergence(student, teacher, temperature)
    manual = F.kl_div(
        F.log_softmax(student.reshape(-1, 3) / temperature, dim=-1),
        F.softmax(teacher.reshape(-1, 3) / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    assert torch.allclose(actual, manual)
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_kd_divergence_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 5, 7)
    value = lab.kd_divergence(logits, logits, 4.0)
    assert math.isclose(value.item(), 0.0, abs_tol=1e-5)


def test_students_with_same_seed_have_identical_initialization() -> None:
    data = lab.make_data(block_size=96)
    first = lab.new_student(data.tokenizer.vocab_size, 0)
    second = lab.new_student(data.tokenizer.vocab_size, 0)
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_quick_argument_overrides_are_small_and_isolated(tmp_path: Path) -> None:
    args = lab.parse_args(["--mode", "quick", "--out-dir", str(tmp_path)])
    assert args.mode == "full"
    assert args.teacher_iters == 8
    assert args.student_iters == 6
    assert args.abl_iters == 3
    assert Path(args.out_dir).name == "quick"


def test_all_mode_routes_positive_feature_weight_into_kd(tmp_path: Path) -> None:
    args = lab.parse_args(
        [
            "--mode", "all", "--feat", "0.2", "--alpha", "0", "--student-iters", "1",
            "--batch-size", "2", "--log-every", "1", "--out-dir", str(tmp_path),
        ]
    )
    data = lab.make_data(block_size=96)
    teacher = lab.GPT(data.tokenizer.vocab_size, lab.TEACHER_CONFIG)
    record, _, _ = lab.run_baselines(args, data, teacher, tmp_path)
    assert record["kd"]["feature_weight"] == 0.2
    assert record["kd"]["feature_curve"][0] is not None
    assert record["kd"]["kd_curve"][0] is None


def test_stale_teacher_cache_is_rejected_with_retrain_hint(tmp_path: Path) -> None:
    data = lab.make_data(block_size=96)
    teacher = lab.GPT(data.tokenizer.vocab_size, lab.TEACHER_CONFIG)
    torch.save(teacher.state_dict(), tmp_path / "teacher.pt")
    args = lab.parse_args(["--mode", "teacher", "--teacher-iters", "1", "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="--retrain"):
        lab.train_or_load_teacher(args, data, tmp_path)


def test_teacher_cache_preserves_training_record(tmp_path: Path) -> None:
    data = lab.make_data(block_size=96)
    args = lab.parse_args(
        [
            "--mode", "teacher", "--teacher-iters", "1", "--batch-size", "2",
            "--log-every", "1", "--retrain", "--out-dir", str(tmp_path),
        ]
    )
    _, trained = lab.train_or_load_teacher(args, data, tmp_path)
    args.retrain = False
    _, cached = lab.train_or_load_teacher(args, data, tmp_path)
    assert trained["iterations"] == cached["iterations"] == 1
    assert cached["time_s"] == trained["time_s"]
    assert cached["curve"] == trained["curve"]
    assert cached["cached"] is True


def test_dynamic_quantization_uses_reloadable_state_dicts(tmp_path: Path) -> None:
    data = lab.make_data(block_size=96)
    args = lab.parse_args(["--mode", "quantize", "--batch-size", "2", "--out-dir", str(tmp_path)])
    student = lab.new_student(data.tokenizer.vocab_size, args.seed)
    record = lab.run_dynamic_quantization(args, student, data, tmp_path)
    assert record["fp32_state_dict_roundtrip_verified"] is True
    assert record["int8_state_dict_roundtrip_verified"] is True
    assert (tmp_path / record["fp32_file"]).is_file()
    assert (tmp_path / record["int8_file"]).is_file()
    assert record["int8_bytes"] < record["fp32_bytes"]
