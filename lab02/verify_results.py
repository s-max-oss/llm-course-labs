"""Audit saved experimental evidence and regenerate its comparison chart/table."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from convlstm_bounce import (ROOT, Config, build_model, choose_combination,
                             parameter_counts, split_data, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    root = args.results.resolve()
    torch.set_num_threads(4)
    summary = json.loads((root / "results.json").read_text(encoding="utf-8"))
    runs = summary["experiments"]
    expected_names = ["baseline", *[f"exp{i}" for i in range(1, 7)],
                      *[f"best_run{i}" for i in range(1, 4)]]
    assert set(runs) == set(expected_names), "Missing or unexpected experimental groups"
    data = split_data()
    hashes = [hashlib.sha256(part.tobytes()).hexdigest() for part in data]
    source_hash = hashlib.sha256((ROOT / "convlstm_bounce.py").read_bytes()).hexdigest()
    audit = {"source_sha256": source_hash, "data_sha256": hashes, "groups": {}}
    for name in expected_names:
        result = runs[name]
        individual = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        assert result == individual, f"{name}: summary differs from individual result"
        signature = result["signature"]
        assert signature["counts"] == [2000, 200, 200]
        assert signature["data_sha256"] == hashes
        assert signature["source_sha256"] == source_hash
        assert len(result["history"]) == 5
        assert [h["epoch"] for h in result["history"]] == [1, 2, 3, 4, 5]
        assert signature["config"]["seed"] == 42
        assert signature["config"]["batch_size"] == 64
        assert signature["config"]["lr"] == 0.001
        arrays = np.load(root / f"predictions_{name}.npz")
        pred, target = arrays["predictions"], arrays["targets"]
        assert pred.shape == target.shape == (200, 32, 32)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(arrays["sample_indices"], np.arange(200))
        frames = signature["config"]["input_frames"]
        np.testing.assert_array_equal(target, data[1][:, frames, 0])
        error = pred - target
        computed = {"mse": float(np.mean(error**2, dtype=np.float64)),
                    "mae": float(np.mean(np.abs(error), dtype=np.float64))}
        for key in computed:
            np.testing.assert_allclose(computed[key], result["test"][key], atol=1e-12, rtol=1e-10)
            assert result["history"][-1][f"test_{key}"] == result["test"][key]
        saved = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)
        assert saved["config"] == signature["config"]
        model = build_model(Config(**saved["config"]))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == result["parameters"]
        # Different CPU/GPU kernels and PyTorch versions need a numerical tolerance.
        with torch.inference_mode():
            replay = model(torch.from_numpy(data[1][:3, :frames]).float()).numpy()
        max_error = float(np.max(np.abs(replay - pred[:3])))
        np.testing.assert_allclose(replay, pred[:3], atol=2e-5, rtol=2e-4)
        for image in (f"result_{name}.png", f"prediction_{name}.png"):
            assert (root / image).is_file()
        audit["groups"][name] = {"metrics_recomputed": computed,
                                 "cpu_replay_max_absolute_error": max_error,
                                 "passed": True}

    config, selected = choose_combination(runs)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] == selected
    assert selection["test_used_for_selection"] is False
    assert vars(config) == selection["candidate_config"]
    reference_weights = torch.load(root / "best_run1.pt", map_location="cpu", weights_only=True)["state_dict"]
    for i in (2, 3):
        weights = torch.load(root / f"best_run{i}.pt", map_location="cpu", weights_only=True)["state_dict"]
        assert all(torch.equal(reference_weights[k], weights[k]) for k in weights)
        np.testing.assert_array_equal(np.load(root / "predictions_best_run1.npz")["predictions"],
                                      np.load(root / f"predictions_best_run{i}.npz")["predictions"])
    for key in ("mse", "mae"):
        mean = np.mean([runs[f"best_run{i}"]["test"][key] for i in (1, 2, 3)])
        np.testing.assert_allclose(summary["combination_mean"][key], mean, atol=1e-14)
    recommended = min(runs, key=lambda n: runs[n]["validation"]["mse"])
    assert recommended == summary["recommended_model"]
    assert parameter_counts()["verified"]
    audit["fixed_seed_repeats_bitwise_equal"] = True
    audit["passed"] = True
    write_json(root / "verification.json", audit)

    names = expected_names[:7] + ["best_run1"]
    labels = ["基线", "2层", "64通道", "5×5核", "8帧", "全连接LSTM", "L1损失", "组合均值"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, ("mse", "mae")):
        values = [runs[name]["test"][metric] for name in names]
        ax.bar(labels, values, color=["#728696"] + ["#4488ad"] * 6 + ["#cf743b"])
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(runs["baseline"]["references"]["all_black"][metric], color="#bd4949",
                   linestyle="--", label="全黑参照（第5帧）")
        ax.set_title(f"测试 {metric.upper()}（越低越好）")
        ax.set_ylim(0, max(max(values), 0.0122) * 1.18)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("2000条训练 / 200条测试；5轮；8帧组预测第9帧，其余预测第5帧")
    fig.savefig(root / "result_comparison.png", dpi=180)
    plt.close(fig)
    table = ["# 正式实验结果索引", "", "原始数据见同目录 JSON；此文件为实验数据索引，非正式实验报告。", "",
             "| 组 | 测试 MSE | 测试 MAE | 验证 MSE | 参数量 | 纯训练秒 | 训练+逐轮评估秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in expected_names:
        r = runs[name]
        table.append(f"| {name} | {r['test']['mse']:.8f} | {r['test']['mae']:.8f} | "
                     f"{r['validation']['mse']:.8f} | {r['parameters']:,} | "
                     f"{r['training_seconds']:.3f} | {r['train_and_eval_seconds']:.3f} |")
    reduction = 1 - summary["combination_mean"]["mse"] / runs["baseline"]["test"]["mse"]
    table += ["", f"组合：{selected}；测试 MSE 三次均值 {summary['combination_mean']['mse']:.8f}，"
              f"较基线降低 {reduction:.2%}；MAE 均值 {summary['combination_mean']['mae']:.8f}。",
              "", "三次同种子结果逐位相同，属于确定性检查，不能视为三份独立统计样本。",
              "", "计时为GPU同步后的实际时长。纯训练指训练循环（含取批、数据传输、前后向与更新）；训练+评估另含逐轮测试/验证。"
              "两者均不含环境加载、数据生成、绘图、保存、恢复时废弃的未完成轮次。",
              "", "![实验组误差对比](result_comparison.png)", "", "![组合预测](result_pred.png)"]
    (root / "RESULTS.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
