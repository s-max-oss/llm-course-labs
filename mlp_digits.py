# -*- coding: utf-8 -*-
"""
实验作业一:使用大模型(DeepSeek)设计更好的神经网络
数据集:sklearn load_digits(8x8 灰度手写数字,1797 样本,10 类)
任务:基线 MLP + 8 组控制变量改进实验 + 失败实验(学习率过大) + 最优组合 3 次复测

AI 协作说明:
- 代码框架由大模型助手(DeepSeek)依据实验指南生成,作者逐行通读并修改;
- 关键设计决策(实验组规划、exp2 保持 Sigmoid、复测种子选择等)为作者本人决策。
"""
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面后端,直接保存图片
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

DATA_SEED = 42          # 数据划分随机种子(所有实验组共用,保证对比公平)
IMG_DIR = "."           # 曲线图输出目录


def load_data():
    """加载数据并做 8:2 分层划分。全部实验组共用同一份划分。"""
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0            # 归一化到 [0, 1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=DATA_SEED)
    X_tr = torch.tensor(X_tr)
    y_tr = torch.tensor(y_tr, dtype=torch.long)  # CrossEntropyLoss 要求 Long
    X_te = torch.tensor(X_te)
    y_te = torch.tensor(y_te, dtype=torch.long)
    return X_tr, X_te, y_tr, y_te


class MLP(nn.Module):
    """可配置 MLP:隐藏层结构 / 激活函数 / Dropout / BatchNorm 均通过参数控制。"""

    def __init__(self, hidden=(32,), act="sigmoid", dropout=0.0, use_bn=False):
        super().__init__()
        act_layer = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
                     "relu": nn.ReLU, "gelu": nn.GELU}[act]
        layers, prev = [], 64
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:                       # 改进点:BatchNorm(线性层后、激活前)
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:                  # 改进点:Dropout
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run(model, X_tr, X_te, y_tr, y_te, opt_name="sgd", lr=0.1,
        epochs=30, wd=0.0, seed=42, verbose=True):
    """训练 + 评估,返回 (history, 耗时)。history 记录每 epoch 训练损失与测试准确率。"""
    torch.manual_seed(seed)                  # 固定参数初始化种子,保证可复现
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"loss": [], "acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)    # 前向 + 损失
        loss.backward()                      # 反向传播
        opt.step()                           # 参数更新
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
    dt = time.time() - t0
    if verbose:
        print(f"  最终测试准确率: {hist['acc'][-1]*100:.2f}%  耗时: {dt:.2f}s")
    return hist, dt


def plot_curves(hist, title, fname):
    """绘制训练损失 + 测试准确率双曲线图并保存。"""
    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, hist["loss"], "b-", linewidth=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("训练损失")
    ax1.set_title(f"{title} - 训练损失")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, np.array(hist["acc"]) * 100, "r-", linewidth=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("测试准确率 (%)")
    ax2.set_title(f"{title} - 测试准确率")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/{fname}", dpi=150)
    plt.close(fig)


def main():
    X_tr, X_te, y_tr, y_te = load_data()
    results = {}   # 汇总全部实验结果,同时写入 results.json 供报告填表

    # 组别配置:名称 / 模型参数 / 优化器参数 / 文件名
    # 基线与其他组的唯一差别就是每组的改动点(控制变量)
    groups = [
        ("baseline",  dict(hidden=(32,),  act="sigmoid"), dict(opt_name="sgd", lr=0.1),  "result_baseline.png"),
        ("exp1",      dict(hidden=(32,),  act="relu"),    dict(opt_name="sgd", lr=0.1),  "result_exp1.png"),
        ("exp2",      dict(hidden=(128, 128), act="sigmoid"), dict(opt_name="sgd", lr=0.1), "result_exp2.png"),
        ("exp3",      dict(hidden=(32,),  act="sigmoid"), dict(opt_name="adam", lr=0.01), "result_exp3.png"),
        ("exp4",      dict(hidden=(32,),  act="sigmoid"), dict(opt_name="sgd", lr=0.01), "result_exp4.png"),
        ("exp5",      dict(hidden=(32,),  act="sigmoid"), dict(opt_name="sgd", lr=0.1, wd=1e-4), "result_exp5.png"),
        ("exp6",      dict(hidden=(32,),  act="sigmoid", dropout=0.2), dict(opt_name="sgd", lr=0.1), "result_exp6.png"),
        ("exp7",      dict(hidden=(32,),  act="sigmoid", use_bn=True), dict(opt_name="sgd", lr=0.1), "result_exp7.png"),
    ]
    print("=" * 60)
    for name, m_kw, o_kw, fname in groups:
        print(f"[{name}] {m_kw} | {o_kw}")
        torch.manual_seed(42)   # 构造前固定种子:各组共享相同初始参数,保证控制变量公平
        model = MLP(**m_kw)
        hist, dt = run(model, X_tr, X_te, y_tr, y_te, seed=42, **o_kw)
        plot_curves(hist, name, fname)
        results[name] = {"model": m_kw, "opt": o_kw,
                         "acc": round(hist["acc"][-1] * 100, 2), "time_s": round(dt, 3)}

    # 失败实验:学习率过大 lr=10.0(基线其余配置不变)
    print("[lr_too_big] sgd lr=10.0")
    torch.manual_seed(42)
    model = MLP(hidden=(32,), act="sigmoid")
    hist, dt = run(model, X_tr, X_te, y_tr, y_te, opt_name="sgd", lr=10.0, seed=42)
    plot_curves(hist, "失败实验: 学习率过大 (lr=10.0)", "result_lr_too_big.png")
    results["lr_too_big"] = {"acc": round(hist["acc"][-1] * 100, 2), "time_s": round(dt, 3)}

    # 加分项:学习率扫描 0.1/1.0/5.0/10.0/20.0,寻找发散临界点
    print("[lr_scan] lr = 0.1, 1.0, 5.0, 10.0, 20.0")
    scan = {}
    plt.figure(figsize=(9, 4))
    for lr in (0.1, 1.0, 5.0, 10.0, 20.0):
        torch.manual_seed(42)   # 同一初始参数下扫描学习率,仅 lr 一个变量
        model = MLP(hidden=(32,), act="sigmoid")
        hist, dt = run(model, X_tr, X_te, y_tr, y_te,
                       opt_name="sgd", lr=lr, seed=42, verbose=False)
        scan[str(lr)] = {"acc": round(hist["acc"][-1] * 100, 2), "time_s": round(dt, 3)}
        plt.plot(range(1, 31), hist["loss"], label=f"lr={lr}")
        print(f"  lr={lr}: acc={hist['acc'][-1]*100:.2f}%")
    plt.xlabel("Epoch"); plt.ylabel("训练损失")
    plt.title("不同学习率下的训练损失曲线(基线结构, SGD)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/result_lr_scan.png", dpi=150)
    plt.close()
    results["lr_scan"] = scan

    # 最优组合:ReLU + 2层x128 + BatchNorm + Adam(0.001),不同初始化种子复测 3 次
    print("[exp8] 最优组合 relu+2x128+bn+adam(0.001), 复测 3 次")
    best_cfg_m = dict(hidden=(128, 128), act="relu", use_bn=True)
    best_cfg_o = dict(opt_name="adam", lr=0.001)
    runs8 = []
    for i, seed in enumerate([42, 123, 2024], start=1):
        torch.manual_seed(seed)   # 复测:不同初始化种子,数据划分保持 DATA_SEED=42
        model = MLP(**best_cfg_m)
        hist, dt = run(model, X_tr, X_te, y_tr, y_te, seed=seed, **best_cfg_o)
        plot_curves(hist, f"最优组合 (第{i}次复测, seed={seed})", f"result_exp8_run{i}.png")
        runs8.append({"seed": seed, "acc": round(hist["acc"][-1] * 100, 2),
                      "time_s": round(dt, 3)})
    results["exp8"] = {"model": best_cfg_m, "opt": best_cfg_o,
                       "runs": runs8,
                       "acc_mean": round(np.mean([r["acc"] for r in runs8]), 2)}

    with open(f"{IMG_DIR}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print("全部实验完成,结果已写入 results.json")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
