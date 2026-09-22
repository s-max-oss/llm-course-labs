# 大模型课程实验代码仓库

《人工智能——深度学习大模型智能体》配套实验作业代码仓库。

- **作者**:孙驰(2024311281)
- **AI 协作方式**:各实验目录记录实际使用工具；实验一原记录为 DeepSeek，实验二使用 Codex，详见各实验的协作记录。
- **环境**:本地 Anaconda 虚拟环境 `llm_course`；实验二正式结果来自 HPC 的 PyTorch CUDA 环境，详见各实验目录。

## 目录

| 实验 | 内容 | 说明 |
|---|---|---|
| [lab01](./lab01) | 使用大模型设计更好的神经网络 | MLP 基线 + 8 组控制变量对照实验 + 失败实验 + 最优组合(97% 准确率) |
| [lab02](./lab02) | ConvLSTM 的算法应用与改进 | 弹跳小球预测、参数量核对、6组单变量对照、组合3次复测；[Word/PDF报告](./lab02/report/)已完成 |
| [lab03](./lab03) | 手搓最小LLM，CPU训练 | MiniGPT基线、8个训练变体、温度/top-k采样及禁止重复3-gram；代码、结果与[Word/PDF报告](./lab03/report/)已完成 |

## 本地课程资料归档

实验一、二、三采用相同的本地组织方式：

```text
课程实验/
├─ 实验指南与模板/
├─ 提交材料/                  # 各实验正式 Word、PDF
├─ lab01_neural_network/      # 实验一代码、结果、图表
│  └─ tools/                  # 报告生成与检查资料
├─ lab02_convlstm/            # 实验二代码、结果、图表
│  └─ tools/                  # 报告生成与检查资料
├─ lab03_min_llm/             # 实验三代码、分组结果、生成文本
│  └─ tools/                  # 报告生成、指南文字与归档检查
└─ labs_repo/                 # 当前 Git 仓库
   ├─ lab01/
   ├─ lab02/                 # results/ 原始证据；report/ 报告副本
   └─ lab03/                 # CPU字符语言模型、原始结果与report/报告副本
```

本地归档与Git目录保留相同的正式结果文件；后续代码修改、运行说明和上传以Git目录为准。

## 通用环境搭建

```bash
conda create -n llm_course python=3.11 -y
conda activate llm_course
pip install torch torchvision scikit-learn matplotlib numpy
```

> 每个实验目录内的 README 包含该实验的详细说明、运行方法与结果分析;所有实验数据均为真实运行结果,可按对应目录中的说明复现。
