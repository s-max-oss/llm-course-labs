# 大模型课程实验代码仓库

《人工智能——深度学习大模型智能体》配套实验作业代码仓库。

- **作者**:孙驰(2024311281)
- **AI 协作方式**:各实验目录记录实际使用工具；实验一原记录为 DeepSeek，实验二使用 Codex，详见各实验的协作记录。
- **环境**:本地 Anaconda 虚拟环境 `llm_course`；实验二正式结果来自 HPC 的 PyTorch CUDA 环境，详见各实验目录。

## 目录

| 实验 | 内容 | 说明 |
|---|---|---|
| [lab01](./lab01) | 使用大模型设计更好的神经网络 | MLP 基线 + 8 组控制变量对照实验 + 失败实验 + 最优组合(97% 准确率) |
| [lab02](./lab02) | ConvLSTM 的算法应用与改进 | 弹跳小球预测、参数量核对、6组单变量对照、组合3次复测；报告待填写 |
| lab03 | 待更新 | |

## 通用环境搭建

```bash
conda create -n llm_course python=3.11 -y
conda activate llm_course
pip install torch torchvision scikit-learn matplotlib numpy
```

> 每个实验目录内的 README 包含该实验的详细说明、运行方法与结果分析;所有实验数据均为真实运行结果,可按对应目录中的说明复现。
