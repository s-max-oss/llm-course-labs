# 大模型课程实验代码仓库

《人工智能——深度学习大模型智能体》配套实验作业代码仓库。

- **作者**:孙驰(2024311281)
- **AI 协作方式**:实验指南要求使用 TRAE IDE,课程允许使用任意一种大模型;本仓库全部实验使用 **DeepSeek 大模型助手**完成 AI 协作环节(代码生成、调试答疑、实验分析)
- **环境**:Anaconda 虚拟环境 `llm_course`(Python 3.11 + PyTorch 2.x CPU),详见各实验目录

## 目录

| 实验 | 内容 | 说明 |
|---|---|---|
| [lab01](./lab01) | 使用大模型设计更好的神经网络 | MLP 基线 + 8 组控制变量对照实验 + 失败实验 + 最优组合(97% 准确率) |
| lab02 | 待更新 | |
| lab03 | 待更新 | |

## 通用环境搭建

```bash
conda create -n llm_course python=3.11 -y
conda activate llm_course
pip install torch torchvision scikit-learn matplotlib numpy
```

> 每个实验目录内的 README 包含该实验的详细说明、运行方法与结果分析;所有实验数据均为真实运行结果,可按对应目录中的说明复现。
