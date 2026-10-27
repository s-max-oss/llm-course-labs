# 大模型课程实验代码仓库

《人工智能——深度学习大模型智能体》配套实验作业代码仓库。

阅读复习可先看[实验一至六快速理解与复习](STUDY_GUIDE_LABS01-06.md)，其中按任务串起核心原理、真实结果和评测边界。

- **作者**:孙驰(2024311281)
- **AI 协作方式**:各实验目录记录实际使用工具；实验一原记录为 DeepSeek，实验二至六使用 Codex，详见各实验的协作记录。
- **环境**:本地 Anaconda 虚拟环境 `llm_course`；实验二、五、六的正式 GPU 结果来自 HPC，CPU 复跑与 GPU 结果分别记录，详见各实验目录。

## 目录

| 实验 | 内容 | 说明 |
|---|---|---|
| [lab01](./lab01) | 使用大模型设计更好的神经网络 | MLP 基线 + 8 组控制变量对照实验 + 失败实验 + 最优组合(96.30% 准确率均值) |
| [lab02](./lab02) | ConvLSTM 的算法应用与改进 | 弹跳小球预测、参数量核对、6组单变量对照、组合3次复测；[Word/PDF报告](./lab02/report/)已完成 |
| [lab03](./lab03) | 手搓最小LLM，CPU训练 | MiniGPT基线、8个训练变体、温度/top-k采样及禁止重复3-gram；代码、结果与[Word/PDF报告](./lab03/report/)已完成 |
| [lab04](./lab04) | Vector RAG、GraphRAG与WikiRAG对比 | 本地BGE、20题评测、K扫描、RRF及DeepSeek现场知识构建/生成；九例评分由本人参考初评确认 |
| [lab05](./lab05) | SmolVLM QLoRA与DPO偏好对齐 | A30正式SFT60步、DPO40步、12/17偏好对与β扫描；代码、结果及[Word/PDF报告](./lab05/report/)已完成 |
| [lab06](./lab06) | SmolVLM 网格定位与 RSI | 指南定位0/8→8/8；同题新测试定位201→215/256、总分260→290/368；保留RSI回退、工具对照及[三页报告](./lab06/report/) |

## 实验六提交状态（2026年10月27日）

已对照最新课件、实验六指导书和模板完成提交前检查。报告补齐六阶段命令、LoRA 参数与默认可训练比例、工具纠错转移统计；默认 A30/NF4 成绩与 CPU/FP32 复跑证据分开，正文保持三页。

- 正式报告：[Word](./lab06/report/2024311281_孙驰_实验作业六.docx)／[PDF](./lab06/report/2024311281_孙驰_实验作业六.pdf)。
- 检查记录：[报告与要求核验](./lab06/report/validation.json)、[数据与评测审计](./lab06/analysis/optimization_v2.json)。先前通过的12项测试，其训练代码和日志哈希未变；HPC作业均已结束。
- 最新同题结果为 V2 独立定位215/256、总分290/368；检测框辅助总分315/368，工具收益单列。V1旧测试109/128保留为历史，不能与新题直接比较百分比。

## 本地课程资料归档

实验一至六采用相同的本地组织方式：

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
├─ lab04_rag/                # 实验四代码、检索结果、报告与tools
├─ lab05_smolvlm/            # 实验五代码、结果与报告
├─ lab06_rsi/                # 实验六代码、结果与报告
└─ labs_repo/                 # 当前 Git 仓库
   ├─ lab01/
   ├─ lab02/                 # results/ 原始证据；report/ 报告副本
   ├─ lab03/                 # CPU字符语言模型、原始结果与report/报告副本
   ├─ lab04/                 # RAG代码、逐题结果与report/报告副本
   ├─ lab05/                 # SmolVLM SFT、DPO与报告副本
   └─ lab06/                 # 网格定位、RSI与报告副本
```

本地归档与Git目录保留相同的正式结果文件；后续代码修改、运行说明和上传以Git目录为准。

## 通用环境搭建

```bash
conda create -n llm_course python=3.11 -y
conda activate llm_course
pip install torch torchvision scikit-learn matplotlib numpy
```

> 每个实验目录内的 README 包含该实验的详细说明、运行方法与结果分析;所有实验数据均为真实运行结果,可按对应目录中的说明复现。
