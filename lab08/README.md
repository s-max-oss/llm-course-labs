# 实验八 自制语言模型的知识蒸馏

本实验在纯 CPU 上训练字符级 Transformer 教师和学生模型，严格控制学生初始化、训练步数与批次顺序，仅改变损失函数，比较硬标签训练与知识蒸馏。代码还包含完整的 `T × alpha` 消融、FitNet 特征蒸馏、教师规模扫描和动态 INT8 量化。

## 运行方法

```powershell
python distill_lm.py --mode quick
python distill_lm.py --mode teacher --retrain
python distill_lm.py --mode all
python distill_lm.py --mode all --feat 0.1
python distill_lm.py --mode ablate
python distill_lm.py --mode full --retrain
```

`--mode all` 默认比较硬标签学生和普通 KD 学生；显式加入 `--feat 0.1` 时，KD 学生会同时启用特征蒸馏。`--mode full` 则固定保留普通 KD 公平基线，并另跑权重 0.1 的 FitNet 扩展、16 组消融、教师规模扫描与动态量化。所有语料由固定种子生成，无需下载模型或外部数据。教师缓存携带配置、语料哈希和训练记录；参数或数据变化时必须用 `--retrain` 重建，避免误用旧权重。

## 本机正式结果

运行环境为 Windows 11、Python 3.13.12、PyTorch 2.12.0+cpu、Intel Core Ultra 5 125H，训练显式限制为 8 个 PyTorch 线程。

| 模型 | 参数量 | 步数 | 留出集 loss | PPL | 教师一致率 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 教师 | 737,280 | 200 | 0.8322 | 2.2984 | - | 12.47 s |
| 学生 从零 | 147,456 | 120 | 1.2722 | 3.5688 | 0.6128 | 3.16 s |
| 学生 KD | 147,456 | 120 | 1.2107 | 3.3557 | 0.6556 | 6.20 s |

教师到学生的参数压缩比为 `5.0x`。在相同学生参数量、初始化和数据顺序下，KD 将留出集 loss 相对降低 `4.84%`，PPL 降低 `0.2131`，教师 top-1 一致率提高 `4.27` 个百分点。

完整消融共 16 组，每组训练 80 步。最优组合为 `T=8, alpha=0.3`，留出集 loss 为 `1.4008`；最差组合为 `T=8, alpha=0.9`，loss 为 `1.6722`。这说明高温度配合过高教师权重会削弱硬标签约束。

FitNet 权重为 0.1 时留出集 loss 为 `1.2190`，略差于普通 KD 的 `1.2107`，表明额外特征约束并非必然提升。动态 INT8 量化在 FP32/INT8 均保存 `state_dict` 的同口径下，把序列化体积从 673,926 字节减至 323,856 字节，压缩 `2.08x`，且重载验证通过；loss 仅增加 `0.0008`。但此极小模型在单样本 CPU 前向中反而变慢，因此不能把体积收益等同于速度收益。

教师宽度扫描的结果为：`C=96` 时教师 loss `0.9188`、KD 相对改善 `2.44%`；`C=160` 时为 `0.8322` 与 `4.84%`；`C=256` 时为 `0.7912` 与 `1.18%`。更宽教师自身更强，但固定 120 步下的学生收益并不单调，说明教师强度、容量差距和软分布可学习性需要共同考虑。

HPC 作业 `17477` 完成全流程复现，作业 `17478` 完成基线和三教师扫描；修正启动脚本后，作业 `17513` 又用显式 `LAB08_DIR`/`LAB08_PYTHON` 完成教师级冒烟验证，三者均以退出码 0 结束。HPC 环境为 Python 3.10.20、PyTorch 2.6.0+cu124（本实验显式使用 CPU），符合实验指南的 Python 3.10–3.12 范围；本机 Python 3.13.12 结果作为交叉复现。HPC 的从零学生 loss 为 `1.2722`，KD loss 为 `1.2101`，与本机结果一致到约 0.001。

## HPC 提交

先把整个 `lab08` 目录上传到计算节点并创建 Slurm 输出目录，再显式指定实验目录和带 PyTorch 的解释器：

```bash
cd ~/llm_course_lab08
mkdir -p results
sbatch --export=ALL,LAB08_DIR="$PWD",LAB08_PYTHON="$HOME/llm_course_lab08/.venv/bin/python",LAB08_MODE=full hpc/lab08_cpu.sbatch
```

若复用统一课程环境，只需把 `LAB08_PYTHON` 改为该环境的 Python 绝对路径。脚本会先检查实验入口、解释器可执行性和 `import torch`，不会静默借用其他实验的虚拟环境。

## 目录

```text
lab08/
├─ distill_lm.py                 # 单文件实验实现
├─ hpc/lab08_cpu.sbatch          # HPC CPU 复现作业
├─ runs_distill/                 # 正式 JSON、图表与本地权重
├─ tests/test_distill_lm.py      # 核心公式与公平性测试
├─ scripts/validate_results.py   # 结果完整性检查
├─ provenance/                   # 原始控制台日志与失败记录
└─ report/                       # Word/PDF 报告与报告检查
```

训练集与留出集使用不同随机种子生成，但参考代码会把同一组 16 首固定诗句附加到两边。报告据实披露该少量精确重合，不将结果表述为完全无重叠测试集。
