# 实验二：ConvLSTM 的算法应用与改进

依据《作业二_ConvLSTM的算法应用与改进_实验指南》，完成弹跳小球下一帧预测、参数量核对、六组单变量对照和组合配置三次复测。[返回课程仓库](../README.md)

- 作者：孙驰（2024311281）。
- 正式运行采用哈尔滨工业大学（深圳）HPC 的单张 NVIDIA A30；支持 CPU 本地复现。
- AI 协作实际使用 Codex，记录见 [AI_COLLABORATION.md](AI_COLLABORATION.md)。未虚构 TRAE 操作或 DeepSeek 对话。
- 本目录提供代码、运行说明和实验原始证据；Word/PDF 实验报告待另行填写模板。

## 已完成的正式结果

基线测试 MSE 为 **0.00454742**；组合（2层、64通道、5×5细胞核）三次均值为 **0.00276185**，降低 **39.27%**，MAE 均值为 **0.01028703**。完整表格、耗时和图见 [结果索引](results/RESULTS.md)。

本次加深、加宽和扩大卷积核在验证 MSE 上有效；增加输入帧数未带来改善。L1组在5轮内接近全黑预测，体现稀疏前景下只看单一平均误差的局限。预测图呈现了这一失败现象，未隐藏不理想的实验组。

6项行为测试在本机和HPC均通过。另逐组核对了200个保存预测的指标、数据哈希与模型参数量，并在本机CPU重放各模型的3个样本；与GPU保存结果最大绝对差小于8×10⁻⁷。三次组合的参数和测试预测逐位相同，检查记录见 `results/verification.json`。

## 快速开始

需要 Python 3.10+、PyTorch 2.0+、NumPy 和 Matplotlib，无需联网下载数据。

```bash
cd lab02
python -m pip install -r requirements.txt
python -m unittest test_convlstm -v

# 核验仓库中的完整结果，并重建对比图和结果数据表
python verify_results.py

# 小数据流程检查，结果自动进入独立 smoke_results/，不作为正式实验数据
python convlstm_bounce.py --smoke --suite all

# 正式 CPU 全量实验；保存逐轮断点，重复运行会检查配置并跳过已完成组
python convlstm_bounce.py --suite all --output results_cpu

# 单张 CUDA GPU 全量实验；完整批大小仍为64
python convlstm_bounce.py --suite all --device cuda --micro-batch 64 --output results_gpu
```

仓库已提供正式 `results/`。如果用不同设备、软件版本、微批大小或修改后的代码复跑，请指定新的 `--output`，避免把不同条件的结果混在一起。Windows 可直接使用已有 Anaconda 环境 `llm_course`。`requirements.txt` 是兼容范围，正式环境的精确版本以 `results/environment.json` 为准。

## 实验设置

每条序列共 10 帧，尺寸 32×32，单通道二值图，球半径 2。每个速度分量的绝对值从 0.8–1.6 像素/帧均匀采样，符号随机；这里的范围指 x/y 分量，沿用指南代码的定义。所有模型直接输出实数像素，不加 Sigmoid；MSE/MAE 用未裁剪预测计算，图片显示范围统一为 [0,1]。

| 组 | 相对基线仅改变一项 |
|---|---|
| baseline | 单层 ConvLSTM，隐藏通道32，细胞核3×3，输入前4帧预测第5帧 |
| exp1 | ConvLSTM 层数改为2 |
| exp2 | 隐藏通道改为64 |
| exp3 | 细胞卷积核改为5×5；输出卷积仍为3×3 |
| exp4 | 输入前8帧预测第9帧 |
| exp5 | FlattenLSTM：每帧展平1024维，LSTM hidden=256，Linear(256,1024) |
| exp6 | 训练目标从 MSE 改为 L1；评估仍同时报告 MSE、MAE |
| best_run1–3 | 将验证集上有效的 ConvLSTM 改动组合，种子42相同复测3次 |

所有组统一 Adam(lr=0.001)、5 epoch、有效 batch size=64、随机种子42，不做早停或按测试集挑选 epoch，均取第5轮结果。

CPU 默认微批为8，用梯度累计实现有效批64：每个微批损失按 `微批样本数/当前有效批样本数` 加权，最后不足64条的尾批也正确归一化。模型没有 BatchNorm/Dropout，因此不改变批间统计或随机层行为；单元测试验证其梯度与完整批一致。HPC 正式实验使用微批64，不需要累计。

### 对指南参考代码的必要修正

1. **隔离训练与测试。** 原代码生成2000条后，用全部训练、最后200条测试，造成重叠。本实现用种子42一次生成2200条，前2000条训练、后200条测试，并核验序列 SHA-256 集合无交集。
2. **额外验证集。** 独立种子43生成200条序列，用于选组合；不占用指南的训练或测试样本。仅将最终验证 MSE 低于基线的 ConvLSTM 改动合并，结构替换组只作对照。最终推荐模型与组合复测结果分别保存；组合不一定优于每个单组。
3. **边界反弹。** 将越界位移镜像折回，使球心处于 `[r, size-1-r]`，避免只翻转速度却让球暂时越界、边缘被裁掉。
4. **状态与设备。** 通过输入张量的 `new_zeros` 创建每层状态，保证 CUDA/CPU 和 dtype 一致。
5. **真实平均损失。** 训练损失按全 epoch 样本数加权，替代指南示例中只打印最后一个 batch 的损失。测试分批评估；耗时包含 CUDA 同步。

### 解读限制

- 三次复测均使用同一个种子42，是指南要求的确定性复现检查；零标准差不代表跨随机种子的泛化稳定性。
- exp4 按指南预测第9帧，其余组预测第5帧；目标时刻和边界反弹发生比例有所不同，不能将全部差异只归因于输入帧数。
- 小球只占图像很小一部分，输出全黑也能取得看似较低的误差。每组同时保存全黑、复制末帧两个无需学习的参照，结合预测图判断是否真正学会运动。
- 结果仅说明此合成数据、5轮训练和给定初始化下的表现，不能概括成结构的普遍优劣。

## 参数量手算核对

合并四门的 ConvLSTM 细胞参数量为 `4 × (K² × C_in × C_h + K² × C_h² + C_h)`。

| 部件 | 代入 | 参数量 |
|---|---|---:|
| 基线细胞 | 4 × (3²×1×32 + 3²×32² + 32) | 38,144 |
| 基线输出卷积 | 3²×32×1 + 1 | 289 |
| 基线总计 | 38,144 + 289 | 38,433 |
| 64通道细胞 | 4 × (3²×1×64 + 3²×64² + 64) | 150,016 |
| 全连接 LSTM | 4×256×1024 + 4×256×256 + 2×4×256 | 1,312,768 |
| 全连接输出层 | 256×1024 + 1024 | 263,168 |
| FlattenLSTM 总计 | 1,312,768 + 263,168 | 1,575,936 |

64通道细胞约为32通道的3.933倍。PyTorch `nn.LSTM` 有输入侧和隐藏侧两组 bias，必须都计入；4帧是时间长度，输入特征维度仍是1024，不能写成4096。代码计算记录见 `results/parameter_counts.json`。

## HPC 运行

先在登录节点上传代码，再从实验目录执行：

```bash
sbatch run_hpc.slurm
squeue -u "$USER"
tail -f slurm-JOBID.log
```

脚本申请 `gpu-a30` 分区、1张GPU、4个CPU核、最多2小时；依次运行单元测试、小数据检查和正式套件，作业结束自动释放资源。所有训练都在 Slurm 计算节点运行。若环境位置不同，通过 `export LAB02_PYTHON=/path/to/python` 指定；不修改公共或个人的既有软件环境。

HPC 默认输出 `results/`，匹配的已完成组会跳过。若要从头重跑，先执行 `export LAB02_OUTPUT=results_gpu_new` 再提交；新目录也适用于更换设备或软件版本后的运行。

平台的 Slurm 节点将 `RealMemory` 配成1MB，故按官方示例省略 `--mem`，显式申请数GB会被调度器拒绝。实际设备、软件版本和作业记录另行保存。参考：[集群登录与作业提交](http://hpc.hitsz.edu.cn/docs/zh/manual/cluster-login)、[平台常见问题](http://hpc.hitsz.edu.cn/docs/zh/home)。

中文图默认用 Microsoft YaHei。在 Linux 可将自有系统字体文件路径放入 `LAB02_FONT`；字体不纳入 Git 仓库。缺少该字体时可在已安装 Microsoft YaHei 的 Windows 上重新生成图。

### 断点恢复

- 每个 epoch 原子保存 `*_partial.pt`，包含模型、Adam状态、随机状态、历史指标和耗时；重新运行同一命令继续。
- 已完成组同时检查配置、数据哈希、PyTorch版本、设备与代码哈希，吻合才跳过。
- 正常结束或 Python 异常会释放 `.run.lock`。若节点被强制终止，先用 `squeue`/`sacct` 确认旧作业已停止，再删除该输出目录中的 `.run.lock` 后重新提交。不得在仍有训练进程时删除锁。
- 改动代码或实验条件后，应使用新的输出目录；不要把旧断点当成新实验结果。

## 文件索引

| 文件 | 用途 |
|---|---|
| `convlstm_bounce.py` | 数据合成、模型、训练、选择配置、绘图与续跑 |
| `test_convlstm.py` | 参数公式、门方程、梯度与时序、数据隔离、控制变量检查 |
| `verify_results.py` | 核验数据/源码哈希、全部保存指标、模型重放、复测一致性，生成对比图与数据表 |
| `run_hpc.slurm` | GPU批作业入口 |
| `results/results.json` | 全套原始结果和组合均值 |
| `results/baseline.json`、`exp*.json`、`best_run*.json` | 每轮训练、测试、验证指标及耗时 |
| `results/*.pt` | 训练完成的模型状态和配置 |
| `results/predictions_*.npz` | 200个测试样本的原始预测、真值和索引 |
| `results/result_baseline.png`、`result_exp1.png`–`result_exp6.png` | 逐组损失与误差曲线 |
| `results/result_pred.png` | 组合第1次复测的3个固定测试样本对比 |
| `results/result_recommended.png` | 按验证 MSE 选择的推荐模型预测图 |
| `results/environment.json`、`selection.json` | 正式环境、来源校验值与组合选择规则 |
| `results/hpc_job.json`、`slurm-10044.log` | Slurm完成状态与测试、训练原始日志 |

原始合成数据可通过 `split_data()` 按固定种子重建，文件中保存的数据哈希可以核验生成内容。
