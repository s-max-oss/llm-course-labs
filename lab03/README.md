# 实验三 手搓最小LLM

按课程指南实现字符级 MiniGPT。作者：孙驰（2024311281，计算机10班）。实际AI协作工具：Codex。当前交付代码及实验原始材料；已收到报告模板，按用户要求先交付代码，正式报告后续填写。

## 从哪里读代码

建议按 `min_llm.py` 中的编号阅读：

1. **语料与标签**：`CharTokenizer`、`get_batch`。输入与标签错开一个字符。
2. **核心注意力**：`CausalSelfAttention`。手写 Q/K/V 投影、多头拆分、缩放点积和因果掩码，没有调用现成 Transformer 层。
3. **模型组合**：`Block`、`MiniGPT`。Pre-Norm、残差、GELU及真实权重共享。
4. **训练**：`train`。CPU训练、独立取样随机数、逐步记录及断点保存。
5. **生成**：`MiniGPT.generate`。自回归温度/top-k采样，以及选做的禁止重复3-gram。

具体实验配置与参数公式见 [PLAN.md](PLAN.md)，实际协作与审查记录见 [AI_COLLABORATION.md](AI_COLLABORATION.md)。

## 已完成的运行

- 指南72段诗文，2163字符，词表699。精确参数量502,656；近似499,072，误差约0.713%。
- 基线CPU训练2000步，末步训练loss **0.029191**，纯训练 **118.77秒**。
- 初始固定训练窗口loss **6.56685**，接近 `ln(699)=6.54965`。
- 八个1000步训练变体全部完成，保留基线1000步快照用于同预算比较。
- 固定基线完成五组采样及禁止重复3-gram选做对照，每组3段；基线另保存「春」「月」各2段。
- 六项行为测试通过；九个最终模型及1000步基线快照通过本机CPU重放核验。
- HPC作业10765完成，退出码0，墙钟时间4分36秒。申请12CPU，最多三组并行，每组4线程；**没有申请或使用GPU**。环境的PyTorch版本虽含CUDA，模型和数据均在CPU。

完整数据、曲线与生成原文见 [results/RESULTS.md](results/RESULTS.md)。这些loss衡量训练语料拟合，未设置独立测试集；很低的loss及大量原诗续写主要反映记忆，不能据此宣称模型具有通用创作或对话能力。

## 本机运行

已配置的Python：`D:\anaconda3\envs\llm_course\python.exe`。以下命令在本目录执行，或先激活 `llm_course` 环境。

```powershell
python -m unittest test_min_llm -v

# 小步数检查，使用单独目录；不混入正式结果
python min_llm.py --iters 20 --output smoke_results

# 完整基线，使用新目录保留原始结果
python min_llm.py --output results_new/baseline

# 单独运行1000步去位置编码实验
python min_llm.py --experiment exp5_no_pos --output results_new/exp5_no_pos

# 本机顺序运行全部实验及采样；每个训练进程4线程
python run_suite.py --workers 1 --threads 4 --output results_new

# 核验已经保存的正式结果，无需重新训练
python analyze_results.py --verify-only --verification-output results/verification_local.json
```

CLI中的 `--experiment` 使用指南预设配置，会覆盖层数、宽度、步数等模型参数；自定义配置时不传该选项。默认自动追加同目录的 `corpus_extra.txt`（如存在）；正式实验未使用扩充语料，保持原始 `corpus.txt` 即可复现。不同语料或代码版本应使用新输出目录。

## 自己试生成

```powershell
python sample.py --prompt 春 --temperature 1.0 --top_k 20
python sample.py --prompt 月 --temperature 1.5 --top_k 20
python sample.py --prompt 春 --temperature 1.0 --top_k 0
python sample.py --prompt 春 --temperature 1.0 --top_k 20 --no_repeat_ngram 3
```

这些命令只加载模型，不重新训练。`top_k=0` 表示不截断。提示词只能使用词表中的字符；若输入未见过的字符，会清楚报错，不会静默丢字。比较采样策略时保持 `--seed` 与提示词相同。

正式生成原文使用HPC的PyTorch 2.6.0；本机环境为2.14.0。不同PyTorch版本的概率采样实现和浮点运算可能使同种子生成不同文本；本机核验的是模型固定输入的loss，而非保证跨版本逐字复现。严格复现生成文本应匹配正式环境版本。

## HPC复跑

`run_hpc.slurm` 申请同一节点的12个CPU核，三组并行，单组4线程；各组仍保持batch32。节点名gpu4和分区名gpu-a30是平台名称，不代表使用GPU。脚本设置 `CUDA_VISIBLE_DEVICES` 为空，也不申请 `--gres`。

```bash
sbatch run_hpc.slurm
```

可用 `LAB03_PYTHON` 指定Python，`LAB03_OUTPUT` 指定新的输出根目录。当前平台节点内存登记为1MB，按平台示例不显式申请内存。最初的9任务数组提交触及账号提交数量上限，改为一个12CPU作业内三进程并行。

## 断点与文件

每250步保存模型、优化器、取批随机状态、历史loss和已完成训练耗时。重新执行相同命令会恢复；操作系统锁在进程退出时释放，`.run.lock` 文件保留是正常现象，不应手工删除正在使用的锁。已完成组按配置、语料、代码哈希及产物哈希核验后跳过。

|文件或目录|用途|
|---|---|
|`min_llm.py`|训练与模型主文件|
|`corpus.txt`、`corpus_provenance.json`|原始语料及指南来源、计数、哈希|
|`test_min_llm.py`|标签、因果性、参数量、权重共享、采样和对照设计检查|
|`run_suite.py`、`run_hpc.slurm`|本地及HPC运行入口|
|`sample.py`|加载模型后交互式尝试提示词与采样配置|
|`analyze_results.py`|核验模型、执行采样对照、生成比较图和数据表|
|`results/各组/`|逐步loss、配置、环境、模型、曲线及生成原文|
|`results/baseline/model_step1000.pt`|与1000步变体比较的基线快照|
|`results/sampling/`|五组采样与选做的完整生成文本和统计|
|`results/verification*.json`|HPC与本机CPU模型核验|

局限：固定种子只说明此初始化；模型大小变化不能保证收益。嵌入宽度组按指南同时改变头数以保持每头32维；短上下文组每步训练字符数下降。质量星级、六道思考题及正式报告将在后续结合生成原文整理。
