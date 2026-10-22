# 实验六：SmolVLM-256M 网格定位与 RSI

以 4×4 网格定位彩色形状，使用 LoRA SFT 建立格式基线，再用图像高亮自查和像素验证器完成两轮纠错、自举与复评。运行保留全部逐题预测、损失、适配器和反馈记忆。

<!-- IMPROVEMENT_RESULT_BEGIN -->
## 继续优化后的验收结果（V2）

原指南8道定位题从默认10步SFT的0/8提高为8/8。V2在开发集选择`rsi1`之后才生成新的368题，并在同一批题上比较固定V1 SFT与V2：定位 **201→215/256（78.5%→84.0%）**，总计 **260→290/368（70.7%→78.8%）**。旧最终184题已公开，只保留作历史记录，不用于本次调参。

| 新368题上的条件 | 定位 | 计数 | 存在性 | 总计 | 定位IoU |
|---|---:|---:|---:|---:|---:|
| V1独立回答 | 201/256 | 31/80 | 28/32 | 260/368 | 0.2994 |
| V2独立回答 | 215/256 | 45/80 | 30/32 | 290/368 | 0.3200 |
| V2检测框辅助 | 233/256 | 51/80 | 31/32 | 315/368 | 0.3459 |

V2新增3840张常规训练图，强化同色异形等干扰；SFT实际3200微步、1600次优化器更新。两轮各184张新反馈图加15道历史题，只把像素核验通过的模型纠错回答入池；每轮400微步含50次验证样本重放，学习率3e-6，不使用穷举候选纠错。开发定位106→108→111→110→112→111/128，按预定定位优先规则选择第一轮RSI。相对SFT800只多1道开发定位，不能称为稳定的RSI提升；SFT1600总分更高也如实保留。

同题配对定位纠正24题、损失10题，McNemar精确p=0.024；总分纠正47题、损失17题，p<0.001。此结论限于单个训练种子、本合成图分布；定位仍有41题错误。两个模型预算不同，属于连续改进，不作单因素因果归因。

训练作业15408在全部训练与选择完成后触及1小时限时；15427仅续评，选择记录哈希不变，17分16秒完成0:0。两作业累计77.5GPU分钟，分配均已释放，队列为空。12项测试、4760题/4744图隔离与标签审计、2723条评分重算均通过。纯CPU默认六阶段的690.6秒成功记录继续保留。

报告见[Word/PDF](report/)，详解见[定位复盘](LOCALIZATION_REVIEW.md)、[新同题统计与审计](analysis/optimization_v2.json)。V1旧独立109/128、RSI退化及有限候选核验184/184全部保留；后者主要由工具覆盖答案空间保证，不能当模型独立能力。

<!-- IMPROVEMENT_RESULT_END -->

## 与指导书参考脚本的差异

- 40 张训练图与 15 张固定种子 42 的评测图分开。15 条种子数据取自训练图，另 15 条训练图用于 RSI 反馈；评测图不进入训练数据池。因此最后的评测可称为“留出图像”评测。
- 验证器按颜色连通块及矩形/圆形角点区分目标类别，仅通过像素与题目生成证据。纠错触发和数据入池不读 `gt.json` 标准答案。标准答案只用于事后计分。
- RSI 从 10 步种子 SFT 适配器继续训练，每轮再训 10 步；逐轮保存适配器和数据池快照。重复的 `(图像, 题目)` 不重复入池。
- 统一固定 512 像素单视图；语言注意力 q/k/v/o LoRA，r=8、α=16、dropout=0.05。CUDA 默认 NF4 双重量化；CPU 为 FP32 LoRA，二者须在报告中区分。
- 指导书样板中“480px 画布”与指南“256×256”、`make_seed` 与指南 `make_sft` 不一致；本实现为 256×256，命令为 `make_sft`。

## 复现

模型权重使用已有的 `HuggingFaceTB/SmolVLM-256M-Instruct` 本地缓存，不入 Git 仓库。环境版本与完整日志见 `provenance/`；正式运行申请 HPC A30 一张。

```bash
python hw6_rsi_localize.py --mode make_data
python hw6_rsi_localize.py --mode make_sft
python hw6_rsi_localize.py --mode infer --model model --device cuda --offline
python hw6_rsi_localize.py --mode sft --steps 10 --model model --device cuda --offline
python hw6_rsi_localize.py --mode eval --model model --device cuda --offline
python hw6_rsi_localize.py --mode rsi --rounds 2 --steps 10 --model model --device cuda --offline
```

本机已有缓存时，可用 `--device cpu --no-quant` 进行功能验证。`results/baseline.json` 等文件含逐题证据，`rsi_memory.json` 含两轮反馈明细。验证器只向模型提供格式、通过/不通过以及候选格覆盖率，不提供目标答案；内部像素答案保存在JSON用于审计。纠错后的正确率包含工具核验反馈。

单独重跑 `sft` 或 `rsi` 会重新生成15条种子池，并保存 `sft_seed.json`；不接续上一次RSI结束的可变数据池。RSI同一次运行的两轮则持续积累通过核验的样本。高亮自查可从冗长回答中提取候选格以尝试恢复回答，但接受答案与计算IoU必须通过严格格式检查。

## 默认10步参照结果与边界

2026-10-22，Slurm 作业 15391 在 A30 上完成，退出码 0:0，墙钟 4分16秒；运行后队列为空。基线、10步 SFT、SFT+验证器、RSI 第1轮、第2轮的留出集总分分别为 **0/15、4/15、7/15、7/15、4/15**，定位子集均为 **0/8**，定位平均IoU均为0。两轮训练图反馈中分别新增 2、0 条验证样本，池规模从 15 变为 17；反馈样本只有计数与存在性，没有定位。因此不能声称模型已经学会精准定位，第二轮还出现了退化。

本次4道计数题的标准答案均为1，工具后的E10/E11还出现重复数字“1. 1. …”。按指南的数量口径二者计数正确，表达质量则不合格；剔除这两条异常后，工具辅助完整回答为5/15。原始输出及异常清单保留在 `results/` 与 `analysis/summary.json`。

旧作业15350向模型直接提示像素答案，还从无效复述中提取IoU，保留于 `provenance/oracle_assisted_15350/` 作为历史对照；默认参照使用修正后的15391。其验证清单可用 `python analyze_results.py --verify-only` 复查，无需重训。

## 定位改进与复现

定位0/8后的诊断、改进依据、工具归因和学习要点见 [LOCALIZATION_REVIEW.md](LOCALIZATION_REVIEW.md)，逐方案成绩见 [analysis/improvement_results.md](analysis/improvement_results.md)。原10步结果保留，扩展结果写入`improvement/`。

```bash
# 1. 三组语言适配对照；首次运行自动生成完整16格×8类别数据
sbatch run_improvement.slurm
# 2. 视觉注意力、跨模态连接层与语言层共同做LoRA适配
sbatch run_visual_grounding.slurm
# 3. 从视觉1200步适配器出发，4000次更新分别使用不同训练图
sbatch run_augmented_grounding.slurm
# 4. 以上作业全部成功后，再运行选型、联合SFT、两轮RSI和最终测试
sbatch run_final.slurm
# 5. 仅重算评分、验证入池来源、检查数据隔离并画图
python analyze_improvement.py --audit --plot
```

按顺序等待前一作业结束，不并发写同一输出目录。`run_joint_development.slurm`是前期开发对照，只使用开发集和原15题，不打开最终测试。`complete_improvement.py`仅按128题开发定位率选择初始检查点；`finish_grounding.py`再在初始、联合SFT、两轮RSI之间按开发集选择阶段，选择记录先于最终推理保存。最终测试只检验已选模型的独立回答、弱工具、检测框工具及有限候选核验四个条件。

扩展训练中只有LoRA参数更新：视觉+连接层+语言适配共1,614,336参数（0.6255%）。新增4000图从单目标过渡到多干扰物，每128图完整覆盖16格×8类别。图像像素哈希与开发、最终测试分开；每次更新用图可从训练日志追溯。

有限候选核验是额外的工具搜索条件：模型对16格、0–4、yes/no的合法答案序列排序，再由像素验证器依次核验。空间穷尽本合成任务的答案，所以成功率主要由验证器决定；模型贡献看首位命中和核验次数，而不是搜索后正确率。候选完整排序、模型序列对数概率与实际核验次数均留存。该条件也用于RSI反馈收集定位监督，`answer_origin`区分自查、检测框和受限候选。

检测框工具由图像像素计算匹配目标的框，给模型的提示含像素坐标及画布/网格大小，不含GT答案字符串。强提示的内容比原核验反馈更多，结果单列；联合训练部分样本也使用相同中性检测框提示。自举只使用通过像素核验的自由生成或模型评分候选，answer_origin注明来源，GT只用于种子监督和事后评分。

所有训练脚本拒绝在非空输出目录覆盖运行。重现时应使用干净目录，或显式指定新的`--output`并修改后续适配器路径；单独重复运行时不应混用旧数据池、旧预测或旧适配器。

## 纯CPU完整链路复跑

```bash
python run_cpu_validation.py --model /path/to/SmolVLM-256M-Instruct --threads 8
```

本机Python3.11.16、torch2.14.0+cpu、transformers4.57.6、peft0.17.1，默认六阶段（15题基线、10步SFT、带/不带纠错评测、2×10步RSI）全部退出码0，总耗时690.6秒。使用CPU/FP32、不使用NF4，成绩单独保存于`improvement/cpu_reference/`；不能混入A30 NF4主表。代码、基座权重与55张图的RGB像素均核对一致，PNG压缩字节可因环境而异。

Windows初次运行因GBK不能编码模型输出中的异常字符中断，失败日志和原始字节保留于`improvement/cpu_reference_failed_gbk/`。运行器已固定子进程UTF-8输出，随后完整重跑成功。

## V2代码与证据

- `optimize_grounding_v2.py`：协议、历史RGB去重、3840图联合训练、两轮保守RSI、选择后新测试；拒绝覆盖已有目录。
- `resume_v2_evaluation.py`：本次15408限时后的纯评测续跑，不调用训练，不改选择；原始源文件、原日志与续评哈希均保存。
- `analyze_v2.py --audit --plot`：重算像素标签、评分、采样/学习率、来源与选择链接；比较前断言同题ID，再计算配对统计与区间。
- `improvement/final_v2/sources/`及`protocol.json`：实际运行源码和预先固定配置；`selection.json`生成早于新测试。
- `provenance/optimization_v2_run.json`：15408 TIMEOUT与15427 COMPLETED分别记录，不把限时退出写成成功训练作业。
- `provenance/improvement_manifest_v1.json`：提交b50b220的历史清单；`improvement_manifest.json`为当前提交清单。历史清单应对照历史提交，报告更新后不再要求其匹配当前报告。

重现时先完成上文V1，再在干净输出目录提交`run_optimize_v2.slurm`；可用`sbatch --time=02:00:00 run_optimize_v2.slurm`给完整训练和评测预留时间。若出现本次这种“选择与测试生成已完成”的限时，可提交`run_v2_evaluation.slurm`，续评脚本会断言各阶段训练已完整保存。新测试一旦公开不再用于调参。
