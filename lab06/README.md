# 实验六：SmolVLM-256M 网格定位与 RSI

以 4×4 网格定位彩色形状，使用 LoRA SFT 建立格式基线，再用图像高亮自查和像素验证器完成两轮纠错、自举与复评。运行保留全部逐题预测、损失、适配器和反馈记忆。

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

## 正式结果与边界

2026-10-22，Slurm 作业 15391 在 A30 上完成，退出码 0:0，墙钟 4分16秒；运行后队列为空。基线、10步 SFT、SFT+验证器、RSI 第1轮、第2轮的留出集总分分别为 **0/15、4/15、7/15、7/15、4/15**，定位子集均为 **0/8**，定位平均IoU均为0。两轮训练图反馈中分别新增 2、0 条验证样本，池规模从 15 变为 17；反馈样本只有计数与存在性，没有定位。因此不能声称模型已经学会精准定位，第二轮还出现了退化。

本次4道计数题的标准答案均为1，工具后的E10/E11还出现重复数字“1. 1. …”。按指南的数量口径二者计数正确，表达质量则不合格；剔除这两条异常后，工具辅助完整回答为5/15。原始输出及异常清单保留在 `results/` 与 `analysis/summary.json`。

旧作业15350向模型直接提示像素答案，还从无效复述中提取IoU，保留于 `provenance/oracle_assisted_15350/` 作为历史对照；正式报告和主结果已使用修正后的15391。验证清单可用 `python analyze_results.py --verify-only` 复查，无需重训。
