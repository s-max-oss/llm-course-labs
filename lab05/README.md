# 实验五 SmolVLM QLoRA 与 DPO

孙驰，2024311281，计算机10班。完成15图/15题基线、NF4 QLoRA SFT 60步、原12对与扩展17对DPO各40步、β=0.05/0.1/0.5扫描、5个新增偏好对、4组手算和9份确认评分。正式计算在HPC A30完成，无LLM API调用。

## 结果

下表采用逐题语义答案复核。正确的属性答案与附加幻觉分开评；“没。”在存在性问句中按口语否定判定。原始规则分数全部保留。

| 阶段 | count /5 | attr /5 | exist /5 | 主集 /15 | 新增题 /5 |
|---|---:|---:|---:|---:|---:|
| BF16基线 | 0 | 3 | 3 | 6 | 1 |
| NF4基线 | 1 | 4 | 2 | 7 | 2 |
| SFT 60步 | 3 | 0 | 3 | 6 | 4 |
| DPO原12对 β=.1 | 1 | 0 | 3 | 4 | 2 |
| DPO扩展17对 β=.05/.1/.5 | 3 | 0 | 3 | 各6 | 各4 |

主三阶段正确率为46.7%→40%→40%。SFT学会简短中文计数格式，属性回答退化，存在性全部否定并错失正例。DPO改善训练偏好目标，未提高任务语义正确率。扩展17对β=.1的全池loss为0.5113→0.3398，chosen平均序列logp却为−8.1749→−9.0331，rejected降得更多（−29.3112→−41.6473）。不能凭loss下降认定视觉事实能力增强。

原规则主三阶段6/15→6/15→4/15：NF4 P08明确green square但附加了圆形/位置错误；DPO的“没”不匹配标准词形。全140输出共有10条语义审计修正，详见`analysis/summary.json`的`semantic_review_changes`；当前规则代码复算原始140条则无变化。语义审计由Codex及独立Codex复核，学生未做独立盲审。所有15题均用于SFT，5道新增题用于扩展DPO，均是训练集复测，不能据此宣称泛化。只有一个训练种子。

## 实现和原框架修正

- 冻结NF4基座、双重量化、BF16计算、分页AdamW32bit；仅语言30层q/k/v/o的LoRA，r=8/α=16，921600/257406528=0.358033%。视觉塔不训练，因此不用样板的0.529%替代实测。
- 原图256×256；所有阶段同为512单视图，64视觉token，不用默认2048多裁剪。
- 问题和答案分开编码并追加EOS，整个prompt标签为−100；因果shift后只对答案位置求序列对数概率和。
- DPO按指南取关闭适配器后的冻结基座作reference，非冻结SFT。参考`eval+no_grad`，缓存detached标量；policy关闭dropout但保留梯度。各组载入同一SFT，无适配器嵌套。
- `softplus(-margin)`为正损失。指南给定0.01例的精算值为0.688159681；正数题设仅作代数输入，不是真实正log概率。
- 训练精确跑60/40个优化步，不受参考两轮/一轮上限限制，始终保存最终适配器。
- 区分有/没有、中文复合数词、不相关否定和矛盾分句。自动规则之外保留可追溯的语义审计。
- `--mode eval`自动逐阶段加载新基座，不把多个PeftModel层层包装；缺少适配器时明确提示跳过。

## 复现

正式环境Python3.10.20、torch2.6.0+cu124、transformers4.57.6、peft0.17.1、bitsandbytes0.48.1，GPU A30。平台已有CUDA版PyTorch，另建环境安装本目录依赖；不修改平台原conda环境。固定模型revision为`7e3e67edbbed1bf9888184d9df282b700a323964`，基座约490MB不入仓库，下载及文件哈希记录在`provenance/model_provenance.json`。

在本目录执行：

```bash
python -m pip install -r requirements.txt
python bootstrap_model.py
python -m unittest test_hw5_dpo -v
python hw5_dpo.py --mode make_data
python hw5_dpo.py --mode infer --model model --offline --device cuda --require_nf4 --output rerun/base
python hw5_dpo.py --mode sft --model model --offline --device cuda --require_nf4 --steps 60 --output rerun/sft
python hw5_dpo.py --mode dpo --model model --offline --device cuda --require_nf4 --steps 40 --pairs augmented --beta .1 --sft_adapter rerun/sft/adapter --output rerun/dpo
python hw5_dpo.py --mode eval --model model --offline --device cuda --require_nf4 --sft_adapter rerun/sft/adapter --dpo_adapter rerun/dpo/adapter --output rerun/eval
python hw5_dpo.py --dpo_check=1.2,0.8,0.9,0.6,0.1
python hw5_dpo.py --dpo_check=-4,-3,-2,-3,0.1
```

完整对照套件（用新的结果目录，保护已存实验）：

```bash
python run_suite.py --model model --results rerun_full --sft_steps 60 --dpo_steps 40
```

HPC使用`setup_hpc.slurm`准备环境和模型、`run_hpc.slurm`申请1张A30运行套件；路径相对于脚本所在目录。两者的默认`.venv`以平台yolo环境为系统包基础。正式作业14595于2026-10-14 00:02:53完成，0:0，耗时5:12；准备14588和烟测14593也已结束。`provenance/jobs.tsv`与空`queue_after.txt`记录资源收尾。

CPU只演示时可明确指定`--device cpu --no_quant --steps 5`，这是FP32+LoRA，不能称为NF4 QLoRA正式结果。当前报告使用GPU正式数据。

## 核验与报告

```bash
python -m unittest test_hw5_dpo -v
python analyze_results.py --verify-only
python analyze_results.py
```

11项行为测试覆盖损失符号/极值、梯度方向、reference无梯度、答案EOS边界、因果shift、否定/计数匹配、图像/真值、默认目录、三阶段新基座评测与语义审计。另经独立小模型PEFT集成检查及真实GPU烟测。

`results/manifest.json`绑定115个原始文件；实际运行源码与套件快照在`provenance/source_at_run.py`和`suite_at_run.py`。当前源码追加三阶段eval入口及评分边界修正，训练函数未改。原始输出JSON和manifest不覆写；派生审计、图和复核记录另存。五份训练适配器均保存了921600个参数，四份DPO的240个张量均相对SFT改变。

报告按更新模板九章填写，完整保留六道思考题、两组自选加两组题设手算、新增偏好对、训练曲线、真实失败案例。三道题×三阶段是九份回答；学生已参考AI建议与依据确认评分，未独立盲审。忠实度/表达评分独立于任务语义判分，P12“没”扣表达完整性，不算事实方向错误。见[报告与检查记录](report/)。

| 目录 | 内容 |
|---|---|
| `data/` | 15张图、图像位置/哈希、15题、SFT与12+5偏好对 |
| `results/` | 7组原始20题输出、220步日志、5份适配器、前后偏好池与manifest |
| `analysis/` | 双口径逐题审计、独立手算CLI结果与汇总 |
| `provenance/` | 原始运行源码、模型版本、作业/资源状态和Slurm日志 |
| `figures/` | 正确率、训练/β曲线、图像与参数日志摘录 |
| `report/` | Word/PDF、9例学生确认分数、独立核验与排版验证 |
