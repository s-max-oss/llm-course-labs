# 期末报告证据索引

更新时间：2026-11-10

本索引只收录已经运行、可复核的结果。RF-DETR、VLM 候选级实验和 development-only RF+B2 级联证伪均已完成；锁定级联和多视角未运行，不得填写推测数字。

最终报告交付物为 `deliverables/2024311281_孙驰_期末报告_RF-DETR_SmolVLM.docx`，可编辑来源为 `final-report.md`；报告中的五张图由封存指标生成，数值边界以本索引和对应 JSON 证据为准。

## 可直接引用的结论

| 报告结论 | 数值 | 一级证据 | 边界 |
|---|---:|---|---|
| 原始数据规模 | 500 张图像、426 个标注框、117 个缺标签背景约定帧 | `data/manifests/batch_500.summary.json`、`batch_500.candidates.summary.json` | 历史仿真数据，不是新增独立场景 |
| 评测语料内部重复控制后规模 | 隔离 99 张，保留 401 张、444 个候选 | `data/splits/pilot_v1/protocol.json` | 仅控制当前语料内部重复；不能证明与 checkpoint 训练数据隔离 |
| RF-DETR 五折运行 | 401/401 图像成功，0 推理错误 | `experiments/pilot_v1/rfdetr_run.json`、五个 `rfdetr_fold_*.jsonl` | 固定生产权重；标签不进入推理；训练重叠未知 |
| RF-DETR 三目标类 | Macro-F1 0.9635 | `experiments/pilot_v1/results/rfdetr_pooled_metrics.json` | box IoU=0.5；green wall 为未计分辅助类 |
| RF-DETR 危险目标 | Precision 0.9734、Recall 0.9892、F1 0.9812；FN=2、FP=5 | 同上 | 对象框级；五 seed 探索性试点 |
| RF-DETR 危险图像存在性 | Recall 0.9944（178 TP、1 FN） | 同上 | 图像级；缺标签帧不是审计负样本 |
| RF-DETR CPU 延迟 | mean 658.2 ms、P50 653.1 ms、P95 743.3 ms | 同上 | 本机 CPU；不能外推 GPU 或机器人端延迟 |
| B1 全五折 | 444/444 严格输出无效；覆盖率、准确率、Macro-F1、危险召回均为 0 | `experiments/pilot_v1/smolvlm_zero_shot_run.json`、`results/smolvlm_zero_shot_metrics.json` | oracle crop；负基线；不能从解释句猜标签 |
| B2-A development | 覆盖率 1.0，准确率 0.4870，Macro-F1 0.2433 | `experiments/pilot_v1/results/b2_a_development_metrics.json` | 只做开发诊断；未运行锁定集 |
| B2-B development | 193/193 有效，准确率 0.8342，Macro-F1 0.6240，危险召回 0.9886 | `experiments/pilot_v1/results/b2_b_development_gate.json` | 背景召回 0；只用于开锁门槛 |
| B2-B locked | 251/251 有效，准确率 0.9084，Macro-F1 0.7217，危险召回 1.0 | `experiments/pilot_v1/results/b2_b_locked_evaluation_metrics.json` | 探索性、oracle crop、非部署 |
| B2-B locked 失败 | 23/23 背景均误判为红方块；背景召回 0 | 同上混淆矩阵 | 不得称为四分类存在性检测器 |
| B1→B2 同子集变化 | 覆盖率 +1.0，准确率 +0.9084，Macro-F1 +0.7217，危险召回 +1.0 | `experiments/pilot_v1/results/smolvlm_b1_b2_locked_comparison.json` | B1 已先观察这些 seed，不是确认性比较 |
| B2-B locked 延迟 | CPU mean 5758.1 ms，P50 5829.7 ms，P95 6064.2 ms | locked metrics | 不满足实时机器人部署；GPU速度尚未测量 |
| RF+B2 development 路由 | 24 个 proposal、22 张图像，占 172 个目标 proposal 的 13.95% | `experiments/pilot_v1/cascade_development_run.json` | 只含 seed 101/202；真实 RF bbox；无 GT/oracle 输入 |
| RF-only→安全级联 | Macro-F1 0.96918→0.96918；危险 Recall 0.98864→0.98864；FP 11→11 | `experiments/pilot_v1/results/cascade_feasibility_development_metrics.json` | 24/24 B2 类别与 RF 相同；无净增益 |
| 真实 proposal B2 延迟 | CPU P50 5114.7 ms、P95 5539.6 ms | 同上 | 每次 VLM 调用；不能外推 GPU |
| 级联决策 | 严格 Macro-F1 增益门槛失败；停止并推荐 RF-only | `experiments/pilot_v1/cascade_development_run.json` | 未在 seed 303/404/505 上运行级联推理，不是确认性结论 |

## 完整性与可复现性

- B2 运行封存：`experiments/pilot_v1/smolvlm_b2_locked_run.json`。
- B2 冻结协议：`configs/smolvlm_b2_protocol.json` 与 `experiments/pilot_v1/smolvlm_b2_protocol.lock.json`。
- 对照复算入口：`scripts/compare_smolvlm_b1_b2_locked.py`。
- RF-DETR 封存：`experiments/pilot_v1/rfdetr_run.json`；完整复算入口：`scripts/validate_rfdetr_run.py`。
- RF-DETR 预运行环境、代码提交、权重和冒烟哈希：`experiments/pilot_v1/rfdetr_preflight.json`。
- RF-DETR checkpoint 谱系审计：`experiments/pilot_v1/rfdetr_checkpoint_provenance.json`；训练样本清单、seed、拆分与哈希不可得，重叠状态为 unknown。
- 级联停止封存：`experiments/pilot_v1/cascade_development_run.json`；完整复算入口：`scripts/validate_cascade_run.py`。
- 一键核查入口：`scripts/run_checks.ps1`。封存主报告对应 57/57；加入 Level A、Level B 调试回放及位姿真值防回退检查后，当前完整回归为 87/87。
- 独立复审首轮发现验证器可接受“结果、指标、清单同步替换”的一致伪造，后续增加 Git evidence commit 字节级信任锚和联合替换测试并通过复审。级联协议预实现复审另纠正一处未匹配 proposal 统计转录；纠正发生在任何级联实现或推理前。

## 机器人接口扩展与完整性更正

| 扩展 | 已验证内容 | 一级证据 | 严格边界 |
|---|---|---|---|
| Level A | 172 帧、226 个检测事件；`HELD_UNLOCALIZED=84`、`REJECTED=77`、`SEARCHING=11` | `experiments/replay_v1/replay_run.json` | 非时序、无三维、无正式结果；仍待 Git evidence anchor |
| Level B v1 | 同步 RGB-D、CameraInfo、TF 和位姿输入；深度质量门失败并停止 | `experiments/level_b_v1/level_b_run.json` | 负结果；不得生成世界坐标结论 |
| Level B v2 调试回放 | 14 帧、28 个检测、确定性时序确认和正式格式结果写入 | `experiments/level_b_v2/level_b_run.json` | 使用 `/Odometry_gazebo` 位姿真值，只是 pose-oracle debug replay |
| Level B 完整性更正 | G2 真值隔离保持失败；危险目标真值与位姿真值分别记录 | `experiments/level_b_v2/research_integrity_correction.json` | 不支持自主定位、比赛合规或完整空间闭环结论 |

Level B 更正不改变主报告的 RF-DETR、SmolVLM 或级联结论。主报告 DOCX 仍与 `e93c619` 字节一致。

## 禁止写入正式结论的内容

- 不得写 RF-DETR+B2 优于 RF-only；development 实测无任何指标变化且增益门槛失败。
- 不得写 B2 能识别空背景；现有证据相反。
- 不得写 B2 是端到端检测器；对象输入来自离线标注框 oracle crop。
- 不得写结果可泛化到新场景；五个历史 seed 已参与方法观察。
- 不得把 RF-DETR 数字称为“严格留出检测器泛化”或“无泄漏检测器评测”；checkpoint 训练来源不足以排除评测重叠。
- 不得把本机 CPU 延迟外推为 3080 Ti 或 3090 GPU 延迟。
- 不得把 Level B v2 称为真值隔离、自主定位或比赛合规空间闭环；其世界坐标使用 Gazebo 位姿真值。

## 当前报告占位项

| 待补结果 | 前置条件 | 当前状态 |
|---|---|---|
| 锁定 seed 上的 RF+B2 对照 | development 全门槛通过 | stopped；门槛失败，禁止运行 |
| 多视角融合 | 单帧级联通过 | postponed；单帧无净增益，当前不执行 |
| 新场景正式泛化验证 | 新增独立且未参与选模的数据 | not started |
| 自主位姿 Level B 回放 | 使用合同允许的 `/odom_fused` 或其他自主位姿重新采集 | optional；不属于当前课程报告完成条件 |
