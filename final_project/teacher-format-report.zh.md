# 从竞赛危险源搜索到可审计的多模态感知

From Competition Hazard Search to Auditable Multimodal Perception

第一作者：孙驰    第二作者：曾国坤  
哈尔滨工业大学（深圳）

## Abstract

This study grew out of my participation in the DG-202602 quadruped-robot competition project, in which a robot must search an unfamiliar building, distinguish red spherical hazards from visually similar red blocks and green balls, and return hazard coordinates to the task system. During development, I found that a visually plausible prediction was not enough: a missed proposal could never reach localization, a false alarm could create a nonexistent hazard, and a free-form model response could break the robot interface even when its description sounded reasonable. I therefore extracted a narrower course research question from the competition pipeline: can a compact vision-language model improve a fixed real-time detector on ambiguous proposals without weakening safety, reproducibility, or latency? The study audits 500 historical RGB frames from five simulation seeds, globally isolates 99 duplicate-risk images, and evaluates a fixed RF-DETR-Seg Medium checkpoint on 401 retained images. RF-DETR reaches 0.9635 three-class Macro-F1 and 0.9892 hazard recall, although unavailable training lineage limits the claim to in-domain exploratory performance. Free-form SmolVLM-256M produces no strictly valid single-label outputs. Frozen candidate scoring restores 100% output coverage and reaches 0.7217 Macro-F1 on 251 oracle crops, but rejects none of 23 background cases. On 24 real low-confidence RF-DETR proposals, SmolVLM repeats every detector class, adds 5.11 s CPU median latency, and yields no metric gain. The preregistered continuation gate therefore stops the cascade. The final recommendation is RF-DETR-only, with Level A interface replay accepted and Level B retained only as Gazebo-pose-oracle debugging evidence.

Keywords: quadruped robot; hazard recognition; RF-DETR; SmolVLM; multimodal intelligence

## 摘要

本研究源于我参与的 DG-202602 四足机器人竞赛项目：机器人必须在陌生建筑中搜索，把红色球形危险源与视觉上相近的红色方块、绿色球体区分开，并把危险源坐标回传给任务系统。研发过程中我发现，视觉上“看起来合理”的预测远远不够：漏检的候选永远无法进入定位环节；一次误报会凭空制造出一个并不存在的危险源；而自由形式的模型输出哪怕描述再合理，也可能直接破坏机器人接口。因此我从竞赛流程中抽取出一个范围更窄的课程研究问题：在模糊候选上，紧凑的视觉语言模型能否在不损害安全性、可复现性与时延的前提下，改进一个固定的实时检测器？本研究审计了来自五个仿真种子的 500 张历史 RGB 帧，全局隔离 99 张存在重复风险的图像，并在保留的 401 张图像上评测固定的 RF-DETR-Seg Medium 检查点。RF-DETR 达到 0.9635 的三分类 Macro-F1 与 0.9892 的危险源召回率，但由于训练数据溯源不可得，该结论仅限于域内探索性表现。自由生成的 SmolVLM-256M 无法产出任何严格合法的单标签输出。冻结候选打分恢复了 100% 的输出覆盖率，并在 251 个基于真值框的裁剪（oracle crop）上达到 0.7217 的 Macro-F1，但对 23 个背景样本无一拒识。在 24 个真实低置信度 RF-DETR 候选上，SmolVLM 复现了检测器给出的每一个类别，额外带来 5.11 s 的 CPU 中位时延，且没有任何指标增益。因此，预注册的继续门限判定停止级联。最终建议为仅使用 RF-DETR：Level A 接口回放予以接受，Level B 仅作为使用 Gazebo 位姿真值的调试证据保留。

关键词：四足机器人；危险源识别；RF-DETR；SmolVLM；多模态智能

## 1 引言 Introduction

### 1.1 研究背景与问题定义

本研究的动机并不始于某个榜单排名，而是来自名为“四足机器人自主危险源搜索与识别”的 DG-202602 竞赛项目。在团队会议与仿真器开发中，任务被划分为自主探索与危险源感知两部分。机器人必须进入未知建筑中的房间，识别红色球形危险源，排除红色方块与绿色球体，最终按规定格式给出危险源的三维坐标。因此，视觉决策会影响导航、定位、重复抑制与最终评分，而不止于输出一个边界框。

实践工作暴露了三类失效模式。第一，检测器一旦漏检便不再产生候选，下游语义模块无从复核。第二，视觉上相近的干扰物可能被误报为危险源，污染坐标列表。第三，生成式模型可能正确描述图像，却仍然违反机器人所要求的固定标签或事件模式。运动模糊、部分遮挡、远处小目标、画面边缘截断与重复框，使这些失效在竞赛流程中反复出现。课程项目正源于这一经验，并有意识地把完整的竞赛问题收窄为一个关于视觉语义与接口可靠性的可审计问题。

本研究不宣称复现完整竞赛。它不验证自主探索、在线控制，也不验证真值隔离的世界坐标流水线。它考察的是：在真正可得的那部分证据上，来自专业检测器的视觉智能与来自紧凑 VLM 的多模态智能是否互补。这一范围设定使本报告不会把一项局部感知研究转写为缺乏支撑的系统级主张。

### 1.2 研究目标与核心贡献

第一个目标是：在经过重复风险控制的历史数据上，测量机器人项目已在使用的固定 RF-DETR 检查点的行为。第二个目标是：通过比较自由生成与冻结候选打分，把 SmolVLM 的语义能力与其指令格式合规性分离开。第三个目标是：检验当 VLM 得到的是真实低置信度检测候选，而不是基于真值框的裁剪时，它能否产生可测量的净收益。

本研究做出四项课程层面的贡献：构建了带 SHA-256 封印、按仿真种子组织的数据与证据协议；把生成式 VLM 改造为封闭候选集合打分器；在运行真实候选级联之前预注册路由、融合与停止条件；以及保留级联的负结果，而不是在观察结果后继续调参。Level A 事件回放展示了检测结果如何进入机器人状态机。Level B RGB-D 回放也被保留，但其使用 `/Odometry_gazebo` 的事实已明确披露，且 G2 真值隔离门仍未通过。

### 1.3 论文组织

第 2 节把这一由竞赛衍生的问题与视觉智能、语言智能和多模态智能联系起来，并综述开源与商用方案。第 3 节定义任务、证据边界、模型与冻结级联。第 4 节报告数据集、受控对比、图表与负结果。第 5 节给出 GitHub 仓库与验证入口。第 6 节展示机器人接口回放。第 7 节解释研究发现、局限、课程学习与后续步骤。第 8 节在不引入新证据的前提下收束论证。

## 2 相关工作 Related Work

### 2.1 课程基础知识回顾

本项目融合了课程中的三个智能模块。视觉智能提供目标定位与类别预测。DETR 把检测表述为直接的集合预测，使用目标查询与二分匹配，降低对手工设计锚框（anchor）和非极大值抑制的依赖 [1]。RT-DETR 表明，高效的混合编码器与查询选择可以把这一类方法适配到实时检测 [2]。RF-DETR 进一步通过权重共享的神经架构搜索，为目标数据集探索精度与时延之间的配置 [3]。

语言智能通过自回归的 token 概率进入。VLM 并不直接输出一个机器人类别，它估计的是在给定图像与提示条件下的 token 续写分布。视觉指令微调使这类模型可用于开放式视觉交互 [4]，而 SmolVLM 把模型规模与内存规模压缩到资源受限推理可承受的范围 [5]。关键的课程认识是：生成式接口与判别式机器人契约是两种不同的对象。候选打分可以把语言解码器当作分类器使用，但前提是候选字符串、分词方式与聚合规则都已冻结。

多模态智能把图像证据与语言概念连接起来，但系统效用还取决于接口。检测器提供候选的存在性与几何；VLM 可以细化语义身份；状态机消费离散事件。这一三段式关系解释了为什么一个强组件依然可能无法改进整条链路：以候选为条件的 VLM 无法找回检测器从未提出的目标；而一个正确的描述，如果其格式无法解析，也同样不可用。

### 2.2 研究现状与技术全景

开源实时检测器与紧凑 VLM 提供了可复现的权重与本地执行能力。RF-DETR 是一个专业化的闭集检测器，而 Grounding DINO 与 YOLO-World 增加了语言条件下的开放词汇检测 [3,6,7]。SmolVLM 强调小型多模态模型，并可在本地运行 [5]。商用多模态 API 提供广泛的视觉推理能力，但云端依赖、版本漂移、成本、时延与数据治理，使其不适合作为这一离线机器人实验的唯一决策路径。表 1 记录的是选型逻辑，而不是一份通用的产品排名。

| 系统族 | 获取方式 | 相关优势 | 纳入或排除的理由 |
|---|---|---|---|
| RF-DETR | 开源、本地 | 专业化的端到端检测与分割 | 项目采用的固定检查点与候选几何 |
| Grounding DINO / YOLO-World | 开源、本地 | 依据文本概念做开放词汇检测 | 相关替代方案，但未用于冻结实验 |
| SmolVLM-256M | 开源、本地 | 紧凑的图像—文本推理与 token 打分 | 用于在 CPU 约束下检验语义复核 |
| 商用多模态 API | 闭源云服务 | 广泛的视觉问答能力 | 因接口远程且版本由外部管控，被排除在最终决策路径之外 |

现有文献支撑了这些组件，但并未确立它们在本项目中的收益。RF-DETR 报告了跨公开数据集的良好精度—时延折衷 [3]，SmolVLM 报告了高效的多模态表现 [5]。这些结果并不回答 SmolVLM 能否修正该检查点在机器人域上的残余错误。要回答它，需要成对的候选、冻结的门限，以及针对本项目的负例分析。

### 2.3 本研究与课程的关联

竞赛提供了实践问题，课程则提供了系统化拆解方法。DETR 关联到 Transformer 注意力与结构化集合预测 [1,8]。SmolVLM 把语言解码与视觉嵌入连接起来 [5]。候选打分把语言模型的似然用于判别式决策。由此形成的级联是一个小型智能系统，其有效性必须根据任务指标、误差传播、时延与可复现性来评判，而不能仅凭模型的新颖性。

本研究也改变了我对“加一个大模型”的理解。课程实验表明，更大或更通用的模型并不自动成为更好的子系统。正确的问题是：在同样的数据与冻结协议下，该模块是否改变了一个可观测的决策。这一原则直接导向严格的继续门限，也导向在 VLM 净增益为零之后仍然坚持仅用 RF-DETR 的最终决定。

## 3 方法 Methodology

### 3.1 问题定义与符号说明

视觉任务包含三个带框类别：`danger_red_sphere`、`distractor_red_box` 与 `distractor_green_sphere`。按照继承的数据约定，没有标签文件的历史帧被视为背景或未知，但不会被描述为人工核验过的负例。在线推理输入只包含图像数据与冻结的配置。标签、仿真器真值以及世界/布局文件只由离线评估器接入。

对类别 c，精确率是 c 的全部预测中被匹配正确的比例，召回率是 c 的全部真值中被匹配到的比例，F1 是二者的调和平均。三分类 Macro-F1 是三个带框类别的算术平均。匹配使用类别一致、框 IoU 不低于 0.5、置信度排序与一对一分配。掩码头 IoU 不予报告，因为历史语料没有真值掩码。

[[EQ:Precision(c) = TP(c)/(TP(c) + FP(c)),   Recall(c) = TP(c)/(TP(c) + FN(c))   (1)]]

[[EQ:F1(c) = 2 Precision(c) Recall(c)/(Precision(c) + Recall(c))   (2)]]

[[EQ:MacroF1 = (F1(1) + F1(2) + F1(3))/3   (3)]]

主系统决策是：VLM 能否在不降低危险源召回率、不增加目标误报、不超过冻结调用率边界的前提下，改善检测器的结果。规则要求指标严格提升。指标持平被视为不满足继续条件，因为 VLM 增加了计算与运维复杂度。

### 3.2 课程理论预备

DETR 使用 Transformer 编码器—解码器与可学习的目标查询，产出固定大小的预测集合 [1]。训练时用二分匹配把预测分配给真值。固定的 RF-DETR 检查点继承这一端到端检测原理，输出类别、置信度、边界框与掩码。在本研究中，掩码输出仅作溯源保留；由于不存在可匹配的掩码标注，定量评估仅使用边界框。

SmolVLM 以图像与文本表示为条件驱动自回归解码器 [5]。自由生成是选择一段 token 序列，因此即便提示只要求给出一个标签，它仍可能附加解释或标点。冻结候选打分则改为逐一评估每一个允许的续写，并对其 token 对数概率取平均。最高的有限分数成为候选预测。这把开放的输出空间变成确定性的四标签决策，同时保留同一个底层模型。

机器人接口增加了一层契约。一个检测帧记录图像标识、模型溯源、有序候选与状态机输入。Level A 禁止世界坐标与时序跟踪；它校验事件结构与故障闭锁（fail-closed）状态转移。Level B 增加 RGB-D 反投影与坐标变换，但其位姿来源必须与目标真值分开审计。由于 `/Odometry_gazebo` 属于仿真器位姿真值，Level B 被归为使用位姿真值的调试，而非自主定位。

### 3.3 核心模型与算法设计

RF-DETR 基线使用固定的 RF-DETR-Seg Medium 检查点，置信度阈值 0.5。其 SHA-256 为 `ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50`。401 个推理作业只包含图像路径与推理配置，不含标签或世界信息。独立评估器随后接入封印标签，并重算每一项指标。

SmolVLM B1 要求恰好给出一个 snake_case 标签，且只接受完全匹配。它不会从一段较长的解释中抽取关键词。B2-A 是使用自然语言标签短语的开发期诊断。B2-B 冻结四个候选字符串，并计算教师强制（teacher forcing）下的平均词元（token）对数概率。B2 的目标输入是将基于真值框的裁剪向外扩展 15%，背景输入则使用整帧。因此，B2-B 评估的是候选区域的语义，而不是可部署的目标存在性。

真实候选级联只在开发集种子 101 与 202 上运行。它路由置信度落在 `[0.5, 0.7)` 的 RF-DETR 候选，将每个框向外扩展 15% 并裁剪到图像边界，再把该裁剪送入 B2-B。队列包含来自 22 张图像的 24 个候选。失败时回退到 RF-DETR。检测器给出的危险源预测不可下调；非危险源类别只允许在两个干扰类别之间重标；VLM 给出的危险源答案也不会把检测器的非危险源候选直接上调为危险源。

继续条件要求同时满足以下各项：危险源召回率不下降、三分类 Macro-F1 严格提升、目标误报不增加、VLM 调用率不超过 15%。该规则在运行之前冻结，其目的是及时终止技术上能够运行、但对系统没有净收益的级联。

### 3.4 实验实现细节

原始语料包含来自五个历史仿真器种子的 500 张 RGB 帧，以及 383 个标签文件、426 个框。审计了 15 个完全重复组与跨种子的感知近重复候选。完全重复组内的全部图像、以及参与跨种子近重复对的全部图像都被隔离，最终保留 401 张图像与 444 个候选决策。五折划分轮换完整种子，而不是随机切分相邻帧。

RF-DETR 在 Windows 11 CPU 上运行，环境为 Python 3.13.12、PyTorch 2.12.0+cpu、Torchvision 0.27.0+cpu 与 `rfdetr` 1.9.4。SmolVLM 使用 PyTorch 2.15.0+cpu、Transformers 4.57.6 与 float32。两个模型需要不同的 Transformers 环境。报告的时延均为该环境下的 CPU 时延，不得外推到 RTX 3080 Ti、RTX 3090 或机器人机载计算机。

协议、队列、模型标识、源文件、预测、汇总与回放产物均以 SHA-256 封印。最终确定性回归运行 87 个测试。RF-DETR 检查点的训练溯源不可得，因此该评估无法建立严格的训练—测试隔离。这一局限被放在结果旁边陈述，而不是藏在文末的脚注里。

## 4 实验与结果 Experiments and Results

### 4.1 实验设置

表 2 区分了检测器基线、VLM 诊断、锁定候选实验与真实候选级联。锁定的 B2 集合属于探索性而非确证性，因为 B1 已经观测过同一批种子。真实级联只使用开发划分，停止规则阻止了在增益门限失败之后再补做事后锁定的级联实验。

| 实验 | 种子 | 规模 | 证据边界 |
|---|---|---|---|
| RF-DETR 五折评估 | 101/202/303/404/505 | 401 张图像 | 重复受控的域内探索 |
| B2 开发 | 101/202 | 193 个候选 | 仅用于门限选择与可行性 |
| B2 锁定候选研究 | 303/404/505 | 251 个真值框裁剪 | 候选语义，而非端到端检测 |
| 真实候选级联 | 101/202 | 24 个候选 | 仅开发集上的净收益检验 |

主检测指标是三分类 Macro-F1。危险源的精确率、召回率、F1、目标 FP/FN 与图像级存在召回率分别报告，因为危险源漏检与干扰物混淆在系统后果上并不相同。VLM 评估还报告严格输出覆盖率与背景召回率。级联评估比较同一候选集在融合前后的表现，并记录 CPU 时延与调用率。

### 4.2 主要结果

RF-DETR 完成 401/401 个推理作业。它达到 0.9635 的 Macro-F1、0.9892 的危险源召回率与 0.9944 的危险源图像级存在召回率。结果包含 183 个危险源真正例（TP）、5 个假正例（FP）与 2 个假负例（FN）。其中一张假负例图像中仍含另一个被检出的危险源；另一张则完全没有产生候选。后者体现了以候选为条件的复核器的结构性上限。

| 指标 | RF-DETR 结果 |
|---|---|
| 三分类 Macro-F1 | **0.9635** |
| 危险源精确率 / 召回率 / F1 | **0.9734 / 0.9892 / 0.9812** |
| 红方块 F1 | 0.9619 |
| 绿球 F1 | 0.9474 |
| 危险源图像级存在召回率 | 0.9944 |
| CPU P50 / P95 时延 | 653.1 / 743.3 ms |

![RF-DETR 固定检查点的各项指标](figures/rfdetr_pilot_metrics.png)

B1 产出 444 条流畅回答，但没有任何严格单标签输出。B2-A 修复了格式，却出现类别坍缩，只预测少数几个标签。B2-B 在 251 个锁定的真值框裁剪上达到 100% 输出覆盖率、0.9084 准确率、0.7217 Macro-F1 与 1.0 危险源召回率。它正确分类 97 个红球、90 个红方块与 41 个绿球，但全部 23 个背景样本都被判为红方块。候选打分修复了输出契约层面的失效，却没有解决目标存在性判断。

![自由生成与冻结候选打分的对比](figures/b1_b2_locked_metrics.png)

![B2 开发集与锁定集的类别召回](figures/b2_development_locked_recall.png)

![B2 锁定集的混淆矩阵](figures/b2_locked_confusion_matrix.png)

### 4.3 消融研究

受控变体隔离了三项设计选择：自由生成与约束打分、自然标签与冻结 snake_case 候选、真值框裁剪诊断与真实候选级联。这些属于组件研究，而不是重训模型的消融；基础权重保持不变。

| 变体 | 输出覆盖率 | Macro-F1 | 主要观察 |
|---|---|---|---|
| B1 自由生成 | 0.0000 | 0.0000 | 语义流畅但不符合输出契约 |
| B2-A 自然候选诊断 | 1.0000 | 开发集 0.2433 | 格式修复，类别坍缩仍在 |
| B2-B 冻结候选打分 | 1.0000 | 锁定真值框裁剪 0.7217 | 目标类别可区分；背景召回率为 0 |
| RF-DETR + B2 真实候选融合 | 24 次调用全覆盖 | 开发集 0.96918 | 无类别或指标变化 |

真实候选级联是决定性的受控比较。SmolVLM 返回 6 个危险源与 18 个绿球标签，在全部 24 个候选上完全复现了 RF-DETR 的类别。安全融合与直接采用 B2 top-1 诊断都没有改变预测。因此 Macro-F1、危险源召回率与 FP 完全不变，而每次 VLM 调用额外增加 5.11 s 的 CPU 中位时延。

| 指标 | 仅 RF-DETR | RF-DETR + B2 |
|---|---|---|
| 三分类 Macro-F1 | 0.96918 | 0.96918 |
| 危险源召回率 | 0.98864 | 0.98864 |
| 目标误报 | 11 | 11 |
| VLM 调用率 | 0 | 13.95% |
| B2 CPU P50 / P95 | 0 | 5114.7 / 5539.6 ms |

![开发集真实候选级联对比](figures/cascade_development_comparison.png)

### 4.4 可视化分析

RF-DETR 的图表显示三个目标类别的 F1 较为均衡，最弱的是绿球。混淆矩阵给出 SmolVLM 的另一视角：目标裁剪大多可分，但每一张背景帧都被强行判入某个目标类别。这不是轻微的格式问题。它意味着在冻结的四标签设计下，VLM 无法独立判断候选是否存在。

Level A 分镜把有序检测结果转换为不带坐标的机器人事件与状态。它包含 12 个预选案例，涵盖正确的危险源确认、干扰物排除、空候选与低置信度处理。重复相同输入会产生逐字节一致的事件、日志与 GIF 输出。该可视化是确定性的接口演示，而不是一条连续轨迹。

![Level A 机器人接口回放](figures/level_a_replay_overview.png)

Level B 图像展示 RGB-D 反投影与时序确认。四份规范输出产物在两次回放之间逐字节一致。一个危险源被匹配、一个被漏检，且未报告虚假危险源；技术得分为 23/37。被匹配位置的误差 0.1238 m 依赖 Gazebo 位姿真值，因此它证明的是内部调试链路的一致性，而不是自主定位精度。

![Level B Gazebo 位姿真值回放](figures/level_b_pose_oracle_overview.png)

### 4.5 结果讨论

严格增益条件因 Macro-F1 未提升而未通过。因此，级联在锁定种子执行与多视角扩展之前终止。按照预注册规则，这一负结果同样是有效的实验结果。若仅为追求正结果而继续，就会把开发反馈变成未报告的调参，并削弱报告的正确性。

现有证据支持在当前语义决策任务中仅使用 RF-DETR，但并不证明所有 VLM 级联都无效。被测试的 VLM 只在检测器给出候选之后才看到裁剪，使用冻结候选集，并在 CPU 上运行。不同的数据集、一个显式的“无目标”候选、图像与候选的联合上下文，或一个监督式误差预测器，都可能改变结论。这些是需要新冻结协议加以检验的新假设，而不是对本次运行结果的重新解释。

## 5 可复现性与可复用代码 Reproducibility

课程仓库地址为：

https://github.com/s-max-oss/llm-course-labs/tree/main/final_project

提交的期末项目目录包含 `configs`、`scripts`、`src`、`tests`、数据清单、封印的实验汇总、图表与操作手册。大型虚拟环境与缓存目录已排除。RF-DETR 检查点以 SHA-256 标识，模型与运行时版本均已记录。数据源路径仅保存在未提交的本地配置文件中。

完整研究目录可用一条命令检查：`powershell -ExecutionPolicy Bypass -File scripts/run_checks.ps1`。最终运行报告 `Ran 87 tests` 与 `OK`。课程压缩包省略了约 677 MB 的机器人 Git 历史包与部署 tar，但保留了它们的清单、哈希、补丁、更正记录与回放输出。因此完整的 Level B 封印必须在原始研究目录中重做。

可复现性有两项明确的限制。RF-DETR 的训练图像溯源缺失，因此该检查点不能被称为无泄漏。GitHub 链接指向课程仓库；除非在记录的修订版本上发布同一个期末项目目录，否则提交的压缩包仍是本报告的权威快照。

## 6 系统演示 System Demonstration

演示按证据级别组织，而不是用一段打磨过的视频充当完整机器人任务的证明。Level A 消费冻结的 RF-DETR 预测，发出 `robot_replay_detection_frame_v1` 事件。它校验类别映射、置信度、输入哈希、从 SEARCHING 到 REPORTED 的状态转移以及故障闭锁行为。它有意禁止世界坐标、时序跟踪与在线控制。

Level B v2 消费一段同步的 11.17 s RGB-D 记录，含 92 个可用配对，选出 14 帧并处理 28 个检测。它校验深度质量、相机反投影、坐标格式、三次命中确认、结果写入与确定性回放。然而其变换链读取 `/Odometry_gazebo`。权威机器人契约将该话题判定为 Gazebo 位姿真值，因此 G2 真值隔离门必须保持未通过。该演示对调试数据通路有用，但不能报告为符合竞赛要求的自主定位。

| 证据级别 | 已验证能力 | 不被支持的结论 |
|---|---|---|
| Level A | 事件模式、状态机、确定性故障闭锁回放 | 连续轨迹、三维定位、在线控制 |
| Level B v1 | 同步 RGB-D 的失败证据 | 正式空间结果或完整真值隔离 |
| Level B v2 | RGB-D 几何、确认、输出格式、回放确定性 | 自主位姿、竞赛合规、完整建筑搜索 |

提交包中的两段 GIF 直接展示接口行为。它们配有清单与校验器，使读者能区分视觉解释与封印结果。这对应竞赛给出的教训：过程证据必须能追溯到确切的输入、状态与输出，而不能由一张最终截图推断。

## 7 讨论 Discussion

### 7.1 结果的深入解读

本研究区分了“模型有效”的三重含义。B1 在视觉上理解得足够好，能产出相关描述，但无法通过接口契约。B2-B 服从接口并能区分目标裁剪，但无法拒识背景。真实级联同时服从输入与输出契约，却没有改变任何决策。这是三个不同的失效层次；若把它们压缩成一个准确率数字，便会掩盖最终工程选择的真正理由。

RF-DETR 的高基线也改变了复核器的价值。被路由的区域里，候选之所以被选中只是因为检测器置信度不确定，而不确定并不等于检测器出错。SmolVLM 看到的多数是检测器本来就判对的样本。而剩下那类“没有候选的漏检”根本不在复核器的可达集合内。更好的路由器应预测检测器出错的概率，而不是只用置信度。

### 7.2 优势与局限

主要优势是证据纪律。数据划分、任务队列、输出模式、候选字符串、融合规则、停止条件与哈希都被冻结并独立重算。负结果与未通过的门限保留在最终叙述中。报告还把一段真实的竞赛工作流与一个有界的课程问题连接起来，同时不宣称课程实验完成了竞赛。

局限是实质性的。历史种子只有五个。99 张图像因重复风险被隔离，但这无法建立检测器的训练—测试分离。117 帧缺少标签文件，不能算作经过严格人工核验的负例。B2 使用真值框裁剪。B1 已观测过锁定种子。级联只在开发种子上运行。CPU 时延不能代表 GPU 或机载运行速度。Level B 使用仿真器位姿真值。这些条件把结论限制在受测产物与协议之内。

### 7.3 课程学习反思

竞赛经历最初让我把感知问题理解为尽可能提高检出率。课程学习改变了这一看法。机器人结果由一条完整链路产生：图像证据变成候选，候选变成语义决策，决策变成事件，事件再变成任务记录。即使前一个模型看起来很准确，每一个接口边界仍可能失效。

最有用的收获是学会区分模型能力与系统价值。我原本预期一个紧凑 VLM 能在模糊的红色目标上发挥作用，因为它可以借助语言语义进行描述。实验却显示：同一个模型可能无法满足标签契约；修复契约之后又无法拒识背景；随后在真实候选上也没有净增益。冻结协议使接受这一结果成为可能。按照预注册规则终止级联，比为了追求正结果而继续调参更符合研究规范。

### 7.4 后续工作

下一步的优先项是新的可审计数据，而不是更大的 GPU。新的仿真或实景场景应记录种子、房间、相机位姿、目标身份、完整标签与采集溯源。检测器应在有公开训练清单的情况下重训或重选。评估应纳入未见过的布局与严格人工核验的负帧。

对 VLM 而言，值得检验的假设包括：加入显式的“无目标”候选、联合使用整帧与候选区域上下文、引入经过校准的拒答机制，以及训练以检测器出错概率为目标的监督式路由器。多视角融合应当只在单帧复核器于相同停止条件下展现净增益之后才开始。Level B 应先用经过审计的自主位姿链替换 Gazebo 位姿真值，才谈得上重新考虑任何定位或竞赛合规主张。

## 8 结论 Conclusion

本论文把一个具体的四足机器人竞赛问题，转化为一项可审计的视觉与多模态智能课程研究。竞赛流程要求机器人找到红色球形危险源、排除相近的干扰物，并把稳定结果交给定位与任务逻辑。对固定 RF-DETR 检查点进行重复风险控制后的评估，在 401 张历史图像上达到 0.9635 的三分类 Macro-F1 与 0.9892 的危险源召回率，但训练数据溯源缺失使该结果仅限于域内探索。SmolVLM 的自由生成无法通过严格的标签接口。冻结候选打分恢复了完整的输出覆盖率并能识别目标裁剪，但其背景召回率为零。在 24 个真实检测候选上，VLM 复现了每一个 RF-DETR 类别，没有改变任何指标，并额外增加 5.11 s 的 CPU 中位时延。因此，预注册的增益门限终止了级联，仅使用 RF-DETR 仍是推荐的语义检测方案。Level A 验证了事件与状态机接口。Level B 验证了一条 RGB-D 调试通路，但它使用 Gazebo 位姿真值，未通过 G2 隔离要求。最终贡献不是宣称已经完成竞赛所需的完整自主系统，而是给出一份可复现的说明：检测器、VLM 与机器人接口的证据在何处一致、在何处失效，以及系统决策为何由这些证据边界推出。

## References 参考文献

[1] Carion N, Massa F, Synnaeve G, et al. End-to-End Object Detection with Transformers. ECCV, 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf

[2] Zhao Y, Lv W, Xu S, et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2024: 16965-16974. https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html

[3] Robinson I, Robicheaux P, Popov M, et al. RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. arXiv:2511.09554, 2025. https://arxiv.org/abs/2511.09554

[4] Liu H, Li C, Wu Q, Lee Y J. Visual Instruction Tuning. NeurIPS, 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html

[5] Marafioti A, Zohar O, Farré M, et al. SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299, 2025. https://arxiv.org/abs/2504.05299

[6] Liu S, Zeng Z, Ren T, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv:2303.05499, 2023. https://arxiv.org/abs/2303.05499

[7] Cheng T, Song L, Ge Y, et al. YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR, 2024: 16901-16911. https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

[8] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017. https://arxiv.org/abs/1706.03762

[9] Cao C, Zhu H, Choset H, Zhang J. TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments. Robotics: Science and Systems, 2021. https://www.roboticsproceedings.org/rss17/p018.pdf

[10] Hugging Face. SmolVLM: Small Yet Mighty Vision Language Model. 2024. https://huggingface.co/blog/smolvlm

[11] Southwest Institute of Technical Physics. DG-202602 Competition Scheme for Autonomous Hazard Search and Recognition with a Quadruped Robot. 2026.

[12] Course Teaching Group. Comprehensive Teaching Materials for Foundations and Applications of Large Models. Harbin Institute of Technology, Shenzhen, 2026.

## 附录 Appendix

### A 代码与产物获取

GitHub repository: https://github.com/s-max-oss/llm-course-labs/tree/main/final_project

提交的压缩包是本报告的权威产物快照。它包含报告、操作手册、源代码目录、500 张 RGB 图像、383 个标签文件、数据清单、实验汇总、RF-DETR 检查点、图表、两段回放 GIF、完整的材料清单与 SHA-256 校验和。

### B 补充证据与主张边界

检测器运行由 `experiments/pilot_v1/rfdetr_run.json` 封印。锁定的 B2 研究由 `experiments/pilot_v1/smolvlm_b2_locked_run.json` 封印。已停止的真实候选级联由 `experiments/pilot_v1/cascade_development_run.json` 封印。Level A 证据位于 `experiments/replay_v1`，Level B v2 证据位于 `experiments/level_b_v2`。

以下陈述为证据所禁止：无泄漏的检测器泛化、严格的背景负例验证、GPU 实时性能、Level B 自主定位、符合竞赛要求的空间闭合，以及完整的建筑搜索验证。有效的结论更为收窄：在冻结的探索性协议下，仅用 RF-DETR 是推荐的语义检测器；SmolVLM 的结果记录了模型与接口的边界；机器人回放验证了其声明证据级别上的具体契约。
