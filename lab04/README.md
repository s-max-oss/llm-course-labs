# 实验四 Vector RAG GraphRAG 与 WikiRAG

孙驰，2024311281，计算机10班。依据作业四指南与 Lesson08，在同一份虚构校园语料上比较三种知识组织方式。使用真实 BGE 中文嵌入，全部检索在本机 CPU 完成，无生成 API 调用，无 HPC 作业。

## 实际结果

默认 K=5，15 道内置题加 5 道新增题，共20题。Recall 按题宏平均；MRR 按去重后的文档首次命中计算。

| 方案 | 平均 Recall@5 | MRR | 全证据命中 |
|---|---:|---:|---:|
| Vector RAG | 0.966667 | 1.000000 | 19/20 |
| GraphRAG Local | 0.808333 | 0.739167 | 14/20 |
| GraphRAG Global | 0.716667 | 0.580833 | 11/20 |
| WikiRAG | 0.858333 | 0.750000 | 16/20 |
| Hybrid RRF 选做 | 0.958333 | 0.975000 | 18/20 |

本小语料上的向量检索更强，不能预设图谱必胜。Graph Global 的全局题 Recall 为0.785714，高于 Local 的0.642857，但事实题较弱。RRF 没有提高事实题分数，全题平均略降，保留负结果。

## 运行

Python 3.11，安装依赖后，在本目录执行：

```powershell
python -m pip install -r requirements.txt
python hw4_rag.py --mode vector --q Q01 --k 5 --revision 7999e1d3359715c523056ef9478215996d62a620
python hw4_rag.py --mode graph --q Q11 --scope global --revision 7999e1d3359715c523056ef9478215996d62a620
python hw4_rag.py --mode eval --questions all --scope local --output eval20.json --revision 7999e1d3359715c523056ef9478215996d62a620
python run_experiments.py --revision 7999e1d3359715c523056ef9478215996d62a620 --output results_reproduced
python -m unittest test_hw4_rag -v
python analyze_results.py --verify-only
```

模型为 `BAAI/bge-small-zh-v1.5`，版本 `7999e1d3359715c523056ef9478215996d62a620`，512维，单位归一化，查询加官方推荐中文指令，文档不加。固定CPU四线程；模型下载和加载时间不计入索引构建时间。`results/environment.json` 记录实际版本和源代码哈希。

官方模型站点连接受限时，可在当前终端设置 `$env:HF_ENDPOINT='https://hf-mirror.com'`。本次使用镜像下载同一模型版本；若终端残留失效代理，仅在该终端清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`。不修改全局网络设置。

`--emb char3` 仅供快速逻辑测试，必须显式指定；正式套件拒绝这种后端。真实模型加载失败会报错，不会用字符向量替换正式结果。

## 实现与评测口径

- Vector：块长120中文字符、重叠20，余弦排名，文档去重。15篇原文都不足120字符，所以本次实际15块，不能据此评价跨块碎片化。
- Graph：40条参考三元组、45节点的多重有向图；greedy modularity 检测10个社区。Global 使用指南预置3段主题摘要，它们没有根据这10个社区重新生成。Local 从最多2个匹配实体展开，最多3跳、每轮前沿6节点，无可走边时回退Global；保存 `fallback` 字段。
- Wiki：保留9条参考条目，以正文嵌入取Top-3条目，再按条目分数及原有来源顺序扩展文档。Graph Global取Top-2摘要。条目/摘要内部不额外重排原文，可能让答案文档排在后面。
- Gold仅在检索完成后用于评测，没有传给检索器。K扫描固定内置15题，每类型5题；20题中fact6、multi7、global7，避免混淆两个汇总口径。
- 选做：中文字符bigram词面余弦与稠密排名按 RRF `1/(60+r_dense)+1/(60+r_sparse)` 融合，再去重文档。

Q11 的指南 gold 包括doc10、doc02、doc03，但完整平台事实主要就在doc10。为与指南一致，没有修改标注；报告区分 gold 覆盖和实际答题充分性。

## 来源核查与答案评分

参考知识库存在原文不支持的改写：计算机学院“参与共建”、联合培养“每年15人”、课程教材名称。主实验保留参考数据，`--corrected` 单独修正以上三处，原文与gold不变。修正后的指标另存，未择优替换基线。

`--mode extract` 只导出参考三元组与两版提示词草案，**没有调用LLM抽取**。JSON校验只验证格式，关系语义仍需查原文。`--normalize-entities` 启用空格/书名号归一与“云大”别名；默认基线保留严格实体匹配，用诊断题展示差异。

`results/answer_review.json` 保存每范式Q01/Q06/Q11共9个一次性Codex答案、检索上下文、原子声明及AI初评。答案使用实际返回的原文；缺资料时拒答，不调用外部生成API。高忠实度不等于答案完整；没有事实声明时比例不定义。**human_score为空，人工评分尚未完成**，报告中也明确保留此状态。提交前请逐条阅读，确认或调整分数并填写人工评分。

## 文件

| 路径 | 内容 |
|---|---|
| `hw4_rag.py` | 自包含实现、指南数据、20题、CLI |
| `reference_data.json` | 从指南提取的原始数据 |
| `run_experiments.py` | 完整真实嵌入评测与K扫描 |
| `analyze_results.py` | 图表、答案初评与算术复核 |
| `results/eval*.json` | 15/20题逐题文档、分数、命中、Local路径/回退 |
| `results/k_scan.json` | 内置15题Local/Global的K扫描 |
| `results/embedding_vectors.npz` | 实际使用的92个文本及512维向量，便于离线回放 |
| `results/mrr_manual.json` | 三题手算与函数核对 |
| `results/manifest.json` | 结果文件SHA-256 |
| `figures/` | 从保存结果生成的图 |
| `report/` | 模板报告、检查记录与人工评分待确认清单 |

单次缓存构建：Vector 0.1509秒、Graph 0.1520秒、Wiki 0.0889秒。这些只是预置知识的嵌入/建图，不含人工整理或LLM抽取时间，样本太小且有模型预热差异，不能代表生产工程成本。没有进行新增文档计时实验。

模型权重和临时缓存不入Git。与前三次实验一致，本地正式报告存入`课程实验/提交材料`，代码、结果与报告副本归档至`课程实验/lab04_rag`，报告生成资料存`tools/`。

## 参考

- 课程作业四指南、实验报告样板、Lesson08第八章。
- [BGE官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
- [Sentence Transformers官方文档](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。
- [NetworkX greedy modularity文档](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)。
