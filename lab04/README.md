# 实验四 Vector RAG GraphRAG 与 WikiRAG

孙驰，2024311281，计算机10班。依据作业四指南与 Lesson08，在同一份虚构校园语料上比较三种知识组织方式。使用真实 BGE 中文嵌入，全部检索在本机 CPU 完成；随后使用 DeepSeek 官方 API 现场生成答案、抽取知识并复测，无 HPC 作业。

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

### DeepSeek 现场构建补充

`results/live_api/` 单独保存现场结果。请求与响应模型均为 `deepseek-flash`，temperature=0、关闭 thinking。61 次真实调用，累计输入15088、输出3768、合计18856 token：9个参考基线答案、3次v1/15次v2抽取、14个社区摘要、9个Wiki正文、2个现场Q11答案及9次同模型AI评审。

| 现场知识方案 | 平均 Recall@5 | MRR | 全证据命中 |
|---|---:|---:|---:|
| Vector 原文基线 | 0.966667 | 1.000000 | 19/20 |
| Graph Local | 0.833333 | 0.816667 | 15/20 |
| Graph Global | 0.900000 | 0.837500 | 16/20 |
| Wiki | 0.908333 | 0.779167 | 15/20 |

v2抽取65条三元组，头尾归一后仍65条、73节点、14个实际社区，各社区根据内部边关联原文生成摘要；Wiki沿用参考9个主题和来源列表，正文重新生成。现场索引与参考索引分开保存，20题及K扫描口径相同。Graph还改变了实体归一、事实覆盖、关系表述及摘要来源，因此不是只改变提示词的控制变量实验。Wiki平均Recall提高但全命中由16降至15题；仍不能说现场生成全面优于参考。

原始失败保留：v2 doc04补入任职边却漏掉研究方向与职称，doc03“设”被抽成“设计”，doc13漏藏书/开放时间，三个社区摘要加入无依据安排或分类。v1/v2预先定义，共同比较文档仅doc04/doc10/doc15。`live_knowledge_audit.json`逐条列出错误；不事后删除错误或选择最高成绩。最终答案只使用检索返回的原始文档，未直接使用有问题的生成摘要作事实证据。

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

### 现场 API 入口

`deepseek_live.py`读取进程环境变量`DEEPSEEK_API_KEY`，请求固定发往官方`https://api.deepseek.com/chat/completions`。不读取聊天日志或本机凭据文件，不写密钥到结果、源码或Git。通过当前终端安全输入密钥后，按需执行下列命令；不要把密钥文字放入命令历史。

```powershell
# 当前会话安全输入；按需将 try 内的命令替换为下面的其他入口。
$apiSecret = Read-Host "DeepSeek API key" -AsSecureString
$apiPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($apiSecret)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($apiPointer)
    python deepseek_live.py generate --system vector --q Q01 --output results_live_reproduced
} finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($apiPointer)
    $apiSecret.Dispose()
}

# 设置进程密钥后可选择的入口；完整实验会产生实际API费用。
python deepseek_live.py extract --doc doc10 --prompt-version v2 --output results_live_reproduced
python deepseek_live.py experiment --output results_live_reproduced
python -m unittest test_hw4_rag test_deepseek_live -v
```

`generate`从保存的参考检索记录读取上下文；`experiment`才会构建现场图谱/Wiki并复测。最多三请求并行，完整请求体及响应保存于`calls/`，不含鉴权头。相同标签和请求哈希的完整响应可复用；HTTP失败、空响应、截断或不合格JSON会报错，不伪造成功或回退为预置答案。本次整段运行24.94秒，单调用耗时之和47.20秒，二者因并发而不同。

`run_summary.json`记录实际运行的源代码哈希，`source_at_run.py`保留当时源码。运行后只增加三项AI评审字段类型校验，正式结果不重写；当前版本与运行版本均可复核。新六项测试使用离线模拟，测试不产生API调用。

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

`results/answer_review.json`保留早期Codex整理的九例作为历史记录。当前报告表5使用`results/live_api/answer_review.json`中的九个真实DeepSeek答案与同模型AI初评。回答使用各自参考检索返回原文；高忠实度不等于答案完整。九例初评均5，AI却把含学院事实的Wiki Q01拒答错拆为零声明，独立来源审计已指出此误判，原AI响应仍保留。纯拒答没有事实声明时比例才不定义。本次未运行RAGAS。

**human_score均为空，人工评分尚未完成**。请在`report/human_review.md`中逐例核查完整答案与原文，确认或调整分数；不能把同模型初评当作人工评分。

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
| `deepseek_live.py` / `test_deepseek_live.py` | 现场API入口与6项离线安全/边界测试 |
| `results/live_api/calls/` | 61个不含鉴权的真实请求/响应、使用量、时间 |
| `results/live_api/` | 抽取、14摘要、9条目、现场评测、123×512向量、答案、原始运行源码及manifest |
| `results/live_api/live_knowledge_audit.json` | 三元组、摘要、Wiki和AI评审的逐条来源审计 |
| `figures/` | 从保存结果生成的图 |
| `report/` | 模板报告、检查记录与人工评分待确认清单 |

单次缓存构建：Vector 0.1509秒、Graph 0.1520秒、Wiki 0.0889秒。这些只是预置知识的嵌入/建图，不含人工整理或LLM抽取时间，样本太小且有模型预热差异，不能代表生产工程成本。没有进行新增文档计时实验。

模型权重和临时缓存不入Git。与前三次实验一致，本地正式报告存入`课程实验/提交材料`，代码、结果与报告副本归档至`课程实验/lab04_rag`，报告生成资料存`tools/`。

## 参考

- 课程作业四指南、实验报告样板、Lesson08第八章。
- [BGE官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
- [Sentence Transformers官方文档](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。
- [NetworkX greedy modularity文档](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)。
- [DeepSeek官方API](https://api-docs.deepseek.com/)与[JSON输出说明](https://api-docs.deepseek.com/guides/json_mode/)。
