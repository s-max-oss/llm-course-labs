"""Three-page report with default reference and the frozen v2 comparison."""
import json


def create_content(builder):
    root = builder.ROOT
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    out = root/'improvement/final_v2'
    summary = read(out/'summary.json'); selection = read(out/'selection.json')
    old = read(root/'analysis/summary.json')
    study = read(root/'analysis/optimization_v2.json')
    run = read(root/'provenance/optimization_v2_run.json')
    get = lambda name: read(out/f'{name}.json')
    metrics = lambda s: [f"{s[t]['correct']}/{s[t]['n']}" for t in ('loc','count','exist','total')]+[f"{s.get('loc_iou',s.get('loc_mean_iou')):.3f}"]
    score = lambda name: metrics(get(name)['summary'])
    blob = []
    p = lambda s, **kw: blob.append(builder.paragraph(s, **kw))
    heading = lambda s, before=False: p(s, heading=True, size=22, after=55, before=before)
    cap = lambda s: p(s, bold=True, size=17, after=25, keep=True)
    tb = lambda rows, widths, size=16: blob.append(builder.table(rows, widths, size))
    base = get('v1_fresh_test')['summary']; final = get('v2_fresh_test')['summary']
    rich = get('v2_rich_tool_fresh_test')['summary']
    rounds = summary['rounds']
    def examples_table():
        cap('代表性路径（新测试题号缩写，逐题图文与原始输出均留存）')
        records = get('v2_rich_tool_fresh_test')['rows']
        examples = [('直接正确',next((r for r in records if r['raw_correct']),None)),
                    ('高亮改对',next((r for r in records if not r['raw_correct'] and r['self_check'] and r['self_check']['verification']['passed']),None)),
                    ('框提示改对',next((r for r in records if not r['raw_correct'] and r['tool_check'] and r['correct']),None)),
                    ('工具仍错',next((r for r in records if not r['correct']),None))]
        tb([['路径','题号／目标','原始→最终','GT'],
            *[[label,r['id'].replace('fresh_test_','')+' '+r['target'],r['raw'][:15]+' → '+r['final'][:15],r['gold']] if r else [label,'无此样例','据实记录','—'] for label,r in examples]], [1300,2600,3300,1800],15)
    p('《大模型通识》实验六实验报告', bold=True, size=29, after=40, center=True)
    p('SmolVLM 精准定位与 RSI 改进验证', size=19, after=65, center=True)
    tb([['项目','实际信息'],['姓名／学号／班级','孙驰／2024311281／计算机10班'],
        ['日期／环境','2026年10月22日；HPC NVIDIA A30 24GB；NF4 QLoRA'],
        ['代码与证据','github.com/s-max-oss/llm-course-labs/tree/main/lab06']], [1800,7200],17)
    heading('一、实验目的与任务概述')
    p('完成256×256合成图的4×4定位、计数和存在性判断，运行高亮自查、像素核验、纠错入池和两轮再训练。保留指南默认10步参照，随后诊断定位失败并连续优化。改进模型在同一份全新368题上比上一版多答对30题；模型与工具收益分开统计。')
    heading('二、实验环境与复现步骤')
    p('Python 3.10.20；torch 2.6.0+cu124；transformers 4.57.6；peft 0.17.1；bitsandbytes 0.48.1。SmolVLM-256M离线基座，NF4双重量化、BF16计算，512px单视图；原图为指南256px，修正样板480px笔误。')
    tb([['脚本／阶段','实际配置'],['run_hpc.slurm（默认）','40训练图、15评测图、15种子；SFT 10步；RSI 2×10步'],
        ['视觉与变化图适配（V1）','视觉+连接层+语言LoRA；1200固定图更新、4000变化图更新及联合SFT'],
        ['run_optimize_v2.slurm（V2）','3840张新训练图；1600次优化器更新×2累积；RSI 2×200次更新'],
        ['run_v2_evaluation.slurm','从已保存选择记录续评，不重训、不更换模型或测试题']], [3300,5700],16)
    p(f"HPC训练作业15408达到1小时限时，全部训练及选择已完成；作业{run['evaluation_job']}仅续评，{run['evaluation_state']}、退出码{run['evaluation_exit_code']}。两作业累计{run['gpu_job_minutes']:.1f}GPU分钟，分配已释放。CPU/FP32默认六阶段另行验证，8线程耗时690.6秒、6/6成功。", size=18)
    heading('三、方法与关键实现')
    p('框中心(cx,cy)映射为1起始r=floor(cy/64)+1、c=floor(cx/64)+1。模型输出rXcY，严格解析后映射中文坐标；IoU比较预测格框与目标框，格位正确时也通常小于1。问题token标签为−100，只学习答案。')
    p('默认语言LoRA参数921600；扩展视觉注意力、连接层与语言层后为1614336（0.6255%）。非重入检查点、三模块非零梯度、累积梯度除2及范数裁剪1保证训练有效。V2用3840新图，SFT实际3200微步：定位1920、计数960、存在320；半数定位图强化同色异形等干扰。', size=18)
    p('V2从V1 SFT继续，学习率3e−5余弦降至3e−6；开发集仍为128定位+40计数+16存在。比较400/800/1600更新，再从开发最优SFT做两轮RSI；每轮184张新反馈图另加15历史题。仅像素核验通过的模型纠错回答入池，3e−6训练，每8微步重放1次，其余抽常规池；不采用穷举候选纠错。', size=18)
    p('全部新图逐张核验并排除历史及跨集RGB重复。按开发定位率、再总正确率选择，平分取早阶段；选择落盘后才生成新368题。V1与V2同图、同量化、同提示、同解码比较。旧最终184题已公开，仅留作历史；本次是累计改进，不能把收益归于某个单独改动。', size=18)
    log = (root/'provenance'/f"v2-evaluation-{run['evaluation_job']}.log").read_text(encoding='utf-8').splitlines()
    lines = [next(line for line in log if line.startswith(prefix)) for prefix in ('[eval] v1_fresh_test', '[eval] v2_fresh_test')]
    p('真实日志节选（截取定位字段，完整输出见provenance）：\n'+'\n'.join(line.split(", 'count':")[0]+' …' for line in lines),size=13,after=20)
    heading('四、实验结果', before=True)
    cap('表6  指南15题默认五阶段及扩展对照（定位8／计数4／存在3）')
    rows = [['阶段','定位','计数','存在','总计','IoU']]
    for label, name in [('原基线','baseline'),('原10步SFT','sft_eval'),('原SFT+验证器','sft_tool_eval'),('原RSI第1轮','rsi1_eval'),('原RSI第2轮','rsi2_eval')]:
        rows.append([label]+metrics(old['stages'][name]))
    for label, name in [('V2+弱提示','selected_weak_tool_legacy'),('V2+检测框','selected_rich_tool_legacy'),('V2 RSI第1轮*','rsi1_legacy'),('V2 RSI第2轮','rsi2_legacy')]:
        rows.append([label]+score(name))
    tb(rows,[2700,1000,1000,1000,1200,2100],16)
    p('* 第一轮RSI即开发集已选模型；独立13/15，8道定位全对。',size=16,after=20)
    cap('表7  默认与V2两轮反馈（自查／工具列为重答尝试次数）')
    rows = [['轮次','池规模','直接对','自查／工具次数','纠错后对','新增','轮末独立']]
    memory = read(root/'rsi_memory.json')['records']
    for r in old['rounds']:
        records = [x for x in memory if x['round']==r['round']]
        rows.append([f"默认{r['round']}",str(r['pool_size']),f"{r['direct_correct']}/15",f"{r['self_rewrites']}／{r['tool_rewrites']}",f"{r['corrected_correct']}/15",str(r['added']),f"{r['heldout_correct']}/15"])
    for r in rounds:
        records = get(f"rsi{r['round']}_feedback")['rows']
        rows.append([f"V2第{r['round']}轮",f"3840+{r['verified_pool_size']}",f"{r['raw_correct_count']}/199",f"{r['self_attempts']}／{r['tool_attempts']}",f"{r['corrected_correct']}/199",str(r['added']),f"{r['legacy']['total']['correct']}/15"])
    tb(rows,[1200,1400,1150,1900,1300,650,1400],15)
    p('默认初始池15；V2常规池3840，验证池0→15→27。两轮各50重放微步，分别覆盖15、27条；入池答案保留自查／检测框来源，不用GT替代。',size=16,after=25)
    p(f"开发集选{summary['selected']}。全新同题测试：定位{base['loc']['correct']}→{final['loc']['correct']}/256、计数{base['count']['correct']}→{final['count']['correct']}/80、存在{base['exist']['correct']}→{final['exist']['correct']}/32，总计{base['total']['correct']}→{final['total']['correct']}/368；V2定位IoU {final['loc_iou']:.3f}。检测框辅助定位{rich['loc']['correct']}/256、总计{rich['total']['correct']}/368。",size=18)
    blob.append('FIGURE_PLACEHOLDER')
    p('图1  开发阶段与同题新测试；独立模型和工具条件分列。',size=16,after=25,center=True)
    heading('五、结果分析与问题调试', before=True)
    p('原10步只有6次定位更新，8条种子仅覆盖6格、没有第二行；两轮新增2条均非定位。延长种子400步虽训练8/8，开发仅10/128；固定128图视觉适配训练抽样16/16，开发16/128。变化图和联合训练后V1旧独立测试109/128，说明训练拟合与新图能力必须分开看。')
    stages = summary['stages']
    p('V2开发定位依次106→108→111→110→112→111/128；计数22→26→29→34→28→28/40。SFT1600总分159/184最高，但预定定位优先规则选择RSI1（155/184）；没有根据新测试改规则。SFT800→RSI1仅定位多1题，不能声称RSI获得稳定显著提升。')
    loc = study['paired_comparison']['loc']; total = study['paired_comparison']['total']
    lo, hi = loc['new_accuracy_wilson95']
    total_p = '<0.001' if total['mcnemar_exact_p'] < .001 else f"={total['mcnemar_exact_p']:.3f}"
    p(f"同题新测试V2定位{final['loc']['correct']/256:.1%}，较V1提高{100*(final['loc']['correct']-base['loc']['correct'])/256:.1f}个百分点；定位纠正{loc['gained']}题、损失{loc['lost']}题，配对精确检验p={loc['mcnemar_exact_p']:.3f}；总分配对p{total_p}。定位Wilson 95%区间{lo:.1%}–{hi:.1%}，仅反映本合成集抽样，不含训练种子波动。")
    p(f"两轮自举图训练后独立复测分别{rounds[0]['bootstrap_posttrain']['total']['correct']}/15、{rounds[1]['bootstrap_posttrain']['total']['correct']}/27；分母不同，不能当同题增幅。空白同题16格探针{get('v2_blank16')['correct']}/16。像素框由外部算法识别，工具成绩不能充作模型能力；V1历史穷举核验184/184主要由完整答案空间与验证器保证。",size=18)
    p('训练限时后只续评，保留15408原日志、全部源代码快照及15427选择哈希；不覆盖失败记录。CPU初跑GBK输出异常，固定UTF-8后六阶段完整通过。扩展测试补齐0–4并拒绝重复数字，避免原四道计数均为1造成常数答案高分。',size=18)
    examples_table()
    heading('六、思考题作答')
    p('1．解析式验证器无需训练，便于追溯；但同色连通、遮挡、边框断裂和近似颜色可能误判。本次新图均核验一致；原E10/E11重复数字被宽松解析接受的缺陷已用严格格式检查修正。真实图可换检测器/OCR，低置信度拒绝入池。',size=18)
    p('2．默认SFT、工具、两轮RSI定位均0/8；扩展V1、V2及其两轮为8/8。新测试独立定位与框辅助成绩单列；关闭工具后的新图结果才反映参数是否学会。3．高亮自查重看自身候选，验证器提供独立外部证据；后者更符合第11章调用工具并据返回结果行动的定义。',size=18)
    p('4．V2两轮新增15、12条，第二轮仍有定位与总分回退。纠错样本数量、分布、学习率、多任务遗忘和验证噪声都会影响上限；本次审计未见入池标签污染，不据此保证更长递归必然更好。',size=18)
    p('5．离散格位减轻小模型坐标回归与解析负担。4×4→8×8→16×16使类别16→64→256，格框变小可能提高命中时IoU，但覆盖样本、训练量和边界敏感性增加。6．无GT可用预训练模型提候选、独立检测或多源一致性核验并少量人工复核启动；低标注成本不等于算力和工具免费。',size=18)
    p('样板补充：ChartQA用OCR/表格提取数值，按坐标轴、单位和聚合公式复算；证据冲突拒绝入池。以上覆盖指南六题与样板四题。',size=18)
    heading('七、结论与反思')
    p(f"默认实验完整保留。多样图和视觉适配解决了原定位0/8；继续联合训练及保守RSI，使全新同题总分260→290/368、定位201→215/256。RSI仅有小幅开发定位收益，第二轮回退，不能夸大。教学上闭环值得复现；纯色图工程应用直接解析更便宜。仍有{256-final['loc']['correct']}道定位错误，且仅单训练种子、合成分布，下一步应做真实图核验与多种子复测。")
    return blob
