"""Three-page template content for the completed improvement study."""
import json


def create_content(builder):
    root=builder.ROOT
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    folder=root/'improvement/final_v1'; summary=read(folder/'summary.json')
    protocol=read(folder/'protocol.json'); selection=read(folder/'selection.json')
    old=read(root/'analysis/summary.json'); development=read(root/'analysis/improvement_summary.json')
    formal=read(root/'provenance/improvement_run.json')
    blob=[]
    p=lambda s,**kw:blob.append(builder.paragraph(s,**kw))
    def heading(s,before=False): p(s,heading=True,size=22,after=65,before=before)
    cap=lambda s:p(s,bold=True,size=17,after=30,keep=True)
    tb=lambda rows,widths,size=16:blob.append(builder.table(rows,widths,size))
    def score(result):
        s=result['summary']
        return [f"{s[t]['correct']}/{s[t]['n']}" for t in ('loc','count','exist','total')]+[f"{s['loc_iou']:.3f}"]
    def get(name): return read(folder/f'{name}.json')
    selected=summary['selected']; final=get('selected_final_test'); rich=get('selected_rich_tool_final_test')
    p('《大模型通识》实验六实验报告',bold=True,size=29,after=45,center=True)
    p('SmolVLM 精准定位与 RSI 改进验证',size=19,after=70,center=True)
    tb([['项目','实际信息'],['姓名／学号／班级','孙驰／2024311281／计算机10班'],
        ['日期／环境','2026年10月22日；HPC NVIDIA A30 24GB；NF4 QLoRA'],
        ['代码与记录','github.com/s-max-oss/llm-course-labs/tree/main/lab06']], [1800,7200],17)
    heading('一、实验目的与任务概述')
    p('完成256×256合成图的4×4网格定位、计数和存在性判断，比较基线、SFT、工具纠错及两轮RSI。先按指南10步设置复现，再针对定位0/8开展数据、输出编码、视觉适配和图像多样性对照；以留出新图检验改进，保留失败记录。')
    heading('二、实验环境与复现步骤')
    p('Python 3.10.20；torch 2.6.0+cu124；transformers 4.57.6；peft 0.17.1；bitsandbytes 0.48.1。SmolVLM-256M离线权重，NF4双重量化、BF16计算，固定512px单视图；原始图仍为指南256px，修正样板480px笔误。')
    tb([['命令／脚本','配置与产物'],['run_hpc.slurm','40张训练图、15张评测图；15种子；10步SFT、2×10步RSI'],
        ['run_improvement／run_visual_grounding.slurm','语言LoRA三组各400步；视觉+连接层LoRA 1200步'],
        ['run_augmented_grounding／run_final.slurm',f"4000张变化图课程训练；联合SFT {protocol['joint_steps']}步、RSI 2×{protocol['rsi_steps']}步"]], [3700,5300],16)
    cpu=read(root/'improvement/cpu_reference/metadata.json')
    p(f"扩展HPC累计{formal['gpu_job_minutes']:.1f}GPU分钟（含开发），最终作业{formal['final_job']}耗时{formal['elapsed']}、退出码{formal['exit_code']}，结束后队列为空。CPU为Python {cpu['python'].split()[0]}/torch {cpu['torch']}，FP32、8线程完整六阶段耗时{cpu['seconds']/60:.1f}分钟，6/6退出码0；与NF4成绩分开。",size=18)
    heading('三、方法与关键实现')
    p('目标框中心按r=floor(cy/64)+1、c=floor(cx/64)+1求1起始行列，输出rXcY，评分映射为中文坐标；仅完整有效答案计分。IoU为预测格框与目标对象框的交并比，正确格不等于IoU=1。prompt标签置−100，只训练答案token。')
    p('默认语言q/k/v/o LoRA有921600个参数。扩展到视觉注意力、跨模态连接层及语言模块后，可训练1614336个参数，占0.6255%；非重入检查点与首步非零梯度检查保证视觉模块实际更新。固定128图明显过拟合后，改用逐步不同的4000图，从单目标递进到2–4目标。')
    p('定位训练、开发验证、最终测试各覆盖16格×8类别；验证及测试各128张，另补0–4计数与均衡存在性，总计各184题，图像哈希不跨集重复。按开发定位率、再按总正确率选阶段，平分取较早者；最终测试不参与选型或自举。')
    p('RSI反馈只用训练图：紫框自查、检测框提示后仍错，则模型给16格/0–4/有无的合法候选评分，验证器依次核验；仅通过核验、首次纠正且未重复的候选入池，来源注明自查、检测框或候选核验。弱提示、检测框、有限候选核验均单列；搜索依赖工具。',size=18)
    log_path=root/'provenance'/f"final-{formal['final_job']}.log"
    log=log_path.read_text(encoding='utf-8',errors='replace').splitlines()
    excerpts=[next(line for line in log if line.startswith(key)) for key in ('[rsi1]','[rsi2]','[final]','[complete]')]
    p('运行日志节选（完整输出见provenance/'+log_path.name+'）：\n'+'\n'.join(excerpts),size=14,after=25)
    heading('四、实验结果',before=True)
    cap('表6  指南15题的默认参照与扩展对照（定位8、计数4、存在性3）')
    baseline=old['stages']['baseline']; old_sft=old['stages']['sft_eval']
    oldscore=lambda s:[f"{s[t]['correct']}/{s[t]['n']}" for t in ('loc','count','exist','total')]+[f"{s['loc_mean_iou']:.3f}"]
    tb([['阶段','定位','计数','存在','总计','IoU'],['原基线']+oldscore(baseline),['原10步SFT']+oldscore(old_sft),
        ['原SFT+验证器']+oldscore(old['stages']['sft_tool_eval']),
        ['原RSI第1轮']+oldscore(old['stages']['rsi1_eval']),['原RSI第2轮']+oldscore(old['stages']['rsi2_eval']),
        ['改进SFT独立']+score(get('sft_legacy')),['改进SFT+弱提示']+score(get('sft_weak_tool_legacy')),
        ['改进SFT+检测框']+score(get('sft_tool_legacy')),['改进SFT+候选核验']+score(get('sft_search_tool_legacy')),['RSI第1轮独立']+score(get('rsi1_legacy')),
        ['RSI第2轮独立']+score(get('rsi2_legacy'))], [2700,1000,1000,1000,1200,2100],16)
    p('默认两轮：池15→17→17，新增2、0条，轮末独立7/15、4/15。下表为扩展闭环。',size=16,after=25)
    cap('表7  扩展两轮反馈（轮末独立依次12/15、13/15，见表6）')
    tb([['轮次','池规模','直接对/15','自查/框/搜索次数','纠错后/15','新增'],
        *[[str(r['round']),str(r['pool_size']),str(r['direct_correct']),f"{r['self_attempts']}/{r['tool_attempts']}/{r['search_attempts']}",str(r['corrected_correct']),str(r['added'])] for r in summary['rounds']]],
       [900,1400,1400,2700,1500,1100],16)
    s=final['summary']; t=rich['summary']; weak=get('selected_weak_tool_final_test')['summary']
    searched=get('selected_search_tool_final_test')['summary']
    p(f"按开发集选择{selected}。最终独立测试：定位{s['loc']['correct']}/128、计数{s['count']['correct']}/40、存在性{s['exist']['correct']}/16，总计{s['total']['correct']}/184；定位IoU={s['loc_iou']:.3f}。同模型弱提示、检测框、候选核验的定位依次{weak['loc']['correct']}/128、{t['loc']['correct']}/128、{searched['loc']['correct']}/128；工具分数单列。",size=18)
    cap('代表性路径（E为旧评测，F为最终测试题，逐题原文可查）')
    rows=get('sft_tool_legacy')['rows']
    feedback=read(folder/'rsi_memory.json')['records']
    choose=lambda predicate:next((r for r in rows if predicate(r)),None)
    self_example=next((r for r in rows+feedback+get('selected_rich_tool_final_test')['rows'] if r['self_check'] and r['self_check']['verification']['passed']),None)
    search_example=next((r for r in get('sft_search_tool_legacy')['rows'] if r['candidate_search']),None)
    examples=[('直接正确',choose(lambda r:r['raw_correct'])),('自查改对',self_example),
              ('检测框改对',choose(lambda r:not r['raw_correct'] and r['tool_check'] and r['correct'])),
              ('候选核验',search_example),('检测框仍错',choose(lambda r:not r['correct']))]
    def short(x): return x.replace('\n',' ')[:26]
    tb([['路径','题号／目标','原始→最终','GT／判断'],
        *[[name,r['id'].replace('final_test_','F')+' '+r['target'],short(r['raw'])+' → '+short(r['final']),r['gold']+('／正确' if r['correct'] else '／仍错')] if r else [name,'无此样例','按实际注明','—'] for name,r in examples]],
       [1300,1850,3800,2050],15)
    blob.append('FIGURE_PLACEHOLDER')
    p('图1  自绘开发阶段独立成绩及最终测试；工具条件单列。',size=16,after=30,center=True)
    heading('五、结果分析与问题调试',before=True)
    p('原10步中定位只有6次更新，8条种子覆盖6格且缺第二行；两轮没有新增定位样本。延长原种子至400步后种子8/8，但开发仅10/128；补齐128图的中文/紧凑编码仅8/128、11/128。说明格式学会、训练拟合，均不能证明学会看图定位。')
    aug=[r for r in development['development_localization'] if r['recipe'].startswith('augmented_v1/')]
    best=max(aug,key=lambda r:r['accuracy'])
    p(f"视觉适配固定128图训练后训练抽样16/16、开发16/128；变化图最佳开发为{best['correct']}/128（{best['recipe'].split('/')[-1]}）。开发重复比较可能乐观，所以最终结论以未参与选择的184题为准。变化图同时增加数量、多样性和课程安排，不能只归因于一个因素。")
    p(f"最终独立定位{s['loc']['correct']}/128，空白图16格诊断为{read(folder/'selected_blank16.json')['correct']}/16；空白图题干相同、格位均衡，作为题目偏置检查。两轮RSI是否提升以开发/留出独立回答判断，若回退则保留并由开发选择较早阶段，不挑最终测试最高分。")
    p('拓展B：自举样本再训后独立答对分别为'+'、'.join(f"第{r['round']}轮{r['bootstrap_independent_correct']}/{r['bootstrap_n']}" for r in summary['rounds'])+'；这是训练反馈图复测，不能充作泛化成绩。',size=18)
    p('检测框提供外部视觉证据；候选核验在22道前序失败题上平均核验2.73次，首位命中0/22（失败子集），模型共评分195条候选。搜索全对依赖验证器，不能归为模型独立能力。自举保存核验后的生成/受限候选，未以GT替换。原计数全为1，新测试补齐0–4并拒绝重复数字。')
    p('纯CPU初跑遇到Windows GBK无法输出异常字符而中断，指定UTF-8后完整重跑。验证器适用纯色无重叠图，真实背景与遮挡未测；仅单次训练种子。128定位测试区间、逐格误差及全量像素核验见补充分析，不能外推通用能力。',size=18)
    heading('六、思考题作答')
    p('1．解析验证器无需训练、便于追溯；遮挡、同色连通和近似颜色是边界。本次4000新图核验无误判；实际遇到旧E10/E11重复1也被宽松数值解析接受，已严格格式化。真实图可换检测器/OCR，低置信度拒绝入池，闭环仍可成立。',size=18)
    p(f"2．表6的SFT独立定位{get('sft_legacy')['summary']['loc']['correct']}/8，检测框辅助{get('sft_tool_legacy')['summary']['loc']['correct']}/8，RSI两轮独立{get('rsi1_legacy')['summary']['loc']['correct']}/8、{get('rsi2_legacy')['summary']['loc']['correct']}/8；参数更新与即时工具收益分别由这些对照体现。",size=18)
    checks=get('selected_rich_tool_final_test')['rows']
    p(f"3．最终测试高亮自查在19次中改对9题；检测框重答在31次中通过9题（处理的是不同失败子集）。自查重看自己的候选，验证器则提供独立外部证据，更符合第11章调用工具并用返回结果行动的定义。",size=18)
    p(f"4．两轮新增{summary['rounds'][0]['added']}、{summary['rounds'][1]['added']}条。第二轮增益受重复题去重、分布窄、短步数和多任务遗忘影响；核验错误也可能污染池，但本次审计未发现。出现回退就保留并按开发集选阶段。",size=18)
    p('5．离散格位降低小模型输出与解析难度。4×4→8×8→16×16使答案从16→64→256类；格变小可减轻离散误差，命中时IoU可能提高，但每类覆盖、训练量及边界敏感性均增加。',size=18)
    p('6．无GT可用预训练模型提出候选，以高置信度检测/多源一致性核验，少量人工复核启动；未验证样本不入池。合成参数和可靠验证器提供低标注成本教师，但训练算力、工具调用及真实图复核并不免费。',size=18)
    p('样板补充：ChartQA用OCR/表格提取数值，按坐标轴、单位及聚合公式复算，证据冲突时拒绝入池。以上同时覆盖指南六题与样板四题。',size=18)
    heading('七、结论与反思')
    p(f"定位提升主要来自视觉适配与多样图SFT：原0/8→8/8；RSI开发106→104→100/128，未见泛化增益。最终独立{s['loc']['correct']}/128、检测框辅助{t['loc']['correct']}/128，后者体现工具贡献。教学上值得观察闭环边界；工程上纯色图直接解析更省算力，当前准确率不足以支撑精准定位。局限为合成分布、验证器覆盖与格位误差；下一步做真实图核验、多种子复测。")
    return blob
