"""Fill the supplied experiment-six Word template using text-only OOXML edits.

The document body is rebuilt with the template's section properties/styles and
all seven requested headings. No lxml serialization is used for authoring.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
COURSE = next(parent for parent in ROOT.parents if (parent / '实验指南与模板').is_dir())
TEMPLATE = COURSE / "实验指南与模板" / "实验六_RSI递归自进化_实验报告样板.docx"
OUT = COURSE / "提交材料" / "2024311281_孙驰_实验作业六.docx"
SUMMARY = json.loads((ROOT / "analysis" / "summary.json").read_text(encoding="utf-8"))
FORMAL = json.loads((ROOT / 'provenance' / 'formal_run.json').read_text(encoding='utf-8'))
RUN_LOG = ROOT / "provenance" / f"slurm-{FORMAL['job_id']}.log"
RSID = "0261016A"


def paragraph(value, *, heading=False, bold=False, size=19, after=50, keep=False, center=False, before=False):
    style = '<w:pStyle w:val="1"/>' if heading else ""
    props = style + ('<w:keepNext/>' if keep or heading else '')
    if before:
        props += '<w:pageBreakBefore/>'
    props += f'<w:keepLines/><w:spacing w:after="{after}" w:line="240" w:lineRule="auto"/>'
    if center:
        props += '<w:jc w:val="center"/>'
    font = '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="微软雅黑"/>'
    color = '008577' if heading else '17212B'
    font += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/>'
    if bold or heading:
        font += '<w:b/><w:bCs/>'
    runs = '<w:br/>'.join('<w:t xml:space="preserve">'+escape(s)+'</w:t>' for s in str(value).split('\n'))
    return f'<w:p w:rsidR="{RSID}" w:rsidRDefault="{RSID}"><w:pPr>{props}</w:pPr><w:r><w:rPr>{font}</w:rPr>{runs}</w:r></w:p>'


def table(rows, widths, size=17):
    assert all(len(row) == len(widths) for row in rows)
    grid = '<w:tblGrid>'+''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)+'</w:tblGrid>'
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for j, value in enumerate(row):
            fill = 'DCEAF2' if i == 0 else ('F8FAFC' if i % 2 == 0 else 'FFFFFF')
            props = (f'<w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>'
                     f'<w:shd w:fill="{fill}"/><w:vAlign w:val="center"/>'
                     '<w:tcMar><w:top w:w="30" w:type="dxa"/><w:bottom w:w="30" w:type="dxa"/>'
                     '<w:left w:w="55" w:type="dxa"/><w:right w:w="55" w:type="dxa"/></w:tcMar></w:tcPr>')
            cells.append('<w:tc>'+props+paragraph(value, bold=i==0, size=size, after=0)+'</w:tc>')
        lines.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'+''.join(cells)+'</w:tr>')
    borders = '<w:tblBorders>'+''.join(f'<w:{side} w:val="single" w:sz="4" w:color="D9D9D9"/>'
                                    for side in ('top','left','bottom','right','insideH','insideV'))+'</w:tblBorders>'
    return '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/>'+borders+'<w:tblLayout w:type="fixed"/></w:tblPr>'+grid+''.join(lines)+'</w:tbl>'


def score_line(name):
    m = SUMMARY['stages'][name]
    return [f"{m[k]['correct']}/{m[k]['n']}" for k in ('loc','count','exist','total')] + [f"{m['loc_mean_iou']:.3f}"]


def show_example(row):
    if not row:
        return ['本轮无符合条件的样例', '—', '—', '—', '—']
    path = '—'
    if row['self_check'] and row['self_check']['verification']['passed']:
        path = '高亮自查'
    elif row['tool_check']:
        path = '验证器重答'
    def trim(x, n=34):
        x = x.replace('\n',' ') if isinstance(x,str) else str(x)
        return x[:n] + ('…' if len(x)>n else '')
    return [row['img'], trim(row['question'], 28), trim(row['raw'], 30), path+'→'+trim(row['final'], 24),
            ('正确' if row['correct'] else '仍错')+'；GT '+trim(row['gold'],10)]


def create_content():
    stage = SUMMARY['stages']
    rounds = SUMMARY['rounds']
    trans = SUMMARY['transitions']
    examples = SUMMARY['examples']
    total = lambda name: stage[name]['total']['correct']
    loc = lambda name: stage[name]['loc']['correct']
    blob = []
    p = lambda s, **kw: blob.append(paragraph(s, **kw))
    h = lambda s: p(s, heading=True, size=22, after=70, before=s.startswith('四、'))
    cap = lambda s: p(s, bold=True, size=18, after=35, keep=True)
    tb = lambda rows, widths, size=17: blob.append(table(rows, widths, size))
    p('《大模型通识》实验六实验报告', bold=True, size=29, after=45, center=True)
    p('SmolVLM-256M 精准定位的 RSI 递归自进化实验', size=19, after=90, center=True)
    tb([['项目','实际信息'],['姓名／学号／班级','孙驰／2024311281／计算机10班'],
        ['日期／平台',f"{FORMAL['date']}；HPC {FORMAL['node']}，NVIDIA A30 24GB"],
        ['代码仓库','https://github.com/s-max-oss/llm-course-labs/tree/main/lab06']], [1800,7200], 18)
    h('一、实验目的与任务概述')
    p('把彩色几何图形的目标位置映射到256×256画布上的4×4网格，比较基线、10步SFT、推理时工具纠错和两轮RSI后续训练。完成合成数据、颜色/形状像素验证器、高亮回灌、验证样本入池、逐题审计与图表；重点分辨参数更新与工具辅助的贡献。')
    h('二、实验环境与复现步骤')
    p('Python 3.10.20；torch 2.6.0+cu124；transformers 4.57.6；peft 0.17.1；bitsandbytes 0.48.1。A30上NF4双重量化、BF16计算；模型复用上次实验的离线权重，图像输入统一512像素单视图。'+f"作业号{FORMAL['job_id']}，墙钟{FORMAL['elapsed']}；运行后队列已清空。")
    tb([['步骤／命令','作用与产物'],['make_data；make_sft','40张训练图、15张留出图；15条GT种子样本'],
        ['infer；sft --steps 10','15题基线；SFT适配器与逐步损失'],
        ['eval；rsi --rounds 2 --steps 10','三阶段评测；两轮反馈、数据池快照、适配器']], [2850,6150], 17)
    p('完整命令、原始Slurm日志和逐题JSON见仓库lab06/run_hpc.slurm、provenance及results；评测阶段计时分别为'+
      '、'.join(f"{n} {SUMMARY['seconds'][n]:.1f}s" for n in ('baseline','sft_eval','sft_tool_eval'))+'。')
    h('三、方法与关键实现')
    p('定位：框中心(cx,cy)按floor(4·cy/256)、floor(4·cx/256)转成行列，输出“第r行第c列”，兼容rXcY与X,Y；评测另算预测格框对GT对象框的IoU。输入和答案分别编码，prompt标签设为−100，只对答案token计算损失。')
    p('SFT：冻结基座，仅在语言注意力q/k/v/o注入LoRA（r=8，α=16，dropout=0.05），可训练921600/257406528=0.3580%。RSI：错误答案若有可解析格则高亮回灌自查；像素验证器按颜色连通块及角点区分圆/方，提供核验结论与候选格覆盖率，不提示目标答案；只有验证正确且未重复的训练图样本入池，再训练10步。')
    p('参考脚本会把评测图用于种子SFT，且以标准答案判断纠错触发。本实现用t00–t14做种子、t15–t29做反馈、e00–e14固定留出评测；反馈决策只读图像和题目，GT仅事后评分。样板的“480px”和“make_seed”与指南不一致，实际采用256px及make_sft。')
    h('四、实验结果')
    cap('表6  留出15题逐阶段结果（loc 8、count 4、exist 3）')
    tb([['阶段','loc','count','exist','总计','平均IoU'],
        ['基线']+score_line('baseline'), ['SFT 10步']+score_line('sft_eval'),
        ['SFT+验证器']+score_line('sft_tool_eval'), ['RSI第1轮']+score_line('rsi1_eval'),
        ['RSI第2轮']+score_line('rsi2_eval')], [2600,1060,1060,1060,1100,2120], 16)
    cap('表7  训练图反馈轮次（轮末准确率为留出评测）')
    tb([['轮次','池规模','直接对','自查','工具重答','纠错后对','轮末/15'],
        ['初始种子','15','—','—','—','—',f"{total('sft_eval')}/15"],
        *[[f"第{r['round']}轮",str(r['pool_size']),f"{r['direct_correct']}/15",str(r['self_rewrites']),
           str(r['tool_rewrites']),f"{r['corrected_correct']}/15",f"{r['heldout_correct']}/15"] for r in rounds]],
       [1450,1100,1100,1100,1400,1600,1250],16)
    p(f"留出集工具纠错尝试{trans['self_attempts']}次高亮自查、{trans['tool_attempts']}次验证器重答；由错变对{trans['wrong_to_right']}题，"
      f"由对变错{trans['right_to_wrong']}题。两轮反馈新增{rounds[0]['added']}、{rounds[1]['added']}条，最终池{SUMMARY['final_pool']}条。")
    cap('代表性逐题输出摘录（完整原文见results；未发生的路径按实际注明）')
    tb([['样例','图名','问题','原始输出','路径与最终输出','结论'],
        ['直接正确']+show_example(examples['direct_correct']),
        ['自查改对']+show_example(examples['self_corrected']),
        ['验证器改对']+show_example(examples['tool_corrected']),
        ['仍然失败']+show_example(examples['still_failed'])],
       [1200,900,2200,1600,2200,900],15)
    blob.append('FIGURE_PLACEHOLDER')
    p('图1  自绘留出集定位与总正确率折线；SFT+工具列是同一模型的推理时辅助，其余列为模型独立回答。',size=16,after=40,center=True)
    log = RUN_LOG.read_text(encoding='utf-8',errors='replace') if RUN_LOG.exists() else ''
    keys = ('[baseline] loc=', '[sft_eval] loc=', '[sft_tool_eval] loc=',
            '[rsi1] pool=', '[rsi2] pool=', '[complete]')
    excerpts = [next((line for line in log.splitlines() if line.startswith(key)), '') for key in keys]
    p('日志节选：\n'+'\n'.join(line for line in excerpts if line),size=16,after=45)
    h('五、结果分析与问题调试')
    p(f"10步SFT将总分{total('baseline')}/15变为{total('sft_eval')}/15，定位仍为{loc('sft_eval')}/8；"
      f"工具提示后的定位为{loc('sft_tool_eval')}/8，总分{total('sft_tool_eval')}/15。两轮再训后定位依次{loc('rsi1_eval')}/8、{loc('rsi2_eval')}/8。"
      '基线常把“4×4”当算术题。试跑曾直接提示答案并从无效复述计算IoU；审查后取消这两种做法并重跑。正式IoU只从独立完整的有效坐标计算；旧试跑另存，不混入表6。')
    repeated = SUMMARY['repetitive_count_outputs']['sft_tool_eval']
    p('表6计数按指南的数量口径判分：'+('、'.join(repeated) if repeated else '无')+'出现重复数字，虽数量一致但表达异常。剔除这类异常后，工具辅助的完整回答为'+str(SUMMARY['without_repetitive_count_correct']['sft_tool_eval'])+'/15；不能只用数量命中宣称输出质量提高。')
    type_labels = {'loc': '定位', 'count': '计数', 'exist': '存在性'}
    added_types = '、'.join(type_labels[t] for t in SUMMARY['bootstrap_types']) or '无'
    p('解析式验证器只适用于本实验的纯色、无重叠几何图；同色异形靠角点判别，真实图片、遮挡、抗锯齿或颜色接近阈值时可能误判。此次4道计数题的标准答案均为1；两轮新增'+str(SUMMARY['final_pool']-15)+'条，涉及'+added_types+'。15题、单一种子和两轮训练不足以证明泛化；纠错分数仍包含外部工具的核验反馈。')
    h('六、思考题作答')
    p('1．8×8会把输出空间从16类扩大到64类，定位命中更难；若命中，格框更小，IoU可能上升，但也更易受边界误差影响。')
    p('2．同色目标重叠、边框断裂或目标颜色偏离阈值会使连通块分裂/合并。此次用角点判别并对全部45道生成题与GT核对，45/45一致；真实图尚未验证。')
    p('3．合成图的绘制参数可直接给出种子标签，验证通过的反馈近乎无人工标注成本；迁移真实图片后需人工复核或更强视觉工具，成本和噪声都会上升。')
    p('4．ChartQA可解析数据表或OCR数值，按坐标轴刻度、单位、聚合公式复算答案；当文字识别置信度低或图表不规范时返回“不确定”，避免错误样本入池。')
    h('七、结论与反思')
    p(f"本次留出集独立定位：基线{loc('baseline')}/8→SFT{loc('sft_eval')}/8→两轮{loc('rsi1_eval')}/8、{loc('rsi2_eval')}/8；"
      f"同模型加工具后为{loc('sft_tool_eval')}/8。应按上述数字分别评估参数更新与工具反馈；此次尚未观察到模型独立定位的提升。"
      '下一步应扩充训练图和定位样本、延长SFT步数，并用多种子留出集与人工核验的真实图片检验效果。')
    return blob


def main():
    assert TEMPLATE.exists(), TEMPLATE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(TEMPLATE) as source:
        package = {name: source.read(name) for name in source.namelist()}
    xml = package['word/document.xml'].decode('utf-8')
    prefix, body = xml.split('<w:body>',1)
    _, suffix = body.split('</w:body>',1)
    sect = re.search(r'<w:sectPr\b.*?</w:sectPr>',body,re.S)
    assert sect, 'Template section properties missing'
    if 'xmlns:v=' not in prefix:
        prefix = prefix.replace('<w:document ', '<w:document xmlns:v="urn:schemas-microsoft-com:vml" ',1)
    if (ROOT / 'improvement/final_v2/summary.json').exists():
        import optimized_content
        content = optimized_content.create_content(sys.modules[__name__])
    elif (ROOT / 'improvement/final_v1/summary.json').exists():
        import improved_content
        content = improved_content.create_content(sys.modules[__name__])
    else:
        content = create_content()
    rels = package['word/_rels/document.xml.rels'].decode('utf-8')
    next_id = max(map(int,re.findall(r'Id="rId(\d+)"',rels)))+1
    rid = f'rId{next_id}'
    figure = ROOT / 'figures' / ('optimized_v2_accuracy.png' if (ROOT / 'improvement/final_v2/summary.json').exists() else 'improved_accuracy.png' if (ROOT / 'improvement/final_v1/summary.json').exists() else 'accuracy.png')
    figure_width = 440 if (ROOT / 'improvement/final_v1/summary.json').exists() else 390
    from PIL import Image
    with Image.open(figure) as image:
        height = figure_width * image.height / image.width
    figure_xml = (f'<w:p w:rsidR="{RSID}" w:rsidRDefault="{RSID}"><w:pPr><w:keepNext/><w:keepLines/>'
                  f'<w:jc w:val="center"/><w:spacing w:after="30"/></w:pPr><w:r><w:pict>'
                  f'<v:shape id="lab06_accuracy" type="#_x0000_t75" style="width:{figure_width}pt;height:{height:.1f}pt">'
                  f'<v:imagedata r:id="{rid}" o:title="accuracy"/></v:shape></w:pict></w:r></w:p>')
    content[content.index('FIGURE_PLACEHOLDER')] = figure_xml
    rels = rels.replace('</Relationships>',f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/lab06_accuracy.png"/></Relationships>')
    package['word/_rels/document.xml.rels'] = rels.encode('utf-8')
    package['word/media/lab06_accuracy.png'] = figure.read_bytes()
    types = package['[Content_Types].xml'].decode('utf-8')
    if 'Extension="png"' not in types:
        types = types.replace('</Types>','<Default Extension="png" ContentType="image/png"/></Types>')
        package['[Content_Types].xml'] = types.encode('utf-8')
    package['word/document.xml'] = (prefix+'<w:body>'+''.join(content)+sect.group()+'</w:body>'+suffix).encode('utf-8')
    with ZipFile(OUT,'w',ZIP_DEFLATED) as destination:
        for name, payload in package.items():
            destination.writestr(name,payload)
    print(OUT)


if __name__ == '__main__':
    main()
