"""Recompute the localization study from saved predictions, without inference."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import hw6_rsi_localize as h
import finish_grounding as final

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(result):
    if 'summary' in result: return result['summary']
    return {k:result[k] for k in ('n','correct','valid','mean_iou') if k in result}


def wilson(correct,n):
    z=1.96; p=correct/n; d=1+z*z/n
    middle=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [middle-half,middle+half]


def collect():
    root=ROOT/'improvement'; rows=[]
    for path in sorted(root.glob('ablation_v1/*/validation.json')):
        rows.append({'recipe':path.parent.name,'path':str(path.relative_to(ROOT)),**metrics(read(path))})
    for run in ('vision_compact_v1','augmented_v1'):
        for path in sorted((root/run).glob('step*/validation.json')):
            rows.append({'recipe':run+'/'+path.parent.name,'path':str(path.relative_to(ROOT)),**metrics(read(path))})
    for row in rows:
        row['accuracy']=row['correct']/row['n']; row['wilson95']=wilson(row['correct'],row['n'])
    results={'development_localization':rows,'final_test_used_for_selection':False}
    for name in ('joint_development_v1','final_v1'):
        path=root/name/'summary.json'
        if path.exists(): results[name]=read(path)
    return results


def audit_final(folder):
    """Independently check strict scores, pool provenance and heldout isolation."""
    data=read(folder/'joint_questions_at_run.json'); hashes={}
    for split,questions in data.items():
        for q in questions:
            digest=hashlib.sha256(final.exp.load_image(ROOT,q).tobytes()).hexdigest()
            assert hashes.get(digest,split)==split
            hashes[digest]=split
            assert h.pixel_answer(final.exp.load_image(ROOT,q),q)[0]==q['gold']
    selection=read(folder/'selection.json')
    assert selection['final_test_evaluated_at_selection'] is False
    scores=selection['validation']
    selected=max(scores,key=lambda k:(scores[k]['loc']['correct']/scores[k]['loc']['n'],scores[k]['total']['correct']/scores[k]['total']['n']))
    assert selected==selection['selected']
    qs={q['id']:q for questions in data.values() for q in questions}
    for split in ('eval','rsi'):
        qs.update({q['id']:q for q in final.legacy_questions(ROOT,split)})
    for path in folder.glob('*.json'):
        result=read(path)
        if not isinstance(result,dict) or 'summary' not in result or 'rows' not in result: continue
        for row in result['rows']:
            q=qs[row['id']]
            assert row['correct']==(final.canonical(q,row['final'])==q['gold'])
            if q['type']=='loc': assert abs(row['iou']-h.loc_iou(q,row['final']))<1e-10
            searched=row.get('candidate_search')
            if searched:
                ranking=searched['ranking'];attempts=searched['attempts']
                assert len(ranking)==searched['candidate_count']
                assert len({r['candidate'] for r in ranking})==len(ranking)
                scores=[r['sequence_log_probability'] for r in ranking]
                assert all(math.isfinite(s) for s in scores) and scores==sorted(scores,reverse=True)
                assert [r['candidate'] for r in attempts]==[r['candidate'] for r in ranking[:len(attempts)]]
                assert searched['attempt_count']==len(attempts)
                assert all(not r['passed'] for r in attempts[:-1])
                for attempt in attempts:
                    assert attempt['passed']==final.check(final.exp.load_image(ROOT,q),q,attempt['candidate'],False)['passed']
                if searched['answer'] is not None:assert attempts[-1]['passed'] and attempts[-1]['candidate']==row['final']
        assert result['summary']['total']['correct']==sum(r['correct'] for r in result['rows'])
        for tp in ('loc','count','exist'):
            subset=[r for r in result['rows'] if r['type']==tp]
            assert result['summary'][tp]['correct']==sum(r['correct'] for r in subset)
    memory=read(folder/'rsi_memory.json')['records']
    for number in (1,2):
        for q in read(folder/f'pool_round{number}.json'):
            assert not q['image'].startswith('improvement/data/final_test/')
            assert not q['image'].startswith('improvement/data/validation/')
            if 'verified_answer' in q:
                assert q['image'].startswith('imgs/train/') or '/train/' in q['image']
                assert final.check(final.exp.load_image(ROOT,q),q,q['verified_answer'])['passed']
                assert any(r['id']==q['id'] and r['final']==q['verified_answer'] and r['final_verified'] for r in memory)
    return {'joint_questions_checked':sum(map(len,data.values())),'unique_images':len(hashes),
            'strict_scores':'passed','bootstrap_provenance':'passed','selection':'passed','split_isolation':'passed'}


def error_breakdown():
    path=ROOT/'improvement/final_v1/selected_final_test.json'
    if not path.exists():return None
    questions={q['id']:q for q in read(ROOT/'improvement/final_v1/joint_questions_at_run.json')['final_test']}
    rows=[r for r in read(path)['rows'] if r['type']=='loc']
    stats={'n':len(rows),'row_correct':0,'column_correct':0,'selected_any_object':0,
           'wrong_selected_same_color':0,'wrong_selected_same_shape':0,'class_correct':{},'cell_correct':{}}
    confusion=[[0]*17 for _ in range(16)]
    for r in rows:
        q=questions[r['id']];cell=h.parse_cell(r['final']) if final.canonical(q,r['final']) else None
        if cell:
            stats['row_correct']+=cell[0]==q['cell'][0];stats['column_correct']+=cell[1]==q['cell'][1]
            matched=[o for o in q['objects'] if h.cell_of(o['box'])==cell]
            stats['selected_any_object']+=bool(matched)
            if not r['correct']:
                stats['wrong_selected_same_color']+=any(o['class'][0]==q['target'][0] for o in matched)
                stats['wrong_selected_same_shape']+=any(o['class'][1]==q['target'][1] for o in matched)
        truth=q['cell'][0]*4+q['cell'][1];pred=cell[0]*4+cell[1] if cell else 16
        confusion[truth][pred]+=1
        for group,key in [('class_correct',q['target']),('cell_correct',str(truth))]:
            score=stats[group].setdefault(key,{'correct':0,'n':0});score['n']+=1;score['correct']+=r['correct']
    stats['confusion_16_cells_plus_invalid']=confusion
    total=sum(r['correct'] for r in rows);stats['accuracy_wilson95']=wilson(total,len(rows))
    return stats


def search_costs(folder):
    results={}
    for path in folder.glob('*search_tool*.json'):
        rows=read(path)['rows'];fallback=[r['candidate_search'] for r in rows if r.get('candidate_search')]
        if fallback:
            results[path.stem]={'questions':len(rows),'fallback_questions':len(fallback),
                    'mean_verifier_attempts_in_fallback':statistics.mean(r['attempt_count'] for r in fallback),
                    'median_verifier_attempts_in_fallback':statistics.median(r['attempt_count'] for r in fallback),
                    'top_rank_correct_in_fallback':sum(r['top_rank_passed'] for r in fallback),
                    'all_model_scored_sequences_in_fallback':sum(r['candidate_count'] for r in fallback)}
    return results


def draw(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=results['development_localization']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={'width_ratios':[1.5,1]})
    colors=['#2876A1' if 'vision' not in r['recipe'] and 'augmented' not in r['recipe'] else '#138A72' for r in rows]
    axes[0].barh([r['recipe'] for r in rows],[r['accuracy']*100 for r in rows],color=colors)
    axes[0].axvline(6.25,color='#B44E4E',linestyle='--',label='Random 1/16')
    for i,r in enumerate(rows): axes[0].text(r['accuracy']*100+.8,i,f"{r['correct']}/{r['n']}",va='center',fontsize=8)
    axes[0].set_xlim(0,105);axes[0].set_xlabel('Development localization accuracy (%)');axes[0].invert_yaxis();axes[0].legend(fontsize=8)
    for run,label in [('vision_compact_v1','Fixed 128 images'),('augmented_v1','Fresh image every step')]:
        path=ROOT/'improvement'/run/'training.json'
        if path.exists():
            logs=read(path);logs=logs['records'] if isinstance(logs,dict) else logs
            points=list(range(49,len(logs),50))
            offset=1200 if run=='augmented_v1' else 0
            axes[1].plot([offset+logs[i]['step'] for i in points],[sum(r['loss'] for r in logs[i-49:i+1])/50 for i in points],label=label)
    axes[1].set_xlabel('Cumulative visual-adapter updates');axes[1].set_ylabel('Mean answer-token loss (50 updates)');axes[1].legend(fontsize=8)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'localization_diagnosis.png',dpi=180);fig.savefig(folder/'localization_diagnosis.pdf');plt.close(fig)
    completed=ROOT/'improvement/final_v1'
    if not (completed/'summary.json').exists():return
    stages=['initial_validation','sft_validation','rsi1_validation','rsi2_validation']
    labels=['Initial','Joint SFT','RSI 1','RSI 2']
    fig,axes=plt.subplots(1,2,figsize=(12,3.6))
    for task,label,color in [('loc','Localization','#2876A1'),('total','All tasks','#138A72')]:
        rates=[100*read(completed/(n+'.json'))['summary'][task]['correct']/read(completed/(n+'.json'))['summary'][task]['n'] for n in stages]
        axes[0].plot(range(4),rates,'o-',label=label,color=color)
    axes[0].set_xticks(range(4),labels);axes[0].set_title('Development set',fontsize=18);axes[0].set_ylim(-3,105);axes[0].set_ylabel('Correct (%)',fontsize=17)
    axes[0].legend(fontsize=16,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.32),frameon=False)
    for i,(name,label,color) in enumerate([('selected_final_test','Independent','#2876A1'),('selected_weak_tool_final_test','Weak tool','#69A9C9'),('selected_rich_tool_final_test','Box tool','#138A72'),('selected_search_tool_final_test','Tool search','#B58237')]):
        s=read(completed/(name+'.json'))['summary']
        values=[100*s[t]['correct']/s[t]['n'] for t in ('loc','count','exist')]
        axes[1].bar([j+(i-1.5)*.19 for j in range(3)],values,width=.19,label=label,color=color)
    axes[1].set_xticks(range(3),['Localization\n(128)','Counting\n(40)','Existence\n(16)']);axes[1].set_ylim(0,105);axes[1].set_ylabel('Correct (%)',fontsize=17);axes[1].set_title('Sealed final test',fontsize=18)
    axes[1].legend(fontsize=16,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.32),frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=16)
    fig.subplots_adjust(bottom=.36,top=.88,wspace=.30,left=.07,right=.98)
    fig.savefig(folder/'improved_accuracy.png',dpi=200);fig.savefig(folder/'improved_accuracy.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plot',action='store_true');p.add_argument('--audit',action='store_true');args=p.parse_args()
    results=collect()
    if args.audit: results['audit']=audit_final(ROOT/'improvement/final_v1')
    errors=error_breakdown()
    if errors: results['final_localization_errors']=errors
    results['tool_search_costs']=search_costs(ROOT/'improvement/final_v1')
    out=ROOT/'analysis';out.mkdir(exist_ok=True);h.write_json(out/'improvement_summary.json',results)
    lines=['# 定位改进实测记录','','所有方案比较只使用开发验证集；最终测试不参与训练或选择。','',
           '| 方案 | 定位正确 | 格式有效 | 平均 IoU |','|---|---:|---:|---:|']
    for row in results['development_localization']:
        lines.append(f"| {row['recipe']} | {row['correct']}/{row['n']} | {row['valid']}/{row['n']} | {row['mean_iou']:.4f} |")
    lines+=['','训练损失下降和格式正确都不能代替新图定位正确率；固定128图与在线变化图同时改变数据数量和多样性，不把收益单独归因于一种因素。',
            '同一开发集上尝试过多个方案，开发成绩可能乐观；最终测试仅用于报告已选方案。工具反馈的弱提示、像素框证据与独立回答分别统计。']
    complete=ROOT/'improvement/final_v1'
    if (complete/'summary.json').exists():
        lines+=['','## 已选模型的最终测试','','| 条件 | 定位 | 计数 | 存在性 | 总计 | 定位 IoU |','|---|---:|---:|---:|---:|---:|']
        for name,label in [('selected_final_test','独立回答'),('selected_weak_tool_final_test','弱工具'),('selected_rich_tool_final_test','检测框工具'),('selected_search_tool_final_test','有限候选核验工具')]:
            s=read(complete/(name+'.json'))['summary']
            values=[f"{s[t]['correct']}/{s[t]['n']}" for t in ('loc','count','exist','total')]
            lines.append('| '+label+' | '+' | '.join(values)+f" | {s['loc_iou']:.4f} |")
        if errors:
            lo,hi=errors['accuracy_wilson95']
            lines+=['',f"独立定位正确率的 Wilson 95% 区间为 {lo:.1%}–{hi:.1%}，仅描述本合成测试集抽样不确定性，不包含训练种子波动。",
                    f"行正确 {errors['row_correct']}/128，列正确 {errors['column_correct']}/128；预测落在某个实际对象所在格 {errors['selected_any_object']}/128。逐类别、逐格和混淆矩阵见JSON。"]
    (out/'improvement_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if args.plot: draw(results)
    print(json.dumps({'recipes':len(results['development_localization']),'audit':results.get('audit')},ensure_ascii=False))


if __name__=='__main__': main()
