"""Make figures and source-grounded answer review records from saved retrievals."""
from pathlib import Path
import json, argparse, hashlib, textwrap
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import hw4_rag as m

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'

def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))

def row(system, qid, scope='local', k=5):
    data = load('eval20_'+scope+'.json') if k==5 else load('k_scan.json')[scope][str(k)]
    return next(r for r in data['rows'] if r['system']==system and r['id']==qid)

def write_review():
    # These are one-off Codex answers written against the saved raw contexts.
    # No LLM endpoint is called and no human review is claimed.
    common = {
        'Q06': ('云山大学的校训是“格物致知”[doc01]。', [('校训是格物致知',['doc01'])]),
        'Q01': ('《大模型通识课》由李文瀚授课[doc06]；李文瀚属于人工智能学院[doc04]；该学院院长是王启明教授[doc02]。因此行政负责人为王启明。',
                [('李文瀚授课',['doc06']),('李文瀚属于人工智能学院',['doc04']),('学院院长为王启明',['doc02'])]),
    }
    answers=[]
    for system in ['Vector RAG','GraphRAG','WikiRAG']:
        for qid in ['Q01','Q06','Q11']:
            r=row(system,qid); abstained=False
            if qid=='Q01' and system=='WikiRAG':
                answer='授课教师是李文瀚[doc06]，其所在学院为人工智能学院[doc04]。本次检索未返回学院院长资料，无法确认行政负责人。'
                claims=[('李文瀚授课',['doc06']),('李文瀚属于人工智能学院',['doc04'])]
                abstained=True
            elif qid=='Q11' and system=='Vector RAG':
                answer='认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理[doc10]。'
                claims=[('重点实验室依托人工智能学院',['doc10']),('超算中心由大学与省科技厅共建',['doc10']),('计算机学院参与运行管理',['doc10'])]
            elif qid=='Q11':
                answer='本次检索上下文没有科研平台与建设依托关系的资料，无法据此列出完整答案。'
                claims=[];abstained=True
            else:
                answer,claims=common[qid]
            assert all(set(ds)<=set(r['retrieval']['documents']) for _,ds in claims)
            answers.append({'system':system,'question_id':qid,'question':r['q'],
                'context':r['retrieval']['context'],'answer':answer,
                'author':'Codex one-off grounded synthesis; no generation API',
                'claims':[{'claim':c,'sources':ds,'ai_supported':True} for c,ds in claims],
                'abstained_or_incomplete':abstained,
                'ai_initial_score':5,'human_score':None,'human_review_status':'pending',
                'faithfulness_fraction':1.0 if claims else None,
                'score_note':'AI 初评：无无依据的事实断言。拒答的事实声明分母为0，忠实度比例不定义；5分仅表示未编造，不能当作答题成功。'})
    m.save_json(RESULTS/'answer_review.json',{
        'protocol':'Three preselected questions Q01/Q06/Q11 per paradigm, reference knowledge, Graph Local, K=5; answers use only returned raw documents, not gold evidence or wiki body.',
        'human_review_complete':False,
        'rubric':'5=所有事实断言有上下文支持或谨慎拒答；4=个别表述边界含糊；3=部分无依据；2=多项无依据；1=主要内容无依据。覆盖率另记，不用高忠实度掩盖拒答。',
        'answers':answers})
    cases=[]
    for system,qid,k,reason in [
        ('Vector RAG','Q11',5,'平台原文doc10排第1，但两学院背景doc02/doc03均未入Top-5；保留指南gold，语义排序未覆盖完整标注。'),
        ('Vector RAG','Q01',3,'桥接的学院院长doc02排第4；K=3时被计算机学院教师doc05挤出。K=5恢复，并非Top-5失败。'),
        ('GraphRAG','Q02',5,'AI 协会含空格，节点AI协会不含空格，未匹配实体而回退Global；未返回doc11。'),
        ('GraphRAG','Q11',5,'只匹配云山大学，图中大学只连基本信息（均doc01）；院系与平台子图未接入，局部走边只返回doc01。Global可补齐gold。'),
        ('WikiRAG','Q01',5,'Top-3条目未选中人工智能学院，返回doc06/doc04/doc11/doc05，缺少doc02；条目数与文档K是两层预算。'),
        ('WikiRAG','Q13',5,'学院教师条目未进Top-3，学院条目并不含教师研究方向，doc04遗漏。')]:
        r=row(system,qid,k=k)
        assert r['recall']<1
        cases.append({'system':system,'question_id':qid,'k':k,'documents':r['retrieval']['documents'],'missed':r['missed'],'recall':r['recall'],'reason':reason})
    m.save_json(RESULTS/'failure_cases.json',cases)

def figures():
    f=BASE/'figures';f.mkdir(exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    r=row('Vector RAG','Q01');lines=['Q01  '+r['q'],'实际保存检索结果排版  |  余弦相似度  |  归一化 BGE 512维','']
    for i,d in enumerate(r['retrieval']['documents'],1):
        hit=next(h for h in r['retrieval']['units'] if h['doc_id']==d)
        lines.append(f'{i}. {d}   cosine={hit["score"]:.6f}   字符区间[{hit["start"]},{hit["end"]})')
        lines.append('   '+hit['text'][:38]+('…' if len(hit['text'])>38 else ''));lines.append('')
    fig,ax=plt.subplots(figsize=(10,6.2));ax.axis('off');ax.text(.015,.99,'\n'.join(lines),va='top',fontsize=14,linespacing=1.55);fig.savefig(f/'vector_q01.png',dpi=160,bbox_inches='tight');plt.close(fig)

    inventory=load('knowledge_inventory.json');G=nx.MultiDiGraph()
    for h,rel,t,src in inventory['triples']:G.add_edge(h,t,relation=rel,source=src)
    nodes=['大模型通识课','李文瀚','人工智能学院','王启明','检索增强生成','AI协会']
    sub=G.subgraph(nodes);pos={'大模型通识课':(0,1),'李文瀚':(2,1),'人工智能学院':(4,1),'王启明':(6,1),'检索增强生成':(2,-.4),'AI协会':(4,-.4)}
    fig,ax=plt.subplots(figsize=(11,4.3));nx.draw_networkx_nodes(sub,pos,node_size=2800,node_color='#e5effa',edgecolors='#6784a5',ax=ax)
    nx.draw_networkx_labels(sub,pos,font_family=plt.rcParams['font.family'],font_size=11,ax=ax)
    nx.draw_networkx_edges(sub,pos,arrows=True,node_size=2800,connectionstyle='arc3,rad=0.1',edge_color='#56708f',ax=ax)
    # Label one direction of the reciprocal guidance pair to keep it readable.
    for h,t in [('李文瀚','大模型通识课'),('李文瀚','人工智能学院'),('人工智能学院','王启明'),('李文瀚','检索增强生成'),('李文瀚','AI协会')]:
        data=next(iter(sub[h][t].values()));x=(pos[h][0]+pos[t][0])/2;y=(pos[h][1]+pos[t][1])/2
        ax.text(x,y+.18,data['relation']+'\n'+data['source'],ha='center',fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    ax.set_title(f'Q01 实际图中的子图  |  完整图 {G.number_of_nodes()} 节点 / {G.number_of_edges()} 三元组\n沿入边找到教师，第二跳到学院，第三跳取得院长来源',fontsize=12,pad=16)
    ax.set_xlim(-1,7);ax.set_ylim(-1.1,1.65);ax.axis('off');fig.savefig(f/'graph_q01.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,9));pos2=nx.spring_layout(G,seed=42,k=.8);nx.draw_networkx(G,pos2,ax=ax,node_size=280,font_size=7,font_family=plt.rcParams['font.family'],arrows=False,node_color='#e5effa');ax.axis('off');fig.savefig(f/'graph_full.png',dpi=180,bbox_inches='tight');plt.close(fig)
    entries=inventory['entries'];lines=['预置条目原文与来源（保留参考数据，另做修正对照）','']
    for i in [1,5]:
        title,body,ds=entries[i];lines.append(f'[{title}]  sources: '+', '.join(ds));lines.extend(textwrap.wrap(body,width=34));lines.append('')
    lines.extend(['来源核查：doc15未写“每年15人”；doc06未写教材名称。','条目重写能聚合关系，也会遗漏原文细节或加入无依据内容。'])
    fig,ax=plt.subplots(figsize=(10,5.7));ax.axis('off');ax.text(.02,.99,'\n'.join(lines),va='top',fontsize=14,linespacing=1.7);fig.savefig(f/'wiki_entries.png',dpi=160,bbox_inches='tight');plt.close(fig)
    scan=load('k_scan.json');fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,tp,label in zip(axes,['fact','multi','global'],['事实','多跳','全局']):
        for name,scope in [('Vector RAG','local'),('GraphRAG','local'),('GraphRAG','global'),('WikiRAG','local')]:
            ys=[next(a['recall'] for a in scan[scope][str(k)]['aggregate'] if a['system']==name and a['type']==tp) for k in [1,2,3,5]]
            ax.plot([1,2,3,5],ys,'o-',label=name+(' '+scope if name=='GraphRAG' else ''))
        ax.set_title(label+'（内置5题）');ax.set_xticks([1,2,3,5]);ax.set_xlabel('文档 K');ax.set_ylim(0,1.05);ax.grid(alpha=.2)
    axes[0].set_ylabel('平均 Recall@K');axes[1].legend(fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(f/'k_scan.png',dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    if not args.verify_only:write_review();figures()
    # Independent arithmetic here does not call the implementation's metrics().
    files=['eval15_local.json','eval15_global.json','eval20_local.json','eval20_global.json','hybrid20.json','corrected20_local.json','corrected20_global.json']
    rows=0
    for name in files:
        data=load(name)
        for r in data['rows']:
            docs=list(dict.fromkeys(r['retrieval']['documents']))[:data['k']];gold=set(r['evidence'])
            rr=next((1/(i+1) for i,d in enumerate(docs) if d in gold),0)
            assert abs(r['recall']-len(set(docs)&gold)/len(gold))<1e-12
            assert abs(rr-r['reciprocal_rank'])<1e-12;rows+=1
        for a in data['aggregate']:
            rs=[r for r in data['rows'] if r['system']==a['system'] and (a['type']=='all' or r['type']==a['type'])]
            assert a['n']==len(rs)
            assert abs(a['recall']-sum(r['recall'] for r in rs)/len(rs))<1e-12
            assert abs(a['mrr']-sum(r['reciprocal_rank'] for r in rs)/len(rs))<1e-12
    env=load('environment.json');assert env['source_sha256']==hashlib.sha256((BASE/'hw4_rag.py').read_bytes()).hexdigest()
    npz=np.load(RESULTS/'embedding_vectors.npz',allow_pickle=False)
    assert npz['vectors'].shape[1]==512 and np.isfinite(npz['vectors']).all()
    assert np.allclose(np.linalg.norm(npz['vectors'],axis=1),1,atol=1e-5)
    m.save_json(RESULTS/'analysis_verification.json',{'arithmetic_rows_checked':rows,'source_hash_match':True,'unit_vectors':len(npz['texts']),'human_review_complete':False})
    m.save_json(RESULTS/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RESULTS.iterdir() if p.is_file() and p.name!='manifest.json'})
    print('Verified',rows,'rows;',len(npz['texts']),'actual unit vectors; human scores pending.')

if __name__=='__main__':main()
