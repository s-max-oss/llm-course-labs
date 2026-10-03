"""Record all course comparisons with one CPU embedding model and fixed inputs."""
from pathlib import Path
import argparse,sys,json,time,platform,hashlib,importlib.metadata as md
import numpy as np
import hw4_rag as m

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--emb',default=m.DEFAULT_MODEL);ap.add_argument('--revision');ap.add_argument('--output',default='results');args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);m.validate_data()
    start=time.perf_counter();emb=m.Embedder(args.emb,args.revision,Path(__file__).parent/'.embedding_cache');load_seconds=time.perf_counter()-start
    if emb.backend!='sentence-transformers':raise ValueError('formal suite requires a real neural embedding model')
    times={};systems={}
    for name,ctor in [('Vector RAG',m.VectorRAG),('GraphRAG',m.GraphRAG),('WikiRAG',m.WikiRAG)]:
        start=time.perf_counter();systems[name]=ctor(emb);times[name]=time.perf_counter()-start
    config={'embedding':{'model':emb.model_name,'revision':emb.revision,'dimension':emb.dimension,'query_prefix':m.QUERY_PREFIX,'device':'cpu','threads':4},'corpus_docs':15,'triples':40,'wiki_entries':9,'discovered_communities':len(systems['GraphRAG'].communities),'preset_summaries':3,'chunk_size':120,'overlap':20,'chunks':len(systems['Vector RAG'].chunks),'entry_top_n':3,'summary_top_n':2,'max_hops':3,'frontier_limit':6,'ranking_ties':'stable insertion order','knowledge':'reference','model_load_seconds':load_seconds,'index_seconds':times,'python':platform.python_version(),'platform':platform.platform(),'versions':{pkg:md.version(pkg) for pkg in ['torch','numpy','networkx','sentence-transformers','transformers']},'source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
    m.save_json(out/'environment.json',config)
    m.save_json(out/'questions.json',m.QUESTIONS+m.ADDED_QUESTIONS)
    evaluations={}
    for scope in ['local','global']:
        for count,qs in [('15',m.QUESTIONS),('20',m.QUESTIONS+m.ADDED_QUESTIONS)]:
            name=f'eval{count}_{scope}';evaluations[name]=m.evaluate(systems,qs,5,scope);m.save_json(out/(name+'.json'),evaluations[name]);print('Completed',name,flush=True)
    scans={}
    for scope in ['local','global']:
        scans[scope]={}
        for k in [1,2,3,5]:scans[scope][str(k)]=m.evaluate(systems,m.QUESTIONS,k,scope)
    m.save_json(out/'k_scan.json',scans)
    hybrid=m.VectorRAG(emb,hybrid=True);m.save_json(out/'hybrid20.json',m.evaluate({'Hybrid RRF':hybrid},m.QUESTIONS+m.ADDED_QUESTIONS,5))
    m.save_json(out/'hybrid_k_scan.json',{str(k):m.evaluate({'Hybrid RRF':hybrid},m.QUESTIONS,k) for k in [1,2,3,5]})
    corrected={'GraphRAG':m.GraphRAG(emb,corrected=True),'WikiRAG':m.WikiRAG(emb,corrected=True)}
    m.save_json(out/'corrected20_local.json',m.evaluate(corrected,m.QUESTIONS+m.ADDED_QUESTIONS,5,'local'))
    m.save_json(out/'corrected20_global.json',m.evaluate(corrected,m.QUESTIONS+m.ADDED_QUESTIONS,5,'global'))
    normalized=m.GraphRAG(emb,normalize=True)
    diagnostic=[]
    for q in ['AI 协会的指导教师主讲课程的课程代码是什么？','云大的校训是什么？']:
        diagnostic.append({'query':q,'strict':systems['GraphRAG'].retrieve(q,5,'local'),'normalized':normalized.retrieve(q,5,'local')})
    m.save_json(out/'entity_normalization.json',diagnostic)
    # Select the three questions before inspecting rank values: fact/multi/global.
    selected=['Q01','Q06','Q11'];manual=[]
    for name in systems:
        rows=[r for r in evaluations['eval20_local']['rows'] if r['system']==name and r['id'] in selected]
        rows.sort(key=lambda r:selected.index(r['id']))
        ranks=[r['first_rank'] for r in rows]
        manual.append({'system':name,'question_ids':selected,**m.mrr_check(ranks),'documents':[r['retrieval']['documents'] for r in rows]})
    m.save_json(out/'mrr_manual.json',manual)
    m.save_json(out/'knowledge_inventory.json',{'corpus':m.CORPUS,'triples':m.PRE_TRIPLES,'preset_summaries':m.COMMUNITY_SUMMARIES,'detected_communities':systems['GraphRAG'].communities,'entries':m.PRE_ENTRIES,'source_audit':[{'kind':'triple and summary','problem':'计算机学院参与共建','source':'doc10','supported':'计算机学院参与运行管理'},{'kind':'wiki','problem':'每年15人','source':'doc15','supported':None},{'kind':'wiki','problem':'教材名称','source':'doc06','supported':None}],'prompts_drafted_only':[m.EXTRACT_PROMPT_V1,m.EXTRACT_PROMPT_V2],'live_extraction_calls':0})
    # Preserve actual neural vectors, allowing independent offline ranking replay.
    texts=list(emb.memory);np.savez_compressed(out/'embedding_vectors.npz',texts=np.asarray(texts,dtype=str),vectors=np.stack([emb.memory[t] for t in texts]))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    m.save_json(out/'manifest.json',manifest)
    for row in evaluations['eval20_local']['aggregate']:print(row)
    print('Saved:',out)

if __name__=='__main__':main()
