# -*- coding: utf-8 -*-
# 课程指南提供的虚构语料及预置知识；学生2024311281，Codex辅助。
import argparse,json,re,zlib,math,hashlib,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx

CORPUS = {'doc01': '云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。',
 'doc02': '人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。',
 'doc03': '计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。',
 'doc04': '人工智能学院教师：李文瀚，教授，2019 '
          '年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。',
 'doc05': '计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。',
 'doc06': '《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。',
 'doc07': '《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。',
 'doc08': '选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。',
 'doc09': '奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年10 人；云山一等奖学金 3000 元/年，比例约 5%。',
 'doc10': '科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。',
 'doc11': '学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。',
 'doc12': '校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月15 日放寒假。',
 'doc13': '图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3层，收藏大模型与智能体专题图书。',
 'doc14': '校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。',
 'doc15': '国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。'}

QUESTIONS = [{'id': 'Q01',
  'type': 'multi',
  'q': '《大模型通识课》授课教师所在学院的行政负责人是谁？',
  'evidence': ['doc04', 'doc06', 'doc02']},
 {'id': 'Q02',
  'type': 'multi',
  'q': 'AI 协会的指导教师主讲课程的课程代码是什么？',
  'evidence': ['doc11', 'doc04', 'doc06']},
 {'id': 'Q03', 'type': 'multi', 'q': '认知计算全国重点实验室依托的学院成立于哪一年？', 'evidence': ['doc10', 'doc02']},
 {'id': 'Q04', 'type': 'multi', 'q': '《知识图谱导论》授课教师的研究方向是什么？', 'evidence': ['doc07', 'doc04']},
 {'id': 'Q05', 'type': 'multi', 'q': '机器人战队的指导教师的研究方向是什么？', 'evidence': ['doc11', 'doc04']},
 {'id': 'Q06', 'type': 'fact', 'q': '云山大学的校训是什么？', 'evidence': ['doc01']},
 {'id': 'Q07', 'type': 'fact', 'q': '国家奖学金的金额是多少？', 'evidence': ['doc09']},
 {'id': 'Q08', 'type': 'fact', 'q': '图书馆每天几点到几点开放？', 'evidence': ['doc13']},
 {'id': 'Q09', 'type': 'fact', 'q': '人工智能学院成立于哪一年？', 'evidence': ['doc02']},
 {'id': 'Q10', 'type': 'fact', 'q': '春季学期什么时候开学？', 'evidence': ['doc12']},
 {'id': 'Q11',
  'type': 'global',
  'q': '云山大学有哪些科研平台？分别依托谁建设？',
  'evidence': ['doc10', 'doc02', 'doc03']},
 {'id': 'Q12', 'type': 'global', 'q': '学校的学生奖励体系包含哪些项目？', 'evidence': ['doc09']},
 {'id': 'Q13', 'type': 'global', 'q': '人工智能学院有哪些教师？各自做什么研究方向？', 'evidence': ['doc04']},
 {'id': 'Q14', 'type': 'global', 'q': '本科生选课有哪些限制和先修要求？', 'evidence': ['doc08']},
 {'id': 'Q15',
  'type': 'global',
  'q': '对想深入学习大模型的学生，学校提供哪些课程和课外活动？',
  'evidence': ['doc06', 'doc11']}]

PRE_TRIPLES = [['李文瀚', '任职于', '人工智能学院', 'doc04'],
 ['李文瀚', '职称', '教授', 'doc04'],
 ['李文瀚', '入职年份', '2019年', 'doc04'],
 ['李文瀚', '研究方向', '检索增强生成', 'doc04'],
 ['李文瀚', '主讲', '大模型通识课', 'doc06'],
 ['李文瀚', '指导', 'AI协会', 'doc11'],
 ['赵婉晴', '任职于', '人工智能学院', 'doc04'],
 ['赵婉晴', '研究方向', '知识图谱', 'doc04'],
 ['赵婉晴', '主讲', '知识图谱导论', 'doc07'],
 ['孙浩然', '任职于', '人工智能学院', 'doc04'],
 ['孙浩然', '研究方向', '强化学习', 'doc04'],
 ['孙浩然', '指导', '机器人战队', 'doc11'],
 ['人工智能学院', '院长', '王启明', 'doc02'],
 ['人工智能学院', '成立于', '2018年', 'doc02'],
 ['人工智能学院', '下设系', '机器学习系', 'doc02'],
 ['人工智能学院', '下设系', '智能科学系', 'doc02'],
 ['人工智能学院', '下设系', '认知计算系', 'doc02'],
 ['认知计算全国重点实验室', '依托', '人工智能学院', 'doc10'],
 ['岭南超算中心', '参与共建', '计算机学院', 'doc10'],
 ['计算机学院', '院长', '陈国峰', 'doc03'],
 ['计算机学院', '前身', '计算机系', 'doc03'],
 ['周天宇', '任职于', '计算机学院', 'doc05'],
 ['周天宇', '研究方向', '分布式系统', 'doc05'],
 ['林小雨', '任职于', '计算机学院', 'doc05'],
 ['林小雨', '研究方向', '数据库系统', 'doc05'],
 ['大模型通识课', '课程代码', 'AI2101', 'doc06'],
 ['大模型通识课', '学分', '3学分', 'doc06'],
 ['大模型通识课', '开设学期', '春季学期', 'doc06'],
 ['知识图谱导论', '课程代码', 'AI3305', 'doc07'],
 ['知识图谱导论', '先修课程', '数据结构', 'doc08'],
 ['AI协会', '指导教师', '李文瀚', 'doc11'],
 ['机器人战队', '指导教师', '孙浩然', 'doc11'],
 ['国家奖学金', '金额', '8000元/年', 'doc09'],
 ['校长奖学金', '金额', '20000元/年', 'doc09'],
 ['云山一等奖学金', '金额', '3000元/年', 'doc09'],
 ['云山大学', '校训', '格物致知', 'doc01'],
 ['云山大学', '建校于', '1958年', 'doc01'],
 ['云山大学', '位于', '岭南省云州市', 'doc01'],
 ['图书馆', '藏书量', '380万册', 'doc13'],
 ['春季学期', '开学日期', '3月2日', 'doc12']]

COMMUNITY_SUMMARIES = [['社区C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。',
  ['doc02', 'doc04', 'doc06', 'doc07', 'doc08']],
 ['社区C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；学院与新加坡南洋理工大学有联合培养项目。',
  ['doc02', 'doc03', 'doc10', 'doc15']],
 ['社区C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。',
  ['doc09', 'doc11', 'doc13']]]

PRE_ENTRIES = [['云山大学',
  '云山大学位于岭南省云州市，建于 1958 年，省属重点大学，本科生约 2.1 万人，校训「格物致知」。地铁 3 号线「云大站」距东门 200 米，校巴 5 条线路。校历：春季学期 3 月 2 '
  '日开学。',
  ['doc01', 'doc12', 'doc14']],
 ['人工智能学院',
  '人工智能学院成立于 2018 年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目（每年 15 人）。',
  ['doc02', 'doc10', 'doc15']],
 ['计算机学院', '计算机学院前身是 1985 年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。参与岭南超算中心运行管理。', ['doc03', 'doc10']],
 ['人工智能学院教师',
  '李文瀚：教授，2019 年入职，研究检索增强生成，主讲《大模型通识课》，指导AI 协会。赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。',
  ['doc04', 'doc11']],
 ['计算机学院教师', '周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。', ['doc05']],
 ['大模型通识课',
  '《大模型通识课》（AI2101）：3 学分 32 学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求，教材《人工智能——深度学习大模型智能体》。',
  ['doc06']],
 ['知识图谱导论', '《知识图谱导论》（AI3305）：2 学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，需先修《数据结构》。', ['doc07', 'doc08']],
 ['奖学金体系', '国家奖学金 8000 元/年（约 2%）；校长奖学金 20000 元/年（全校 10 人）；云山一等奖学金 3000 元/年（约 5%）。', ['doc09']],
 ['学生社团与校园生活',
  'AI 协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00-22:00 开放，人工智能分馆在理科楼 B 座。',
  ['doc11', 'doc13']]]

DEFAULT_MODEL = 'BAAI/bge-small-zh-v1.5'
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'
ADDED_QUESTIONS = [
    {'id':'Q16','type':'fact','q':'校内校巴共有几条线路，多久一班？','evidence':['doc14']},
    {'id':'Q17','type':'multi','q':'研究知识图谱的教师所授课程要求先修什么课程？','evidence':['doc04','doc07','doc08']},
    {'id':'Q18','type':'multi','q':'机器人战队指导教师所在学院下设哪些系？','evidence':['doc11','doc04','doc02']},
    {'id':'Q19','type':'global','q':'学校在图书资源和国际交流方面有哪些安排？','evidence':['doc13','doc15']},
    {'id':'Q20','type':'global','q':'两所学院分别开设哪些本科专业或下属系，负责人是谁？','evidence':['doc02','doc03']},
]

def canonical(text):
    return re.sub(r'[\s《》「」“”]', '', text)

def validate_data(corpus=CORPUS, questions=None, triples=PRE_TRIPLES, entries=PRE_ENTRIES):
    qs=questions or (QUESTIONS+ADDED_QUESTIONS)
    assert len({q['id'] for q in qs})==len(qs)
    for q in qs:
        assert q['type'] in ('fact','multi','global')
        assert q['evidence'] and len(set(q['evidence']))==len(q['evidence'])
        assert set(q['evidence'])<=set(corpus)
    for h,r,t,src in triples:
        assert all(isinstance(v,str) and v for v in (h,r,t,src)) and src in corpus
    for title,body,sources in entries:
        assert title and body and sources and set(sources)<=set(corpus)

def knowledge(corrected=False):
    triples=list(PRE_TRIPLES)
    summaries=[(t,list(ds)) for t,ds in COMMUNITY_SUMMARIES]
    entries=[(title,body,list(ds)) for title,body,ds in PRE_ENTRIES]
    if corrected:
        triples=[(h,'参与运行管理' if h=='岭南超算中心' and t=='计算机学院' else r,t,src) for h,r,t,src in triples]
        summaries=[(text.replace('由计算机学院参与共建','由计算机学院参与运行管理'),ds) for text,ds in summaries]
        entries=[(title,body.replace('（每年 15 人）','').replace('，教材《人工智能——深度学习大模型智能体》',''),ds) for title,body,ds in entries]
    return triples,summaries,entries

def chunk_docs(size=120, overlap=20, corpus=CORPUS):
    if not (isinstance(size,int) and isinstance(overlap,int) and size>0 and 0<=overlap<size):
        raise ValueError('chunk size must be positive and 0 <= overlap < size')
    chunks=[]
    for doc_id,text in corpus.items():
        if not text:continue
        for start in range(0,len(text),size-overlap):
            chunks.append({'doc_id':doc_id,'text':text[start:start+size],'start':start,'end':min(start+size,len(text))})
            if start+size>=len(text):break
    return chunks

class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, revision=None, cache_dir=None):
        self.model_name=model_name;self.revision=revision;self.memory={};self.cache_dir=Path(cache_dir) if cache_dir else None
        self.backend='char3-explicit' if model_name=='char3' else 'sentence-transformers'
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)
        if self.backend!='char3-explicit':
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(4)
            # A failed neural load is a real error, never a silent formal fallback.
            self.model=SentenceTransformer(model_name,revision=revision,device='cpu')
            self.revision=getattr(self.model[0].auto_model.config, '_commit_hash', None) or revision
            self.dimension=self.model.get_sentence_embedding_dimension()
        else:self.dimension=512

    def encode(self,texts,query=False):
        texts=[(QUERY_PREFIX+t if query and self.backend!='char3-explicit' and ('bge' in self.model_name.lower()) else t) for t in texts]
        missing=list(dict.fromkeys(t for t in texts if t not in self.memory))
        for text in missing[:]:
            key=hashlib.sha256(json.dumps([self.model_name,self.revision,text],ensure_ascii=False).encode()).hexdigest()
            path=self.cache_dir/(key+'.npy') if self.cache_dir else None
            if path and path.exists():self.memory[text]=np.load(path,allow_pickle=False);missing.remove(text)
        if missing:
            if self.backend!='char3-explicit':
                vectors=self.model.encode(missing,batch_size=32,normalize_embeddings=True,show_progress_bar=False)
            else:
                vectors=np.zeros((len(missing),512),dtype=np.float32)
                for row,t in enumerate(missing):
                    t=canonical(t)
                    for i in range(max(1,len(t)-2)):
                        g=t[i:i+3] if len(t)>=3 else t
                        vectors[row,zlib.crc32(g.encode())%512]+=1
                    vectors[row]/=np.linalg.norm(vectors[row]) or 1
            for t,v in zip(missing,vectors):
                self.memory[t]=np.asarray(v,dtype=np.float32)
                if self.cache_dir:
                    key=hashlib.sha256(json.dumps([self.model_name,self.revision,t],ensure_ascii=False).encode()).hexdigest()
                    np.save(self.cache_dir/(key+'.npy'),self.memory[t],allow_pickle=False)
        return np.stack([self.memory[t] for t in texts]) if texts else np.empty((0,self.dimension),dtype=np.float32)

def ranked_scores(values):
    # Stable ties make sparse collisions and replay deterministic.
    return sorted(range(len(values)),key=lambda i:(-float(values[i]),i))

def dedup(hits,k):
    if k<1:raise ValueError('k must be >= 1')
    docs=[]
    for hit in hits:
        if hit['doc_id'] not in docs:docs.append(hit['doc_id'])
        if len(docs)==k:break
    return docs

def result(docs,units,**extra):
    return {'documents':docs,'units':units,'context':[{'doc_id':d,'text':CORPUS[d]} for d in docs],**extra}

def bigrams(text):
    text=canonical(text)
    return Counter(text[i:i+2] for i in range(len(text)-1))

def lexical_score(a,b):
    av,bv=bigrams(a),bigrams(b)
    dot=sum(n*bv.get(g,0) for g,n in av.items())
    den=math.sqrt(sum(n*n for n in av.values())*sum(n*n for n in bv.values()))
    return dot/den if den else 0.0

class VectorRAG:
    name='Vector RAG'
    def __init__(self,emb,hybrid=False,size=120,overlap=20):
        self.emb=emb;self.hybrid=hybrid;self.chunks=chunk_docs(size,overlap)
        self.M=emb.encode([c['text'] for c in self.chunks])
    def retrieve(self,query,k=5,scope=None):
        scores=self.M@self.emb.encode([query],query=True)[0]
        dense_order=ranked_scores(scores);dense_rank={i:r+1 for r,i in enumerate(dense_order)}
        lexical=[lexical_score(query,c['text']) for c in self.chunks]
        sparse_rank={i:r+1 for r,i in enumerate(ranked_scores(lexical))}
        fused=[1/(60+dense_rank[i])+1/(60+sparse_rank[i]) for i in range(len(scores))]
        values=fused if self.hybrid else scores
        hits=[{**self.chunks[i],'score':float(values[i]),'dense_cosine':float(scores[i]),'bigram_cosine':lexical[i],'dense_rank':dense_rank[i],'sparse_rank':sparse_rank[i]} for i in ranked_scores(values)]
        return result(dedup(hits,k),hits,ranking='RRF k0=60' if self.hybrid else 'cosine')

class GraphRAG:
    name='GraphRAG'
    def __init__(self,emb,corrected=False,normalize=False,hops=3,frontier_limit=6):
        self.emb=emb;self.normalize=normalize;self.hops=hops;self.frontier_limit=frontier_limit
        self.triples,self.summaries,_=knowledge(corrected)
        self.G=nx.MultiDiGraph()
        for i,(h,r,t,src) in enumerate(self.triples):self.G.add_edge(h,t,key=i,rel=r,src=src,triple_id=i)
        simple=nx.Graph();simple.add_nodes_from(self.G.nodes);simple.add_edges_from(self.G.edges())
        self.communities=[sorted(c) for c in nx.community.greedy_modularity_communities(simple)]
        self.triple_M=emb.encode([f'{h} —{r}→ {t}' for h,r,t,src in self.triples])
        self.summ_M=emb.encode([t for t,ds in self.summaries])
    def entities(self,query,topn=2):
        if self.normalize:
            query=canonical(query).replace('云大','云山大学')
            found=[n for n in self.G.nodes if canonical(n) in query]
        else:found=[n for n in self.G.nodes if n in query]
        return sorted(found,key=lambda n:(-len(n),list(self.G.nodes).index(n)))[:topn]
    def retrieve(self,query,k=5,scope='local'):
        if scope not in ('local','global'):raise ValueError('scope must be local or global')
        return self.local(query,k) if scope=='local' else self.global_(query,k)
    def local(self,query,k):
        ents=self.entities(query);frontier=ents[:];visited_nodes=set();seen_edges=set();walk=[]
        for hop in range(1,self.hops+1):
            nxt=[]
            for node in frontier:
                if node in visited_nodes:continue
                visited_nodes.add(node)
                edges=list(self.G.out_edges(node,keys=True,data=True))+list(self.G.in_edges(node,keys=True,data=True))
                for h,t,key,data in edges:
                    if key not in seen_edges:
                        seen_edges.add(key);walk.append({'hop':hop,'from':node,'head':h,'relation':data['rel'],'tail':t,'doc_id':data['src'],'triple_id':key})
                    other=t if h==node else h
                    if other not in visited_nodes and other not in nxt:nxt.append(other)
            nxt.sort(key=lambda n:0 if self.G.out_degree(n)>0 else 1)
            frontier=nxt[:self.frontier_limit]
        if not walk:
            fallback=self.global_(query,k);fallback.update(scope='local',fallback='global: no entity edge matched',entities=ents,walk=[]);return fallback
        qv=self.emb.encode([query],query=True)[0]
        for hit in walk:hit['score']=float(self.triple_M[hit['triple_id']]@qv)
        order=sorted(walk,key=lambda h:(-h['score'],h['triple_id']))
        return result(dedup(order,k),order,scope='local',entities=ents,walk=walk,fallback=None)
    def global_(self,query,k):
        scores=self.summ_M@self.emb.encode([query],query=True)[0];order=ranked_scores(scores)[:2];units=[];hits=[]
        for i in order:
            text,docs=self.summaries[i]
            units.append({'summary_id':i+1,'text':text,'sources':docs,'score':float(scores[i])})
            hits.extend({'doc_id':d,'score':float(scores[i]),'summary_id':i+1} for d in docs)
        return result(dedup(hits,k),units,scope='global',document_order='summary cosine then listed source order')

class WikiRAG:
    name='WikiRAG'
    def __init__(self,emb,corrected=False):
        self.emb=emb;_,_,self.entries=knowledge(corrected)
        self.M=emb.encode([body for title,body,sources in self.entries])
    def retrieve(self,query,k=5,scope=None):
        scores=self.M@self.emb.encode([query],query=True)[0];order=ranked_scores(scores)[:3];units=[];hits=[]
        for i in order:
            title,body,sources=self.entries[i]
            units.append({'entry_id':i+1,'title':title,'body':body,'sources':sources,'score':float(scores[i])})
            hits.extend({'doc_id':d,'score':float(scores[i]),'entry_id':i+1} for d in sources)
        return result(dedup(hits,k),units,document_order='entry cosine then listed source order')

def metrics(documents,evidence,k=5):
    docs=list(dict.fromkeys(documents))[:k];ev=set(evidence)
    if not ev:raise ValueError('evidence must not be empty')
    hit=ev&set(docs);rank=next((i+1 for i,d in enumerate(docs) if d in ev),0)
    return {'recall':len(hit)/len(ev),'reciprocal_rank':1/rank if rank else 0,'first_rank':rank,'full_hit':hit==ev,'matched':sorted(hit),'missed':sorted(ev-set(docs))}

def evaluate(systems,questions,k=5,scope='local'):
    rows=[]
    for name,sys_ in systems.items():
        for q in questions:
            retrieval=sys_.retrieve(q['q'],k,scope if isinstance(sys_,GraphRAG) else None)
            rows.append({'system':name,**q,**metrics(retrieval['documents'],q['evidence'],k),'retrieval':retrieval})
    aggregate=[]
    for name in systems:
        for tp in ('fact','multi','global','all'):
            sub=[r for r in rows if r['system']==name and (tp=='all' or r['type']==tp)]
            aggregate.append({'system':name,'type':tp,'n':len(sub),'recall':float(np.mean([r['recall'] for r in sub])),'mrr':float(np.mean([r['reciprocal_rank'] for r in sub])),'full_hits':sum(r['full_hit'] for r in sub)})
    return {'k':k,'scope':scope,'question_count':len(questions),'aggregate':aggregate,'rows':rows}

def mrr_check(ranks):
    if not ranks or any(r<0 for r in ranks):raise ValueError('ranks must be nonnegative; 0 means no hit')
    terms=[1/r if r else 0 for r in ranks]
    return {'ranks':ranks,'terms':terms,'mrr':sum(terms)/len(terms),'formula':'('+' + '.join(f'1/{r}' if r else '0' for r in ranks)+f') / {len(ranks)}'}

def parse_triples(raw,source,corpus=CORPUS):
    """Validate an extraction export; never execute text as Python."""
    if source not in corpus:raise ValueError('unknown source')
    raw=raw.strip()
    if raw.startswith('```'):
        raw=re.sub(r'^```(?:json)?\s*','',raw);raw=re.sub(r'\s*```$','',raw)
    arr=json.loads(raw)
    if not isinstance(arr,list):raise ValueError('expected a JSON array')
    out=[]
    for row in arr:
        if not isinstance(row,list) or len(row)!=3 or any(not isinstance(v,str) or not v.strip() for v in row):raise ValueError('each triple needs three nonempty strings')
        out.append(tuple(v.strip() for v in row)+(source,))
    return out

EXTRACT_PROMPT_V1='从文档抽取头实体、关系、尾实体，只返回JSON三元数组。'
EXTRACT_PROMPT_V2='只抽取原文明确支持的关系，保留人物和机构全称，不将运行管理写成共建。只返回三字符串数组组成的JSON数组；不补充原文没有的年份、人数或教材。'

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def make_systems(emb,corrected=False,normalize=False):
    return {'Vector RAG':VectorRAG(emb),'GraphRAG':GraphRAG(emb,corrected,normalize),'WikiRAG':WikiRAG(emb,corrected)}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser(description='实验四：CPU检索评测，不依赖生成API')
    ap.add_argument('--mode',choices=['eval','vector','graph','wiki','hybrid','extract'],default='eval')
    ap.add_argument('--q',default='Q01');ap.add_argument('--scope',choices=['local','global'],default='local')
    ap.add_argument('--k',type=int,default=5);ap.add_argument('--emb',default=DEFAULT_MODEL);ap.add_argument('--revision')
    ap.add_argument('--questions',choices=['builtin','all'],default='builtin')
    ap.add_argument('--corrected',action='store_true');ap.add_argument('--normalize-entities',action='store_true')
    ap.add_argument('--cache-dir',default=str(Path(__file__).parent/'.embedding_cache'))
    ap.add_argument('--output');ap.add_argument('--mrr_check')
    args=ap.parse_args();validate_data()
    if args.mrr_check:
        print(json.dumps(mrr_check([int(v) for v in args.mrr_check.split(',')]),ensure_ascii=False,indent=2));return
    if args.mode=='extract':
        out={'mode':'export_reference_only_no_llm_call','prompt_candidates':[EXTRACT_PROMPT_V1,EXTRACT_PROMPT_V2],'triples':knowledge(args.corrected)[0]}
    else:
        if args.k<1:ap.error('--k must be >= 1')
        emb=Embedder(args.emb,args.revision,args.cache_dir)
        systems=make_systems(emb,args.corrected,args.normalize_entities)
        if args.mode=='eval':
            out=evaluate(systems,QUESTIONS if args.questions=='builtin' else QUESTIONS+ADDED_QUESTIONS,args.k,args.scope)
            print('system | type | Recall@'+str(args.k)+' | MRR | full hit')
            for r in out['aggregate']:print(f"{r['system']} | {r['type']} | {r['recall']:.6f} | {r['mrr']:.6f} | {r['full_hits']}/{r['n']}")
        else:
            qs=QUESTIONS+ADDED_QUESTIONS;q=next((q for q in qs if q['id'].upper()==args.q.upper()),None)
            if q is None:ap.error('unknown question ID')
            system={'vector':'Vector RAG','graph':'GraphRAG','wiki':'WikiRAG','hybrid':'Hybrid RRF'}[args.mode]
            sys_=VectorRAG(emb,hybrid=True) if args.mode=='hybrid' else systems[system]
            out={'system':system,**q,'retrieval':sys_.retrieve(q['q'],args.k,args.scope)}
            print(json.dumps(out,ensure_ascii=False,indent=2))
        out['embedding']={'model':emb.model_name,'revision':emb.revision,'backend':emb.backend,'dimension':emb.dimension,'query_prefix':QUERY_PREFIX if 'bge' in emb.model_name.lower() else ''}
    if args.output:save_json(args.output,out)
    elif args.mode=='extract':print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
