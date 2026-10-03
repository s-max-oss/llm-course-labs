import unittest,json
import numpy as np
import hw4_rag as m

class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.emb=m.Embedder('char3')

    def test_chunk_boundaries_and_validation(self):
        text='甲乙丙丁戊己庚辛壬癸'
        chunks=m.chunk_docs(6,2,{'d':text})
        self.assertEqual([(c['start'],c['end']) for c in chunks],[(0,6),(4,10)])
        self.assertEqual(''.join([chunks[0]['text'],chunks[1]['text'][2:]]),text)
        for size,overlap in [(0,0),(3,3),(3,-1)]:
            with self.assertRaises(ValueError):m.chunk_docs(size,overlap)
        self.assertEqual(m.chunk_docs(6,2,{'d':''}),[])

    def test_document_metrics_deduplicate_and_cutoff(self):
        r=m.metrics(['x','a','a','b'],['a','b'],2)
        self.assertEqual(r['recall'],.5);self.assertEqual(r['reciprocal_rank'],.5)
        self.assertEqual(m.metrics(['x'],['a'],5)['reciprocal_rank'],0)
        self.assertAlmostEqual(m.mrr_check([2,1,4])['mrr'],7/12)
        self.assertAlmostEqual(m.mrr_check([0,2])['mrr'],.25)
        with self.assertRaises(ValueError):m.mrr_check([-1])

    def test_reference_and_added_data(self):
        m.validate_data()
        self.assertEqual((len(m.CORPUS),len(m.QUESTIONS),len(m.PRE_TRIPLES),len(m.PRE_ENTRIES)),(15,15,40,9))
        self.assertEqual(len(m.ADDED_QUESTIONS),5)
        self.assertEqual({q['type'] for q in m.ADDED_QUESTIONS},{'fact','multi','global'})

    def test_normalization_and_bounded_walk(self):
        a=m.GraphRAG(self.emb);b=m.GraphRAG(self.emb,normalize=True)
        self.assertNotIn('AI协会',a.entities('AI 协会的教师'))
        self.assertIn('AI协会',b.entities('AI 协会的教师'))
        self.assertIn('云山大学',b.entities('云大的校训'))
        hit=a.retrieve('大模型通识课授课教师所在学院负责人',5,'local')
        self.assertTrue(all(row['hop']<=3 for row in hit['walk']))
        self.assertEqual(len({row['triple_id'] for row in hit['walk']}),len(hit['walk']))
        self.assertEqual(a.G.number_of_edges(),40)

    def test_scope_forwarding_and_gold_not_used(self):
        class Stub(m.GraphRAG):
            def __init__(self):self.scopes=[]
            def retrieve(self,q,k,scope):
                self.scopes.append(scope);return {'documents':['doc01'],'units':[]}
        s=Stub();m.evaluate({'Graph':s},m.QUESTIONS,5,'global')
        self.assertEqual(s.scopes,['global']*15)
        vector=m.VectorRAG(self.emb);query='云山大学的校训是什么'
        before=vector.retrieve(query,5)['documents']
        original=m.QUESTIONS;m.QUESTIONS=[]
        try:self.assertEqual(vector.retrieve(query,5)['documents'],before)
        finally:m.QUESTIONS=original

    def test_extraction_json_shape_without_execution(self):
        self.assertEqual(m.parse_triples('```json\n[["甲","位于","乙"]]\n```','doc01'),[('甲','位于','乙','doc01')])
        for raw in ['{"x":1}','[["甲","乙"]]','[["甲",0,"乙"]]','explanation [["甲","乙","丙"]]']:
            with self.assertRaises((ValueError,json.JSONDecodeError)):m.parse_triples(raw,'doc01')

    def test_rrf_and_corrected_knowledge(self):
        v=m.VectorRAG(self.emb,hybrid=True)
        hit=v.retrieve('国家奖学金8000元',5)
        for row in hit['units']:
            self.assertAlmostEqual(row['score'],1/(60+row['dense_rank'])+1/(60+row['sparse_rank']))
        triples,summaries,entries=m.knowledge(True)
        self.assertTrue(any(r=='参与运行管理' for h,r,t,d in triples))
        self.assertFalse(any('参与共建' in txt for txt,ds in summaries))
        self.assertFalse(any('每年 15 人' in txt or '教材《' in txt for title,txt,ds in entries))
        self.assertTrue(any('每年 15 人' in txt for title,txt,ds in m.PRE_ENTRIES))

if __name__=='__main__':unittest.main()
