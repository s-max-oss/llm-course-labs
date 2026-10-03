"""Offline tests of credential isolation, caching, grounded prompts and review validation."""
import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import deepseek_live as d


class LiveTests(unittest.TestCase):
    def test_missing_key_fails_before_request(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}), patch.object(d.httpx, 'Client') as network:
            with self.assertRaises(ValueError): d.DeepSeekClient()
            network.assert_not_called()

    def test_cache_preserves_payload_without_credentials(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-test-credential'}):
            response=MagicMock(status_code=200)
            response.json.return_value={'choices':[{'message':{'content':'answer'},'finish_reason':'stop'}], 'model':'offline', 'usage':{'total_tokens':4}}
            with patch.object(d.httpx, 'Client') as network:
                network.return_value.__enter__.return_value.post.return_value=response
                client=d.DeepSeekClient(root)
                first=client.chat('test',[{'role':'user','content':'question'}])
                second=client.chat('test',[{'role':'user','content':'question'}])
                self.assertEqual(first,second)
                self.assertEqual(network.return_value.__enter__.return_value.post.call_count,1)
            stored=next((Path(root)/'calls').glob('*.json')).read_text(encoding='utf-8')
            self.assertNotIn('offline-test-credential',stored)
            self.assertNotIn('Authorization',stored)

    def test_http_errors_and_truncation_are_not_success(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'DEEPSEEK_API_KEY':'offline-test-credential'}):
            with patch.object(d.httpx,'Client') as network:
                response=MagicMock(status_code=401)
                response.text='offline-test-credential'
                network.return_value.__enter__.return_value.post.return_value=response
                with self.assertRaisesRegex(RuntimeError,'HTTP 401'):d.DeepSeekClient(root).chat('err',[])
                self.assertEqual(list((Path(root)/'calls').glob('*.json')),[])
                response.status_code=200
                response.json.return_value={'choices':[{'message':{'content':'unfinished'},'finish_reason':'length'}]}
                with self.assertRaisesRegex(ValueError,'truncated'):d.DeepSeekClient(root).chat('cut',[])

    def test_generator_uses_returned_sources_not_gold(self):
        client=MagicMock(model='offline')
        client.chat.return_value={'response':{'content':'answer','model':'offline'}}
        row={'system':'Vector RAG','id':'Q01','q':'question','evidence':['NEVER_SEND_GOLD'],
             'retrieval':{'documents':['doc01'],'context':[{'doc_id':'doc01','text':'actual returned text'}]}}
        result=d.generate(client,row,'generated')
        payload=json.dumps(client.chat.call_args.args[1],ensure_ascii=False)
        self.assertIn('actual returned text',payload)
        self.assertNotIn('NEVER_SEND_GOLD',payload)
        self.assertIsNone(result['human_score'])

    def test_extraction_rejects_nontriple_schema(self):
        client=MagicMock()
        client.chat.return_value={'label':'offline','elapsed_seconds':0,'response':{'content':'{"triples":[["x",0,"y"]]}'}}
        with self.assertRaises(ValueError):d.extract(client,'doc01','v2')

    def test_judge_checks_sources_quotes_and_zero_claims(self):
        client=MagicMock()
        answer={'question':'q','answer':'a','context':[{'doc_id':'doc01','text':'原文支持事实'}], 'human_score':None}
        valid={'score':5,'abstained_or_incomplete':False,'reason':'initial',
               'claims':[{'claim':'fact','supported':True,'sources':['doc01'],'exact_quote':'不存在的引文'}]}
        def review(value):
            client.chat.return_value={'label':'offline','response':{'content':json.dumps(value)}}
            return d.evaluate_claims(client,answer,0)
        self.assertFalse(review(valid)['claims'][0]['quote_found_in_source'])
        for update in [{'exact_quote':1},{'sources':['doc99']},{'sources':[{}]},{'supported':'true'}]:
            bad={**valid,'claims':[{**valid['claims'][0],**update}]}
            with self.assertRaises(ValueError):review(bad)
        self.assertIsNone(review({**valid,'claims':[]})['faithfulness_fraction'])
        with self.assertRaises(ValueError):review({**valid,'abstained_or_incomplete':None})


if __name__=='__main__': unittest.main()
