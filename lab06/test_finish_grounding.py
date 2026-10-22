import tempfile
import unittest
from pathlib import Path
from collections import Counter

import hw6_rsi_localize as h
import improve_localization as exp
import finish_grounding as final


class TestJointGrounding(unittest.TestCase):
    def test_joint_distribution_and_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            exp.make_balanced_data(root)
            data=final.make_joint_data(root)
            self.assertEqual(Counter(q['type'] for q in data['train']),{'loc':128,'count':80,'exist':32})
            self.assertEqual(Counter(q['type'] for q in data['validation']),{'loc':128,'count':40,'exist':16})
            self.assertEqual(Counter(q['gold'] for q in data['train'] if q['type']=='exist'),{'有':16,'没有':16})
            counts=[q for q in data['train'] if q['type']=='count']
            self.assertEqual(set(Counter(q['gold'] for q in counts).values()),{16})
            for q in counts[:10]:
                image=exp.load_image(root,q)
                self.assertEqual(h.pixel_answer(image,q)[0],q['gold'])
                answer=final.training_answer(q,True)
                self.assertTrue(final.check(image,q,answer)['passed'])

    def test_repeated_count_is_not_valid(self):
        q={'type':'count','gold':'1个'}
        self.assertIsNone(final.canonical(q,'1. 1. 1.'))
        self.assertEqual(final.canonical(q,'1.'),'1个')
        self.assertIsNone(final.canonical({'type':'exist'},'yes and no'))
        self.assertEqual(final.canonical({'type':'exist'},'No.'),'没有')

    def test_bootstrap_origin_separates_tools_from_raw(self):
        row={'final':'r2c3','raw':'r1c1','tool_check':{'prediction':'r2c3'},'self_check':{'prediction':'r4c4'}}
        self.assertEqual(final.answer_origin(row),'bbox_tool')
        row['candidate_search']={'answer':'r2c3'}
        self.assertEqual(final.answer_origin(row),'constrained_candidate')
        row['candidate_search']=None;row['tool_check']=None;row['self_check']['prediction']='r2c3'
        self.assertEqual(final.answer_origin(row),'self_check')


if __name__=='__main__': unittest.main()
