"""Check the new generator's isolation, coverage, and difficult target cases."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import hw6_rsi_localize as h
import optimize_grounding_v2 as opt


class TestContinuation(unittest.TestCase):
    def test_balanced_fresh_pixels_and_hard_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excluded = set()
            counts = {'loc': 128, 'count': 40, 'exist': 16}
            train = opt.make_set(root, root/'train', counts, 123, excluded, True)
            first = set(excluded)
            test = opt.make_set(root, root/'test', counts, 124, excluded, False)
            self.assertEqual(len(excluded), 368)
            self.assertFalse(first & {q['pixel_sha256'] for q in test})
            self.assertEqual(len({(tuple(q['cell']), q['target']) for q in train if q['type']=='loc'}), 128)
            self.assertEqual(Counter(q['gold'] for q in train if q['type']=='count'), {f'{i}个': 8 for i in range(5)})
            self.assertEqual(Counter(q['gold'] for q in train if q['type']=='exist'), {'有': 8, '没有': 8})
            for q in train + test:
                self.assertEqual(h.pixel_answer(opt.exp.load_image(root, q), q)[0], q['gold'])
                if q['type']=='loc':
                    self.assertEqual(sum(o['class']==q['target'] for o in q['objects']), 1)
            hard = [q for q in train if q['type']=='loc' and any(o['class'][0]==q['target'][0] and o['class']!=q['target'] for o in q['objects'])]
            self.assertGreaterEqual(len(hard), 64)

    def test_selection_does_not_reward_overall_at_expense_of_localization(self):
        make = lambda loc, total: {'loc': {'correct': loc, 'n': 128}, 'total': {'correct': total, 'n': 184}}
        stages = {'baseline': make(106, 142), 'forgetting': make(100, 150), 'improved': make(106, 160), 'tie': make(106, 160)}
        self.assertEqual(max(stages, key=lambda name: opt.key(stages[name])), 'improved')


if __name__ == '__main__': unittest.main()
