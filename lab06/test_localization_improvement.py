import tempfile
import unittest
from collections import Counter
from pathlib import Path

import hw6_rsi_localize as h
import improve_localization as improve
import train_augmented_grounding as augmented


class TestBalancedGrounding(unittest.TestCase):
    def test_balanced_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = improve.make_balanced_data(root)
            self.assertEqual({k:len(v) for k,v in splits.items()},
                             {'train':128, 'validation':128, 'final_test':128})
            self.assertEqual(len(h.read_json(root/'improvement/data/pixel_hashes.json')), 384)
            for split, repetitions in [('train',8),('validation',8),('final_test',8)]:
                cells = Counter(tuple(q['cell']) for q in splits[split])
                self.assertEqual(len(cells), 16)
                self.assertEqual(set(cells.values()), {repetitions})
                pairs = Counter((tuple(q['cell']), q['target']) for q in splits[split])
                self.assertEqual(len(pairs), 128)
                self.assertEqual(set(pairs.values()), {1})
            probes = splits['validation'][::8]
            self.assertEqual(len({tuple(q['cell']) for q in probes}), 16)
            self.assertEqual(len({q['target'] for q in probes}), 1)
            for q in splits['train'][::8]:
                got, _ = h.pixel_answer(improve.load_image(root,q),q)
                self.assertEqual(got, q['gold'])

    def test_encoding_preserves_cell(self):
        q = {'cell':[2,1], 'type':'loc', 'gold':'第3行第2列'}
        self.assertEqual(h.parsed_answer(q, improve.response(q, True)), q['gold'])
        self.assertEqual(h.parsed_answer(q, improve.response(q, False)), q['gold'])
        self.assertNotIn('4×4', improve.prompt('红圆', True))

    def test_curriculum_and_full_epoch_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); improve.make_balanced_data(root)
            rows=augmented.make_examples(root,800)
            self.assertEqual(len({q['pixel_sha256'] for q in rows}),800)
            for start in range(0,768,128):
                self.assertEqual(len({(tuple(q['cell']),q['target']) for q in rows[start:start+128]}),128)
            self.assertTrue(all(len(q['objects'])==1 for q in rows[:256]))
            self.assertTrue(all(len(q['objects'])==2 for q in rows[256:768]))
            self.assertTrue(all(2<=len(q['objects'])<=4 for q in rows[768:]))
            for index in (0,255,256,767,768,799):
                q=rows[index]
                self.assertEqual(h.pixel_answer(improve.load_image(root,q),q)[0],q['gold'])


if __name__ == '__main__':
    unittest.main()
