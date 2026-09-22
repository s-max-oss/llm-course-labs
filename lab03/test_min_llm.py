"""Behavioral checks for causal language modelling, not training-quality claims."""
from dataclasses import asdict, replace
import math
import unittest

import torch

from min_llm import (Config, CharTokenizer, MiniGPT, build_corpus, experiment_configs,
                     get_batch, parameter_counts)


class LanguageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tokenizer_and_shift_including_last_window(self):
        tok = CharTokenizer('春眠不觉晓')
        ids = torch.tensor(tok.encode('春眠不觉晓'))
        self.assertEqual(tok.decode(ids.tolist()), '春眠不觉晓')
        with self.assertRaises(KeyError): tok.encode('？')
        x, y = get_batch(ids, 3, 1, starts=torch.tensor([1]))
        self.assertEqual(tok.decode(x[0].tolist()), '眠不觉')
        self.assertEqual(tok.decode(y[0].tolist()), '不觉晓')
        # A corpus of T+1 tokens has exactly one valid window.
        x, y = get_batch(ids, 4, 8, torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(torch.equal(y[:, -1], ids[-1].expand(8)))

    def test_causal_outputs_and_gradients_with_or_without_position(self):
        for use_pos in (True, False):
            torch.manual_seed(7)
            model = MiniGPT(17, Config(n_embd=16, n_head=2, n_layer=2, block_size=8, use_pos=use_pos))
            x = torch.tensor([[1,2,3,4,5,6]])
            changed = torch.tensor([[1,2,3,9,10,11]])
            a, b = model(x)[0], model(changed)[0]
            torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
            self.assertFalse(torch.equal(a[:, 3:], b[:, 3:]))
            # Distinct token IDs: future embeddings cannot affect the logit at position 2.
            a[0, 2, 0].backward()
            self.assertEqual(float(model.tok_emb.weight.grad[4:7].abs().sum()), 0.0)
            self.assertGreater(float(model.tok_emb.weight.grad[1:3].abs().sum()), 0.0)

    def test_exact_parameters_and_actual_weight_sharing(self):
        vocab = CharTokenizer(build_corpus()).vocab_size
        self.assertEqual(vocab, 699)
        for config in [*experiment_configs().values(), Config(n_embd=256, n_head=8, n_layer=3)]:
            model = MiniGPT(vocab, config)
            self.assertIs(model.head.weight, model.tok_emb.weight)
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_counts(vocab, config)['exact'])
        counts = parameter_counts(vocab, Config())
        self.assertEqual(counts['approximate'], 499072)
        self.assertEqual(counts['exact'], 502656)
        self.assertLess(counts['approximation_relative_error'], .01)

    def test_initial_loss_and_loss_shape(self):
        torch.manual_seed(42)
        tok = CharTokenizer(build_corpus()); data = torch.tensor(tok.encode(build_corpus()))
        model = MiniGPT(tok.vocab_size)
        x, y = get_batch(data, 128, 2, torch.Generator().manual_seed(42))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 128, 699))
        self.assertLess(abs(loss.item()-math.log(699)), .25)
        with self.assertRaises(ValueError): model(torch.zeros((1,129), dtype=torch.long))

    def test_sampling_determinism_top1_and_repetition_rule(self):
        torch.manual_seed(8)
        model = MiniGPT(5, Config(n_embd=8, n_head=2, n_layer=1, block_size=4))
        start = torch.tensor([[0]])
        a = model.generate(start, 40, generator=torch.Generator().manual_seed(9))
        b = model.generate(start, 40, generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(a, b))
        c = model.generate(start, 20, top_k=1, generator=torch.Generator().manual_seed(1))
        d = model.generate(start, 20, top_k=1, generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(c, d))
        blocked = model.generate(start, 25, no_repeat_ngram=3, generator=torch.Generator().manual_seed(9))[0].tolist()
        triples = [tuple(blocked[i:i+3]) for i in range(len(blocked)-2)]
        self.assertEqual(len(set(triples)), len(triples))
        with self.assertRaises(ValueError): model.generate(start, temperature=0)

    def test_controlled_configurations(self):
        runs = experiment_configs(); base = asdict(replace(Config(), iters=1000))
        expected = {'exp1_lr_high': {'lr'}, 'exp1_lr_low': {'lr'},
                    'exp2_shallow': {'n_layer'}, 'exp2_deep': {'n_layer'},
                    'exp3_small': {'n_embd','n_head'}, 'exp3_large': {'n_embd','n_head'},
                    'exp4_short': {'block_size'}, 'exp5_no_pos': {'use_pos'}}
        for name, changes in expected.items():
            actual = {k for k,v in asdict(runs[name]).items() if v != base[k]}
            self.assertEqual(actual, changes)
            self.assertEqual(runs[name].n_embd//runs[name].n_head, 32)


if __name__ == '__main__':
    unittest.main()
