"""Behavior checks: equations, batch weighting, temporal state and data isolation."""
import unittest
import numpy as np
import torch
from torch import nn
from convlstm_bounce import (BASE, EXPERIMENTS, ConvLSTM, ConvLSTMCell,
                             FlattenLSTM, make_sequences, parameter_counts, split_data)


class ConvLSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_counts(self):
        counts = parameter_counts()["code_counts"]
        self.assertEqual(counts["cell"], 38144)
        self.assertEqual(counts["output"], 289)
        self.assertEqual(counts["baseline_total"], 38433)
        self.assertEqual(counts["wide_cell"], 150016)
        self.assertEqual(counts["flatten_total"], 1575936)

    def test_cell_matches_gate_equations(self):
        cell = ConvLSTMCell(1, 1, k=1).double()
        with torch.no_grad():
            cell.conv.weight.zero_()
            cell.conv.bias.copy_(torch.tensor([0.2, -0.3, 0.4, 0.5]))
        x = torch.zeros(2, 1, 3, 3, dtype=torch.float64)
        old_c = torch.full_like(x, 0.7)
        h, c = cell(x, (x, old_c))
        i, f, g, o = cell.conv.bias
        expected_c = f.sigmoid() * old_c + i.sigmoid() * g.tanh()
        torch.testing.assert_close(c, expected_c)
        torch.testing.assert_close(h, o.sigmoid() * expected_c.tanh())

    def test_shapes_temporal_gradient_and_stateless_calls(self):
        torch.manual_seed(42)
        model = ConvLSTM(hid=3, layers=2, k=5)
        x = torch.rand(2, 4, 1, 9, 9, requires_grad=True)
        pred = model(x)
        self.assertEqual(tuple(pred.shape), (2, 9, 9))
        pred.square().mean().backward()
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
        torch.testing.assert_close(model(x.detach()), pred.detach())
        self.assertEqual(tuple(FlattenLSTM()(torch.rand(2, 4, 1, 32, 32)).shape), (2, 32, 32))
        with self.assertRaises(ValueError):
            model(torch.rand(2, 1, 9, 9))

    def test_gradient_accumulation_matches_full_batch_including_tail(self):
        torch.manual_seed(4)
        model = ConvLSTM(hid=2)
        x, y = torch.rand(5, 2, 1, 6, 6), torch.rand(5, 6, 6)
        nn.MSELoss()(model(x), y).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for start in range(0, 5, 2):
            batch = x[start:start+2]
            (nn.MSELoss()(model(batch), y[start:start+2]) * len(batch) / 5).backward()
        for parameter, gradient in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-7)

    def test_data_reproducibility_no_clipping_and_independent_splits(self):
        data = make_sequences(24, T=60)
        np.testing.assert_array_equal(data, make_sequences(24, T=60))
        self.assertEqual(tuple(data.shape), (24, 60, 1, 32, 32))
        self.assertEqual(set(np.unique(data)), {0, 1})
        # Radius-2 disks stay wholly within the raster; first/last pixel rows stay empty.
        self.assertEqual(int(data[..., 0, :].sum() + data[..., -1, :].sum()), 0)
        self.assertEqual(int(data[..., :, 0].sum() + data[..., :, -1].sum()), 0)
        self.assertTrue(np.all(data.sum(axis=(2, 3, 4)) >= 9))
        train, test, val = split_data(20, 10, 10)
        self.assertEqual((len(train), len(test), len(val)), (20, 10, 10))
        self.assertFalse(np.array_equal(train[:10], test))

    def test_ablations_change_exactly_one_variable(self):
        for name, config in EXPERIMENTS.items():
            if name == "baseline":
                continue
            changes = [key for key in vars(BASE) if getattr(BASE, key) != getattr(config, key)]
            self.assertEqual(len(changes), 1, (name, changes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
