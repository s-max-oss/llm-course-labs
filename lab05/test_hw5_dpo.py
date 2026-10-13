"""Offline behavioural tests for the lab's easily silent correctness errors."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from hw5_dpo import (QUESTIONS, EXTRA_PAIRS, IMG_SPEC, PRE_PAIRS, scalar_dpo,
                     dpo_loss, sequence_logprob, judge, legacy_judge, make_images,
                     encode_answer)


class CoreTests(unittest.TestCase):
    def test_dpo_hand_computation_and_stability(self):
        r = scalar_dpo(1.2, .8, .9, .6, .1)
        self.assertAlmostEqual(r["margin"], .01)
        self.assertAlmostEqual(r["loss"], .6881596805078623)
        self.assertAlmostEqual(scalar_dpo(1.5, 1., .2, .6, .1)["margin"], .09)
        for m in [-1000., 0., 1000.]:
            expected = F.softplus(torch.tensor(-m, dtype=torch.float64)).item()
            self.assertAlmostEqual(scalar_dpo(m, 0, 0, 0, 1)["loss"], expected)

    def test_dpo_gradient_direction_and_detached_reference(self):
        pw, pl, rw, rl = [torch.tensor(x, requires_grad=True) for x in [-4., -5., -3., -3.]]
        loss, margin = dpo_loss(pw, rw, pl, rl, .1)
        loss.backward()
        self.assertGreater(margin.item(), 0)
        self.assertLess(pw.grad.item(), 0)   # descent increases chosen logp
        self.assertGreater(pl.grad.item(), 0)  # descent decreases rejected logp
        self.assertIsNone(rw.grad)
        self.assertIsNone(rl.grad)

    def test_sequence_sum_causal_shift_and_prompt_mask(self):
        logits = torch.tensor([[[2., 1., 0.], [1., 2., 0.], [0., 1., 2.], [2., 0., 1.]]], requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 1]])
        actual = sequence_logprob(logits, labels).squeeze()
        expected = logits[0, 1].log_softmax(0)[2] + logits[0, 2].log_softmax(0)[1]
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(logits.grad[0, 0].eq(0).all())
        self.assertTrue(logits.grad[0, 3].eq(0).all())
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            sequence_logprob(logits, torch.full_like(labels, -100))

    def test_answer_boundary_and_eos_masks(self):
        class Tokenizer:
            eos_token_id = 9
            def __call__(self, text, **kw):
                return {"input_ids": [7, 8]}
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, *a, **kw):
                return "prompt"
            def __call__(self, *a, **kw):
                return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                        "pixel_values": torch.zeros(1, 1, 3, 2, 2)}
        batch, plen = encode_answer(Processor(), None, "q", "answer")
        self.assertEqual(plen, 3)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 7, 8, 9]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 7, 8, 9]])
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)

    def test_exist_negation_is_not_positive(self):
        q = {"type": "exist", "gold": "有"}
        self.assertTrue(legacy_judge(q, "没有。"))  # demonstrate original false positive
        self.assertFalse(judge(q, "没有。"))
        self.assertFalse(judge(q, "没有，但也有。"))
        self.assertTrue(judge(q, "有。"))
        self.assertTrue(judge(q, "Yes."))
        self.assertTrue(judge(q, "There is a yellow square."))
        self.assertFalse(judge(q, "No."))
        self.assertTrue(judge(dict(q, gold="没有"), "No, it does not exist."))
        self.assertFalse(judge(dict(q, gold="没有"), "Yes and no."))
        negative_question = next(q for q in QUESTIONS if q["id"] == "P12")
        self.assertFalse(judge(negative_question, "The image does not contain any text."))
        self.assertTrue(judge(negative_question, "There is no red circle."))
        self.assertTrue(judge(negative_question, "No."))
        self.assertFalse(judge(negative_question, "There is a red circle, but no blue squares."))
        self.assertFalse(judge(negative_question, "There is a red circle with no blue squares."))
        self.assertTrue(judge(negative_question, "No blue squares, and there is no red circle."))
        self.assertFalse(judge(negative_question, "There is a red circle. There is no red circle."))

    def test_counts_and_shapes_reject_ambiguous_outputs(self):
        q = {"type": "count", "gold": "2"}
        for p in ["2个。", "两个。", "Two circles."]:
            self.assertTrue(judge(q, p), p)
        for p in ["12个。", "十二个。", "二十二个。", "一百零二个。", "2或3个。", "没有。"]:
            self.assertFalse(judge(q, p), p)
        q = {"type": "attr", "gold": "方形"}
        self.assertTrue(judge(q, "square"))
        self.assertFalse(judge(q, "圆形或方形"))
        self.assertFalse(judge(q, "circle"))

    def test_gold_matches_declared_image_objects(self):
        color_map = {"红色": "红", "蓝色": "蓝", "绿色": "绿", "黄色": "黄"}
        for q in QUESTIONS + EXTRA_PAIRS:
            text = q.get("q", q.get("prompt"))
            color = next(v for k, v in color_map.items() if k in text)
            objects = [(c, s, n) for c, s, n in IMG_SPEC[q["img"]] if c == color]
            if q["type"] == "count":
                shape = "圆" if "圆" in text else "方"
                gold = str(sum(n for c, s, n in objects if s == shape))
            elif q["type"] == "attr":
                self.assertEqual(len(objects), 1)
                gold = objects[0][1] + "形"
            else:
                shape = "圆" if "圆" in text else "方" if "方" in text else None
                gold = "有" if any(s == shape or shape is None for c, s, n in objects) else "没有"
            self.assertEqual(q["gold"], gold, q["id"])
        self.assertEqual(len(PRE_PAIRS), 12)
        self.assertEqual(len(EXTRA_PAIRS), 5)

    def test_image_generation_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ma, mb = make_images(a), make_images(b)
            self.assertEqual(ma, mb)
            self.assertEqual(len(ma), 15)
            self.assertEqual(len(ma["img01"]["objects"]), 3)

    def test_cli_training_modes_use_matching_adapter_locations(self):
        from hw5_dpo import main
        with tempfile.TemporaryDirectory() as images:
            make_images(images)
            for mode in ["sft", "dpo"]:
                with patch("sys.argv", ["hw5_dpo.py", "--mode", mode, "--images", images]), patch("hw5_dpo.train") as call:
                    main()
                    args, kind = call.call_args.args
                    self.assertEqual(args.output, "results/" + mode)
                    self.assertEqual(args.sft_adapter, "results/sft/adapter")
                    self.assertEqual(kind, mode)

    def test_semantic_review_separates_answer_from_style_and_extra_claims(self):
        from analyze_results import semantic_grade
        negative = next(q for q in QUESTIONS if q["id"] == "P12")
        positive = next(q for q in QUESTIONS if q["id"] == "P11")
        self.assertTrue(semantic_grade(negative, "没。", False)[0])
        self.assertFalse(semantic_grade(positive, "没。", False)[0])
        attr = next(q for q in QUESTIONS if q["id"] == "P08")
        self.assertTrue(semantic_grade(attr, "The image is a circular shape with a green square in the center.", False)[0])

    def test_eval_loads_fresh_base_for_each_available_stage(self):
        from hw5_dpo import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            make_images(root/"images")
            for stage in ["sft","dpo"]:
                (root/stage).mkdir()
                (root/stage/"adapter_config.json").write_text("{}")
            argv=["hw5_dpo.py","--mode","eval","--images",str(root/"images"),
                  "--sft_adapter",str(root/"sft"),"--dpo_adapter",str(root/"dpo"),"--output",str(root/"eval")]
            with patch("sys.argv",argv),patch("hw5_dpo.load_model",return_value=(None,None,torch.device("cpu"),{})) as loader,patch("hw5_dpo.evaluate",return_value={"rows":[]}) as evaluator:
                main()
                self.assertEqual([call.args[1] for call in loader.call_args_list],[None,str(root/"sft"),str(root/"dpo")])
                self.assertEqual(evaluator.call_count,6)
                self.assertTrue((root/"eval/sft_dpo/evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
