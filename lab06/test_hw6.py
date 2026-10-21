import tempfile
import unittest
from pathlib import Path

from PIL import Image

import hw6_rsi_localize as h


class TestGrounding(unittest.TestCase):
    def test_grid_and_parsing(self):
        self.assertEqual(h.cell_of([64, 128, 104, 168]), (2, 1))
        for text in ("第3行第2列", "r3c2", "3,2"):
            self.assertEqual(h.parse_cell(text), (2, 1))
        self.assertIsNone(h.parse_cell("第5行第2列"))
        self.assertAlmostEqual(h.iou([0, 0, 2, 2], [1, 1, 3, 3]), 1/7)
        loc = {"type": "loc", "gold": "第2行第4列"}
        self.assertEqual(h.parsed_answer(loc, "答案：第2行第4列。"), loc["gold"])
        self.assertIsNone(h.parsed_answer(loc, "请用第2行第4列的问题样式回答。"))
        self.assertEqual(h.loc_iou({**loc, 'box': [192,64,240,112]}, "请用第2行第4列的问题样式回答。"), 0.0)

    def test_splits_and_pixel_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = h.make_data(root)
            self.assertEqual({k: len(v) for k, v in splits.items()},
                             {"seed": 15, "rsi": 15, "eval": 15})
            self.assertTrue(set(q["img"] for q in splits["seed"]).isdisjoint(
                q["img"] for q in splits["rsi"]))
            self.assertTrue(set(q["img"] for q in splits["eval"]).isdisjoint(
                q["img"] for q in splits["seed"] + splits["rsi"]))
            for rows in splits.values():
                for q in rows:
                    with Image.open(h.image_path(root, q)) as image:
                        result, _ = h.pixel_answer(image, q)
                        self.assertEqual(result, q["gold"], q["id"])
                        self.assertTrue(h.verify(image, q, result)["passed"])
            seed = h.make_seed(root)
            h.write_json(root / 'sft_data.json', seed + [seed[0]])
            self.assertEqual(len(h.make_seed(root)), 15)
            self.assertEqual(h.read_json(root / 'sft_seed.json'), seed)

    def test_wrong_shape_does_not_validate(self):
        image = Image.new("RGB", (256, 256), "white")
        from PIL import ImageDraw
        ImageDraw.Draw(image).ellipse([12, 12, 50, 50], outline=h.COLORS["红"], width=4)
        question = {"type": "exist", "target": "红方", "gold": "没有"}
        self.assertEqual(h.pixel_answer(image, question)[0], "没有")
        self.assertFalse(h.verify(image, question, "有")["passed"])

    def test_location_feedback_and_overlay(self):
        image = Image.new("RGB", (256, 256), "white")
        from PIL import ImageDraw
        ImageDraw.Draw(image).rectangle([140, 70, 178, 108], outline=h.COLORS["蓝"], width=4)
        question = {"type": "loc", "target": "蓝方", "gold": "第2行第3列", "box": [140, 70, 179, 109]}
        check = h.verify(image, question, "第1行第1列")
        self.assertFalse(check["passed"])
        self.assertNotIn(question["gold"], check["hint"])
        with tempfile.TemporaryDirectory() as directory:
            marked = h.overlay(image, (0, 0), Path(directory) / "overlay.png")
            self.assertEqual(marked.getpixel((0, 0)), (255, 0, 255))
        self.assertTrue(h.verify(image, question, "第2行第3列")["passed"])
        self.assertTrue(h.verify(image, {**question, 'gold': '第1行第1列'}, "第2行第3列")["passed"])


if __name__ == "__main__":
    unittest.main()
