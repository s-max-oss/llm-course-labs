#!/usr/bin/env python3
"""Lab 6: held-out grid grounding with a verifiable two-round RSI loop.

The verifier sees pixels and the question, never the answer key. Ground truth is
used only to build the seed set and to score predictions after inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220),
          "绿": (40, 170, 70), "黄": (230, 190, 40)}
SHAPES = ("圆", "方")
GRID, SIZE = 4, 256
KINDS = ("loc",) * 8 + ("count",) * 4 + ("exist",) * 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_of(box, grid=GRID):
    return (min(grid - 1, int((box[1] + box[3]) / 2 * grid / SIZE)),
            min(grid - 1, int((box[0] + box[2]) / 2 * grid / SIZE)))


def cell_box(cell, grid=GRID):
    r, c = cell
    w = SIZE / grid
    return [int(c * w), int(r * w), int((c + 1) * w), int((r + 1) * w)]


def cell_text(cell):
    return f"第{cell[0]+1}行第{cell[1]+1}列"


def parse_cell(text, grid=GRID):
    patterns = (r"第\s*(\d+)\s*行\s*第\s*(\d+)\s*列",
                r"\br\s*(\d+)\s*c\s*(\d+)\b",
                r"\b(\d+)\s*[,，、-]\s*(\d+)\b")
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            row, col = map(int, match.groups())
            return (row - 1, col - 1) if 1 <= row <= grid and 1 <= col <= grid else None
    return None


def iou(a, b):
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    intersection = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    shared = area(intersection)
    union = area(a) + area(b) - shared
    return shared / union if union else 0.0


def draw_scene(rng):
    """Put nonoverlapping outline figures in separate 4x4 cells."""
    cells = rng.sample(range(16), rng.randint(2, 4))
    objects = []
    for index in cells:
        color, shape = rng.choice(tuple(COLORS)), rng.choice(SHAPES)
        side = rng.randint(34, 48)
        row, col = divmod(index, 4)
        x = col * 64 + rng.randint(6, 58-side)
        y = row * 64 + rng.randint(6, 58-side)
        objects.append({"class": color + shape, "box": [x, y, x+side, y+side]})
    return objects


def render_scene(objects, path):
    image = Image.new("RGB", (SIZE, SIZE), "white")
    pen = ImageDraw.Draw(image)
    for obj in objects:
        painter = pen.ellipse if obj["class"][1] == "圆" else pen.rectangle
        painter(obj["box"], outline=COLORS[obj["class"][0]], width=4)
    image.save(path)


def build_questions(records, prefix, start):
    questions = []
    for i, kind in enumerate(KINDS):
        image_id = f"{prefix}{start+i:02d}"
        objects = records[image_id]
        classes = Counter(o["class"] for o in objects)
        if kind == "loc":
            unique = sorted(c for c, n in classes.items() if n == 1)
            if not unique:
                raise ValueError(f"No uniquely identifiable target: {image_id}")
            target = unique[i % len(unique)]
            box = next(o["box"] for o in objects if o["class"] == target)
            gold = cell_text(cell_of(box))
        elif kind == "count":
            target = sorted(classes)[i % len(classes)]
            box = None
            gold = f"{classes[target]}个"
        else:
            present = i != 14  # two positive, one negative, for all three splits
            options = sorted(classes) if present else sorted(set(a+b for a in COLORS for b in SHAPES) - set(classes))
            target = options[i % len(options)]
            box = None
            gold = "有" if present else "没有"
        name = target[0] + "色" + ("圆形" if target[1] == "圆" else "方形")
        question = (f"图中的{name}位于第几行第几列？请用“第X行第Y列”回答，网格为4×4。"
                    if kind == "loc" else f"图中有几个{name}？" if kind == "count" else f"图中有没有{name}？")
        questions.append({"id": f"{prefix.upper()}{start+i:02d}", "type": kind,
                          "img": image_id, "target": target, "question": question,
                          "gold": gold, "box": box})
    return questions


def make_data(root):
    root = Path(root)
    records = {}
    for split, n, seed, prefix in (("train", 40, 2026, "t"), ("eval", 15, 42, "e")):
        folder = root / "hw6_imgs" / split
        folder.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        for i in range(n):
            image_id = f"{prefix}{i:02d}"
            for _ in range(100):
                objects = draw_scene(rng)
                if any(n == 1 for n in Counter(o["class"] for o in objects).values()):
                    break
            else:
                raise RuntimeError("Could not draw a uniquely localizable scene")
            render_scene(objects, folder / f"{image_id}.png")
            records[image_id] = objects
    questions = {"seed": build_questions(records, "t", 0),
                 "rsi": build_questions(records, "t", 15),
                 "eval": build_questions(records, "e", 0)}
    write_json(root / "hw6_imgs" / "gt.json", records)
    write_json(root / "hw6_imgs" / "questions.json", questions)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / "hw6_imgs").glob("*/*.png"))}
    write_json(root / "hw6_imgs" / "image_hashes.json", hashes)
    print(f"[data] 40 train + 15 held-out eval images; seed/rsi/eval: 15/15/15; root={root}", flush=True)
    return questions


def load_questions(root):
    path = Path(root) / "hw6_imgs" / "questions.json"
    return read_json(path) if path.exists() else make_data(root)


def image_path(root, q):
    return Path(root) / "hw6_imgs" / ("eval" if q["img"].startswith("e") else "train") / (q["img"] + ".png")


def make_seed(root):
    data = [{"img": q["img"], "prompt": q["question"], "answer": q["gold"],
             "source": "train-GT", "type": q["type"]} for q in load_questions(root)["seed"]]
    write_json(Path(root) / "sft_data.json", data)
    print(f"[seed] {len(data)} training-only GT records", flush=True)
    return data


def components(image, color):
    """8-connected exact-color components; shape from corner occupancy."""
    pixels = image.convert("RGB").load()
    reference = COLORS[color]
    remaining = {(x, y) for y in range(SIZE) for x in range(SIZE)
                 if sum((pixels[x, y][k]-reference[k])**2 for k in range(3)) < 9000}
    found = []
    while remaining:
        todo = [remaining.pop()]
        points = []
        while todo:
            x, y = todo.pop()
            points.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x+dx, y+dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        todo.append(neighbor)
        xs, ys = zip(*points)
        box = [min(xs), min(ys), max(xs)+1, max(ys)+1]
        if len(points) < 20:
            continue
        w, h = box[2]-box[0], box[3]-box[1]
        corners = sum(any(abs(x-cx) <= 3 and abs(y-cy) <= 3 for x, y in points)
                      for cx in (box[0], box[2]-1) for cy in (box[1], box[3]-1))
        shape = "方" if corners >= 2 else "圆"
        found.append({"class": color+shape, "box": box, "pixels": len(points), "corners": corners})
    return found


def pixel_answer(image, q):
    matches = [o for o in components(image, q["target"][0]) if o["class"] == q["target"]]
    if q["type"] == "count":
        return f"{len(matches)}个", {"components": len(matches)}
    if q["type"] == "exist":
        return "有" if matches else "没有", {"components": len(matches)}
    if len(matches) != 1:
        return None, {"components": len(matches), "reason": "target not unique"}
    return cell_text(cell_of(matches[0]["box"])), {"box": matches[0]["box"], "components": 1}


def parse_count(text):
    numbers = [int(x) for x in re.findall(r"(?<!\d)\d+(?!\d)", text)]
    for zh, n in (("零", 0), ("一", 1), ("二", 2), ("两", 2), ("三", 3), ("四", 4)):
        if re.search(zh + r"\s*(?:个|只|枚|[。.!！]?$)", text):
            numbers.append(n)
    for en, n in (("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4)):
        if re.search(r"\b" + en + r"\b", text, re.I):
            numbers.append(n)
    return numbers[0] if len(set(numbers)) == 1 and numbers else None


def parse_exist(text):
    stripped = text.strip().lower()
    if re.match(r"^(?:没有|不存在|无|no\b|not\b|none\b)", stripped):
        return "没有"
    if re.match(r"^(?:有|存在|yes\b)", stripped):
        return "有"
    return None


def parsed_answer(q, prediction):
    if q["type"] == "loc":
        clean = prediction.strip()
        clean = re.sub(r"^(?:答案|位置)\s*(?:是|为|[:：])?\s*", "", clean)
        clean = re.sub(r"[。.!！]\s*$", "", clean)
        # A copied instruction such as "请用第2行第4列的问题样式" is not an answer.
        valid = re.fullmatch(r"第\s*\d+\s*行\s*第\s*\d+\s*列|r\s*\d+\s*c\s*\d+|\d+\s*[,，、-]\s*\d+", clean, re.I)
        cell = parse_cell(clean) if valid else None
        return cell_text(cell) if cell is not None else None
    if q["type"] == "count":
        count = parse_count(prediction)
        return f"{count}个" if count is not None else None
    return parse_exist(prediction)


def verify(image, q, prediction):
    expected, evidence = pixel_answer(image, q)
    parsed = parsed_answer(q, prediction)
    passed = expected is not None and parsed == expected
    # The checker supplies pixel-derived evidence, never GT from gt.json.
    hint = ("解析式验证器无法唯一识别目标；请谨慎回答。" if expected is None else
            f"像素检查结果：{expected}。只输出这个结果，不要复述提示或列出步骤。")
    return {"passed": passed, "parsed": parsed, "pixel_result": expected,
            "evidence": evidence, "hint": hint}


def score(q, prediction):
    return parsed_answer(q, prediction) == q["gold"]


def loc_iou(q, prediction):
    cell = parse_cell(prediction)
    return iou(cell_box(cell), q["box"]) if cell is not None and q["box"] else 0.0


def overlay(image, cell, path):
    marked = image.copy()
    painter = ImageDraw.Draw(marked)
    painter.rectangle(cell_box(cell), outline=(255, 0, 255), width=5)
    marked.save(path)
    return marked


def load_model(args, adapter=None, train=False):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    quantized = use_cuda and not args.no_quant
    options = {"local_files_only": args.offline}
    processor = AutoProcessor.from_pretrained(args.model, **options)
    processor.image_processor.size = {"longest_edge": 512}
    if hasattr(processor.image_processor, "do_image_splitting"):
        processor.image_processor.do_image_splitting = False
    kwargs = dict(options, torch_dtype=dtype, attn_implementation="eager")
    if quantized:
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForVision2Seq.from_pretrained(args.model, **kwargs)
    if not quantized:
        model.to(device)
    model.config.use_cache = not train
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train)
    elif train:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
    elif train:
        targets = [name for name, _ in model.named_modules()
                   if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and "vision" not in name]
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                                target_modules=targets, task_type="CAUSAL_LM"))
    stats = None
    if train:
        numerator, denominator = model.get_nb_trainable_parameters()
        stats = {"trainable": numerator, "total": denominator,
                 "percent": 100*numerator/denominator}
        print(f"[model] trainable={numerator}/{denominator} ({stats['percent']:.4f}%) NF4={quantized}", flush=True)
    return processor, model, device, {"device": str(device), "dtype": str(dtype), "nf4": quantized,
                                       "processor_longest_edge": 512, "image_splitting": False,
                                       "parameters": stats}


def build_prompt(processor, question):
    return processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ], add_generation_prompt=True)


def encode_answer(processor, image, question, response):
    import torch
    batch = dict(processor(text=[build_prompt(processor, question)], images=[image], return_tensors="pt"))
    length = batch["input_ids"].shape[1]
    extra_ids = processor.tokenizer(" " + response, add_special_tokens=False)["input_ids"]
    extra_ids.append(processor.tokenizer.eos_token_id)
    extra = torch.tensor([extra_ids], dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat([batch["input_ids"], extra], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(extra)], dim=1)
    labels = batch["input_ids"].clone()
    labels[:, :length] = -100
    batch["labels"] = labels
    return batch


def answer(processor, model, device, image, question, max_new=32):
    import torch
    model.eval()
    with torch.inference_mode():
        batch = processor(text=[build_prompt(processor, question)], images=[image], return_tensors="pt")
        batch = {key: value.to(device) for key, value in dict(batch).items()}
        prefix = batch["input_ids"].shape[1]
        tokens = model.generate(**batch, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                pad_token_id=processor.tokenizer.eos_token_id)
        return processor.tokenizer.decode(tokens[0, prefix:], skip_special_tokens=True).strip()


def train_steps(processor, model, device, args, pool, count, tag, priority=None):
    import torch
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    logs = []
    priority = priority or []
    schedule = priority[:count]
    schedule += [pool[(i * len(pool)) // max(1, count-len(schedule))]
                 for i in range(count-len(schedule))]
    for index, sample in enumerate(schedule):
        image = Image.open(image_path(args.root, sample)).convert("RGB")
        batch = encode_answer(processor, image, sample["prompt"], sample["answer"])
        batch = {key: value.to(device) for key, value in batch.items()}
        model.train()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logs.append({"step": index+1, "loss": float(loss.detach().cpu())})
        print(f"[{tag}] {index+1}/{count} loss={logs[-1]['loss']:.4f}", flush=True)
    return logs


def evaluate(processor, model, device, args, questions, stage, correction=False):
    rows = []
    started = time.perf_counter()
    for q in questions:
        image = Image.open(image_path(args.root, q)).convert("RGB")
        raw = answer(processor, model, device, image, q["question"], args.max_new)
        first = verify(image, q, raw)
        final = raw
        self_check = None
        tool_check = None
        if correction and not first["passed"]:
            cell = parse_cell(raw) if q["type"] == "loc" else None
            if cell is not None:
                folder = Path(args.root) / "results" / "overlays" / stage
                folder.mkdir(parents=True, exist_ok=True)
                marked = overlay(image, cell, folder / f"{q['id']}.png")
                revised = answer(processor, model, device, marked,
                                 q["question"] + "\n紫色框是你刚才选的位置。请自查是否圈中目标，必要时改写。", args.max_new)
                self_check = {"prediction": revised, "verification": verify(image, q, revised)}
                if self_check["verification"]["passed"]:
                    final = revised
            if not verify(image, q, final)["passed"]:
                prompt = q["question"] + "\n验证器反馈：" + first["hint"]
                revised = answer(processor, model, device, image, prompt, args.max_new)
                tool_check = {"prediction": revised, "verification": verify(image, q, revised)}
                final = revised
        row = {"id": q["id"], "type": q["type"], "img": q["img"], "question": q["question"],
               "gold": q["gold"], "raw": raw, "raw_correct": score(q, raw),
               "verification": first, "self_check": self_check, "tool_check": tool_check,
               "final": final, "correct": score(q, final),
               "iou": loc_iou(q, final) if q["type"] == "loc" else None}
        rows.append(row)
        print(f"[{stage} {q['id']}] {int(row['correct'])} {raw!r} -> {final!r} | {q['gold']}", flush=True)
    summary = {kind: {"correct": sum(r["correct"] for r in rows if r["type"] == kind),
                      "n": sum(r["type"] == kind for r in rows)} for kind in ("loc", "count", "exist")}
    summary["total"] = {"correct": sum(r["correct"] for r in rows), "n": len(rows)}
    summary["loc_mean_iou"] = sum(r["iou"] for r in rows if r["iou"] is not None) / max(1, summary["loc"]["n"])
    result = {"stage": stage, "split": "held-out eval" if questions[0]["img"].startswith("e") else "train feedback",
              "correction": correction, "rows": rows, "summary": summary,
              "seconds": time.perf_counter()-started}
    write_json(Path(args.root) / "results" / f"{stage}.json", result)
    print(f"[{stage}] loc={summary['loc']['correct']}/8 count={summary['count']['correct']}/4 "
          f"exist={summary['exist']['correct']}/3 total={summary['total']['correct']}/15 "
          f"IoU={summary['loc_mean_iou']:.4f}", flush=True)
    return result


def run_rsi(args):
    import torch
    root = Path(args.root)
    questions = load_questions(root)
    pool_path = root / "sft_data.json"
    pool = read_json(pool_path) if pool_path.exists() else make_seed(root)
    seed_adapter = root / "results" / "sft_adapter"
    if not seed_adapter.exists():
        raise FileNotFoundError("Run --mode sft before RSI; this loop continues the SFT adapter")
    torch.manual_seed(args.seed)
    processor, model, device, metadata = load_model(args, adapter=seed_adapter, train=True)
    all_memory = []
    rounds = []
    for number in range(1, args.rounds+1):
        direct = evaluate(processor, model, device, args, questions["rsi"], f"rsi{number}_direct")
        corrected = evaluate(processor, model, device, args, questions["rsi"], f"rsi{number}_corrected", correction=True)
        additions = []
        for q, row in zip(questions["rsi"], corrected["rows"]):
            if verify(Image.open(image_path(root, q)).convert("RGB"), q, row["final"])["passed"] and not row["verification"]["passed"]:
                candidate = {"img": q["img"], "prompt": q["question"], "answer": row["final"],
                             "source": f"verified-rsi-{number}", "type": q["type"]}
                if not any(p["img"] == candidate["img"] and p["prompt"] == candidate["prompt"] for p in pool):
                    additions.append(candidate)
            all_memory.append({"round": number, **row})
        pool.extend(additions)
        write_json(pool_path, pool)
        write_json(root / f"sft_data_round{number}.json", pool)
        log = train_steps(processor, model, device, args, pool, args.steps, f"rsi{number}", priority=additions)
        adapter_dir = root / "results" / f"rsi{number}_adapter"
        model.save_pretrained(adapter_dir)
        model.eval()
        held_out = evaluate(processor, model, device, args, questions["eval"], f"rsi{number}_eval")
        rounds.append({"round": number, "pool_size": len(pool), "added": len(additions),
                       "direct_correct": direct["summary"]["total"]["correct"],
                       "corrected_correct": corrected["summary"]["total"]["correct"],
                       "self_rewrites": sum(r["self_check"] is not None for r in corrected["rows"]),
                       "tool_rewrites": sum(r["tool_check"] is not None for r in corrected["rows"]),
                       "heldout_correct": held_out["summary"]["total"]["correct"],
                       "heldout_loc": held_out["summary"]["loc"]["correct"],
                       "training": log})
        write_json(root / "rsi_memory.json", {"model": metadata, "records": all_memory, "rounds": rounds})
        print(f"[rsi{number}] pool={len(pool)} added={len(additions)} heldout={rounds[-1]['heldout_correct']}/15", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("make_data", "make_sft", "infer", "sft", "eval", "rsi"), required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.root)
    if args.mode == "make_data":
        make_data(root)
        return
    questions = load_questions(root)
    if args.mode == "make_sft":
        make_seed(root)
        return
    import torch
    torch.manual_seed(args.seed)
    if args.mode == "rsi":
        run_rsi(args)
        return
    if args.mode == "sft":
        pool = read_json(root / "sft_data.json") if (root / "sft_data.json").exists() else make_seed(root)
        processor, model, device, metadata = load_model(args, train=True)
        logs = train_steps(processor, model, device, args, pool, args.steps, "sft")
        model.save_pretrained(root / "results" / "sft_adapter")
        write_json(root / "results" / "sft_training.json", {"metadata": metadata, "steps": logs})
        return
    processor, model, device, _ = load_model(args)
    baseline = evaluate(processor, model, device, args, questions["eval"], "baseline")
    if args.mode == "infer":
        return
    if (root / "results" / "sft_adapter").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, root / "results" / "sft_adapter").to(device)
        evaluate(processor, model, device, args, questions["eval"], "sft_eval")
        evaluate(processor, model, device, args, questions["eval"], "sft_tool_eval", correction=True)
    else:
        print("[eval] no SFT adapter; only baseline available", flush=True)
    print(f"[baseline] {baseline['summary']['total']['correct']}/15", flush=True)


if __name__ == "__main__":
    main()
