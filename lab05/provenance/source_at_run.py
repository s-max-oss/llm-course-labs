#!/usr/bin/env python3
"""Lab 05: reproducible SmolVLM QLoRA SFT and handwritten DPO.

The 15 images/questions and 12 original preference pairs follow the course guide.
Only answer tokens (including EOS) contribute to SFT and sequence log probability.
DPO's frozen reference is the same NF4 base with the adapter disabled, as required
by this guide; it is NOT the more usual frozen SFT reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220),
          "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],
    "img12": [("红", "方", 1), ("蓝", "方", 1)],
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]
SFT_DATA = [{**q, "answer": q["gold"] + ("个。" if q["type"] == "count" else "。")}
            for q in QUESTIONS]

# Original guide pairs. Keep their wording so the original-vs-added comparison
# changes only the five explicitly documented new examples.
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？", "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？", "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？", "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？", "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？", "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？", "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？", "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？", "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？", "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？", "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？", "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？", "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]
EXTRA_PAIRS = [
    {"id": "A01", "type": "count", "img": "img04", "prompt": "图中有几个红色方形？", "gold": "3", "chosen": "3个。", "rejected": "4个。", "failure_mode": "计数加一"},
    {"id": "A02", "type": "count", "img": "img07", "prompt": "图中有几个蓝色方形？", "gold": "2", "chosen": "2个。", "rejected": "1个。", "failure_mode": "颜色对象混淆导致计数错误"},
    {"id": "A03", "type": "attr", "img": "img13", "prompt": "图中绿色的图形是什么形状？", "gold": "圆形", "chosen": "圆形。", "rejected": "方形。", "failure_mode": "形状幻觉"},
    {"id": "A04", "type": "exist", "img": "img14", "prompt": "图里有没有红色圆形？", "gold": "没有", "chosen": "没有。", "rejected": "有，一个红色圆形。", "failure_mode": "把红色方形误认成圆形，硬答有"},
    {"id": "A05", "type": "count", "img": "img15", "prompt": "图中有几个绿色圆形？", "gold": "2", "chosen": "2个。", "rejected": "2个。旁边还有一个红色方形，图中物体排列得非常整齐。", "failure_mode": "数量正确但冗长补充虚构物体及排列"},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_images(root):
    """Reproduce the guide's random coordinates exactly; retain shape positions."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(42)
    manifest = {}
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(im)
        objects = []
        for color, shape, n in spec:
            for _ in range(n):
                x, y, s = rnd.randint(10, 200), rnd.randint(10, 200), rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                fn = draw.ellipse if shape == "圆" else draw.rectangle
                fn(box, outline=COLORS[color], width=4)
                objects.append({"color": color, "shape": shape, "box": box})
        path = root / (img_id + ".png")
        im.save(path)
        manifest[img_id] = {"objects": objects, "sha256": sha256(path)}
    dump(root / "image_manifest.json", manifest)
    dump(root / "questions.json", QUESTIONS)
    dump(root / "sft_data.json", SFT_DATA)
    dump(root / "preference_pairs.json", {"original": PRE_PAIRS, "added": EXTRA_PAIRS})
    print(f"[data] {len(manifest)} guide images; {len(QUESTIONS)} questions; 12+5 preferences", flush=True)
    return manifest


def load_image(root, img_id):
    with Image.open(Path(root) / (img_id + ".png")) as im:
        return im.convert("RGB")


def legacy_judge(q, pred):
    """Original guide matcher, retained ONLY as a named secondary diagnostic."""
    p, g = pred.replace(" ", ""), q["gold"]
    return ("没有" in p and "有" in p) if g == "没有" else g in p


def judge(q, pred):
    """Loose answer matching with disjoint yes/no, shape and count alternatives.

    This is task-answer accuracy, not a claim-by-claim faithfulness judgement.
    English equivalents are accepted because the official model is English.
    Ambiguous/contradictory alternatives are rejected rather than substring-hit.
    """
    p = re.sub(r"\s+", "", pred.lower())
    if q["type"] == "exist":
        question = q.get("q", q.get("prompt", ""))
        # 'The image does not contain any text' is unrelated to 'red circle?'.
        # Accept an explicit binary answer, or require the queried object in a
        # descriptive sentence. Bare not/no in an unrelated sentence is not a
        # correct negative answer.
        explicit = bool(re.match(r"\s*(?:yes|no)\b", pred.lower()) or
                        re.match(r"\s*(?:有|没有|不存在|无)(?:[。.!！,，\s]|$)", pred))
        if question and not explicit:
            color_terms = next(([cn, en] for cn, en in [("红", "red"), ("蓝", "blue"), ("绿", "green"), ("黄", "yellow")]
                                if cn in question), [])
            shape_terms = ["圆", "circle", "round"] if "圆" in question else ["方", "square", "rectangle"] if "方" in question else []
            if not any(t in p for t in color_terms) or (shape_terms and not any(t in p for t in shape_terms)):
                return False
        negative = bool(re.search(r"没有|不存在|未见|未看到|未发现|无|\bno\b|\bnot\b|\bnone\b", pred.lower()))
        # Remove entire negative tokens before checking the positive Chinese 有.
        residue = re.sub(r"没有|不存在|未见|未看到|未发现|无", "", p)
        english_affirmative = bool(re.search(r"\b(?:there|it)\s+(?:is|are)\b|\b(?:contains|has)\b", pred.lower()))
        positive = "有" in residue or bool(re.search(r"\byes\b", pred.lower())) or (english_affirmative and not negative)
        if negative and positive:
            return False
        return negative if q["gold"] == "没有" else positive and not negative
    if q["type"] == "attr":
        square = bool(re.search(r"方形|正方|矩形|square|rectangle", p))
        circle = bool(re.search(r"圆|circle|circular|round", p))
        return (square and not circle) if q["gold"] == "方形" else (circle and not square)
    numbers = {int(x) for x in re.findall(r"(?<!\d)\d+(?!\d)", p)}
    cn = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
          "七": 7, "八": 8, "九": 9, "十": 10}
    def chinese_number(token):
        if len(token) == 1:
            return cn.get(token)
        m = re.fullmatch(r"([一二两三四五六七八九])?十([一二两三四五六七八九])?", token)
        return (cn[m[1]] if m[1] else 1) * 10 + (cn[m[2]] if m[2] else 0) if m else None
    tokens = re.findall(r"([零一二两三四五六七八九十百千]+)(?:个|只|枚|种)", p)
    if re.fullmatch(r"[零一二两三四五六七八九十百千]+[。.!！]?", p):
        tokens.append(p.rstrip("。.!！"))
    numbers.update(n for token in tokens if (n := chinese_number(token)) is not None)
    en = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    numbers.update(en[x] for x in re.findall(r"\b(zero|one|two|three|four|five)\b", pred.lower()))
    return numbers == {int(q["gold"])}


def scalar_dpo(pw, rw, pl, rl, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = beta * ((pw - rw) - (pl - rl))
    # softplus(-margin), including stable branches at very large |margin|.
    loss = max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))
    return {"policy_chosen": pw, "reference_chosen": rw, "policy_rejected": pl,
            "reference_rejected": rl, "beta": beta, "margin": margin, "loss": loss}


def dpo_loss(pol_w, ref_w, pol_l, ref_l, beta):
    margin = beta * ((pol_w - ref_w.detach()) - (pol_l - ref_l.detach()))
    return -F.logsigmoid(margin), margin


def sequence_logprob(logits, labels):
    """Causal shift; sum (not mean) over answer and EOS, excluding prompt/pad."""
    target = labels[:, 1:]
    mask = target.ne(-100)
    if not mask.any():
        raise ValueError("No supervised answer tokens")
    lp = logits[:, :-1].float().log_softmax(-1)
    selected = lp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1)


def build_prompt(processor, question):
    return processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ], add_generation_prompt=True)


def encode_answer(processor, image, question, response):
    """Tokenise answer separately, ensuring the multimodal prompt is exact prefix.

    Re-tokenising prompt+answer together can change the final BPE token at the
    boundary. Appending explicitly avoids an off-by-one prompt_len mask.
    """
    batch = dict(processor(text=[build_prompt(processor, question)], images=[image], return_tensors="pt"))
    plen = batch["input_ids"].shape[1]
    response_ids = processor.tokenizer(" " + response, add_special_tokens=False)["input_ids"]
    eos = processor.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define EOS")
    response_ids = response_ids + [eos]
    extra = torch.tensor([response_ids], dtype=batch["input_ids"].dtype)
    batch["input_ids"] = torch.cat([batch["input_ids"], extra], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(extra)], dim=1)
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][:, :plen] = -100
    assert batch["labels"][:, :plen].eq(-100).all()
    assert batch["labels"][:, plen:].ne(-100).all()
    return batch, plen


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def parameter_stats(model):
    # PEFT compensates for Params4bit packed storage in get_nb_trainable_parameters.
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if names:
        assert all("lora_" in n for n in names), "A base weight is unexpectedly trainable"
        assert not any("vision" in n for n in names), "Vision adapter unexpectedly trainable"
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percent": 100 * trainable / total, "trainable_tensors": names}


def load_model(args, adapter=None, train=False, full_precision=False):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    source = args.model
    kwargs = {"local_files_only": args.offline}
    if args.revision:
        kwargs["revision"] = args.revision
    processor = AutoProcessor.from_pretrained(source, **kwargs)
    # Native processor otherwise may upsample a 256px synthetic image to 2048px
    # and split it into many crops. Fixed one 512px view for every stage.
    processor.image_processor.size = {"longest_edge": 512}
    if hasattr(processor.image_processor, "do_image_splitting"):
        processor.image_processor.do_image_splitting = False
    cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if cuda else "cpu")
    quantized = cuda and not full_precision and not args.no_quant
    if not cuda and args.require_nf4:
        raise RuntimeError("Formal NF4 run requires NVIDIA CUDA; no silent CPU fallback")
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    model_kwargs = dict(kwargs, torch_dtype=dtype, attn_implementation="eager")
    if quantized:
        model_kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForVision2Seq.from_pretrained(source, **model_kwargs)
    if not quantized:
        model.to(device)
    model.config.use_cache = not train
    # Use the same frozen nonquantized FP32 embeddings/norms for the NF4
    # baseline and both training stages, rather than changing precision at SFT.
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train)
    if train:
        if not quantized:
            for p in model.parameters():
                p.requires_grad_(False)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
    elif train:
        targets = [n for n, m in model.named_modules()
                   if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
                   and "vision" not in n]
        if not targets:
            raise RuntimeError("No language attention targets found")
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    if train:
        model.print_trainable_parameters()
    metadata = {"device": str(device), "nf4": quantized, "double_quant": quantized,
                "compute_dtype": str(dtype), "processor_longest_edge": 512,
                "image_splitting": False, "reference": "adapter-disabled base",
                "rank": args.rank, "alpha": 2 * args.rank, "lora_dropout": 0.05,
                "parameters": parameter_stats(model) if train else None}
    return processor, model, device, metadata


@torch.inference_mode()
def answer(processor, model, device, image, question, max_new=32):
    model.eval()
    batch = processor(text=[build_prompt(processor, question)], images=[image], return_tensors="pt")
    batch = to_device(dict(batch), device)
    plen = batch["input_ids"].shape[1]
    out = model.generate(**batch, max_new_tokens=max_new, do_sample=False, use_cache=True,
                         pad_token_id=processor.tokenizer.eos_token_id)
    return processor.tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()


def evaluate(processor, model, device, args, stage, questions=QUESTIONS):
    start = time.perf_counter()
    rows = []
    for q in questions:
        tick = time.perf_counter()
        question = q.get("q", q.get("prompt"))
        pred = answer(processor, model, device, load_image(args.images, q["img"]), question, args.max_new)
        row = {"id": q["id"], "type": q["type"], "img": q["img"], "question": question,
               "gold": q["gold"], "prediction": pred, "correct": bool(judge(q, pred)),
               "legacy_correct": bool(legacy_judge(q, pred)), "seconds": time.perf_counter() - tick}
        rows.append(row)
        print(f"[{stage} {q['id']}] {'OK' if row['correct'] else 'FAIL'} {pred!r} gold={q['gold']}", flush=True)
    metrics = {}
    for tp in ["count", "attr", "exist", "overall"]:
        subset = rows if tp == "overall" else [r for r in rows if r["type"] == tp]
        if subset:
            metrics[tp] = {"correct": sum(r["correct"] for r in subset), "n": len(subset),
                           "accuracy": sum(r["correct"] for r in subset) / len(subset),
                           "legacy_correct": sum(r["legacy_correct"] for r in subset)}
    return {"stage": stage, "rows": rows, "metrics": metrics, "seconds": time.perf_counter() - start,
            "max_new_tokens": args.max_new, "decoding": "greedy", "seed": args.seed,
            "evaluation_scope": "training-image resubstitution, NOT held-out generalization"}


def preference_diagnostic(model, pairs):
    """Dropout-free exact full-pool loss, separated from noisy training-step loss."""
    model.eval()
    rows = []
    with torch.no_grad():
        for p, bw, bl, rw, rl in pairs:
            pw = sequence_logprob(model(**{k: v for k, v in bw.items() if k != "labels"}).logits, bw["labels"]).squeeze()
            pl = sequence_logprob(model(**{k: v for k, v in bl.items() if k != "labels"}).logits, bl["labels"]).squeeze()
            rows.append({"id": p["id"], "policy_chosen": float(pw), "policy_rejected": float(pl),
                         "reference_chosen": float(rw), "reference_rejected": float(rl),
                         "chosen_tokens": int(bw["labels"].ne(-100).sum()),
                         "rejected_tokens": int(bl["labels"].ne(-100).sum())})
    return rows


def train(args, kind):
    from peft import PeftModel
    adapter = args.sft_adapter if kind == "dpo" else None
    if adapter and not (Path(adapter) / "adapter_config.json").exists():
        raise FileNotFoundError("DPO requires the completed SFT adapter")
    torch.manual_seed(args.seed)
    processor, model, device, metadata = load_model(args, adapter=adapter, train=True)
    steps = args.steps if args.steps is not None else (60 if kind == "sft" else 40)
    lr = args.lr if args.lr is not None else (1e-4 if kind == "sft" else 5e-5)
    if steps <= 0:
        raise ValueError("steps must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {out}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if metadata["nf4"]:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW32bit([p for p in model.parameters() if p.requires_grad], lr=lr)
        metadata["optimizer"] = "bitsandbytes.PagedAdamW32bit"
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        metadata["optimizer"] = "torch.optim.AdamW (unquantized fallback)"
    start = time.perf_counter()
    logs, mask_audit, cached = [], [], []
    if kind == "sft":
        for s in SFT_DATA:
            batch, plen = encode_answer(processor, load_image(args.images, s["img"]), s["q"], s["answer"])
            mask_audit.append({"id": s["id"], "prompt_tokens": plen,
                               "answer_tokens_including_eos": int(batch["labels"].ne(-100).sum()),
                               "prompt_mask_all_ignored": bool(batch["labels"][:, :plen].eq(-100).all())})
            cached.append(to_device(batch, device))
        model.train()
    else:
        assert isinstance(model, PeftModel)
        pool = [dict(p, id=f"R{i+1:02d}") for i, p in enumerate(PRE_PAIRS)]
        if args.pairs == "augmented":
            pool += EXTRA_PAIRS
        # Frozen base has no stochastic dropout. Cache only its detached scalar
        # sequence probabilities, not policy activations or model parameters.
        model.eval()
        for p in pool:
            image = load_image(args.images, p["img"])
            bw, plenw = encode_answer(processor, image, p["prompt"], p["chosen"])
            bl, plenl = encode_answer(processor, image, p["prompt"], p["rejected"])
            assert plenw == plenl
            bw, bl = to_device(bw, device), to_device(bl, device)
            with torch.no_grad(), model.disable_adapter():
                rw = sequence_logprob(model(**{k: v for k, v in bw.items() if k != "labels"}).logits, bw["labels"]).squeeze().detach()
                rl = sequence_logprob(model(**{k: v for k, v in bl.items() if k != "labels"}).logits, bl["labels"]).squeeze().detach()
            assert not rw.requires_grad and not rl.requires_grad
            mask_audit.append({"id": p["id"], "prompt_tokens": plenw,
                               "chosen_tokens": int(bw["labels"].ne(-100).sum()),
                               "rejected_tokens": int(bl["labels"].ne(-100).sum()),
                               "reference_requires_grad": False})
            cached.append((p, bw, bl, rw, rl))
        dump(out / "preferences_used.json", pool)
        dump(out / "preference_before.json", preference_diagnostic(model, cached))
        # Keep dropout disabled for both policy/reference; eval() does not turn
        # off gradients. This makes log ratios deterministic and comparable.
        model.eval()
    dump(out / "mask_audit.json", mask_audit)
    order, epoch = [], 0
    for step in range(steps):
        if not order:
            order = list(range(len(cached)))
            random.Random(args.seed + epoch).shuffle(order)
            epoch += 1
        idx = order.pop()
        tick = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if kind == "sft":
            loss = model(**cached[idx]).loss
            row = {"step": step + 1, "epoch": epoch, "sample_id": SFT_DATA[idx]["id"]}
        else:
            p, bw, bl, rw, rl = cached[idx]
            pw = sequence_logprob(model(**{k: v for k, v in bw.items() if k != "labels"}).logits, bw["labels"]).squeeze()
            pl = sequence_logprob(model(**{k: v for k, v in bl.items() if k != "labels"}).logits, bl["labels"]).squeeze()
            loss, margin = dpo_loss(pw, rw, pl, rl, args.beta)
            row = {"step": step + 1, "epoch": epoch, "sample_id": p["id"], "margin": float(margin.detach()),
                   "policy_chosen": float(pw.detach()), "policy_rejected": float(pl.detach()),
                   "reference_chosen": float(rw), "reference_rejected": float(rl)}
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {kind} loss at step {step+1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        row.update(loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.perf_counter() - tick)
        logs.append(row)
        print(f"[{kind}] {step+1}/{steps} loss={row['loss']:.6f}" +
              (f" margin={row['margin']:.6f}" if kind == "dpo" else ""), flush=True)
        dump(out / "training.json", logs)
    # Always save AFTER exact requested steps, including non-epoch-aligned counts.
    model.save_pretrained(out / "adapter", safe_serialization=True)
    processor.save_pretrained(out / "adapter")
    if kind == "dpo":
        dump(out / "preference_after.json", preference_diagnostic(model, cached))
    train_seconds = time.perf_counter() - start
    result = evaluate(processor, model, device, args, out.name)
    dump(out / "evaluation.json", result)
    dump(out / "added_evaluation.json", evaluate(processor, model, device, args, out.name + "_added", EXTRA_PAIRS))
    metadata.update(kind=kind, steps=steps, learning_rate=lr, beta=args.beta if kind == "dpo" else None,
                    pairs=args.pairs if kind == "dpo" else None, seed=args.seed, batch_size=1,
                    train_seconds_including_preprocessing_and_diagnostics=train_seconds,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                    source_sha256=sha256(__file__), parent_sft_adapter_sha256=
                    sha256(Path(adapter) / "adapter_model.safetensors") if adapter else None)
    dump(out / "run.json", metadata)
    return result


def environment():
    packages = {}
    for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "pillow", "huggingface_hub", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(), "source_sha256": sha256(__file__),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": platform.node()}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval", "environment"], default="infer")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no_quant", action="store_true", help="Explicit unquantized LoRA fallback")
    ap.add_argument("--require_nf4", action="store_true")
    ap.add_argument("--images", default="data")
    ap.add_argument("--output", help="Default: results/sft, results/dpo or results/baseline by mode")
    ap.add_argument("--sft_adapter", default="results/sft/adapter")
    ap.add_argument("--adapter", help="Evaluate a single adapter on a fresh base")
    ap.add_argument("--stage", default="baseline")
    ap.add_argument("--full_precision", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--pairs", choices=["original", "augmented"], default="augmented")
    ap.add_argument("--dpo_check", help="policy_chosen,ref_chosen,policy_rejected,ref_rejected,beta")
    args = ap.parse_args()
    if args.output is None:
        args.output = "results/" + (args.mode if args.mode in ["sft", "dpo"] else "baseline")
    if args.dpo_check:
        values = [float(x) for x in args.dpo_check.split(",")]
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError("Five finite numbers are required")
        print(json.dumps(scalar_dpo(*values), ensure_ascii=False, indent=2))
        return
    if args.beta <= 0 or not math.isfinite(args.beta) or args.rank <= 0 or args.threads <= 0:
        ap.error("beta/rank/threads must be positive")
    if args.no_quant and args.require_nf4:
        ap.error("--no_quant conflicts with --require_nf4")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == "make_data":
        make_images(args.images)
        return
    if args.mode == "environment":
        dump(Path(args.output) / "environment.json", environment())
        print(json.dumps(environment(), ensure_ascii=False, indent=2))
        return
    if not (Path(args.images) / "image_manifest.json").exists():
        make_images(args.images)
    if args.mode in ["sft", "dpo"]:
        train(args, args.mode)
    else:
        processor, model, device, metadata = load_model(args, args.adapter, full_precision=args.full_precision)
        result = evaluate(processor, model, device, args, args.stage)
        dump(Path(args.output) / "evaluation.json", result)
        dump(Path(args.output) / "added_evaluation.json", evaluate(processor, model, device, args, args.stage + "_added", EXTRA_PAIRS))
        metadata.update(source_sha256=sha256(__file__), inference_seconds=result["seconds"])
        dump(Path(args.output) / "run.json", metadata)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
