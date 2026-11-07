# -*- coding: utf-8 -*-
"""实验八：自制字符级语言模型的知识蒸馏。

该脚本仅依赖 PyTorch；语料由固定种子现场生成，不下载模型或数据。
它实现教师训练、学生硬标签基线、知识蒸馏、公平性校验、T×alpha
完整消融、FitNet 特征蒸馏与动态量化。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "runs_distill"
TEACHER_CONFIG = {"n_layer": 2, "n_head": 4, "n_embd": 160, "block_size": 96, "dropout": 0.0}
STUDENT_CONFIG = {"n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 96, "dropout": 0.0}


SUBJECTS = [
    "小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
    "少年", "故人", "归客", "山僧", "孤客", "游子",
]
PLACES = [
    "江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
    "溪桥", "松间", "篱下", "窗前", "陌上", "舟中",
]
VERBS = [
    "望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
    "静听", "又逢", "恰遇", "空忆",
]
OBJECTS = [
    "明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
    "白雪", "寒江", "疏星", "暮云", "秋草", "远帆",
]
MOODS = [
    "心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
    "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别",
]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]


def build_corpus(n_lines: int = 3000, seed: int = 0) -> str:
    """生成可复现的仿古诗句；不同 seed 产生同分布样本，固定诗句会重合。"""

    rng = random.Random(seed)
    lines: list[str] = []
    for index in range(n_lines):
        subject, place, verb, obj, mood = (
            rng.choice(items) for items in (SUBJECTS, PLACES, VERBS, OBJECTS, MOODS)
        )
        if index % 3 == 0:
            line = f"{subject}{place}{verb}{obj}，{mood}。"
        elif index % 3 == 1:
            action = rng.choice(["看", "听", "守", "忆"])
            line = f"{subject}在{place}{obj}了一夜{action}，{mood}。"
        else:
            adverb = rng.choice(["正好", "可怜", "可惜", "依旧"])
            line = f"{subject}{verb}道：“{obj}{adverb}。”{mood}。"
        lines.append(line + "\n")
    lines.extend(POEMS)
    return "".join(lines)


class CharacterTokenizer:
    def __init__(self, text: str):
        self.itos = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text if char in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(index)] for index in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = int(config["n_embd"])
        heads = int(config["n_head"])
        if width % heads != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = heads
        self.c_attn = nn.Linear(width, 3 * width)
        self.c_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(float(config["dropout"]))
        mask = torch.tril(torch.ones(int(config["block_size"]), int(config["block_size"])))
        self.register_buffer("mask", mask.view(1, 1, mask.shape[0], mask.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        query, key, value = self.c_attn(x).split(width, dim=2)
        head_width = width // self.n_head
        query = query.view(batch, seq_len, self.n_head, head_width).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_head, head_width).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_head, head_width).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(head_width)
        attention = attention.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attention = self.dropout(F.softmax(attention, dim=-1))
        output = (attention @ value).transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = int(config["n_embd"])
        self.fc = nn.Linear(width, 4 * width)
        self.proj = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(float(config["dropout"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        width = int(config["n_embd"])
        self.ln1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        width = int(config["n_embd"])
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(int(config["block_size"]), width)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(int(config["n_layer"]))]
        )
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._initialize)
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        need_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        _, seq_len = tokens.shape
        if seq_len > int(self.config["block_size"]):
            raise ValueError("输入序列长度超过 block_size")
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, hidden
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.9,
        top_k: int | None = 20,
        seed: int = 2026,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        output = tokens
        for _ in range(max_new_tokens):
            context = output[:, -int(self.config["block_size"]):]
            logits = self(context)[0][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)
            output = torch.cat((output, next_token), dim=1)
        self.train(was_training)
        return output


@dataclass(frozen=True)
class DataBundle:
    tokenizer: CharacterTokenizer
    train_ids: torch.Tensor
    validation_ids: torch.Tensor
    train_text: str
    validation_text: str


def make_data(block_size: int) -> DataBundle:
    train_text = build_corpus(3000, seed=0)
    validation_text = build_corpus(300, seed=999)
    tokenizer = CharacterTokenizer(train_text)
    unknown = sorted(set(validation_text) - set(tokenizer.itos))
    if unknown:
        raise ValueError(f"留出集存在训练词表外字符: {unknown}")
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    if len(train_ids) <= block_size + 1 or len(validation_ids) <= block_size + 1:
        raise ValueError("语料长度不足以按给定 block_size 采样")
    return DataBundle(tokenizer, train_ids, validation_ids, train_text, validation_text)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[index : index + block_size] for index in starts])
    y = torch.stack([data[index + 1 : index + 1 + block_size] for index in starts])
    return x, y


def kd_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """计算 KL(p_teacher || p_student)，按 token 平均并乘 T²。"""

    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    vocab_size = student_logits.size(-1)
    student_log_probability = F.log_softmax(
        student_logits.reshape(-1, vocab_size) / temperature, dim=-1
    )
    teacher_probability = F.softmax(
        teacher_logits.reshape(-1, vocab_size) / temperature, dim=-1
    )
    return F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature**2)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    iterations: int = 20,
    seed: int = 1234,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _ in range(iterations):
        x, y = get_batch(data, batch_size, block_size, generator)
        loss = model(x, y)[1]
        if loss is None:
            raise RuntimeError("评估时未返回 loss")
        losses.append(float(loss.item()))
    model.train(was_training)
    return statistics.fmean(losses)


@torch.no_grad()
def teacher_agreement(
    student: nn.Module,
    teacher: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    iterations: int = 10,
    seed: int = 4321,
) -> float:
    student_state, teacher_state = student.training, teacher.training
    student.eval()
    teacher.eval()
    generator = torch.Generator().manual_seed(seed)
    matches = 0
    total = 0
    for _ in range(iterations):
        x, _ = get_batch(data, batch_size, block_size, generator)
        student_tokens = student(x)[0].argmax(dim=-1)
        teacher_tokens = teacher(x)[0].argmax(dim=-1)
        matches += int((student_tokens == teacher_tokens).sum().item())
        total += teacher_tokens.numel()
    student.train(student_state)
    teacher.train(teacher_state)
    return matches / total


def train_model(
    model: GPT,
    data: torch.Tensor,
    iterations: int,
    batch_size: int,
    block_size: int,
    learning_rate: float,
    seed: int,
    tag: str,
    teacher: GPT | None = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    feature_weight: float = 0.0,
    log_every: int = 50,
) -> tuple[dict[str, Any], nn.Linear | None]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必须位于 [0, 1]")
    if feature_weight < 0:
        raise ValueError("feature_weight 不能为负")
    projection = None
    if teacher is not None and feature_weight > 0:
        projection = nn.Linear(int(model.config["n_embd"]), int(teacher.config["n_embd"]))
    parameters = list(model.parameters()) + ([] if projection is None else list(projection.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed + 1)
    if teacher is not None:
        teacher.eval()
    model.train()
    started = time.perf_counter()
    total_curve: list[float] = []
    ce_curve: list[float] = []
    kd_curve: list[float | None] = []
    feature_curve: list[float | None] = []

    for step in range(iterations):
        x, y = get_batch(data, batch_size, block_size, generator)
        if projection is None:
            student_logits = model(x)[0]
            student_hidden = None
        else:
            student_logits, _, student_hidden = model(x, need_hidden=True)
        ce_loss = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), y.reshape(-1))
        loss = ce_loss
        kd_loss = None
        feature_loss = None
        if teacher is not None and (alpha > 0 or projection is not None):
            with torch.no_grad():
                if projection is None:
                    teacher_logits = teacher(x)[0]
                    teacher_hidden = None
                else:
                    teacher_logits, _, teacher_hidden = teacher(x, need_hidden=True)
            if alpha > 0:
                kd_loss = kd_divergence(student_logits, teacher_logits, temperature)
                loss = (1 - alpha) * ce_loss + alpha * kd_loss
            if projection is not None:
                if student_hidden is None or teacher_hidden is None:
                    raise RuntimeError("特征蒸馏缺少隐状态")
                feature_loss = F.mse_loss(projection(student_hidden), teacher_hidden)
                loss = loss + feature_weight * feature_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"{tag} 第 {step + 1} 步出现非有限梯度")
        optimizer.step()
        total_curve.append(float(loss.item()))
        ce_curve.append(float(ce_loss.item()))
        kd_curve.append(None if kd_loss is None else float(kd_loss.item()))
        feature_curve.append(None if feature_loss is None else float(feature_loss.item()))
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == iterations:
            components = f"ce={ce_loss.item():.4f}"
            if kd_loss is not None:
                components += f" kd_T2={kd_loss.item():.4f}"
            if feature_loss is not None:
                components += f" feat={feature_loss.item():.4f}"
            print(
                f"[{tag}] step={step + 1:4d}/{iterations} loss={loss.item():.4f} "
                f"{components} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    history = {
        "curve": total_curve,
        "ce_curve": ce_curve,
        "kd_curve": kd_curve,
        "feature_curve": feature_curve,
        "time_s": elapsed,
        "iterations": iterations,
        "batch_seed": seed + 1,
    }
    return history, projection


def environment_record(thread_count: int) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": thread_count,
        "gpu_used": False,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_checkpoint_metadata(args: argparse.Namespace, data: DataBundle) -> dict[str, Any]:
    data_digest = hashlib.sha256(data.train_ids.numpy().tobytes()).hexdigest()
    return {
        "schema_version": 1,
        "teacher_config": TEACHER_CONFIG,
        "vocab_size": data.tokenizer.vocab_size,
        "train_ids_sha256": data_digest,
        "seed": args.seed,
        "teacher_iters": args.teacher_iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }


def load_teacher(path: Path, vocab_size: int, seed: int) -> GPT:
    torch.manual_seed(seed)
    teacher = GPT(vocab_size, TEACHER_CONFIG)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    teacher.load_state_dict(state)
    return teacher


def new_student(vocab_size: int, seed: int) -> GPT:
    torch.manual_seed(seed + 7)
    return GPT(vocab_size, STUDENT_CONFIG)


def model_result(
    model: GPT,
    teacher: GPT,
    validation_ids: torch.Tensor,
    tokenizer: CharacterTokenizer,
    history: dict[str, Any],
    batch_size: int,
    block_size: int,
    prompt: str,
    generation_seed: int,
) -> dict[str, Any]:
    validation_loss = evaluate(model, validation_ids, batch_size, block_size)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    sample = tokenizer.decode(model.generate(prompt_ids, seed=generation_seed)[0].tolist())
    return {
        "params": model.parameter_count,
        "iterations": history["iterations"],
        "time_s": history["time_s"],
        "val_loss": validation_loss,
        "ppl": math.exp(validation_loss),
        "agreement": teacher_agreement(model, teacher, validation_ids, batch_size, block_size),
        "sample": sample,
        "curve": history["curve"],
        "ce_curve": history["ce_curve"],
        "kd_curve": history["kd_curve"],
        "feature_curve": history["feature_curve"],
        "batch_seed": history["batch_seed"],
    }


def train_or_load_teacher(args: argparse.Namespace, data: DataBundle, out_dir: Path) -> tuple[GPT, dict[str, Any]]:
    teacher_path = out_dir / "teacher.pt"
    expected_metadata = teacher_checkpoint_metadata(args, data)
    torch.manual_seed(args.seed)
    teacher = GPT(data.tokenizer.vocab_size, TEACHER_CONFIG)
    if teacher_path.exists() and not args.retrain:
        payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
        required_keys = {"state_dict", "metadata", "training_record"}
        if not isinstance(payload, dict) or not required_keys.issubset(payload):
            raise RuntimeError(
                f"教师缓存 {teacher_path} 是旧格式或不完整；请加 --retrain 重新训练。"
            )
        if payload["metadata"] != expected_metadata:
            raise RuntimeError(
                f"教师缓存 {teacher_path} 与当前配置/数据不匹配；请加 --retrain 重新训练。"
            )
        teacher.load_state_dict(payload["state_dict"])
        validation_loss = evaluate(
            teacher, data.validation_ids, args.batch_size, TEACHER_CONFIG["block_size"]
        )
        record = copy.deepcopy(payload["training_record"])
        record.update(
            {
                "params": teacher.parameter_count,
                "val_loss": validation_loss,
                "ppl": math.exp(validation_loss),
                "sample": data.tokenizer.decode(
                    teacher.generate(
                        torch.tensor([data.tokenizer.encode(args.prompt)], dtype=torch.long),
                        seed=args.generation_seed,
                    )[0].tolist()
                ),
                "cached": True,
            }
        )
        print(f"[teacher] loaded={teacher_path} val_loss={validation_loss:.4f}", flush=True)
        return teacher, record

    history, _ = train_model(
        teacher,
        data.train_ids,
        args.teacher_iters,
        args.batch_size,
        TEACHER_CONFIG["block_size"],
        args.lr,
        args.seed,
        "teacher",
        log_every=args.log_every,
    )
    validation_loss = evaluate(
        teacher, data.validation_ids, args.batch_size, TEACHER_CONFIG["block_size"]
    )
    sample = data.tokenizer.decode(
        teacher.generate(
            torch.tensor([data.tokenizer.encode(args.prompt)], dtype=torch.long),
            seed=args.generation_seed,
        )[0].tolist()
    )
    record = {
        "params": teacher.parameter_count,
        "iterations": history["iterations"],
        "time_s": history["time_s"],
        "val_loss": validation_loss,
        "ppl": math.exp(validation_loss),
        "sample": sample,
        "curve": history["curve"],
        "cached": False,
    }
    torch.save(
        {
            "state_dict": teacher.state_dict(),
            "metadata": expected_metadata,
            "training_record": record,
        },
        teacher_path,
    )
    print(
        f"[teacher] params={teacher.parameter_count / 1e6:.3f}M "
        f"val_loss={validation_loss:.4f} ppl={math.exp(validation_loss):.3f} "
        f"time={history['time_s']:.1f}s",
        flush=True,
    )
    return teacher, record


def run_baselines(
    args: argparse.Namespace,
    data: DataBundle,
    teacher: GPT,
    out_dir: Path,
) -> tuple[dict[str, Any], GPT, GPT]:
    block_size = STUDENT_CONFIG["block_size"]
    scratch = new_student(data.tokenizer.vocab_size, args.seed)
    kd_student = new_student(data.tokenizer.vocab_size, args.seed)
    initial_scratch = {name: tensor.detach().clone() for name, tensor in scratch.state_dict().items()}
    initialization_identical = all(
        torch.equal(initial_scratch[name], tensor) for name, tensor in kd_student.state_dict().items()
    )
    if not initialization_identical:
        raise RuntimeError("scratch 与 KD 学生初始化不一致")

    scratch_history, _ = train_model(
        scratch,
        data.train_ids,
        args.student_iters,
        args.batch_size,
        block_size,
        args.lr,
        args.seed,
        "scratch",
        log_every=args.log_every,
    )
    baseline_feature_weight = args.feature_weight if args.mode == "all" else 0.0
    kd_history, _ = train_model(
        kd_student,
        data.train_ids,
        args.student_iters,
        args.batch_size,
        block_size,
        args.lr,
        args.seed,
        "kd",
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=baseline_feature_weight,
        log_every=args.log_every,
    )
    scratch_record = model_result(
        scratch,
        teacher,
        data.validation_ids,
        data.tokenizer,
        scratch_history,
        args.batch_size,
        block_size,
        args.prompt,
        args.generation_seed,
    )
    kd_record = model_result(
        kd_student,
        teacher,
        data.validation_ids,
        data.tokenizer,
        kd_history,
        args.batch_size,
        block_size,
        args.prompt,
        args.generation_seed,
    )
    kd_record.update(
        {
            "temperature": args.temperature,
            "alpha": args.alpha,
            "feature_weight": baseline_feature_weight,
        }
    )
    scratch_record.update({"temperature": None, "alpha": 0.0, "feature_weight": 0.0})
    torch.save(scratch.state_dict(), out_dir / "student_scratch.pt")
    torch.save(kd_student.state_dict(), out_dir / "student_kd.pt")
    relative_improvement = (
        (scratch_record["val_loss"] - kd_record["val_loss"]) / scratch_record["val_loss"] * 100
    )
    record = {
        "fairness": {
            "same_architecture": True,
            "same_initialization": initialization_identical,
            "same_batch_seed": scratch_history["batch_seed"] == kd_history["batch_seed"],
            "same_iterations": scratch_history["iterations"] == kd_history["iterations"],
            "only_objective_differs": True,
        },
        "scratch": scratch_record,
        "kd": kd_record,
        "relative_val_loss_improvement_percent": relative_improvement,
    }
    print(
        f"[comparison] scratch={scratch_record['val_loss']:.4f} "
        f"kd={kd_record['val_loss']:.4f} improvement={relative_improvement:.2f}%",
        flush=True,
    )
    return record, scratch, kd_student


def run_ablation(
    args: argparse.Namespace,
    data: DataBundle,
    teacher: GPT,
) -> dict[str, Any]:
    block_size = STUDENT_CONFIG["block_size"]
    grid_t = [1.0, 2.0, 4.0, 8.0]
    grid_alpha = [0.3, 0.5, 0.7, 0.9]
    generator = torch.Generator().manual_seed(2024)
    cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(min(32, args.abl_iters)):
            x, y = get_batch(data.train_ids, args.batch_size, block_size, generator)
            cache.append((x, y, teacher(x)[0]))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for temperature in grid_t:
        for alpha in grid_alpha:
            student = new_student(data.tokenizer.vocab_size, args.seed)
            optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
            student.train()
            train_curve: list[float] = []
            for step in range(args.abl_iters):
                x, y, teacher_logits = cache[step % len(cache)]
                student_logits = student(x)[0]
                ce_loss = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)), y.reshape(-1)
                )
                kd_loss = kd_divergence(student_logits, teacher_logits, temperature)
                loss = (1 - alpha) * ce_loss + alpha * kd_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                train_curve.append(float(loss.item()))
            validation_loss = evaluate(
                student, data.validation_ids, args.batch_size, block_size
            )
            agreement = teacher_agreement(
                student, teacher, data.validation_ids, args.batch_size, block_size
            )
            row = {
                "temperature": temperature,
                "alpha": alpha,
                "val_loss": validation_loss,
                "ppl": math.exp(validation_loss),
                "agreement": agreement,
                "iterations": args.abl_iters,
                "train_final_loss": train_curve[-1],
                "curve": train_curve,
            }
            rows.append(row)
            print(
                f"[ablation] T={temperature:g} alpha={alpha:.1f} "
                f"val_loss={validation_loss:.4f} agreement={agreement:.3f}",
                flush=True,
            )
    best = min(rows, key=lambda item: item["val_loss"])
    worst = max(rows, key=lambda item: item["val_loss"])
    return {
        "rows": rows,
        "best": best,
        "worst": worst,
        "cache_batches": len(cache),
        "cache_seed": 2024,
        "same_student_initialization": True,
        "time_s": time.perf_counter() - started,
    }


def run_feature_distillation(
    args: argparse.Namespace,
    data: DataBundle,
    teacher: GPT,
) -> tuple[dict[str, Any], GPT]:
    student = new_student(data.tokenizer.vocab_size, args.seed)
    feature_weight = args.feature_weight if args.feature_weight > 0 else 0.1
    history, projection = train_model(
        student,
        data.train_ids,
        args.student_iters,
        args.batch_size,
        STUDENT_CONFIG["block_size"],
        args.lr,
        args.seed,
        "feature-kd",
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        feature_weight=feature_weight,
        log_every=args.log_every,
    )
    record = model_result(
        student,
        teacher,
        data.validation_ids,
        data.tokenizer,
        history,
        args.batch_size,
        STUDENT_CONFIG["block_size"],
        args.prompt,
        args.generation_seed,
    )
    record.update(
        {
            "temperature": args.temperature,
            "alpha": args.alpha,
            "feature_weight": feature_weight,
            "projection_params": 0 if projection is None else sum(p.numel() for p in projection.parameters()),
        }
    )
    return record, student


def run_teacher_scale_scan(
    args: argparse.Namespace,
    data: DataBundle,
    base_teacher: GPT,
    teacher_record: dict[str, Any],
    baseline_record: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """比较三种教师宽度，并保持学生、初始化、批次与训练步数不变。"""

    rows: list[dict[str, Any]] = []
    scratch_loss = baseline_record["scratch"]["val_loss"]
    for width in (96, 160, 256):
        if width == 160:
            teacher = base_teacher
            current_teacher_record = teacher_record
            kd_record = baseline_record["kd"]
        else:
            config = dict(TEACHER_CONFIG)
            config["n_embd"] = width
            torch.manual_seed(args.seed)
            teacher = GPT(data.tokenizer.vocab_size, config)
            teacher_history, _ = train_model(
                teacher,
                data.train_ids,
                args.teacher_iters,
                args.batch_size,
                config["block_size"],
                args.lr,
                args.seed,
                f"teacher-c{width}",
                log_every=args.log_every,
            )
            teacher_loss = evaluate(
                teacher, data.validation_ids, args.batch_size, config["block_size"]
            )
            current_teacher_record = {
                "params": teacher.parameter_count,
                "iterations": args.teacher_iters,
                "time_s": teacher_history["time_s"],
                "val_loss": teacher_loss,
                "ppl": math.exp(teacher_loss),
                "curve": teacher_history["curve"],
            }
            torch.save(teacher.state_dict(), out_dir / f"teacher_c{width}.pt")

            student = new_student(data.tokenizer.vocab_size, args.seed)
            kd_history, _ = train_model(
                student,
                data.train_ids,
                args.student_iters,
                args.batch_size,
                STUDENT_CONFIG["block_size"],
                args.lr,
                args.seed,
                f"kd-teacher-c{width}",
                teacher=teacher,
                temperature=args.temperature,
                alpha=args.alpha,
                log_every=args.log_every,
            )
            kd_record = model_result(
                student,
                teacher,
                data.validation_ids,
                data.tokenizer,
                kd_history,
                args.batch_size,
                STUDENT_CONFIG["block_size"],
                args.prompt,
                args.generation_seed,
            )
        kd_loss = kd_record["val_loss"]
        rows.append(
            {
                "teacher_width": width,
                "teacher_params": teacher.parameter_count,
                "teacher_val_loss": current_teacher_record["val_loss"],
                "teacher_ppl": current_teacher_record["ppl"],
                "teacher_time_s": current_teacher_record["time_s"],
                "teacher_iterations": current_teacher_record["iterations"],
                "teacher_curve": current_teacher_record["curve"],
                "teacher_batch_seed": args.seed + 1,
                "student_kd_val_loss": kd_loss,
                "student_kd_ppl": kd_record["ppl"],
                "student_teacher_agreement": kd_record["agreement"],
                "student_kd_time_s": kd_record["time_s"],
                "student_kd_iterations": kd_record["iterations"],
                "student_kd_curve": kd_record["curve"],
                "student_kd_batch_seed": kd_record["batch_seed"],
                "scratch_val_loss": scratch_loss,
                "kd_relative_improvement_percent": (scratch_loss - kd_loss) / scratch_loss * 100,
            }
        )
    return {
        "rows": rows,
        "controlled_student_width": STUDENT_CONFIG["n_embd"],
        "same_student_initialization": True,
        "same_student_batches": True,
        "same_student_iterations": True,
    }


def serialized_size(path: Path, object_to_save: Any) -> int:
    torch.save(object_to_save, path)
    return path.stat().st_size


@torch.no_grad()
def median_forward_latency_ms(model: nn.Module, tokens: torch.Tensor, repeats: int = 50) -> float:
    model.eval()
    for _ in range(5):
        model(tokens)
    measurements: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(tokens)
        measurements.append((time.perf_counter() - started) * 1000)
    return statistics.median(measurements)


def run_dynamic_quantization(
    args: argparse.Namespace,
    student: GPT,
    data: DataBundle,
    out_dir: Path,
) -> dict[str, Any]:
    fp32_model = copy.deepcopy(student).cpu().eval()
    try:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
    except AttributeError:
        quantize_dynamic = torch.quantization.quantize_dynamic
    int8_model = quantize_dynamic(copy.deepcopy(fp32_model), {nn.Linear}, dtype=torch.qint8)
    fp32_path = out_dir / "student_kd_fp32_state.pt"
    int8_path = out_dir / "student_kd_int8_state.pt"
    fp32_size = serialized_size(fp32_path, fp32_model.state_dict())
    int8_size = serialized_size(int8_path, int8_model.state_dict())
    reloaded_fp32_model = new_student(data.tokenizer.vocab_size, args.seed).cpu().eval()
    reloaded_fp32_model.load_state_dict(
        torch.load(fp32_path, map_location="cpu", weights_only=True)
    )
    reloaded_int8_model = quantize_dynamic(copy.deepcopy(fp32_model), {nn.Linear}, dtype=torch.qint8)
    reloaded_int8_model.load_state_dict(
        torch.load(int8_path, map_location="cpu", weights_only=True)
    )
    generator = torch.Generator().manual_seed(9001)
    latency_tokens, _ = get_batch(
        data.validation_ids,
        1,
        STUDENT_CONFIG["block_size"],
        generator,
    )
    fp32_latency = median_forward_latency_ms(reloaded_fp32_model, latency_tokens)
    int8_latency = median_forward_latency_ms(int8_model, latency_tokens)
    fp32_loss = evaluate(
        reloaded_fp32_model,
        data.validation_ids,
        args.batch_size,
        STUDENT_CONFIG["block_size"],
    )
    int8_loss = evaluate(
        reloaded_int8_model,
        data.validation_ids,
        args.batch_size,
        STUDENT_CONFIG["block_size"],
    )
    return {
        "method": "PyTorch dynamic qint8 for nn.Linear",
        "serialization": "state_dict for both fp32 and int8",
        "fp32_state_dict_roundtrip_verified": True,
        "int8_state_dict_roundtrip_verified": True,
        "fp32_file": fp32_path.name,
        "int8_file": int8_path.name,
        "fp32_bytes": fp32_size,
        "int8_bytes": int8_size,
        "size_compression_ratio": fp32_size / int8_size,
        "fp32_latency_ms": fp32_latency,
        "int8_latency_ms": int8_latency,
        "latency_speedup": fp32_latency / int8_latency,
        "fp32_val_loss": fp32_loss,
        "int8_val_loss": int8_loss,
        "val_loss_delta": int8_loss - fp32_loss,
        "latency_batch_size": 1,
        "latency_block_size": STUDENT_CONFIG["block_size"],
        "latency_repeats": 50,
    }


def plot_results(summary: dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[plot] skipped: {error}", flush=True)
        return

    plt.rcParams["axes.unicode_minus"] = False
    if "baseline" in summary:
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        teacher_curve = summary["teacher"].get("curve", [])
        if teacher_curve:
            axes[0].plot(teacher_curve, label="Teacher", linewidth=1.3)
        axes[0].plot(summary["baseline"]["scratch"]["curve"], label="Student scratch", linewidth=1.3)
        axes[0].plot(summary["baseline"]["kd"]["curve"], label="Student KD", linewidth=1.3)
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        labels = ["Teacher", "Scratch", "KD"]
        values = [
            summary["teacher"]["val_loss"],
            summary["baseline"]["scratch"]["val_loss"],
            summary["baseline"]["kd"]["val_loss"],
        ]
        bars = axes[1].bar(labels, values, color=["#365f91", "#a5a5a5", "#5b9bd5"])
        axes[1].set_title("Validation loss")
        axes[1].set_ylabel("Loss (lower is better)")
        axes[1].grid(alpha=0.25, axis="y")
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
        figure.tight_layout()
        figure.savefig(out_dir / "curves.png", dpi=180)
        plt.close(figure)

    if "ablation" in summary:
        rows = summary["ablation"]["rows"]
        grid_t = [1.0, 2.0, 4.0, 8.0]
        grid_alpha = [0.3, 0.5, 0.7, 0.9]
        matrix = [
            [next(row["val_loss"] for row in rows if row["temperature"] == t and row["alpha"] == a) for a in grid_alpha]
            for t in grid_t
        ]
        figure, axis = plt.subplots(figsize=(6.3, 4.8))
        image = axis.imshow(matrix, cmap="Blues_r", aspect="auto")
        axis.set_xticks(range(len(grid_alpha)), [str(value) for value in grid_alpha])
        axis.set_yticks(range(len(grid_t)), [str(int(value)) for value in grid_t])
        axis.set_xlabel("alpha")
        axis.set_ylabel("temperature T")
        axis.set_title("T x alpha ablation: validation loss")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="black")
        figure.colorbar(image, ax=axis, label="Validation loss")
        figure.tight_layout()
        figure.savefig(out_dir / "ablation_heatmap.png", dpi=180)
        plt.close(figure)

    if "teacher_scale_scan" in summary:
        rows = summary["teacher_scale_scan"]["rows"]
        widths = [row["teacher_width"] for row in rows]
        teacher_losses = [row["teacher_val_loss"] for row in rows]
        benefits = [row["kd_relative_improvement_percent"] for row in rows]
        figure, left = plt.subplots(figsize=(6.6, 4.2))
        right = left.twinx()
        left.plot(widths, teacher_losses, marker="o", color="#365f91", label="Teacher val loss")
        right.plot(widths, benefits, marker="s", color="#c55a11", label="KD improvement")
        left.set_xlabel("Teacher embedding width")
        left.set_ylabel("Teacher validation loss", color="#365f91")
        right.set_ylabel("KD improvement vs scratch (%)", color="#c55a11")
        left.set_xticks(widths)
        left.grid(alpha=0.25)
        lines = left.lines + right.lines
        left.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(out_dir / "teacher_scale_scan.png", dpi=180)
        plt.close(figure)


def base_summary(args: argparse.Namespace, data: DataBundle, out_dir: Path) -> dict[str, Any]:
    train_lines = set(data.train_text.splitlines())
    validation_lines = set(data.validation_text.splitlines())
    exact_line_overlap = sorted(train_lines & validation_lines)
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "output_dir": str(out_dir),
        "environment": environment_record(torch.get_num_threads()),
        "data": {
            "vocab_size": data.tokenizer.vocab_size,
            "train_characters": len(data.train_ids),
            "validation_characters": len(data.validation_ids),
            "train_seed": 0,
            "validation_seed": 999,
            "same_generator_distribution": True,
            "text_identical": data.train_text == data.validation_text,
            "exact_line_overlap_count": len(exact_line_overlap),
            "exact_line_overlap": exact_line_overlap,
        },
        "configs": {"teacher": TEACHER_CONFIG, "student": STUDENT_CONFIG},
        "hyperparameters": {
            "teacher_iters": args.teacher_iters,
            "student_iters": args.student_iters,
            "ablation_iters": args.abl_iters,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "feature_weight": args.feature_weight,
            "seed": args.seed,
        },
    }


def artifact_manifest(out_dir: Path) -> dict[str, Any]:
    checkpoint_names = [
        "teacher.pt",
        "student_scratch.pt",
        "student_kd.pt",
        "student_kd_fp32_state.pt",
        "student_kd_int8_state.pt",
    ]
    checkpoints = {
        name: {"sha256": sha256_file(out_dir / name), "bytes": (out_dir / name).stat().st_size}
        for name in checkpoint_names
        if (out_dir / name).is_file()
    }
    return {
        "source": {
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "checkpoints": checkpoints,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    thread_count = max(1, min(args.threads, os.cpu_count() or args.threads))
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = make_data(TEACHER_CONFIG["block_size"])
    summary = base_summary(args, data, out_dir)
    print(
        f"[data] vocab={data.tokenizer.vocab_size} train={len(data.train_ids)} "
        f"validation={len(data.validation_ids)} batch={args.batch_size} threads={thread_count}",
        flush=True,
    )

    teacher, teacher_record = train_or_load_teacher(args, data, out_dir)
    summary["teacher"] = teacher_record
    if args.mode == "teacher":
        summary["artifacts"] = artifact_manifest(out_dir)
        save_json(out_dir / "results_teacher.json", summary)
        plot_results(summary, out_dir)
        return summary

    kd_student: GPT | None = None
    if args.mode in {"all", "full", "quantize", "scan"}:
        baseline, _, kd_student = run_baselines(args, data, teacher, out_dir)
        summary["baseline"] = baseline

    if args.mode in {"ablate", "full"}:
        summary["ablation"] = run_ablation(args, data, teacher)

    if args.mode in {"feature", "full"}:
        feature_record, _ = run_feature_distillation(args, data, teacher)
        summary["feature_distillation"] = feature_record

    if args.mode in {"scan", "full"}:
        summary["teacher_scale_scan"] = run_teacher_scale_scan(
            args,
            data,
            teacher,
            teacher_record,
            summary["baseline"],
            out_dir,
        )

    if args.mode in {"quantize", "full"}:
        if kd_student is None:
            kd_path = out_dir / "student_kd.pt"
            if not kd_path.exists():
                raise FileNotFoundError("动态量化前需先运行 all/full，或保留 student_kd.pt")
            kd_student = new_student(data.tokenizer.vocab_size, args.seed)
            kd_student.load_state_dict(torch.load(kd_path, map_location="cpu", weights_only=True))
        summary["dynamic_quantization"] = run_dynamic_quantization(args, kd_student, data, out_dir)

    model_counts = {
        "teacher": teacher.parameter_count,
        "student": new_student(data.tokenizer.vocab_size, args.seed).parameter_count,
    }
    model_counts["compression_ratio"] = model_counts["teacher"] / model_counts["student"]
    summary["model_counts"] = model_counts
    summary["artifacts"] = artifact_manifest(out_dir)
    save_json(out_dir / f"results_{args.mode}.json", summary)
    if args.mode == "full":
        save_json(out_dir / "results_all.json", summary)
    plot_results(summary, out_dir)
    print(f"[done] results={out_dir}", flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["quick", "teacher", "all", "ablate", "feature", "quantize", "scan", "full"],
    )
    parser.add_argument("--teacher-iters", type=int, default=200)
    parser.add_argument("--student-iters", type=int, default=120)
    parser.add_argument("--abl-iters", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--temperature", "--T", dest="temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--feature-weight", "--feat", dest="feature_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=2026)
    parser.add_argument("--prompt", default="月")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    if args.mode == "quick":
        args.mode = "full"
        args.teacher_iters = 8
        args.student_iters = 6
        args.abl_iters = 3
        args.batch_size = 4
        args.log_every = 2
        args.out_dir = str(Path(args.out_dir) / "quick")
        args.retrain = True
    for name in ("teacher_iters", "student_iters", "abl_iters", "batch_size", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
