"""Character GPT for lab03. Read stages 1-5 in order; all training is CPU-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 128
    lr: float = 1e-3
    batch_size: int = 32
    iters: int = 2000
    seed: int = 42
    use_pos: bool = True


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def output_lock(path):
    """Kernel-owned lock: released even after process termination; keep its inode."""
    handle = Path(path).open('a+b')
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(f'Output is in use by another process: {path}') from error
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == 'nt':
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


# 1. Corpus and next-character labels. Never silently discard unknown characters.
def build_corpus(extra_file=None):
    text = (ROOT / 'corpus.txt').read_text(encoding='utf-8')
    if extra_file is not None and Path(extra_file).is_file():
        text += '\n' + Path(extra_file).read_text(encoding='utf-8')
    return text


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.chars[i] for i in ids)


def get_batch(data, block_size, batch_size, generator=None, starts=None):
    if len(data) <= block_size:
        raise ValueError('Corpus must contain at least block_size + 1 characters')
    if starts is None:
        # randint's upper bound is exclusive; include the last legal window.
        starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    ix = torch.as_tensor(starts)[:, None] + offsets[None, :]
    return data[ix], data[ix + 1]


# 2. Manual multi-head causal self-attention: (B,T,C) -> (B,H,T,C/H).
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head:
            raise ValueError('n_embd must be divisible by n_head')
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('mask', torch.ones(block_size, block_size, dtype=torch.bool).tril(),
                             persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(c // self.n_head)
        weights = F.softmax(scores.masked_fill(~self.mask[:t, :t], float('-inf')), dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


# 3. Pre-Norm blocks and genuine parameter sharing between input and output.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.n_embd
        self.ln1 = nn.LayerNorm(c)
        self.attn = CausalSelfAttention(c, config.n_head, config.block_size)
        self.ln2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, 4*c), nn.GELU(), nn.Linear(4*c, c))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, config=Config()):
        super().__init__()
        self.config = config
        c = config.n_embd
        self.tok_emb = nn.Embedding(vocab_size, c)
        self.pos_emb = nn.Embedding(config.block_size, c) if config.use_pos else None
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(c)
        self.head = nn.Linear(c, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError('Expected (B,T) with 0 < T <= block_size')
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                            targets.reshape(-1))
        return logits, loss

    # 5. Sampling has its own generator and never modifies a trained model.
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=20,
                 generator=None, no_repeat_ngram=0):
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError('temperature must be finite and positive')
        if top_k is not None and top_k < 1:
            raise ValueError('top_k must be positive or None')
        if no_repeat_ngram not in (0, 3):
            raise ValueError('Supported repetition setting: 0 or 3')
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.config.block_size:])[0][:, -1, :] / temperature
            if no_repeat_ngram:
                for row in range(idx.shape[0]):
                    ids = idx[row].tolist()
                    prefix = ids[-2:]
                    banned = {ids[j+2] for j in range(len(ids)-2) if ids[j:j+2] == prefix}
                    if len(banned) == logits.shape[-1]:
                        raise RuntimeError('No token satisfies the repetition constraint')
                    if banned:
                        logits[row, list(banned)] = float('-inf')
            if top_k is not None:
                threshold = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def parameter_counts(vocab_size, config):
    c, l = config.n_embd, config.n_layer
    embedding = vocab_size*c
    position = config.block_size*c if config.use_pos else 0
    approximate = embedding + position + l*12*c*c
    exact = approximate + l*13*c + 2*c
    return {'token_embedding': embedding, 'position_embedding': position,
            'per_block': 12*c*c+13*c, 'final_layernorm': 2*c,
            'approximate': approximate, 'exact': exact,
            'approximation_relative_error': (exact-approximate)/exact}


def experiment_configs():
    base = Config()
    common = replace(base, iters=1000)
    return {'baseline': base,
            'exp1_lr_high': replace(common, lr=1e-2), 'exp1_lr_low': replace(common, lr=1e-4),
            'exp2_shallow': replace(common, n_layer=1), 'exp2_deep': replace(common, n_layer=4),
            'exp3_small': replace(common, n_embd=64, n_head=2),
            'exp3_large': replace(common, n_embd=256, n_head=8),
            'exp4_short': replace(common, block_size=32),
            'exp5_no_pos': replace(common, use_pos=False)}


@torch.inference_mode()
def evaluate(model, data):
    # Fixed windows from the training corpus: an evaluation of fit, NOT a test set.
    model.eval()
    g = torch.Generator().manual_seed(20260922)
    starts = torch.randint(len(data)-max(128, model.config.block_size), (64,), generator=g)
    total = 0.0
    for part in starts.split(8):
        x, y = get_batch(data, model.config.block_size, len(part), starts=part)
        total += model(x, y)[1].item()*len(part)
    return total / len(starts)


def save_weights(path, model, tokenizer, signature, step):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save({'state_dict': model.state_dict(), 'config': asdict(model.config),
                'chars': tokenizer.chars, 'signature': signature, 'step': step}, tmp)
    tmp.replace(path)


def save_samples(model, tok, path, prompts=('春', '春', '月', '月'), temperature=1.0,
                 top_k=20, no_repeat_ngram=0):
    samples = []
    for i, prompt in enumerate(prompts):
        seed = 1001+i
        g = torch.Generator().manual_seed(seed)
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = model.generate(ids, temperature=temperature, top_k=top_k, generator=g,
                             no_repeat_ngram=no_repeat_ngram)
        samples.append({'prompt': prompt, 'seed': seed, 'temperature': temperature,
                        'top_k': top_k, 'no_repeat_ngram': no_repeat_ngram,
                        'text': tok.decode(out[0].tolist())})
    write_json(Path(path).with_suffix('.json'), samples)
    Path(path).write_text('\n\n'.join(f"Prompt={s['prompt']} seed={s['seed']} "
                         f"temperature={temperature} top_k={top_k} no_repeat_ngram={no_repeat_ngram}\n{s['text']}"
                         for s in samples)+'\n', encoding='utf-8')
    return samples


def curve_name(config):
    tag = 'no_pos' if not config.use_pos else f'L{config.n_layer}_E{config.n_embd}_lr{config.lr:g}'
    return f'loss_curve_{tag}.png'


def plot_loss(out, losses, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses)+1), losses, linewidth=0.7)
    ax.set(xlabel='Iteration', ylabel='Training cross-entropy',
           title=f'L{config.n_layer} E{config.n_embd} H{config.n_head} '
                 f'context={config.block_size} lr={config.lr:g} pos={config.use_pos}')
    fig.tight_layout()
    fig.savefig(out/curve_name(config), dpi=150)
    plt.close(fig)


# 4. Training. Independent batch RNG, atomic partial checkpoints, CPU only.
def train(config, out, threads=4, extra_file=None):
    if min(config.n_embd, config.n_head, config.n_layer, config.block_size,
           config.batch_size, config.iters, threads) <= 0 or config.lr <= 0:
        raise ValueError('Configuration values must be positive')
    torch.set_num_threads(threads)
    torch.manual_seed(config.seed)
    text = build_corpus(extra_file)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    if len(data) <= max(128, config.block_size):
        raise ValueError('Corpus is too short')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    signature = {'config': asdict(config), 'corpus_sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'source_sha256': digest(__file__), 'torch': str(torch.__version__),
                 'threads': threads, 'device': 'cpu'}
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text(encoding='utf-8'))
        if result['signature'] != signature:
            raise ValueError('Completed results have a different signature; use a new output directory')
        required = {'model.pt', 'generated_base.txt', 'generated_base.json', 'environment.json', curve_name(config)}
        if config.iters > 1000:
            required |= {'model_step1000.pt', 'step1000.json'}
        if set(result.get('artifacts', {})) != required:
            raise ValueError('Completion manifest is incomplete')
        for name, sha in result['artifacts'].items():
            if not (out/name).is_file() or digest(out/name) != sha:
                raise ValueError(f'Missing or altered completed artifact: {name}; restore it or use a new output directory')
        print('Already completed:', out, flush=True)
        return result
    with output_lock(out/'.run.lock'):
        model = MiniGPT(tok.vocab_size, config)
        counts = parameter_counts(tok.vocab_size, config)
        assert sum(p.numel() for p in model.parameters()) == counts['exact']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
        batches = torch.Generator().manual_seed(config.seed)
        losses, evaluations, seconds = [], [], 0.0
        checkpoint = out/'partial.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['signature'] != signature:
                raise ValueError('Partial checkpoint signature differs')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            batches.set_state(saved['batch_rng']); losses = saved['losses']
            evaluations = saved['evaluations']; seconds = saved['training_seconds']
        else:
            evaluations.append({'step': 0, 'training_window_loss': evaluate(model, data)})
        environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                       'platform': platform.platform(), 'host': platform.node(),
                       'cpu': platform.processor(), 'threads': threads, 'device': 'cpu',
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                       'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')}
        write_json(out/'environment.json', environment)
        print(f'CPU threads={threads} chars={len(text)} V={tok.vocab_size} '
              f'params={counts["exact"]} initial={evaluations[0]["training_window_loss"]:.5f} '
              f'lnV={math.log(tok.vocab_size):.5f}', flush=True)
        for step in range(len(losses)+1, config.iters+1):
            start = time.perf_counter()
            model.train()
            x, y = get_batch(data, config.block_size, config.batch_size, batches)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            if not torch.isfinite(loss):
                write_json(out/'failure.json', {'step': step, 'reason': 'nonfinite loss'})
                raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward(); optimizer.step()
            losses.append(loss.item()); seconds += time.perf_counter()-start
            if step % 100 == 0 or step == config.iters:
                metric = evaluate(model, data)
                evaluations.append({'step': step, 'training_window_loss': metric})
                print(f'step {step}/{config.iters} loss={losses[-1]:.5f} fit={metric:.5f} '
                      f'train_seconds={seconds:.1f}', flush=True)
            if step == 1000 and config.iters > 1000:
                save_weights(out/'model_step1000.pt', model, tok, signature, step)
                write_json(out/'step1000.json', {'step': step, 'last_batch_loss': losses[-1],
                           'last100_mean_loss': sum(losses[-100:])/100,
                           'training_window_loss': evaluations[-1]['training_window_loss'],
                           'training_seconds': seconds})
            if step % 250 == 0 or step == config.iters:
                tmp = out/'partial.tmp'
                torch.save({'signature': signature, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'batch_rng': batches.get_state(),
                            'losses': losses, 'evaluations': evaluations, 'training_seconds': seconds}, tmp)
                tmp.replace(checkpoint)
        result = {'signature': signature, 'characters': len(text), 'vocab_size': tok.vocab_size,
                  'parameters': counts, 'ln_vocab': math.log(tok.vocab_size),
                  'losses': losses, 'evaluations': evaluations, 'last_batch_loss': losses[-1],
                  'last100_mean_loss': sum(losses[-100:])/len(losses[-100:]),
                  'training_window_loss': evaluations[-1]['training_window_loss'],
                  'training_seconds': seconds, 'training_tokens': config.iters*config.batch_size*config.block_size,
                  'evaluation_scope': 'fixed windows from training corpus; not held-out evaluation'}
        save_weights(out/'model.pt', model, tok, signature, config.iters)
        plot_loss(out, losses, config)
        save_samples(model, tok, out/'generated_base.txt')
        required = ['model.pt', 'generated_base.txt', 'generated_base.json', 'environment.json', curve_name(config)]
        if config.iters > 1000:
            required += ['model_step1000.pt', 'step1000.json']
        result['artifacts'] = {name: digest(out/name) for name in required}
        write_json(out/'result.json', result)
        checkpoint.unlink()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['iters', 'batch_size', 'block_size', 'n_embd', 'n_head', 'n_layer', 'seed']:
        p.add_argument('--'+name, type=int, default=getattr(Config(), name))
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no_pos', action='store_true')
    p.add_argument('--extra_corpus', type=Path, default=ROOT/'corpus_extra.txt')
    p.add_argument('--output', type=Path, default=ROOT/'results'/'baseline')
    p.add_argument('--experiment', choices=list(experiment_configs()))
    args = p.parse_args()
    config = Config(**{n: getattr(args, n) for n in asdict(Config()) if n != 'use_pos'},
                    use_pos=not args.no_pos)
    if args.experiment:
        config = experiment_configs()[args.experiment]
    train(config, args.output, args.threads, args.extra_corpus)


if __name__ == '__main__':
    main()
