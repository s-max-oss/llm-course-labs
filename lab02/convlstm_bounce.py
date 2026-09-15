"""Lab 02: reproducible ConvLSTM ablations on synthetic bouncing balls.

Run: python convlstm_bounce.py --suite all
No network, dataset download or external AI service is required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
if os.environ.get("LAB02_FONT"):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(os.environ["LAB02_FONT"])


@dataclass(frozen=True)
class Config:
    architecture: str = "convlstm"
    hidden: int = 32
    kernel: int = 3
    layers: int = 1
    input_frames: int = 4
    loss: str = "mse"
    lr: float = 0.001
    epochs: int = 5
    batch_size: int = 64
    seed: int = 42


BASE = Config()
EXPERIMENTS = {
    "baseline": BASE,
    "exp1": replace(BASE, layers=2),
    "exp2": replace(BASE, hidden=64),
    "exp3": replace(BASE, kernel=5),
    "exp4": replace(BASE, input_frames=8),
    "exp5": replace(BASE, architecture="flatten_lstm"),
    "exp6": replace(BASE, loss="l1"),
}


def make_sequences(n_seq=2200, T=10, size=32, r=2, seed=42):
    """Return uint8 (N,T,1,H,W), with reflected centers inside the image.

    Each velocity component has magnitude U(0.8,1.6), as in the guide.
    Mirror overshoot at the boundary instead of letting the ball clip.
    """
    if n_seq < 1 or T < 2 or size < 12 or r < 1 or 2 * r >= size - 1:
        raise ValueError("Invalid sequence count, length, size or radius")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    seqs = np.zeros((n_seq, T, 1, size, size), dtype=np.uint8)
    lo, hi = float(r), float(size - 1 - r)
    for s in range(n_seq):
        pos = rng.uniform(4, size - 5, 2)
        velocity = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)
        for t in range(T):
            pos += velocity
            for axis in range(2):
                while pos[axis] < lo or pos[axis] > hi:
                    if pos[axis] < lo:
                        pos[axis] = 2 * lo - pos[axis]
                    else:
                        pos[axis] = 2 * hi - pos[axis]
                    velocity[axis] *= -1
            seqs[s, t, 0] = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 <= r**2
    return seqs


def split_data(n_train=2000, n_test=200, n_val=200):
    data = make_sequences(n_train + n_test)
    train, test = data[:n_train], data[n_train:]
    # Additional independent validation data selects combinations without test tuning.
    val = make_sequences(n_val, seed=43)
    hashes = [{hashlib.sha256(s.tobytes()).hexdigest() for s in part}
              for part in (train, test, val)]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Overlapping sequences across splits")
    return train, test, val


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.hid = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.conv(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        if layers < 1 or hid < 1:
            raise ValueError("layers and hidden channels must be positive")
        channels = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(channels[i], hid, k) for i in range(layers)])
        # The output kernel remains 3x3 even in the K=5 cell experiment.
        self.out = nn.Conv2d(hid, 1, 3, padding=1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] < 1 or x.shape[2] != 1:
            raise ValueError("Expected (B,T,1,H,W) with T >= 1")
        b, _, _, height, width = x.shape
        states = [(x.new_zeros(b, cell.hid, height, width),
                   x.new_zeros(b, cell.hid, height, width)) for cell in self.cells]
        for t in range(x.shape[1]):
            current = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(current, states[j])
                current = states[j][0]
        return self.out(states[-1][0]).squeeze(1)


class FlattenLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1024, 256, batch_first=True)
        self.out = nn.Linear(256, 1024)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] < 1 or tuple(x.shape[2:]) != (1, 32, 32):
            raise ValueError("Expected (B,T,1,32,32) with T >= 1")
        hidden, _ = self.lstm(x.flatten(2))
        return self.out(hidden[:, -1]).reshape(-1, 32, 32)


def build_model(config):
    if config.architecture == "flatten_lstm":
        return FlattenLSTM()
    if config.architecture != "convlstm":
        raise ValueError("Unknown architecture")
    return ConvLSTM(hid=config.hidden, k=config.kernel, layers=config.layers)


def parameter_counts():
    cell = 4 * (3 * 3 * 1 * 32 + 3 * 3 * 32 * 32 + 32)
    output = 3 * 3 * 32 * 1 + 1
    wide_cell = 4 * (3 * 3 * 1 * 64 + 3 * 3 * 64 * 64 + 64)
    fc_recurrent = 4 * 256 * 1024 + 4 * 256 * 256 + 2 * 4 * 256
    fc_output = 256 * 1024 + 1024
    expected = {"cell": cell, "output": output, "baseline_total": cell + output,
                "wide_cell": wide_cell, "wide_cell_ratio": wide_cell / cell,
                "flatten_recurrent": fc_recurrent, "flatten_output": fc_output,
                "flatten_total": fc_recurrent + fc_output}
    model = ConvLSTM()
    actual = {"cell": sum(p.numel() for p in model.cells[0].parameters()),
              "output": sum(p.numel() for p in model.out.parameters()),
              "baseline_total": sum(p.numel() for p in model.parameters()),
              "wide_cell": sum(p.numel() for p in ConvLSTMCell(1, 64).parameters()),
              "flatten_total": sum(p.numel() for p in FlattenLSTM().parameters())}
    for key, count in actual.items():
        if expected[key] != count:
            raise AssertionError((key, expected[key], count))
    return {"hand_calculated": expected, "code_counts": actual, "verified": True}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def evaluate(model, data, frames, micro_batch):
    model.eval()
    device = next(model.parameters()).device
    squared = absolute = count = 0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data), micro_batch):
            batch = torch.from_numpy(data[start:start + micro_batch]).to(device=device, dtype=torch.float32)
            pred = model(batch[:, :frames])
            error = pred - batch[:, frames, 0]
            squared += error.square().sum(dtype=torch.float64).item()
            absolute += error.abs().sum(dtype=torch.float64).item()
            count += error.numel()
            predictions.append(pred.cpu().numpy())
    return {"mse": squared / count, "mae": absolute / count}, np.concatenate(predictions)


def reference_metrics(data, frames):
    truth = data[:, frames, 0].astype(np.float32)
    previous = data[:, frames - 1, 0].astype(np.float32)
    def metrics(pred):
        error = pred - truth
        return {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}
    return {"all_black": metrics(np.zeros_like(truth)), "last_frame": metrics(previous)}


def curve(history, path, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="训练目标损失")
    axes[0].plot(epochs, [h["test_mse"] for h in history], "s-", label="测试 MSE")
    axes[0].plot(epochs, [h["validation_mse"] for h in history], "^-", label="验证 MSE")
    axes[1].plot(epochs, [h["test_mae"] for h in history], "o-", label="测试 MAE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title(f"{name}：训练损失与 MSE（L1 组训练损失为 MAE）", fontsize=10)
    axes[1].set_title("下一帧预测误差")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prediction_plot(data, predictions, frames, path, title):
    indices = [0, len(data) // 2, len(data) - 1]
    fig, axes = plt.subplots(3, frames + 2, figsize=(2 * (frames + 2), 6.5),
                             constrained_layout=True)
    for row, index in enumerate(indices):
        truth = data[index, frames, 0]
        mse = float(np.mean((predictions[index] - truth)**2))
        images = [*data[index, :frames, 0], truth, predictions[index]]
        labels = [f"输入 {i+1}" for i in range(frames)] + [f"真实 {frames+1}", "预测"]
        for col, (im, label) in enumerate(zip(images, labels)):
            axes[row, col].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"样本 {index}\nMSE={mse:.6f}")
    fig.suptitle(title + "；显示范围 [0,1]，MSE 使用未裁剪原始输出", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(name, config, data, output, micro_batch, threads, device="cpu"):
    train, test, val = data
    signature = {"config": asdict(config), "counts": [len(p) for p in data],
                 "micro_batch": micro_batch, "threads": threads, "device": device,
                 "torch_version": str(torch.__version__),
                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "data_sha256": [hashlib.sha256(p.tobytes()).hexdigest() for p in data],
                 "protocol_version": 1}
    result_path = output / f"{name}.json"
    checkpoint_path = output / f"{name}_partial.pt"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError(f"Configuration mismatch for {name}; use a new --output directory")
        if not (output / f"{name}.pt").exists():
            raise FileNotFoundError(f"Missing completed model for {name}")
        print(f"SKIP completed {name}", flush=True)
        return saved
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss() if config.loss == "mse" else nn.L1Loss()
    history, elapsed, train_elapsed = [], 0.0, 0.0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        history, elapsed = checkpoint["history"], checkpoint["elapsed"]
        train_elapsed = checkpoint["train_elapsed"]
    print(f"START {name} {asdict(config)} params={sum(p.numel() for p in model.parameters())}",
          flush=True)
    for epoch in range(len(history), config.epochs):
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train)).numpy()
        total_loss = 0.0
        for start in range(0, len(train), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            # Accumulate a true mean over the effective batch of 64, including its tail.
            for m in range(0, len(indices), micro_batch):
                batch = torch.from_numpy(train[indices[m:m + micro_batch]]).to(device=device, dtype=torch.float32)
                loss = loss_fn(model(batch[:, :config.input_frames]), batch[:, config.input_frames, 0])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {name}, epoch {epoch+1}")
                (loss * (len(batch) / len(indices))).backward()
                total_loss += loss.item() * len(batch)
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_elapsed += time.perf_counter() - epoch_start
        test_metrics, pred = evaluate(model, test, config.input_frames, micro_batch)
        val_metrics, _ = evaluate(model, val, config.input_frames, micro_batch)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed += epoch_seconds
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train),
                        "test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"],
                        "validation_mse": val_metrics["mse"],
                        "validation_mae": val_metrics["mae"], "seconds": epoch_seconds})
        print(f"{name} epoch {epoch+1}/{config.epochs} train={total_loss/len(train):.7f} "
              f"test_MSE={test_metrics['mse']:.7f} test_MAE={test_metrics['mae']:.7f} "
              f"val_MSE={val_metrics['mse']:.7f} seconds={epoch_seconds:.1f}", flush=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"signature": signature, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                    "history": history, "elapsed": elapsed, "train_elapsed": train_elapsed}, temporary)
        temporary.replace(checkpoint_path)
    test_metrics, pred = evaluate(model, test, config.input_frames, micro_batch)
    val_metrics, _ = evaluate(model, val, config.input_frames, micro_batch)
    curve(history, output / f"result_{name}.png", name)
    prediction_plot(test, pred, config.input_frames, output / f"prediction_{name}.png", name)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, output / f"{name}.pt")
    np.savez_compressed(output / f"predictions_{name}.npz", predictions=pred,
                        targets=test[:, config.input_frames, 0], sample_indices=np.arange(len(test)))
    result = {"name": name, "signature": signature, "history": history,
              "parameters": sum(p.numel() for p in model.parameters()),
              "test": test_metrics, "validation": val_metrics,
              "training_seconds": train_elapsed, "train_and_eval_seconds": elapsed,
              "references": reference_metrics(test, config.input_frames),
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(result_path, result)
    checkpoint_path.unlink(missing_ok=True)
    return result


def choose_combination(results):
    baseline = results["baseline"]["validation"]["mse"]
    config, selected = BASE, []
    for name, field in (("exp1", "layers"), ("exp2", "hidden"), ("exp3", "kernel"),
                        ("exp4", "input_frames"), ("exp6", "loss")):
        if results[name]["validation"]["mse"] < baseline:
            config = replace(config, **{field: getattr(EXPERIMENTS[name], field)})
            selected.append(name)
    return config, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "ablations", *EXPERIMENTS], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--micro-batch", type=int, default=8,
                        help="Memory-saving gradient accumulation; effective batch stays 64")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Tiny 1-epoch pipeline check, not final evidence")
    args = parser.parse_args()
    if not 1 <= args.micro_batch <= 64 or args.threads < 1:
        parser.error("micro-batch must be 1..64 and threads must be positive")
    if args.smoke and args.output.resolve() == (ROOT / "results").resolve():
        args.output = ROOT / "smoke_results"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f"Run lock exists: {lock}; remove only after confirming the old process stopped")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA GPU is available")
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = split_data(32, 12, 12) if args.smoke else split_data()
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "torch": str(torch.__version__), "numpy": np.__version__,
                       "matplotlib": matplotlib.__version__, "device": args.device,
                       "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                       "cuda_runtime": torch.version.cuda,
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "threads": args.threads, "micro_batch": args.micro_batch,
                       "smoke": args.smoke, "data_counts": [len(p) for p in data]}
        write_json(output / "environment.json", environment)
        write_json(output / "parameter_counts.json", parameter_counts())
        names = list(EXPERIMENTS) if args.suite in ("all", "ablations") else [args.suite]
        results = {}
        for name in names:
            config = replace(EXPERIMENTS[name], epochs=1) if args.smoke else EXPERIMENTS[name]
            results[name] = run(name, config, data, output, args.micro_batch, args.threads, args.device)
        if args.suite == "all":
            config, selected = choose_combination(results)
            if args.smoke:
                config = replace(config, epochs=1)
            write_json(output / "selection.json", {
                "rule": "Combine ConvLSTM ablations whose final validation MSE beats baseline; exclude architecture swap",
                "selected": selected, "candidate_config": asdict(config),
                "validation_seed": 43, "test_used_for_selection": False,
                "repeats": "identical seed 42; reproducibility check, not independent-seed uncertainty"})
            for index in range(1, 4):
                name = f"best_run{index}"
                results[name] = run(name, config, data, output, args.micro_batch, args.threads, args.device)
            # A combination is not guaranteed to beat its components. Select the final
            # recommended model by validation MSE and retain all candidate repeat results.
            best_name = min(results, key=lambda key: results[key]["validation"]["mse"])
            repeats = [results[f"best_run{i}"] for i in range(1, 4)]
            summary = {"experiments": results, "recommended_model": best_name,
                       "combination_mean": {key: float(np.mean([r["test"][key] for r in repeats]))
                                            for key in ("mse", "mae")},
                       "combination_mean_seconds": float(np.mean([r["train_and_eval_seconds"] for r in repeats])),
                       "combination_test_mse_std": float(np.std([r["test"]["mse"] for r in repeats]))}
            write_json(output / "results.json", summary)
            saved = torch.load(output / f"{best_name}.pt", map_location="cpu", weights_only=True)
            model = build_model(Config(**saved["config"])).to(args.device)
            model.load_state_dict(saved["state_dict"])
            frames = saved["config"]["input_frames"]
            _, predictions = evaluate(model, data[1], frames, args.micro_batch)
            prediction_plot(data[1], predictions, frames, output / "result_recommended.png",
                            f"验证集选择：{best_name}（固定样本 0 / 100 / 199）")
            saved_combo = np.load(output / "predictions_best_run1.npz")
            prediction_plot(data[1], saved_combo["predictions"], config.input_frames,
                            output / "result_pred.png", "组合配置第 1 次复测（固定样本 0 / 100 / 199）")
            print(json.dumps({k: v for k, v in summary.items() if k != "experiments"}, indent=2), flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
