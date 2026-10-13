"""One controlled GPU run: common base/SFT, 12-vs-17 pairs, beta scan."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model")
    ap.add_argument("--results", default="results")
    ap.add_argument("--sft_steps", type=int, default=60)
    ap.add_argument("--dpo_steps", type=int, default=40)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    # Root-relative paths have the same meaning regardless of caller cwd.
    results = (root / args.results).resolve()
    model = str((root / args.model).resolve())
    results.mkdir(parents=True, exist_ok=True)
    common = ["--model", model, "--offline", "--device", "cuda", "--require_nf4", "--images", str(root / "data")]
    commands = []
    def run(options, name):
        cmd = [sys.executable, str(root / "hw5_dpo.py")] + options
        start = time.perf_counter()
        print("RUN", name, cmd, flush=True)
        with (results / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
        commands.append({"name": name, "argv": cmd, "seconds": time.perf_counter() - start})
        print("DONE", name, commands[-1]["seconds"], flush=True)
        (results / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "unittest", "test_hw5_dpo", "-v"], check=True, cwd=root)
    run(["--mode", "make_data"], "make_data")
    run(["--mode", "environment", "--output", str(results)], "environment")
    run(common + ["--mode", "infer", "--full_precision", "--stage", "baseline_bf16", "--output", str(results / "baseline_bf16")], "baseline_bf16")
    run(common + ["--mode", "infer", "--stage", "baseline_nf4", "--output", str(results / "baseline_nf4")], "baseline_nf4")
    run(common + ["--mode", "sft", "--steps", str(args.sft_steps), "--output", str(results / "sft")], "sft")
    sft = str(results / "sft/adapter")
    for tag, pairs, beta in [("dpo_original_b01", "original", "0.1"),
                              ("dpo_augmented_b01", "augmented", "0.1"),
                              ("dpo_augmented_b005", "augmented", "0.05"),
                              ("dpo_augmented_b05", "augmented", "0.5")]:
        run(common + ["--mode", "dpo", "--steps", str(args.dpo_steps), "--sft_adapter", sft,
                      "--pairs", pairs, "--beta", beta, "--output", str(results / tag)], tag)
    checks = []
    from hw5_dpo import scalar_dpo
    for numbers in [(1.2, .8, .9, .6, .1), (1.5, 1., .2, .6, .1), (-4., -3., -2., -3., .1)]:
        checks.append(scalar_dpo(*numbers))
    (results / "dpo_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    (results / "manifest.json").write_text(json.dumps({p.relative_to(results).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(results.rglob("*")) if p.is_file() and p.name != "manifest.json"}, indent=2)+"\n", encoding="utf-8")
    print("SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
