#!/usr/bin/env python3
"""Summarize immutable experiment outputs and produce the required self-drawn plot."""
import argparse
import hashlib
import re
from pathlib import Path

import hw6_rsi_localize as h


ROOT = Path(__file__).resolve().parent
NAMES = ("baseline", "sft_eval", "sft_tool_eval", "rsi1_eval", "rsi2_eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest_file = ROOT / 'provenance' / 'manifest.json'
    if args.verify_only:
        manifest = h.read_json(manifest_file)
        for relative, expected in manifest['sha256'].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Checksum mismatch: {relative}')
        print(f"verified {len(manifest['sha256'])} files")
        return
    results = {name: h.read_json(ROOT / "results" / f"{name}.json") for name in NAMES}
    memory = h.read_json(ROOT / "rsi_memory.json")
    formal_run = h.read_json(ROOT / 'provenance' / 'formal_run.json')
    pool = h.read_json(ROOT / "sft_data.json")
    questions = h.load_questions(ROOT)
    eval_ids = {q["img"] for q in questions["eval"]}
    assert not eval_ids.intersection(p["img"] for p in pool), "Held-out image entered training pool"
    assert len(questions["eval"]) == 15
    assert len(memory["rounds"]) == 2
    for result in results.values():
        assert len(result["rows"]) == 15
        summary = result["summary"]
        assert [summary[k]["n"] for k in ("loc", "count", "exist")] == [8, 4, 3]
    corrected = results["sft_tool_eval"]["rows"]
    raw = results["sft_eval"]["rows"]
    assert all(a["raw"] == b["raw"] for a, b in zip(corrected, raw)), "Greedy SFT prediction changed"
    transitions = {"wrong_to_right": sum(not r["raw_correct"] and r["correct"] for r in corrected),
                   "right_to_wrong": sum(r["raw_correct"] and not r["correct"] for r in corrected),
                   "self_attempts": sum(r["self_check"] is not None for r in corrected),
                   "tool_attempts": sum(r["tool_check"] is not None for r in corrected)}
    samples = {
        "direct_correct": next((r for r in raw if r["correct"]), None),
        "self_corrected": next((r for r in corrected if r["self_check"] and
                               r["self_check"]["verification"]["passed"] and not r["raw_correct"]), None),
        "tool_corrected": next((r for r in reversed(corrected) if r["tool_check"] and
                               r["tool_check"]["verification"]["passed"] and not r["raw_correct"]), None),
        "still_failed": next((r for r in corrected if not r["correct"]), None),
    }
    repetition = {name: [r['id'] for r in results[name]['rows'] if r['type'] == 'count'
                        and re.fullmatch(r'(?:\d+[.。]?\s*){2,}', r['final'].strip())]
                  for name in NAMES}
    summary = {"stages": {name: results[name]["summary"] for name in NAMES},
               'repetitive_count_outputs': repetition,
               'without_repetitive_count_correct': {name: sum(r['correct'] and r['id'] not in repetition[name]
                                                              for r in results[name]['rows']) for name in NAMES},
               "seconds": {name: results[name]["seconds"] for name in NAMES},
               "transitions": transitions, "rounds": memory["rounds"],
               "final_pool": len(pool), "examples": samples,
               "bootstrap_types": sorted({p['type'] for p in pool if p['source'].startswith('verified-rsi-')}),
               "train_eval_overlap": 0,
               "limitations": ["one synthetic seed", "15 fixed held-out images",
                               "synthetic pixel verifier", "only two RSI rounds"]}
    h.write_json(ROOT / "analysis" / "summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Baseline", "SFT", "SFT+tool", "RSI-1", "RSI-2"]
    loc = [results[name]["summary"]["loc"]["correct"] / 8 for name in NAMES]
    total = [results[name]["summary"]["total"]["correct"] / 15 for name in NAMES]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=180)
    ax.plot(names, loc, "o-", color="#1769AA", label="Localization /8", linewidth=2)
    ax.plot(names, total, "s--", color="#CB6D24", label="All questions /15", linewidth=1.8)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    target = ROOT / "figures" / "accuracy.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target)
    plt.close(fig)
    immutable = [ROOT / name for name in ('hw6_rsi_localize.py', 'run_hpc.slurm',
                 'rsi_memory.json', 'sft_seed.json', 'sft_data.json', 'sft_data_round1.json', 'sft_data_round2.json')]
    immutable += sorted((ROOT / 'hw6_imgs').rglob('*.*'))
    immutable += sorted(p for p in (ROOT / 'results').rglob('*') if p.is_file())
    h.write_json(manifest_file, {'formal_job': formal_run['job_id'], 'source_at_run': 'provenance/source_at_run.py',
                 'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in immutable if p.is_file()}})
    print("stage scores:")
    for name in NAMES:
        s = results[name]["summary"]
        print(name, s["loc"]["correct"], s["count"]["correct"],
              s["exist"]["correct"], s["total"]["correct"], f"IoU={s['loc_mean_iou']:.4f}")
    print("transitions:", transitions, "pool:", len(pool))


if __name__ == "__main__":
    main()
