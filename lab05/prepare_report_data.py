"""Record user-confirmed sample scores and real CLI hand-check output."""
from pathlib import Path
import json
import subprocess
import sys
from hw5_dpo import sha256

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    summary = read(ROOT / "analysis/summary.json")
    scores = {"baseline_nf4": [2, 1, 3], "sft": [2, 2, 5], "dpo_augmented_b01": [2, 2, 3]}
    truths = {"P04": "img06有4个黄色圆形", "P07": "img05红色对象为圆形，另有2个绿色方形",
              "P12": "img12有红、蓝方形各1，没有红色圆形"}
    reasons = {("baseline_nf4", "P04"): "计数2而非4，主要事实错误",
               ("sft", "P04"): "计数2而非4，主要事实错误",
               ("dpo_augmented_b01", "P04"): "计数2而非4，主要事实错误",
               ("baseline_nf4", "P07"): "将几何图描述为GUI截图，场景幻觉",
               ("sft", "P07"): "将红圆答为三角，形状错误",
               ("dpo_augmented_b01", "P07"): "三字没有给出所问形状，答复质量低",
               ("baseline_nf4", "P12"): "蓝方事实正确，但Yes错误回答红圆存在性",
               ("sft", "P12"): "完整否定与图中事实一致",
               ("dpo_augmented_b01", "P12"): "口语没是正确否定；学生认可按表达完整性扣分，不认定为事实方向错误"}
    rows = []
    for stage, values in scores.items():
        for qid, score in zip(["P04", "P07", "P12"], values):
            row = next(r for r in summary["stage_scores"][stage]["main"]["rows"] if r["id"] == qid)
            rows.append({"stage": stage, "id": qid, "image": row["img"], "question": row["question"],
                         "prediction": row["prediction"], "ground_truth": truths[qid],
                         "ai_proposed_score": score, "human_confirmed_score": score,
                         "reason": reasons[stage, qid], "source_file_sha256": sha256(ROOT / "results" / stage / "evaluation.json")})
    comparisons = []
    for a, b in [("sft", "baseline_nf4"), ("dpo_augmented_b01", "sft"), ("dpo_augmented_b01", "baseline_nf4")]:
        diffs = [x-y for x,y in zip(scores[a], scores[b])]
        comparisons.append({"candidate": a, "reference": b, "wins": sum(d>0 for d in diffs),
                            "ties": sum(d==0 for d in diffs), "losses": sum(d<0 for d in diffs),
                            "n": 3, "win_rate_including_ties": sum(d>0 for d in diffs)/3})
    dump(ROOT / "report/human_scores.json", {"confirmed_by": "孙驰", "confirmed_on": "2026-10-14",
         "confirmation_basis": "用户认可以上九例评分和依据", "human_scores_confirmed": True,
         "scoring_method": "user_confirmation_of_ai_suggested_faithfulness_and_expression_scores",
         "independent_blind_review_performed": False,
         "scale": "5忠实且表达完整；4轻微问题；3部分事实正确或表达不足；2主要答案错误/残缺；1场景编造",
         "scope": "3道题各由3阶段回答，共9份回答；同时考察忠实度与表达质量",
         "answers": rows, "paired_comparisons_derived_from_scores": comparisons})
    checks = []
    for label, numbers in [("指南给定例", "1.2,0.8,0.9,0.6,0.1"),
                           ("思考题2", "1.5,1.0,0.2,0.6,0.1"),
                           ("自选A", "-4,-3,-2,-3,0.1"),
                           ("自选B", "-2,-3,-4,-3,0.5")]:
        cmd = [sys.executable, str(ROOT / "hw5_dpo.py"), "--dpo_check=" + numbers]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        checks.append({"label": label, "arguments": numbers, "command": cmd, "output": json.loads(result.stdout)})
    dump(ROOT / "analysis/dpo_hand_checks.json", checks)
    print("Nine user-confirmed scores and four actual CLI DPO checks recorded.")


if __name__ == "__main__":
    main()
