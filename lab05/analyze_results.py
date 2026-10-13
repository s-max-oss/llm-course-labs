"""Verify original files, rescore saved answers, and plot actual training records."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys

from hw5_dpo import QUESTIONS, EXTRA_PAIRS, judge, scalar_dpo

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DERIVED = ROOT / "analysis"
STAGES = ["baseline_bf16", "baseline_nf4", "sft", "dpo_original_b01",
          "dpo_augmented_b01", "dpo_augmented_b005", "dpo_augmented_b05"]
SHORT = {"baseline_bf16": "Base BF16", "baseline_nf4": "Base NF4", "sft": "SFT",
         "dpo_original_b01": "DPO 12 / b=.1", "dpo_augmented_b01": "DPO 17 / b=.1",
         "dpo_augmented_b005": "DPO 17 / b=.05", "dpo_augmented_b05": "DPO 17 / b=.5"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_grade(question, prediction, auto_correct):
    """Explicit, narrow review rules; preserve the automatic result separately.

    These assess the requested answer, not every incidental claim in the reply.
    Added text can still be unfaithful even if the requested attribute is right.
    """
    if question["type"] == "exist" and prediction.strip().rstrip("。.!！") == "没":
        return question["gold"] == "没有", "单字口语‘没’在有没有问句下是明确否定；表达质量另评"
    query = question.get("q", question.get("prompt", ""))
    if question["type"] == "attr" and "绿色" in query and "green square" in prediction.lower():
        return question["gold"] == "方形", "明确把绿色对象绑定为square；附加圆形/位置描述的忠实度另评"
    return auto_correct, "宽松答案匹配经逐题来源复核；附加断言另作忠实度判断"


def analyze():
    manifest = read(RESULTS / "manifest.json")
    for name, expected in manifest.items():
        assert sha(RESULTS / name) == expected, name
    audits, deltas, semantic_changes, pool_stats = {}, [], [], {}
    for stage in STAGES:
        stage_data = {}
        for filename, questions in [("evaluation.json", QUESTIONS), ("added_evaluation.json", EXTRA_PAIRS)]:
            data = read(RESULTS / stage / filename)
            assert [r["id"] for r in data["rows"]] == [q["id"] for q in questions]
            rows = []
            for row, question in zip(data["rows"], questions):
                assert row["gold"] == question["gold"]
                correct = bool(judge(question, row["prediction"]))
                semantic_correct, reason = semantic_grade(question, row["prediction"], correct)
                rows.append(dict(row, original_auto_correct=row["correct"], correct=correct,
                                 semantic_correct=semantic_correct, semantic_review_reason=reason))
                if semantic_correct != correct:
                    semantic_changes.append({"stage": stage, "file": filename, "id": row["id"],
                                             "auto_correct": correct, "semantic_correct": semantic_correct, "reason": reason})
                if correct != row["correct"]:
                    deltas.append({"stage": stage, "file": filename, "id": row["id"],
                                   "original": row["correct"], "corrected": correct, "prediction": row["prediction"]})
            metrics, semantic_metrics = {}, {}
            for tp in ["count", "attr", "exist", "overall"]:
                subset = rows if tp == "overall" else [r for r in rows if r["type"] == tp]
                if subset:
                    metrics[tp] = {"correct": sum(r["correct"] for r in subset), "n": len(subset),
                                   "accuracy": sum(r["correct"] for r in subset)/len(subset)}
                    semantic_metrics[tp] = {"correct": sum(r["semantic_correct"] for r in subset), "n": len(subset),
                                            "accuracy": sum(r["semantic_correct"] for r in subset)/len(subset)}
            stage_data["main" if filename == "evaluation.json" else "added"] = {"rows": rows, "metrics": metrics, "semantic_metrics": semantic_metrics}
        audits[stage] = stage_data
        run = read(RESULTS / stage / "run.json")
        if stage.startswith("dpo") or stage == "sft":
            logs = read(RESULTS / stage / "training.json")
            assert len(logs) == (60 if stage == "sft" else 40)
            assert [r["step"] for r in logs] == list(range(1, len(logs)+1))
            assert all(math.isfinite(r["loss"]) and r["loss"] >= 0 for r in logs)
            assert run["nf4"] and run["rank"] == 8
            assert run["parameters"]["trainable_parameters"] == 921600
            assert run["parameters"]["total_parameters"] == 257406528
            if stage.startswith("dpo"):
                assert run["parent_sft_adapter_sha256"] == sha(RESULTS / "sft/adapter/adapter_model.safetensors")
                for row in logs:
                    expected = scalar_dpo(row["policy_chosen"], row["reference_chosen"],
                                          row["policy_rejected"], row["reference_rejected"], run["beta"])
                    assert abs(expected["margin"] - row["margin"]) < 1e-4
                    assert abs(expected["loss"] - row["loss"]) < 1e-4
                before, after = [read(RESULTS / stage / ("preference_"+tag+".json")) for tag in ["before", "after"]]
                assert len(before) == len(after) == (12 if stage == "dpo_original_b01" else 17)
                for a, b in zip(before, after):
                    assert a["id"] == b["id"]
                    assert a["reference_chosen"] == b["reference_chosen"]
                    assert a["reference_rejected"] == b["reference_rejected"]
                stats = {}
                for label, rows in [("before", before), ("after", after)]:
                    vals = [scalar_dpo(r["policy_chosen"], r["reference_chosen"], r["policy_rejected"], r["reference_rejected"], run["beta"]) for r in rows]
                    stats[label] = {"loss": statistics.mean(r["loss"] for r in vals),
                                    "margin": statistics.mean(r["margin"] for r in vals),
                                    "positive_margin_pairs": sum(r["margin"] > 0 for r in vals),
                                    "n": len(vals), "policy_chosen": statistics.mean(r["policy_chosen"] for r in rows),
                                    "policy_rejected": statistics.mean(r["policy_rejected"] for r in rows)}
                pool_stats[stage] = stats
    source_at_run = ROOT / "provenance/source_at_run.py"
    assert read(RESULTS / "environment.json")["source_sha256"] == sha(source_at_run)
    output = {"manifest_files_verified": len(manifest), "stage_scores": audits,
              "rescoring_changes": deltas, "full_preference_pool": pool_stats,
              "semantic_review_changes": semantic_changes,
              "semantic_review_method": "Codex source-grounded review, independently checked by a second Codex reviewer; not student blind scoring",
              "current_evaluator_source_sha256": sha(ROOT / "hw5_dpo.py"),
              "original_training_source_sha256": read(RESULTS / "environment.json")["source_sha256"],
              "primary_stages": ["baseline_nf4", "sft", "dpo_augmented_b01"],
              "limits": ["15 evaluation questions reuse SFT images/questions; no generalization claim",
                         "5 added questions are DPO training examples in augmented runs, not held-out",
                         "Rule matching is separate from image-grounded faithfulness review",
                         "Single seed, no statistical uncertainty estimate"]}
    dump(DERIVED / "summary.json", output)
    return output


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    stages = data["primary_stages"]
    types = ["count", "attr", "exist", "overall"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (stage, color) in enumerate(zip(stages, ["#7E8EA4", "#3D739D", "#43A086"])):
        values = [data["stage_scores"][stage]["main"]["semantic_metrics"][t]["accuracy"]*100 for t in types]
        bars = ax.bar(x+(i-1)*.23, values, .23, label=SHORT[stage], color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:.0f}%", ha="center", fontsize=11)
    ax.set(xticks=x, xticklabels=types, ylim=(0, 112), ylabel="Answer accuracy (%)",
           title="Three stages on the same 15 questions")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(figures / "accuracy.png", dpi=180); plt.close(fig)

    fig, grid = plt.subplots(2, 2, figsize=(8.5, 6.0))
    axes = list(grid.flat)
    sft = read(RESULTS / "sft/training.json")
    axes[0].plot([r["step"] for r in sft], [r["loss"] for r in sft], color="#3D739D", lw=1.2)
    axes[0].set(title="SFT: 60 steps", xlabel="Step", ylabel="Answer-token loss")
    for stage, color in [("dpo_original_b01", "#7E8EA4"), ("dpo_augmented_b01", "#43A086")]:
        logs = read(RESULTS / stage / "training.json")
        for ax, key in zip(axes[1:3], ["loss", "margin"]):
            ax.plot([r["step"] for r in logs], [r[key] for r in logs], color=color, lw=1.2, label=SHORT[stage])
    axes[1].set(title="DPO per-pair loss", xlabel="Step", ylabel="Loss")
    axes[2].set(title="DPO per-pair margin", xlabel="Step", ylabel="Scaled log ratio")
    axes[2].axhline(0, color="gray", lw=.6); axes[2].legend(fontsize=10, frameon=False)
    pool=data["full_preference_pool"]["dpo_augmented_b01"]
    xx=np.arange(2)
    axes[3].bar(xx-.17,[pool["before"]["policy_chosen"],pool["before"]["policy_rejected"]],.34,label="Before",color="#7E8EA4")
    axes[3].bar(xx+.17,[pool["after"]["policy_chosen"],pool["after"]["policy_rejected"]],.34,label="After",color="#43A086")
    axes[3].set(xticks=xx,xticklabels=["Chosen","Rejected"],ylabel="Mean sequence logp",title="17-pair pool / beta=.1")
    axes[3].legend(fontsize=10,frameon=False)
    fig.tight_layout(); fig.savefig(figures / "training_curves.png", dpi=180); plt.close(fig)

    scan = ["dpo_augmented_b005", "dpo_augmented_b01", "dpo_augmented_b05"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    axes[0].bar(["0.05", "0.1", "0.5"], [data["stage_scores"][s]["main"]["semantic_metrics"]["overall"]["accuracy"]*100 for s in scan], color=["#7E8EA4", "#43A086", "#3D739D"])
    axes[0].set(xlabel="Beta", ylabel="Answer accuracy (%)", ylim=(0, 100), title="Task accuracy / 17 pairs")
    for s, color in zip(scan, ["#7E8EA4", "#43A086", "#3D739D"]):
        logs = read(RESULTS / s / "training.json")
        axes[1].plot([r["step"] for r in logs], [r["loss"] for r in logs], lw=1.1, label=SHORT[s], color=color)
    axes[1].set(xlabel="Step", ylabel="DPO loss", title="DPO loss vs step")
    axes[1].legend(fontsize=10, frameon=False)
    fig.tight_layout(); fig.savefig(figures / "beta_scan.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(9.5, 6.5))
    for ax, img_id in zip(axes.flat, sorted({q["img"] for q in QUESTIONS} | {p["img"] for p in EXTRA_PAIRS})):
        ax.imshow(Image.open(ROOT / "data" / (img_id + ".png")))
        ax.set_title(img_id, fontsize=14); ax.axis("off")
    fig.tight_layout(); fig.savefig(figures / "image_gallery.png", dpi=180); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(8.0,3.0))
    for ax,img_id,title in zip(axes,["img06","img05","img12"],
                              ["P04 / img06: 4 yellow circles","P07 / img05: red circle","P12 / img12: no red circle"]):
        ax.imshow(Image.open(ROOT/"data"/(img_id+".png")))
        ax.set_title(title,fontsize=12)
        ax.axis("off")
    fig.tight_layout();fig.savefig(figures/"sample_images.png",dpi=200);plt.close(fig)
    line = next(s for s in (RESULTS / "sft.log").read_text(encoding="utf-8").splitlines() if s.startswith("trainable params:"))
    im = Image.new("RGB", (1100, 190), "white")
    draw = ImageDraw.Draw(im)
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 30) if font_path.exists() else ImageFont.load_default()
    draw.text((25, 20), "Actual print_trainable_parameters output / sft.log", fill="black", font=font)
    parts=line.split(" || ",1)
    draw.text((25, 75), parts[0], fill="black", font=font)
    draw.text((25, 125), parts[1], fill="black", font=font)
    im.save(figures / "parameters.png")
    print("Figures saved; verified", data["manifest_files_verified"], "original files; rescoring changes", len(data["rescoring_changes"]))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = analyze()
    if "--verify-only" not in sys.argv:
        plot(summary)
    for s in STAGES:
        print(s, summary["stage_scores"][s]["main"]["semantic_metrics"])
