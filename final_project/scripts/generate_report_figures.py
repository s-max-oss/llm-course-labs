#!/usr/bin/env python3
"""Generate report figures directly from sealed RF-DETR and B1/B2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "experiments" / "pilot_v1" / "results"
FIGURE_ROOT = PROJECT_ROOT / "figures"
CLASS_ORDER = [
    "danger_red_sphere",
    "distractor_red_box",
    "distractor_green_sphere",
    "background_or_unknown",
]
CLASS_LABELS = ["Red ball\n(danger)", "Red box", "Green ball", "Background"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_vector(metrics: dict) -> list[float]:
    return [
        metrics["prediction_coverage"],
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["danger_binary"]["recall"],
        metrics["per_class"]["background_or_unknown"]["recall"],
    ]


def save_metric_comparison(comparison: dict) -> Path:
    b1 = metric_vector(comparison["b1_strict_generation"])
    b2 = metric_vector(comparison["b2_fixed_candidate_scoring"])
    labels = ["Coverage", "Accuracy", "Macro-F1", "Danger recall", "Background recall"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar(x - width / 2, b1, width, label="B1 strict generation", color="#8c8c8c")
    bars = axis.bar(x + width / 2, b2, width, label="B2 fixed scoring", color="#2878b5")
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score")
    axis.set_xticks(x, labels)
    axis.set_title("Same 251-candidate locked exploratory subset (oracle crops)")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = FIGURE_ROOT / "b1_b2_locked_metrics.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_confusion_matrix(metrics: dict) -> Path:
    matrix = np.array(
        [
            [metrics["confusion_matrix"][truth][prediction] for prediction in CLASS_ORDER]
            for truth in CLASS_ORDER
        ],
        dtype=int,
    )
    fig, axis = plt.subplots(figsize=(6.8, 5.8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value > matrix.max() / 2 else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=color, fontsize=12)
    axis.set_xticks(np.arange(4), CLASS_LABELS)
    axis.set_yticks(np.arange(4), CLASS_LABELS)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("B2 locked confusion matrix\n251-candidate exploratory oracle-crop subset")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    output = FIGURE_ROOT / "b2_locked_confusion_matrix.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_development_locked_recall(development: dict, locked: dict) -> Path:
    development_recall = [development["per_class"][name]["recall"] for name in CLASS_ORDER]
    locked_recall = [locked["per_class"][name]["recall"] for name in CLASS_ORDER]
    x = np.arange(4)
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.bar(x - width / 2, development_recall, width, label="Development (193)", color="#9ac9db")
    axis.bar(x + width / 2, locked_recall, width, label="Locked exploratory (251)", color="#2878b5")
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Recall")
    axis.set_xticks(x, CLASS_LABELS)
    axis.set_title("B2 fixed-candidate scoring: per-class recall")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = FIGURE_ROOT / "b2_development_locked_recall.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_rfdetr_metrics(report: dict) -> Path:
    metrics = report["metrics"]
    class_names = [
        "danger_red_sphere",
        "distractor_red_box",
        "distractor_green_sphere",
    ]
    labels = ["Red ball\n(danger)", "Red box", "Green ball"]
    precision = [metrics["per_target_class"][name]["precision"] for name in class_names]
    recall = [metrics["per_target_class"][name]["recall"] for name in class_names]
    f1 = [metrics["per_target_class"][name]["f1"] for name in class_names]
    x = np.arange(3)
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - width, precision, width, label="Precision", color="#9ac9db")
    axis.bar(x, recall, width, label="Recall", color="#2878b5")
    axis.bar(x + width, f1, width, label="F1", color="#f28e2b")
    macro = metrics["three_target_class_macro_f1"]
    axis.axhline(macro, color="#333333", linestyle="--", linewidth=1.2, label=f"Macro-F1 {macro:.3f}")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_xticks(x, labels)
    axis.set_title("RF-DETR fixed-checkpoint in-domain exploratory pilot (box IoU = 0.5)")
    axis.legend(frameon=False, loc="lower left")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = FIGURE_ROOT / "rfdetr_pilot_metrics.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_cascade_comparison(metrics: dict) -> Path:
    rf_only = metrics["rf_only"]
    cascade = metrics["primary_safety_preserving_cascade"]
    labels = ["Macro-F1", "Danger recall", "Target FP / 20"]
    rf_values = [
        rf_only["three_target_class_macro_f1"],
        rf_only["danger_recall"],
        rf_only["target_false_positive_total"] / 20.0,
    ]
    cascade_values = [
        cascade["three_target_class_macro_f1"],
        cascade["danger_recall"],
        cascade["target_false_positive_total"] / 20.0,
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    bars_rf = axis.bar(x - width / 2, rf_values, width, label="RF-DETR only", color="#8c8c8c")
    bars_cascade = axis.bar(
        x + width / 2,
        cascade_values,
        width,
        label="RF-DETR + B2 safety fusion",
        color="#2878b5",
    )
    axis.bar_label(bars_rf, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(bars_cascade, fmt="%.3f", padding=3, fontsize=9)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score (FP scaled by 1/20)")
    axis.set_xticks(x, labels)
    axis.set_title("Development-only real-proposal cascade: no metric change")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = FIGURE_ROOT / "cascade_development_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    comparison = load_json(RESULT_ROOT / "smolvlm_b1_b2_locked_comparison.json")
    development = load_json(RESULT_ROOT / "b2_b_development_gate.json")["metrics"]
    locked = load_json(RESULT_ROOT / "b2_b_locked_evaluation_metrics.json")["metrics"]
    rfdetr = load_json(RESULT_ROOT / "rfdetr_pooled_metrics.json")
    cascade = load_json(RESULT_ROOT / "cascade_feasibility_development_metrics.json")
    outputs = [
        save_rfdetr_metrics(rfdetr),
        save_metric_comparison(comparison),
        save_confusion_matrix(locked),
        save_development_locked_recall(development, locked),
        save_cascade_comparison(cascade),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
