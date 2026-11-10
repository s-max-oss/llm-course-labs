"""Deterministic classification metrics without external ML dependencies."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .contracts import EXPECTED_LABELS, percentile


INVALID_LABEL = "__invalid_or_error__"


def classification_metrics(
    pairs: Iterable[tuple[str, str | None]],
    latencies_ms: Iterable[float] = (),
) -> dict[str, Any]:
    items = [(truth, prediction if prediction in EXPECTED_LABELS else INVALID_LABEL) for truth, prediction in pairs]
    support = Counter(truth for truth, _ in items)
    predicted = Counter(prediction for _, prediction in items)
    confusion_labels = list(EXPECTED_LABELS) + [INVALID_LABEL]
    confusion = {
        truth: {prediction: 0 for prediction in confusion_labels}
        for truth in EXPECTED_LABELS
    }
    for truth, prediction in items:
        confusion[truth][prediction] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for label in EXPECTED_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in EXPECTED_LABELS if other != label
        )
        false_negative = sum(
            count for prediction, count in confusion[label].items() if prediction != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0 if support[label] else None
        )
        if f1 is not None:
            f1_values.append(f1)
        per_class[label] = {
            "support": support[label],
            "predicted": predicted[label],
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = len(items)
    correct = sum(confusion[label][label] for label in EXPECTED_LABELS)
    all_classes_supported = all(support[label] > 0 for label in EXPECTED_LABELS)
    danger = "danger_red_sphere"
    danger_tp = confusion[danger][danger]
    danger_fp = sum(confusion[label][danger] for label in EXPECTED_LABELS if label != danger)
    danger_fn = sum(
        count for prediction, count in confusion[danger].items() if prediction != danger
    )
    danger_precision = danger_tp / (danger_tp + danger_fp) if danger_tp + danger_fp else None
    danger_recall = danger_tp / (danger_tp + danger_fn) if danger_tp + danger_fn else None
    danger_f1 = (
        2 * danger_precision * danger_recall / (danger_precision + danger_recall)
        if danger_precision is not None
        and danger_recall is not None
        and danger_precision + danger_recall
        else 0.0
    )
    latencies = [float(value) for value in latencies_ms]
    return {
        "sample_count": total,
        "valid_prediction_count": total - predicted[INVALID_LABEL],
        "invalid_or_error_count": predicted[INVALID_LABEL],
        "prediction_coverage": (total - predicted[INVALID_LABEL]) / total if total else None,
        "accuracy": correct / total if total else None,
        "all_expected_classes_supported": all_classes_supported,
        "macro_f1": sum(f1_values) / len(EXPECTED_LABELS) if all_classes_supported else None,
        "per_class": per_class,
        "danger_binary": {
            "true_positive": danger_tp,
            "false_positive": danger_fp,
            "false_negative": danger_fn,
            "precision": danger_precision,
            "recall": danger_recall,
            "f1": danger_f1,
        },
        "confusion_matrix": confusion,
        "latency_ms": {
            "count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }
