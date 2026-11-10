"""Offline baseline utilities for the robot VLM pilot study."""

from .contracts import (
    EXPECTED_LABELS,
    BaselineContractError,
    canonical_detector_label,
    parse_strict_label,
    validate_baseline_policy,
    validate_inference_jobs,
)
from .metrics import classification_metrics
from .b2 import (
    load_frozen_b2_protocol,
    parse_b2_a_label,
    rank_candidate_scores,
    validate_b2_jobs,
)

__all__ = [
    "EXPECTED_LABELS",
    "BaselineContractError",
    "canonical_detector_label",
    "parse_strict_label",
    "validate_baseline_policy",
    "validate_inference_jobs",
    "classification_metrics",
    "load_frozen_b2_protocol",
    "parse_b2_a_label",
    "rank_candidate_scores",
    "validate_b2_jobs",
]
