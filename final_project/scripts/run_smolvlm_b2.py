#!/usr/bin/env python3
"""Run frozen exploratory SmolVLM B2 variants on label-free jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.b2 import (  # noqa: E402
    B2_JOBS_MANIFEST_SHA256,
    B2_QUEUE_ARTIFACTS,
    B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
    load_frozen_b2_protocol,
    parse_b2_a_label,
    rank_candidate_scores,
    validate_b2_jobs,
    validate_development_gate_report,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    resolve_image_path,
    sha256_file,
    validate_baseline_policy,
    write_jsonl,
)


VARIANT_KEYS = {
    "b2_a": "b2_a_strict_natural_generation",
    "b2_b": "b2_b_fixed_candidate_scoring",
}


def crop_for_job(image: Image.Image, job: dict[str, Any]) -> Image.Image:
    region = job["target_region_xyxy_pixels"]
    if region is None:
        return image.copy()
    x1, y1, x2, y2 = (float(value) for value in region)
    left = max(0, min(image.width, math.floor(x1)))
    top = max(0, min(image.height, math.floor(y1)))
    right = max(0, min(image.width, math.ceil(x2)))
    bottom = max(0, min(image.height, math.ceil(y2)))
    if right <= left or bottom <= top:
        raise BaselineContractError(f"{job['job_id']}: crop has no pixels")
    return image.crop((left, top, right, bottom))


def verify_local_model(model_ref: str, baseline: dict[str, Any]) -> Path:
    root = Path(model_ref).expanduser().resolve()
    if not root.is_dir():
        raise BaselineContractError(f"SmolVLM model path is not a local directory: {root}")
    expected = set(baseline["model_files"])
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise BaselineContractError(
            "SmolVLM model directory differs from manifest: "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )
    for name, digest in baseline["model_files"].items():
        if sha256_file(root / name) != digest:
            raise BaselineContractError(f"SmolVLM model file hash mismatch: {root / name}")
    return root


def load_model(model_ref: str, longest_edge: int):
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as AutoModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModel
    except ImportError as exc:
        raise BaselineContractError(
            "transformers/torch are unavailable; install requirements-baselines.txt"
        ) from exc
    processor = AutoProcessor.from_pretrained(model_ref, local_files_only=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": longest_edge}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False
    model = AutoModel.from_pretrained(
        model_ref,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    target = torch.device("cpu")
    model.to(target)
    model.eval()
    return processor, model, target


def build_prompt(processor: Any, prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True,
    )


def generate_response(
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    formatted = build_prompt(processor, prompt)
    batch = processor(text=[formatted], images=[image], return_tensors="pt")
    batch = {key: value.to(device) for key, value in dict(batch).items()}
    prefix = batch["input_ids"].shape[1]
    with torch.inference_mode():
        tokens = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(tokens[0, prefix:], skip_special_tokens=True).strip()


def fixed_candidate_scores(
    processor: Any,
    model: Any,
    device: Any,
    image: Image.Image,
    prompt: str,
    completions: list[str],
) -> dict[str, float]:
    """Return mean teacher-forced continuation log probabilities, excluding EOS."""

    import torch

    formatted = build_prompt(processor, prompt)
    base = processor(text=[formatted], images=[image], return_tensors="pt")
    base = {key: value.to(device) for key, value in dict(base).items()}
    prefix_length = int(base["input_ids"].shape[1])
    scores: dict[str, float] = {}
    for completion in completions:
        token_ids = processor.tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        if token_ids.shape[1] == 0:
            raise BaselineContractError(f"empty B2 candidate tokenization: {completion!r}")
        inputs = dict(base)
        inputs["input_ids"] = torch.cat([base["input_ids"], token_ids], dim=1)
        continuation_mask = torch.ones_like(token_ids, device=device)
        inputs["attention_mask"] = torch.cat(
            [base["attention_mask"], continuation_mask], dim=1
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            log_probs = torch.log_softmax(
                logits[:, prefix_length - 1 : prefix_length + token_ids.shape[1] - 1, :],
                dim=-1,
            )
            selected = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
        natural = completion.strip()
        score = float(selected.mean().item())
        if not math.isfinite(score):
            raise BaselineContractError(f"non-finite B2 candidate score: {natural}")
        scores[natural] = score
    return scores


def _validate_jobs_manifest(
    jobs_path: Path,
    jobs: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
) -> None:
    manifest_path = jobs_path.parent / "protocol.json"
    if not manifest_path.is_file():
        raise BaselineContractError(f"missing B2 jobs manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != B2_JOBS_MANIFEST_SHA256:
        raise BaselineContractError("B2 jobs manifest differs from frozen implementation artifact")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise BaselineContractError("B2 jobs manifest uses another protocol")
    if manifest.get("ground_truth_available_to_inference") is not False:
        raise BaselineContractError("B2 jobs manifest violates the ground-truth boundary")
    if manifest.get("deployment_eligible") is not False:
        raise BaselineContractError("B2 jobs manifest incorrectly claims deployment eligibility")
    source_paths = {
        "machine_protocol": protocol_path,
        "protocol_lock": lock_path,
        "baseline_config": config_path,
    }
    for name, path in source_paths.items():
        if manifest.get("source_files", {}).get(name, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"B2 jobs manifest source hash mismatch: {name}")
    matching = [
        item
        for item in manifest.get("queues", {}).values()
        if item.get("path") == jobs_path.name
    ]
    if len(matching) != 1:
        raise BaselineContractError("jobs file is not uniquely registered in B2 manifest")
    if matching[0].get("sha256") != sha256_file(jobs_path):
        raise BaselineContractError("B2 jobs hash differs from jobs manifest")
    if matching[0].get("count") != len(jobs):
        raise BaselineContractError("B2 jobs count differs from jobs manifest")
    registered_name = next(
        name
        for name, item in manifest["queues"].items()
        if item.get("path") == jobs_path.name
    )
    if matching[0] != B2_QUEUE_ARTIFACTS[registered_name]:
        raise BaselineContractError("B2 jobs differ from frozen implementation artifact")


def run(
    jobs_path: Path,
    output_path: Path,
    image_root: Path,
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    variant_short: str,
    model_ref: str | None,
    development_gate: Path | None,
    limit: int | None,
    dry_run: bool,
) -> int:
    protocol, lock = load_frozen_b2_protocol(PROJECT_ROOT, protocol_path, lock_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    variant = VARIANT_KEYS[variant_short]
    jobs = load_jsonl(jobs_path)
    _validate_jobs_manifest(
        jobs_path, jobs, protocol, protocol_path, lock_path, config_path
    )
    partitions = {job.get("partition") for job in jobs}
    if len(partitions) != 1:
        raise BaselineContractError("B2 runner requires jobs from exactly one partition")
    partition = next(iter(partitions))
    validate_b2_jobs(jobs, protocol, set(config["inference_forbidden_fields"]), partition)
    if partition == "locked_evaluation":
        if variant_short == "b2_a":
            raise BaselineContractError("frozen protocol prohibits B2-A on locked evaluation")
        if development_gate is None:
            raise BaselineContractError("locked evaluation requires a passing development gate report")
        if B2_TRUSTED_DEVELOPMENT_GATE_SHA256 is None:
            raise BaselineContractError(
                "locked evaluation is not authorized: no trusted development gate digest"
            )
        validate_development_gate_report(
            development_gate,
            protocol,
            lock["sealed_files"]["machine_protocol"]["sha256"],
            PROJECT_ROOT,
            B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
        )
    if limit is not None:
        jobs = jobs[:limit]
    if dry_run:
        for job in jobs:
            image_path = resolve_image_path(image_root, job)
            with Image.open(image_path) as source:
                crop_for_job(source.convert("RGB"), job)
        print(f"READY: validated {len(jobs)} B2 jobs; model was not loaded")
        return 0
    if output_path.exists():
        raise BaselineContractError(f"refusing to overwrite existing output: {output_path}")
    baseline = config["smolvlm"]
    resolved_model = model_ref or os.environ.get(baseline["model_path_env"], "")
    if not resolved_model:
        raise BaselineContractError(
            f"set {baseline['model_path_env']} or pass --model with a local SmolVLM directory"
        )
    model_root = verify_local_model(resolved_model, baseline)
    processor, model, target = load_model(
        str(model_root), int(protocol["model_contract"]["image_longest_edge"])
    )
    settings = protocol["variants"][variant]
    records: list[dict[str, Any]] = []
    for job in jobs:
        attempt_started = time.perf_counter()
        model_scope_started: float | None = None
        common = {
            "schema_version": 1,
            "b2_protocol_id": protocol["protocol_id"],
            "variant": variant,
            "job_id": job["job_id"],
            "partition": job["partition"],
            "candidate_id": job["candidate_id"],
            "source_sample_id": job["source_sample_id"],
            "image_sha256": job["image_sha256"],
            "input_variant": job["input_variant"],
            "proposal_source": job["proposal_source"],
            "deployment_eligible": False,
        }
        try:
            image_path = resolve_image_path(image_root, job)
            with Image.open(image_path) as source:
                loaded_image = source.convert("RGB")
            model_scope_started = time.perf_counter()
            image = crop_for_job(loaded_image, job)
            if variant_short == "b2_a":
                raw = generate_response(
                    processor,
                    model,
                    target,
                    image,
                    settings["prompt"],
                    int(settings["max_new_tokens"]),
                )
                parsed = parse_b2_a_label(raw, protocol)
                natural, prediction = parsed if parsed is not None else (None, None)
                status = "ok" if parsed is not None else "invalid_output"
                record = {
                    **common,
                    "status": status,
                    "error": None,
                    "predicted_natural_label": natural,
                    "predicted_label": prediction,
                    "raw_response": raw,
                }
            else:
                completions = settings["candidate_completions_in_tie_order"]
                scores = fixed_candidate_scores(
                    processor, model, target, image, settings["prompt"], completions
                )
                order = [item.strip() for item in completions]
                top1, top2, margin = rank_candidate_scores(scores, order)
                record = {
                    **common,
                    "status": "ok",
                    "error": None,
                    "predicted_natural_label": top1,
                    "predicted_label": protocol["label_mapping"][top1],
                    "candidate_scores": scores,
                    "top1_natural_label": top1,
                    "top2_natural_label": top2,
                    "margin": margin,
                }
        except Exception as exc:  # preserve every failed sample
            record = {
                **common,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "predicted_natural_label": None,
                "predicted_label": None,
            }
            if variant_short == "b2_a":
                record["raw_response"] = ""
            else:
                record.update(
                    candidate_scores={},
                    top1_natural_label=None,
                    top2_natural_label=None,
                    margin=None,
                )
        timing_origin = (
            model_scope_started if model_scope_started is not None else attempt_started
        )
        record["latency_ms"] = round(
            (time.perf_counter() - timing_origin) * 1000.0, 3
        )
        records.append(record)
    write_jsonl(output_path, records)
    print(output_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_KEYS), required=True)
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json"
    )
    parser.add_argument(
        "--lock", type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json")
    parser.add_argument("--model")
    parser.add_argument("--development-gate", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            args.jobs.resolve(), args.output.resolve(), args.image_root.resolve(),
            args.protocol.resolve(), args.lock.resolve(), args.config.resolve(),
            args.variant, args.model,
            args.development_gate.resolve() if args.development_gate else None,
            args.limit, args.dry_run,
        )
    except (BaselineContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
