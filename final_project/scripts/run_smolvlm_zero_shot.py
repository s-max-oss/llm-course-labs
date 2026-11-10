#!/usr/bin/env python3
"""Run the offline oracle-crop zero-shot SmolVLM diagnostic baseline."""

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

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    parse_strict_label,
    resolve_image_path,
    sha256_file,
    validate_baseline_policy,
    validate_inference_jobs,
    write_jsonl,
)


def crop_for_job(image: Image.Image, job: dict[str, Any]) -> Image.Image:
    region = job.get("target_region_xyxy_pixels")
    if region is None:
        return image.copy()
    if not isinstance(region, list) or len(region) != 4:
        raise BaselineContractError(f"{job['job_id']}: invalid crop region")
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
    expected_files = set(baseline["model_files"])
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise BaselineContractError(
            f"SmolVLM model directory differs from manifest: "
            f"missing={missing} unexpected={unexpected}"
        )
    for relative_name, expected_hash in baseline["model_files"].items():
        path = root / relative_name
        if not path.is_file():
            raise BaselineContractError(f"SmolVLM model file is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise BaselineContractError(f"SmolVLM model file hash mismatch: {path}")
    return root


def load_model(model_ref: str, device: str, local_files_only: bool, longest_edge: int):
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
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise BaselineContractError("CUDA was requested but torch.cuda.is_available() is false")
    target = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    options = {"local_files_only": local_files_only}
    processor = AutoProcessor.from_pretrained(model_ref, **options)
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"longest_edge": longest_edge}
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False
    model = AutoModel.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        attn_implementation="eager",
        **options,
    )
    model.to(target)
    model.eval()
    return processor, model, target


def build_prompt(processor: Any, prompt: str) -> str:
    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        add_generation_prompt=True,
    )


def generate_label(
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
    return processor.tokenizer.decode(
        tokens[0, prefix:], skip_special_tokens=True
    ).strip()


def run(
    jobs_path: Path,
    output_path: Path,
    image_root: Path,
    config_path: Path,
    model_ref: str | None,
    device: str,
    limit: int | None,
    dry_run: bool,
) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    baseline = config["smolvlm"]
    if baseline["deployment_eligible"] is not False:
        raise BaselineContractError("oracle-crop baseline cannot be deployment eligible")
    jobs = load_jsonl(jobs_path)
    if limit is not None:
        jobs = jobs[:limit]
    validate_inference_jobs(jobs, config, "smolvlm")
    if dry_run:
        for job in jobs:
            image_path = resolve_image_path(image_root, job)
            with Image.open(image_path) as source:
                crop_for_job(source.convert("RGB"), job)
        print(f"READY: validated {len(jobs)} SmolVLM jobs; model was not loaded")
        return 0
    if output_path.exists():
        raise BaselineContractError(f"refusing to overwrite existing output: {output_path}")
    resolved_model = model_ref or os.environ.get(baseline["model_path_env"], "")
    if not resolved_model:
        raise BaselineContractError(
            f"set {baseline['model_path_env']} or pass --model with a local SmolVLM directory"
        )
    model_root = verify_local_model(resolved_model, baseline)
    processor, model, target = load_model(
        str(model_root),
        device,
        local_files_only=True,
        longest_edge=int(baseline["image_longest_edge"]),
    )
    records = []
    for job in jobs:
        started = time.perf_counter()
        try:
            image_path = resolve_image_path(image_root, job)
            with Image.open(image_path) as source:
                model_image = crop_for_job(source.convert("RGB"), job)
            raw_response = generate_label(
                processor,
                model,
                target,
                model_image,
                baseline["prompt"],
                int(baseline["max_new_tokens"]),
            )
            predicted_label = parse_strict_label(raw_response)
            status = "ok" if predicted_label is not None else "invalid_output"
            error = None
        except Exception as exc:  # preserve every failed sample
            raw_response = ""
            predicted_label = None
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "schema_version": 1,
                "job_id": job["job_id"],
                "baseline_id": job["baseline_id"],
                "fold_id": job["fold_id"],
                "candidate_id": job["candidate_id"],
                "source_sample_id": job["source_sample_id"],
                "image_sha256": job["image_sha256"],
                "input_variant": job["input_variant"],
                "proposal_source": job["proposal_source"],
                "deployment_eligible": False,
                "status": status,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "raw_response": raw_response,
                "predicted_label": predicted_label,
            }
        )
    write_jsonl(output_path, records)
    print(output_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            args.jobs.resolve(),
            args.output.resolve(),
            args.image_root.resolve(),
            args.config.resolve(),
            args.model,
            args.device,
            args.limit,
            args.dry_run,
        )
    except BaselineContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
