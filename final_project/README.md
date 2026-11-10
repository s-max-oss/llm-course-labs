# RF-DETR–SmolVLM confidence-aware cascade

Independent experiment workspace for the course final project:

> 面向四足机器人危险源识别的 RF-DETR–SmolVLM 置信度感知级联方法

## Published source snapshot

This directory is the compact source-code snapshot referenced by the final course report. It includes the frozen configurations, implementation, validators, tests, report sources, and document builders. The submitted course ZIP is the authoritative artifact snapshot for the 500-image dataset, sealed experiment outputs, model checkpoint, figures, replay media, and final Word/PDF documents; those large or binary artifacts are intentionally not duplicated in GitHub. Commands that validate sealed results require the corresponding ZIP artifact directories to be restored beside this source tree.

The source robot repository and course laboratories remain external and read-only. The full research workspace stores reproducible manifests, experiment code, configurations, metrics, and report artifacts; this GitHub snapshot retains the source-controlled portion.

## Robot perception replay extension

The Level-A replay extension converts sealed RF-DETR development results into an independent equivalent of the robot's frame-level detection stream and a fail-closed decision trace. A research-integrity re-audit found that Level-B v2 used `/Odometry_gazebo`, which the authoritative robot contract classifies as Gazebo pose ground truth. Level-B v2 is therefore preserved as a target-label-free, pose-oracle debug replay: it validates synchronized RGB-D processing, depth filtering, temporal confirmation, deterministic official-format output, and post-hoc scoring, but not autonomous localization, full truth isolation, competition compliance, online control, or a full match. See `experiments/level_b_v2/research_integrity_correction.json`, `experiments/level_b_v2/level_b_run.json`, and the corrected Word supplement under `deliverables/`; run `scripts/run_checks.ps1` to validate the bundle.

## Current status

- Legacy pilot set: 500 images from five simulator seeds (`101`, `202`, `303`, `404`, `505`).
- Labels: YOLO boxes for `red_ball`, `red_cube`, and `green_ball`; missing label files follow the legacy background convention but are retained explicitly as missing-label records.
- Evaluation scope: a duplicate-controlled five-seed pilot; it is not a formal generalization experiment. The detector checkpoint's training manifest is unavailable, so training/evaluation overlap cannot be excluded.
- Local simulation: VMware was used for the separate Level-B v2 pose-oracle debug replay. The guest has no NVIDIA/CUDA device, but a GPU or full competition entry is not required for the core course report.
- B2 exploratory evaluation is complete and sealed. Fixed-candidate scoring recovered object-class discrimination but failed background rejection; see `experiments/pilot_v1/smolvlm_b2_locked_run.json`.
- The exact production RF-DETR checkpoint was recovered and hash-verified without modifying the source repository. The full 401-image CPU run completed with zero errors: three-target Macro-F1 0.9635, danger-object recall 0.9892, and P50 latency 653.1 ms. These are fixed-checkpoint in-domain exploratory results, not held-out detector-generalization evidence. No GPU rental was needed.

## Build and validate the legacy manifest

Create `configs/data_sources.local.json` from the example and set local absolute paths. Then run:

```powershell
python scripts/build_manifest.py --config configs/data_sources.local.json
python scripts/build_candidate_manifest.py
python scripts/build_pilot_splits.py
python scripts/prepare_baseline_jobs.py
python scripts/validate_manifest.py --config configs/data_sources.local.json --verify-hashes
python scripts/validate_pilot_splits.py
python scripts/validate_baseline_jobs.py
python -m unittest discover -s tests -v
```

Generated files are small metadata artifacts; source images are not copied:

- `data/manifests/batch_500.jsonl`
- `data/manifests/batch_500.summary.json`
- `data/manifests/batch_500.near_duplicates.json`
- `data/manifests/batch_500.candidates.jsonl`
- `data/manifests/batch_500.candidates.summary.json`

## Dataset discipline

The four semantic outputs are:

1. `danger_red_sphere`
2. `distractor_red_box`
3. `distractor_green_sphere`
4. `background_or_unknown`

The immutable source manifest keeps `split="unassigned"`; generated pilot artifacts carry fold assignments. A future formal generalization study still requires independent supplemental scenes.

The candidate manifest represents each annotated object as one VLM decision unit and each legacy background frame as one background unit. Object candidates carry both the full-frame reference and a 15%-padded crop rectangle; no pixels are duplicated.

## Leakage-controlled five-seed pilot

`configs/pilot_cv.json` defines five rotating folds. Each fold holds out one complete scene seed for test, the next seed for validation, and uses the remaining three seeds for training. Every eligible image is test exactly once, validation exactly once, and train exactly three times.

The generator conservatively excludes every image in an exact duplicate group and every image participating in a cross-seed dHash candidate pair. This quarantines 99/500 images and leaves 401 images. The primary result must be computed from pooled out-of-fold predictions. Fold-level four-class Macro-F1 is forbidden when a class has zero support; in particular, the seed-202 test fold contains no eligible red-box candidates.

These outputs define a **five-seed leakage-controlled pilot study**. They do not support a formal generalization claim.

## Baseline inference queues

`configs/baselines.json` freezes the audited production RF-DETR-Seg Medium contract and the SmolVLM-256M zero-shot diagnostic. `scripts/prepare_baseline_jobs.py` produces label-free inference queues under `experiments/pilot_v1/baseline_jobs/`; the independent validator reconstructs every job from the sealed fold files and rejects ground-truth fields.

The SmolVLM B1 queue uses annotation-derived padded crops only as an **offline oracle-proposal diagnostic**. It is marked `deployment_eligible=false` throughout. A deployable cascade must replace those regions with RF-DETR proposals; these oracle-crop results must never be reported as online end-to-end performance.

The model stacks require separate environments: use `requirements-smolvlm.txt` for the frozen SmolVLM runtime and `requirements-rfdetr.txt` for RF-DETR. They cannot share one Transformers installation because RF-DETR 1.9.4 requires Transformers 5.x while the frozen SmolVLM runtime uses 4.57.6. The verified local CPU environment reuses the system CPU Torch/Torchvision installation through `--system-site-packages`, while RF-DETR and Transformers remain environment-local; exact active versions are sealed in the RF-DETR preflight manifest. Both runners default to local models and refuse to overwrite an existing output:

```powershell
$imageRoot = (Get-Content -Raw configs/data_sources.local.json | ConvertFrom-Json).image_root

# Readiness check only: verifies every image hash without loading a model.
python scripts/run_rfdetr_baseline.py `
  --jobs experiments/pilot_v1/baseline_jobs/fold_01/rfdetr_images.jsonl `
  --output results/fold_01_rfdetr.jsonl --image-root $imageRoot --dry-run

python scripts/run_smolvlm_zero_shot.py `
  --jobs experiments/pilot_v1/baseline_jobs/fold_01/smolvlm_candidates.jsonl `
  --output results/fold_01_smolvlm.jsonl --image-root $imageRoot --dry-run
```

For real execution, set `DANGER_VISION_MODEL` to the production RF-DETR checkpoint whose SHA-256 is fixed in the baseline config, and set `SMOLVLM_MODEL` to the local SmolVLM-256M directory whose required files are also SHA-256 pinned. Remove `--dry-run`; use a new result path for every run. Each runner validates the exact task schema and rejects leaked target fields before loading a model. Individual image/hash/decode failures are emitted as error records instead of aborting the batch.

After all five SmolVLM result files exist, the offline-only evaluator joins predictions to sealed labels and refuses missing or duplicate candidates:

```powershell
python scripts/evaluate_smolvlm_baseline.py `
  --results results/fold_01_smolvlm.jsonl results/fold_02_smolvlm.jsonl `
            results/fold_03_smolvlm.jsonl results/fold_04_smolvlm.jsonl `
            results/fold_05_smolvlm.jsonl `
  --output results/smolvlm_zero_shot_metrics.json
```

The evaluator verifies each ground-truth candidate file against the sealed baseline-job protocol, records those hashes in its report, and requires the exact runner result schema with consistent status/label/error semantics and oracle/full-frame proposal provenance. It reports pooled out-of-fold metrics as primary and automatically suppresses four-class fold Macro-F1 where class support is incomplete.

### Recorded B1 result

The frozen five-fold SmolVLM B1 run completed locally on CPU. All 444 generations returned explanation text rather than exactly one allowed label. Under the strict predeclared parser, prediction coverage, pooled Macro-F1, accuracy, and danger recall are therefore all 0. The median generation latency was 2145.0 ms. This is retained as an honest negative zero-shot baseline; explanation text is never post-processed into guessed labels. Raw artifacts remain under `results/`, and their SHA-256 values plus the runtime are sealed in `experiments/pilot_v1/smolvlm_zero_shot_run.json`.

### Recorded B2 exploratory result

B2 was frozen before execution. Seeds 101/202 formed a 193-candidate development partition; seeds 303/404/505 formed a 251-candidate locked exploratory partition. B2-A strict natural-language generation was development-only. B2-B scored the four fixed completions by mean teacher-forced continuation log probability.

B2-B passed all development gates, after which the user authorized exactly one locked run. The locked run produced 251/251 valid predictions, 90.84% accuracy, Macro-F1 0.7217, and 100% danger recall. Red ball, red box, and green ball each had 100% recall, but all 23 background examples were classified as red box. Median CPU latency was 5829.7 ms.

This result is an offline oracle-crop exploratory diagnostic. It supports retaining B2-B only as a proposal-conditioned three-object semantic classifier candidate. It does not establish background detection, deployable four-class performance, or formal generalization. The locked partition must not be used for further prompt, answer, mapping, or threshold tuning.

The same-subset B1/B2 comparison is independently recomputed from the full sealed result contracts. It first validates the frozen B1 run and the B2 locked-run hashes, then invokes both complete evaluators. The evidence is stored in:

- `experiments/pilot_v1/smolvlm_b2_locked_run.json`
- `experiments/pilot_v1/results/smolvlm_b1_b2_locked_comparison.json`
- `report-evidence-index.md`

Run the complete deterministic validation suite with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_checks.ps1
```

The current suite checks the Level-A replay contracts and the Level-B v2 distinction between danger-target truth and Gazebo pose truth, as well as depth, detector provenance, class mapping, claim boundaries, reconstruction, and artifact hashes. G2 truth isolation intentionally remains failed for Level-B v2 while the debug artifact validator passes. The exact `checkpoint_best_ema.pth` matches SHA-256 `ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50`; repeated smoke tests and the full run passed. `experiments/pilot_v1/rfdetr_checkpoint_provenance.json` records that the checkpoint's training-sample lineage is unavailable; therefore the RF-DETR result must be described as fixed-checkpoint in-domain exploratory performance.

The exact pre-run code commit, active package versions, sealed input hashes, five queue counts, clean `pip check`, and second smoke-result hash are recorded in `experiments/pilot_v1/rfdetr_preflight.json`.

Recompute and verify the complete RF-DETR evidence bundle with:

```powershell
python scripts/validate_rfdetr_run.py
```

The development-only cascade feasibility protocol is frozen in `configs/cascade_feasibility_v1.json`. Its deterministic queue contains 24 low-confidence RF-DETR target proposals from 22 seed-101/202 images, uses no ground truth or oracle boxes during generation/inference, and is reconstructed by `scripts/validate_cascade_jobs.py`. The real run completed 24/24 jobs, but B2 repeated every detector label: RF-only and the safety-preserving cascade both had Macro-F1 0.96918, danger recall 0.98864, and 11 target false positives. The strict gain gate failed, so the cascade stopped and no cascade inference was run on seeds 303/404/505. `scripts/validate_cascade_run.py` fully recomputes this stopped outcome.

## Final report

The final Chinese course report is `deliverables/2024311281_孙驰_期末报告_RF-DETR_SmolVLM.docx`; its editable source is `final-report.md`. It was built from the read-only course template by `scripts/build_final_report.py`, opened successfully in Microsoft Word, exported to PDF for QA, and visually inspected page by page. The document has 13 A4 pages, five evidence-derived figures, and three tables. It preserves the report boundary that RF-DETR is fixed-checkpoint in-domain exploratory evidence and that B1, oracle-crop B2, and the stopped development cascade are not deployable end-to-end results.
