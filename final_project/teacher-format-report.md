# From Competition Hazard Search to Auditable Multimodal Perception

First Author: Sun Chi    Second Author: Kuo-Kun Tsengg

Harbin Institute of Technology, Shenzhen

## Abstract

This study grew out of my participation in the DG-202602 quadruped-robot competition project, in which a robot must search an unfamiliar building, distinguish red spherical hazards from visually similar red blocks and green balls, and return hazard coordinates to the task system. During development, I found that a visually plausible prediction was not enough: a missed proposal could never reach localization, a false alarm could create a nonexistent hazard, and a free-form model response could break the robot interface even when its description sounded reasonable. I therefore extracted a narrower course research question from the competition pipeline: can a compact vision-language model improve a fixed real-time detector on ambiguous proposals without weakening safety, reproducibility, or latency? The study audits 500 historical RGB frames from five simulation seeds, globally isolates 99 duplicate-risk images, and evaluates a fixed RF-DETR-Seg Medium checkpoint on 401 retained images. RF-DETR reaches 0.9635 three-class Macro-F1 and 0.9892 hazard recall, although unavailable training lineage limits the claim to in-domain exploratory performance. Free-form SmolVLM-256M produces no strictly valid single-label outputs. Frozen candidate scoring restores 100% output coverage and reaches 0.7217 Macro-F1 on 251 oracle crops, but rejects none of 23 background cases. On 24 real low-confidence RF-DETR proposals, SmolVLM repeats every detector class, adds 5.11 s CPU median latency, and yields no metric gain. The preregistered continuation gate therefore stops the cascade. The final recommendation is RF-DETR-only, with Level A interface replay accepted and Level B retained only as Gazebo-pose-oracle debugging evidence.

Keywords: quadruped robot; hazard recognition; RF-DETR; SmolVLM; multimodal intelligence

## 1 Introduction

### 1.1 Research Background and Problem Statement

The motivation did not begin with a benchmark leaderboard. It came from the DG-202602 competition project entitled Autonomous Hazard Search and Recognition with a Quadruped Robot. In team meetings and simulator development, the task was divided into autonomous exploration and hazard perception. The robot had to enter rooms in an unknown building, observe red spherical hazards, reject red blocks and green balls, and finally provide three-dimensional hazard coordinates in the required result format. The visual decision therefore affected navigation, localization, duplicate suppression, and final scoring rather than ending at a bounding box.

Practical work exposed three failure modes. First, a detector miss creates no proposal, so a downstream semantic module has nothing to review. Second, a visually similar distractor can become a false hazard and contaminate the coordinate list. Third, a generative model may describe an image correctly but still violate the fixed label or event schema expected by the robot. Motion blur, partial occlusion, small distant targets, frame-edge truncation, and repeated boxes made these failures visible during the competition workflow. The course project was inspired by this experience and deliberately narrows the full competition problem to an auditable question about visual semantics and interface reliability.

The study does not claim to reproduce the complete competition. It does not validate autonomous exploration, online control, or a truth-isolated world-coordinate pipeline. Instead, it examines whether visual intelligence from a specialist detector and multimodal intelligence from a compact VLM are complementary on the evidence that is actually available. This scope protects the report from converting a partial perception study into an unsupported system-level claim.

### 1.2 Research Objectives and Core Contributions

The first objective is to measure the behavior of the fixed RF-DETR checkpoint already used by the robot project on a duplicate-controlled historical corpus. The second is to separate SmolVLM's semantic ability from its instruction-format compliance by comparing free generation with frozen candidate scoring. The third is to test whether the VLM produces measurable net benefit when it receives real low-confidence detector proposals rather than oracle boxes.

The work makes four course-level contributions. It constructs a seed-aware data and evidence protocol with SHA-256 seals; turns a generative VLM into a closed candidate scorer; preregisters routing, fusion, and stop conditions before running the real-proposal cascade; and preserves the negative cascade result instead of tuning after observation. A Level A event replay demonstrates how detections enter a robot state machine. A Level B RGB-D replay is also retained, but its use of `/Odometry_gazebo` is disclosed and the G2 truth-isolation gate remains failed.

### 1.3 Paper Organization

Section 2 connects the competition-derived problem to visual, language, and multimodal intelligence and reviews open-source and commercial options. Section 3 defines the task, evidence boundary, models, and frozen cascade. Section 4 reports the dataset, controlled comparisons, figures, and negative result. Section 5 gives the GitHub repository and verification entry. Section 6 presents the robot-interface replays. Section 7 interprets the findings, limitations, course learning, and next steps. Section 8 closes the argument without introducing new evidence.

## 2 Related Work

### 2.1 Course Fundamental Knowledge Review

The project combines three of the course's intelligent modules. Visual intelligence supplies object localization and class prediction. DETR formulates detection as direct set prediction using object queries and bipartite matching, reducing dependence on hand-designed anchors and non-maximum suppression [1]. RT-DETR shows how an efficient hybrid encoder and query selection can adapt this family to real-time detection [2]. RF-DETR further explores accuracy-latency configurations for target datasets through weight-sharing neural architecture search [3].

Language intelligence enters through autoregressive token probabilities. A VLM does not directly emit a robot class; it estimates a distribution over token continuations conditioned on an image and prompt. Visual instruction tuning makes such models useful for open-ended visual interaction [4], while SmolVLM reduces model and memory scale for resource-constrained inference [5]. The key course insight is that a generative interface and a discriminative robot contract are different objects. Candidate scoring can use the language decoder as a classifier, but only after candidate strings, tokenization, and aggregation rules are frozen.

Multimodal intelligence connects image evidence and language concepts, but system utility also depends on interfaces. The detector provides candidate existence and geometry; the VLM may refine semantic identity; the state machine consumes discrete events. This three-stage relationship explains why a strong component can still fail to improve the complete chain. A proposal-conditioned VLM cannot recover an object that the detector never proposed, and a correct description is unusable if its format cannot be parsed.

### 2.2 State of the Art and Technology Landscape

Open-source real-time detectors and compact VLMs provide reproducible weights and local execution. RF-DETR is a specialist closed-set detector, while Grounding DINO and YOLO-World add language-conditioned open-vocabulary detection [3,6,7]. SmolVLM emphasizes small multimodal models and can run locally [5]. Commercial multimodal APIs provide broad visual reasoning, but cloud dependency, version drift, cost, latency, and data governance make them unsuitable as the only decision path for this offline robot experiment. Table 1 records the selection logic rather than presenting a generic product ranking.

| System family | Access model | Relevant strength | Reason for inclusion or exclusion |
|---|---|---|---|
| RF-DETR | Open source, local | Specialist end-to-end detection and segmentation | Fixed production checkpoint and proposal geometry |
| Grounding DINO / YOLO-World | Open source, local | Open-vocabulary detection from text concepts | Relevant alternative, but not used in the frozen experiment |
| SmolVLM-256M | Open source, local | Compact image-text reasoning and token scoring | Used to test semantic review under CPU constraints |
| Commercial multimodal APIs | Closed cloud service | Broad visual question answering | Excluded from the final decision path because the interface is remote and versioned externally |

Current literature supports the components but does not establish their benefit in this project. RF-DETR reports strong accuracy-latency trade-offs across public datasets [3], and SmolVLM reports efficient multimodal performance [5]. Those results do not answer whether SmolVLM can correct this checkpoint's residual robot-domain errors. The answer requires paired proposals, frozen gates, and project-specific negative-case analysis.

### 2.3 Correlation Between This Research and the Course

The competition supplied the practical problem; the course supplied a disciplined way to decompose it. DETR connects to Transformer attention and structured set prediction [1,8]. SmolVLM connects language decoding with visual embeddings [5]. Candidate scoring applies language-model likelihood to a discriminative decision. The cascade then becomes a small intelligent system whose usefulness must be judged by task metrics, error propagation, latency, and reproducibility rather than by model novelty alone.

This research also changes how I interpret “adding a large model.” The course experiments show that a larger or more general model is not automatically a better subsystem. The correct question is whether the module changes an observable decision under the same data and frozen protocol. This principle led directly to the strict continuation gate and to the final decision to keep RF-DETR-only after the VLM produced zero net gain.

## 3 Methodology / Proposed Method

### 3.1 Problem Definition and Symbol Explanation

The visual task contains three boxed object classes: `danger_red_sphere`, `distractor_red_box`, and `distractor_green_sphere`. Historical frames without label files are treated as background or unknown under the inherited data convention, but they are not described as manually verified negatives. The online inference input contains only image data and frozen configuration. Labels, simulator truth, and world/layout files are connected only by offline evaluators.

For class c, precision is the proportion of matched predictions among all predictions of c, recall is the proportion of matched ground-truth objects among all ground-truth objects of c, and F1 is their harmonic mean. Three-class Macro-F1 is the arithmetic mean across the boxed classes. Matching uses class agreement, box IoU at least 0.5, confidence order, and one-to-one assignment. Mask IoU is not reported because the historical corpus has no ground-truth masks.

[[EQ:Precision_c = TP_c/(TP_c + FP_c),   Recall_c = TP_c/(TP_c + FN_c)   (1)]]

[[EQ:F1_c = 2 Precision_c Recall_c/(Precision_c + Recall_c)   (2)]]

[[EQ:MacroF1 = (F1_1 + F1_2 + F1_3)/3   (3)]]

The main system decision is whether the VLM changes the detector outcome without decreasing hazard recall, increasing object false positives, or exceeding the frozen call-rate boundary. A strict metric increase is required. Equal performance is treated as a failed continuation condition because the VLM adds computation and operational complexity.

### 3.2 Course Theoretical Preliminaries

DETR uses a Transformer encoder-decoder and learned object queries to produce a fixed-size prediction set [1]. Bipartite matching assigns predictions to targets during training. The fixed RF-DETR checkpoint inherits this end-to-end detection principle and supplies class, confidence, bounding box, and mask outputs. In this research, the mask output is retained for provenance but the quantitative evaluation uses bounding boxes because matching mask annotations do not exist.

SmolVLM conditions an autoregressive decoder on image and text representations [5]. Free generation selects a sequence of tokens, so it may add explanations or punctuation even when the prompt requests one label. Frozen candidate scoring instead evaluates each allowed continuation and averages its token log probabilities. The highest finite score becomes the candidate prediction. This converts an open output space into a deterministic four-label decision while retaining the same underlying model.

The robot interface adds a contract layer. A detection frame records image identity, model provenance, ordered proposals, and state-machine inputs. Level A forbids world coordinates and temporal tracking; it validates event structure and fail-closed state transitions. Level B adds RGB-D back-projection and coordinate transforms, but its pose source must be audited separately from target truth. Because `/Odometry_gazebo` is simulator pose truth, Level B is classified as pose-oracle debugging rather than autonomous localization.

### 3.3 Core Model and Algorithm Design

The RF-DETR baseline uses the fixed RF-DETR-Seg Medium checkpoint with confidence threshold 0.5. Its SHA-256 is `ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50`. The 401 inference jobs contain image paths and inference configuration but no labels or world information. An independent evaluator later attaches the sealed labels and recomputes every metric.

SmolVLM B1 asks for exactly one snake-case label and accepts only an exact match. It does not extract keywords from a longer explanation. B2-A is a development diagnostic with natural label phrases. B2-B freezes four candidate strings and computes teacher-forced average token log probability. The B2 object inputs are oracle crops expanded by 15%, while background inputs use full frames. B2-B therefore evaluates candidate-region semantics, not deployable object existence.

The real-proposal cascade runs only on development seeds 101 and 202. It routes RF-DETR proposals with confidence in `[0.5, 0.7)`, expands each box by 15%, clips it to the image, and sends the crop to B2-B. The queue contains 24 proposals from 22 images. Failure falls back to RF-DETR. A detector hazard prediction cannot be downgraded. Non-hazard classes may be relabeled only between the two distractor classes. A VLM hazard answer does not directly upgrade a non-hazard detector proposal.

Continuation requires all of the following: hazard recall does not decrease, three-class Macro-F1 strictly increases, object false positives do not increase, and the VLM call rate remains at or below 15%. The rule was frozen before the run. Its purpose is to stop a technically functioning but systemically useless cascade.

### 3.4 Experimental Implementation Details

The raw corpus contains 500 RGB frames from five historical simulator seeds and 383 label files with 426 boxes. Fifteen exact duplicate groups and cross-seed perceptual near-duplicate candidates were audited. All images in exact duplicate groups and all images participating in a cross-seed near-duplicate pair were isolated, leaving 401 images and 444 candidate decisions. Five folds rotate complete seeds instead of randomly splitting adjacent frames.

RF-DETR ran on Windows 11 CPU with Python 3.13.12, PyTorch 2.12.0+cpu, Torchvision 0.27.0+cpu, and `rfdetr` 1.9.4. SmolVLM used PyTorch 2.15.0+cpu, Transformers 4.57.6, and float32. The two models require different Transformers environments. Reported latency is CPU latency from this environment and must not be extrapolated to an RTX 3080 Ti, RTX 3090, or onboard robot computer.

Protocols, queues, model identity, source files, predictions, summaries, and replay artifacts are sealed with SHA-256. The final deterministic regression runs 87 tests. Training lineage for the RF-DETR checkpoint is unavailable, so the evaluation cannot establish strict train-test isolation. This limitation is kept beside the result rather than hidden in a final footnote.

## 4 Experiments and Results

### 4.1 Experimental Setup

Table 2 separates the detector baseline, VLM diagnostics, locked candidate experiment, and real-proposal cascade. The locked B2 set is exploratory rather than confirmatory because B1 had already observed the same seeds. Only the development partition is used for the real cascade, and the stop rule prevents a post-hoc locked cascade after the gain gate fails.

| Experiment | Seeds | Scale | Evidence boundary |
|---|---|---|---|
| RF-DETR five-fold evaluation | 101/202/303/404/505 | 401 images | Duplicate-controlled in-domain exploration |
| B2 development | 101/202 | 193 candidates | Gate selection and feasibility only |
| B2 locked candidate study | 303/404/505 | 251 oracle crops | Candidate semantics, not end-to-end detection |
| Real-proposal cascade | 101/202 | 24 proposals | Development-only net-benefit test |

The primary detector metric is three-class Macro-F1. Hazard Precision, Recall, F1, object FP/FN, and image-level presence recall are reported separately because a hazard miss has a different system consequence from a distractor confusion. VLM evaluation also reports exact output coverage and background recall. Cascade evaluation compares the same proposal set before and after fusion and records CPU latency and call rate.

### 4.2 Main Results

RF-DETR completes 401 of 401 inference jobs. It reaches 0.9635 Macro-F1, 0.9892 hazard recall, and 0.9944 hazard-image presence recall. The result contains 183 hazard true positives, five false positives, and two false negatives. One false-negative image still contains another detected hazard; the other produces no proposal. The latter demonstrates the structural limit of a proposal-conditioned reviewer.

| Metric | RF-DETR result |
|---|---|
| Three-class Macro-F1 | **0.9635** |
| Hazard Precision / Recall / F1 | **0.9734 / 0.9892 / 0.9812** |
| Red-box F1 | 0.9619 |
| Green-sphere F1 | 0.9474 |
| Hazard-image presence Recall | 0.9944 |
| CPU P50 / P95 latency | 653.1 / 743.3 ms |

![RF-DETR fixed-checkpoint metrics](figures/rfdetr_pilot_metrics.png)

B1 produces 444 fluent responses but zero exact single-label outputs. B2-A restores formatting but collapses toward a small subset of labels. B2-B reaches 100% output coverage, 0.9084 accuracy, 0.7217 Macro-F1, and 1.0 hazard recall on 251 locked oracle crops. It correctly classifies 97 red spheres, 90 red boxes, and 41 green spheres, but all 23 background cases become red boxes. Candidate scoring fixes the contract failure but not object-existence judgment.

![Free generation and frozen candidate scoring](figures/b1_b2_locked_metrics.png)

![B2 development and locked class recall](figures/b2_development_locked_recall.png)

![B2 locked confusion matrix](figures/b2_locked_confusion_matrix.png)

### 4.3 Ablation Study

The controlled variants isolate three design choices: free generation versus constrained scoring, natural labels versus frozen snake-case candidates, and oracle-crop diagnosis versus real-proposal cascade. These are component studies rather than retrained model ablations; the base weights remain unchanged.

| Variant | Output coverage | Macro-F1 | Main observation |
|---|---|---|---|
| B1 free generation | 0.0000 | 0.0000 | Semantically fluent but contract-invalid |
| B2-A natural candidate diagnostic | 1.0000 | 0.2433 on development | Format repaired, class collapse remains |
| B2-B frozen candidate scoring | 1.0000 | 0.7217 on locked oracle crops | Object classes separable; background recall is 0 |
| RF-DETR plus B2 real-proposal fusion | 1.0000 for 24 calls | 0.96918 on development | No class or metric changes |

The real-proposal cascade is the decisive controlled comparison. SmolVLM returns six hazard and eighteen green-sphere labels, exactly repeating RF-DETR on all 24 proposals. Safe fusion and direct B2 top-1 diagnosis both leave the predictions unchanged. Macro-F1, hazard recall, and FP therefore remain identical while each VLM call adds 5.11 s CPU median latency.

| Metric | RF-DETR only | RF-DETR plus B2 |
|---|---|---|
| Three-class Macro-F1 | 0.96918 | 0.96918 |
| Hazard Recall | 0.98864 | 0.98864 |
| Object false positives | 11 | 11 |
| VLM call rate | 0 | 13.95% |
| B2 CPU P50 / P95 | 0 | 5114.7 / 5539.6 ms |

![Development cascade comparison](figures/cascade_development_comparison.png)

### 4.4 Visualization Analysis

The RF-DETR chart shows balanced F1 across the three object classes, with the weakest result on green spheres. The confusion matrix gives a different view of SmolVLM: object crops are mostly separable, yet every background frame is forced into an object class. This is not a minor formatting issue. It means the VLM cannot independently determine whether a candidate exists under the frozen four-label design.

The Level A storyboard translates ordered detections into robot events and states without coordinates. It contains twelve preselected cases, including correct hazard confirmation, distractor rejection, empty proposals, and low-confidence handling. Repeating the same input produces byte-identical events, logs, and GIF output. The visual is a deterministic interface demonstration, not a continuous trajectory.

![Level A robot-interface replay](figures/level_a_replay_overview.png)

The Level B image shows RGB-D back-projection and temporal confirmation. Four normative output artifacts are byte-identical across two replays. One hazard is matched, one is missed, and no false hazard is reported; the technical score is 23/37. The matched position error of 0.1238 m uses Gazebo pose truth, so it demonstrates internal debug-chain consistency rather than autonomous localization accuracy.

![Level B Gazebo-pose-oracle replay](figures/level_b_pose_oracle_overview.png)

### 4.5 Result Discussion

The strict gain condition fails because Macro-F1 does not increase. The cascade therefore stops before locked-seed execution and multi-view expansion. This is the correct experimental outcome under the preregistered rule. Continuing only to seek a positive result would turn development feedback into unreported tuning and weaken the report's correctness.

The evidence supports RF-DETR-only for the present semantic decision. It does not prove that every VLM cascade is ineffective. The tested VLM sees crops only after a detector proposal, uses a frozen candidate set, and runs on CPU. A different dataset, an explicit no-object candidate, joint image-proposal context, or supervised error predictor might change the result. Those are new hypotheses requiring new frozen protocols, not reinterpretations of this run.

## 5 Reproducibility and Reusable Code

The course repository is available at:

https://github.com/s-max-oss/llm-course-labs

The submitted final-project directory contains `configs`, `scripts`, `src`, `tests`, data manifests, sealed experiment summaries, figures, and the operating manual. Large virtual environments and cache directories are excluded. The RF-DETR checkpoint is identified by SHA-256, and model/runtime versions are recorded. Data source paths are stored only in a local configuration file that is not committed.

The complete research directory is checked with one command: `powershell -ExecutionPolicy Bypass -File scripts/run_checks.ps1`. The final run reports `Ran 87 tests` and `OK`. The compact course ZIP omits an approximately 677 MB robot Git history bundle and deployment tar, but retains their manifests, hashes, patches, correction record, and replay outputs. Full Level B sealing must therefore be repeated in the original research directory.

Reproducibility has two explicit limits. The RF-DETR training image lineage is missing, so the checkpoint cannot be presented as leakage-free. The GitHub URL points to the course repository; the submitted ZIP remains the authoritative snapshot for this report unless the same final-project directory is published at the recorded revision.

## 6 System Demonstration

The demonstration is organized by evidence level instead of presenting a single polished video as proof of the full robot task. Level A consumes frozen RF-DETR predictions and emits `robot_replay_detection_frame_v1` events. It validates class mapping, confidence, input hashes, state transitions from SEARCHING to REPORTED, and fail-closed behavior. It deliberately prohibits world coordinates, temporal tracking, and online control.

Level B v2 consumes a synchronized 11.17 s RGB-D record with 92 acceptable pairs, selects 14 frames, and processes 28 detections. It verifies depth quality, camera back-projection, coordinate formatting, three-hit confirmation, result writing, and deterministic replay. However, the transform chain reads `/Odometry_gazebo`. The authoritative robot contract classifies this topic as Gazebo pose truth, so the G2 truth-isolation gate must remain failed. The demonstration is useful for debugging the data path but cannot be reported as competition-compliant autonomous localization.

| Evidence level | Verified capability | Unsupported conclusion |
|---|---|---|
| Level A | Event schema, state machine, deterministic fail-closed replay | Continuous trajectory, 3D localization, online control |
| Level B v1 | Synchronized RGB-D failure evidence | Formal spatial result or complete truth isolation |
| Level B v2 | RGB-D geometry, confirmation, output format, replay determinism | Autonomous pose, competition compliance, complete building search |

The two GIFs in the submission package show the interface behavior directly. They are accompanied by manifests and validators so that a reader can distinguish a visual explanation from a sealed result. This matches the competition lesson that process evidence must be traced to the exact input, state, and output rather than inferred from a final screenshot.

## 7 Discussion

### 7.1 In-depth Result Interpretation

The study separates three meanings of “the model works.” B1 visually understands enough to produce relevant descriptions, but it fails the interface. B2-B obeys the interface and distinguishes object crops, but it fails background rejection. The real cascade obeys both the input and output contracts, yet it changes no decision. These are different failure layers, and collapsing them into one accuracy value would hide the engineering reason for the final choice.

RF-DETR's high baseline also changes the value of a reviewer. The routed region contains proposals that are uncertain by detector confidence, but uncertainty does not guarantee detector error. SmolVLM mostly sees cases on which the detector is already correct. The remaining detector miss without a proposal is outside the reviewer's reachable set. A better future router should predict likely detector error rather than use confidence alone.

### 7.2 Advantages and Limitations

The main advantage is evidence discipline. Data partitions, task queues, output schemas, candidate strings, fusion rules, stop conditions, and hashes are frozen and independently recomputed. Negative results and failed gates remain in the final narrative. The report also connects an actual competition workflow to a bounded course question without claiming that the course experiment completed the competition.

The limitations are substantial. There are only five historical seeds. Ninety-nine images are isolated for duplicate risk, but this cannot establish detector train-test separation. One hundred seventeen frames lack label files and are not strict human-audited negatives. B2 uses oracle crops. B1 has observed the locked seeds. The cascade runs only on development seeds. CPU latency cannot represent GPU or onboard speed. Level B uses simulator pose truth. These conditions restrict the conclusion to the tested artifacts and protocol.

### 7.3 Course Learning Reflection

The competition experience originally made me view perception as a practical race to raise detection rate. The course changed that view. A robot result is produced by a chain: image evidence becomes a proposal, a proposal becomes a semantic decision, the decision becomes an event, and the event becomes a task record. Each boundary can fail even when the preceding model looks accurate.

The most useful lesson was learning to distinguish model capability from system value. I expected a compact VLM to help on ambiguous red objects because it could use semantic descriptions. The experiment showed that the same model could fail the label contract, repair the contract but fail background rejection, and then add no value on real proposals. Freezing the protocol made it possible to accept that result. Correctly stopping the cascade was a stronger research outcome than forcing an artificial improvement.

### 7.4 Future Work

The next priority is new auditable data, not a larger GPU. New simulator or physical scenes should record seed, room, camera pose, object identity, complete labels, and collection provenance. The detector should be retrained or selected with a published training manifest. Evaluation should include unseen layouts and strict human-verified negative frames.

For the VLM, useful hypotheses include an explicit no-object candidate, joint full-frame and proposal context, calibrated abstention, and a supervised router trained to predict detector error. Multi-view fusion should begin only after a single-frame reviewer demonstrates net gain under the same stop conditions. Level B should replace Gazebo pose truth with an audited autonomous pose chain before any localization or competition-compliance claim is reconsidered.

## 8 Conclusion

This paper converts a concrete quadruped-robot competition problem into an auditable course study of visual and multimodal intelligence. The competition pipeline required a robot to find red spherical hazards, reject similar distractors, and pass stable results to localization and task logic. A duplicate-controlled evaluation of a fixed RF-DETR checkpoint reaches 0.9635 three-class Macro-F1 and 0.9892 hazard recall on 401 historical images, but missing training lineage limits the result to in-domain exploration. SmolVLM free generation fails the strict label interface. Frozen candidate scoring restores complete output coverage and identifies object crops, yet its background recall is zero. On 24 real detector proposals, the VLM repeats every RF-DETR class, changes no metric, and adds 5.11 s CPU median latency. The preregistered gain gate therefore stops the cascade, and RF-DETR-only remains the recommended semantic detector. Level A verifies the event and state-machine interface. Level B verifies an RGB-D debug path but uses Gazebo pose truth and fails the G2 isolation requirement. The final contribution is not a claim of complete competition autonomy. It is a reproducible account of where detector, VLM, and robot-interface evidence agree, where they fail, and why the system decision follows from those boundaries.

## References

[1] Carion N, Massa F, Synnaeve G, et al. End-to-End Object Detection with Transformers. ECCV, 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf

[2] Zhao Y, Lv W, Xu S, et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2024: 16965-16974. https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html

[3] Robinson I, Robicheaux P, Popov M, et al. RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. arXiv:2511.09554, 2025. https://arxiv.org/abs/2511.09554

[4] Liu H, Li C, Wu Q, Lee Y J. Visual Instruction Tuning. NeurIPS, 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html

[5] Marafioti A, Zohar O, Farré M, et al. SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299, 2025. https://arxiv.org/abs/2504.05299

[6] Liu S, Zeng Z, Ren T, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv:2303.05499, 2023. https://arxiv.org/abs/2303.05499

[7] Cheng T, Song L, Ge Y, et al. YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR, 2024: 16901-16911. https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

[8] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017. https://arxiv.org/abs/1706.03762

[9] Cao C, Zhu H, Choset H, Zhang J. TARE: A Hierarchical Framework for Efficiently Exploring Complex 3D Environments. Robotics: Science and Systems, 2021. https://www.roboticsproceedings.org/rss17/p018.pdf

[10] Hugging Face. SmolVLM: Small Yet Mighty Vision Language Model. 2024. https://huggingface.co/blog/smolvlm

[11] Southwest Institute of Technical Physics. DG-202602 Competition Scheme for Autonomous Hazard Search and Recognition with a Quadruped Robot. 2026.

[12] Course Teaching Group. Comprehensive Teaching Materials for Foundations and Applications of Large Models. Harbin Institute of Technology, Shenzhen, 2026.

## Appendix

### A Code and Artifact Access

GitHub repository: https://github.com/s-max-oss/llm-course-labs

The submitted ZIP is the authoritative artifact snapshot for this report. It contains the report, operating manual, source directory, 500 RGB images, 383 label files, data manifests, experiment summaries, RF-DETR checkpoint, figures, two replay GIFs, a complete material inventory, and SHA-256 checksums.

### B Supplementary Evidence and Claim Boundary

The detector run is sealed by `experiments/pilot_v1/rfdetr_run.json`. The locked B2 study is sealed by `experiments/pilot_v1/smolvlm_b2_locked_run.json`. The stopped real-proposal cascade is sealed by `experiments/pilot_v1/cascade_development_run.json`. Level A evidence is under `experiments/replay_v1`, and Level B v2 evidence is under `experiments/level_b_v2`.

The following statements are prohibited by the evidence: leakage-free detector generalization, strict background-negative validation, GPU real-time performance, autonomous Level B localization, competition-compliant spatial closure, and complete building-search validation. The valid conclusion is narrower: RF-DETR-only is the recommended semantic detector under the frozen exploratory protocol; SmolVLM results document model and interface boundaries; robot replays validate specific contracts at their declared evidence levels.
