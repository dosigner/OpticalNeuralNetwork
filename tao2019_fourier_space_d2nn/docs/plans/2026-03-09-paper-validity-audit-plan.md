# Paper Validity Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the Tao 2019 FD2NN saliency claim is reproducible, under-specified, weak, or unsupported in this codebase, using falsifiable controls rather than more local tuning.

**Architecture:** Replace incremental knob-tuning with a staged audit. First lock and document all paper-vs-code deviations. Then run positive controls, negative controls, and simpler baseline models under the same data protocol. Finally, interpret outcomes with an explicit evidence rubric so we distinguish “implementation mismatch,” “missing paper details,” and “claim does not hold up here.”

**Tech Stack:** Python, PyTorch, YAML configs, existing `tao2019_fd2nn` training CLI, local reports/figures, pytest, JSON/Markdown experiment summaries.

---

### Task 1: Freeze the current evidence and deviation list

**Files:**
- Create: `reports/paper_validity_audit_2026-03-09.md`
- Inspect: `src/tao2019_fd2nn/config/saliency_cifar_fu2013_cat2horse_bs10_100pad160_f2mm.yaml`
- Inspect: `reports/gt_quality_diagnosis_2026-03-09.md`
- Inspect: `reports/sbn_binary_ablation_2026-03-09.md`
- Inspect: `reports/loss_function_experiment_report.md`

**Step 1: Write a paper-vs-code deviation table**

Include at minimum:
- GT source mismatch
- pseudo-mask generation mismatch
- loss mismatch
- SBN placement/normalization mismatch
- saliency metric scope and crop choices
- any training budget or seed differences

**Step 2: State the current strongest local result**

Record:
- best CIFAR saliency pilot under current audit path
- best full run remembered so far
- exact config lineage that produced them

**Step 3: Add an evidence rubric**

Use four buckets:
- `implementation bug`
- `paper under-specified`
- `claim weak but directionally true`
- `claim unsupported in this reproduction`

### Task 2: Add positive controls

**Files:**
- Create: `tmp/audit_mnist_positive_control.yaml`
- Create: `tmp/audit_cifar_binary_reference.yaml`
- Create: `reports/paper_validity_positive_controls_2026-03-09.json`

**Step 1: Re-run a classification positive control**

Purpose:
- verify this codebase can reproduce an easier paper claim before judging saliency

Run:
- 5-layer FD2NN MNIST classification config closest to Fig. 4 / Supp. S7

Expected interpretation:
- if this also fails badly, the issue may still be implementation or optical modeling

**Step 2: Re-run the best current CIFAR saliency reference**

Purpose:
- establish the current strongest in-house saliency baseline before challenge tests

Run:
- binary GT reference pilot
- optional full run only if pilot remains strongest

### Task 3: Add negative controls that should destroy a real effect

**Files:**
- Create: `tmp/audit_cifar_binary_random_gt.yaml`
- Create: `tmp/audit_cifar_binary_shuffled_pairs.yaml`
- Create: `tmp/audit_cifar_binary_random_phase.yaml`
- Create: `reports/paper_validity_negative_controls_2026-03-09.json`

**Step 1: Random-GT control**

Keep input images the same, replace train masks with random binary masks of matched foreground ratio.

Expected:
- Fmax should collapse
- output should lose object correlation

**Step 2: Shuffled-pair control**

Keep real masks, but mismatch images and masks across samples.

Expected:
- if performance remains close to the reference, the system is exploiting priors rather than image-specific cues

**Step 3: Random-phase or frozen-phase control**

Train nothing or use randomized phase masks while keeping evaluation protocol unchanged.

Expected:
- strong drop from the trained model
- otherwise the apparent gain may come mostly from preprocessing/metric bias

### Task 4: Add simpler baseline models under the same data protocol

**Files:**
- Create: `scripts/train_binary_teacher_baseline.py`
- Create: `tmp/audit_cifar_binary_teacher_baseline.yaml`
- Create: `reports/paper_validity_simple_baselines_2026-03-09.json`

**Step 1: Train a small non-optical teacher**

Use a tiny CNN or shallow U-Net-like model on the exact same binary GT data.

Purpose:
- determine whether the GT is learnable in an ordinary model

Interpretation:
- if the tiny CNN cleanly learns object masks while FD2NN remains blob-like, the bottleneck is optical/modeling, not GT alone

**Step 2: Add a center-prior baseline**

Implement a trivial predictor that always emits:
- a centered Gaussian
- or the mean train mask

Expected:
- if this baseline approaches FD2NN Fmax, the task/protocol is too permissive and the claim is weak

### Task 5: Audit whether the metric can be gamed

**Files:**
- Create: `scripts/evaluate_saliency_nulls.py`
- Create: `reports/paper_validity_metric_audit_2026-03-09.md`

**Step 1: Evaluate null predictors**

Test:
- center Gaussian
- mean binary mask
- blurred center square
- all-zero
- all-one

on the same validation split and crop rules.

**Step 2: Compare to FD2NN**

If nulls are too close to the best FD2NN score, conclude the current metric/data protocol does not strongly support the saliency claim.

### Task 6: Add an optical-setup falsification pass

**Files:**
- Create: `tmp/audit_cifar_binary_f1_1mm.yaml`
- Create: `tmp/audit_cifar_binary_spacing_50um.yaml`
- Create: `tmp/audit_cifar_binary_spacing_200um.yaml`
- Create: `reports/paper_validity_optics_2026-03-09.json`

**Step 1: Change one optical degree of freedom at a time**

Use the binary GT and the strongest currently supported SBN setting.

Test:
- `f1=f2=1 mm` vs `2 mm`
- layer spacing `50 um`, `100 um`, `200 um`

Purpose:
- see whether the claim is hypersensitive to undocumented optical settings

Interpretation:
- if tiny optical changes flip the conclusion, the paper claim is fragile or under-specified

### Task 7: Decide the outcome with an explicit verdict rule

**Files:**
- Create: `reports/paper_validity_verdict_2026-03-09.md`

**Step 1: Apply this verdict table**

Use:
- positive control result
- negative control result
- teacher baseline result
- null predictor result
- optical sensitivity result

Verdict logic:
- Positive controls fail: do **not** accuse the paper; conclude reproduction stack still not trustworthy
- Positive controls pass, saliency fails, null baselines are close: conclude saliency claim is weak or protocol-dependent here
- Positive controls pass, saliency fails, teacher succeeds, null baselines are far worse: conclude FD2NN architecture/modeling is the bottleneck
- Negative controls do not collapse: conclude current protocol is not validating image-specific learning

**Step 2: Separate scientific conclusion from accusation**

Allowed conclusion:
- “This reproduction does not support the saliency claim under accessible details.”

Not allowed conclusion without extraordinary evidence:
- “The paper is fraudulent.”

### Task 8: Minimal execution order

**Files:**
- No new files

**Step 1: Execute in this order**

1. deviation table
2. MNIST positive control
3. binary GT reference freeze
4. random-GT + shuffled-pair controls
5. center-prior + mean-mask baselines
6. tiny CNN teacher
7. optical sensitivity pass
8. verdict

**Step 2: Stop condition**

Stop early if either becomes true:
- null predictors are too close to FD2NN
- negative controls do not materially collapse

That means further local tuning is low value until protocol validity is fixed.
