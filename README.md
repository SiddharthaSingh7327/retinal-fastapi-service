# Beyond Abstention: Actionable DR Triage

A research project investigating how a diabetic retinopathy (DR) screening system can decide the appropriate next action for each fundus image. Beyond predicting a disease grade, the system aims to distinguish poor image quality, uncertain predictions, and potentially harmful under-grading.

Our main question is: **Can severity-aware routing reduce missed referable and severe DR cases at the same expert-review workload?**

## Approach

The planned system combines three components:

1. **Image-quality gate** — determines whether the image is adequate for grading.
2. **Frozen DR classifier** — predicts five severity grades, from No DR to Proliferative DR.
3. **Dual-risk router** — estimates the likelihood of any classification error and the expected harm of that error.

These outputs drive an ordered four-action policy:

| Action | When it applies |
| --- | --- |
| **RE-IMAGE** | Image quality is insufficient for reliable grading. |
| **PRIORITY REVIEW** | Severe/PDR probability or harmful-error risk is high. |
| **ROUTINE REVIEW** | An adequate image has high referable-DR probability or classification-error risk. |
| **AUTO-GRADE** | Image quality is adequate and none of the review conditions applies. |

Priority review can be triggered even when the classifier is confident. The aim is to identify potentially serious cases, rather than relying only on uncertainty.

## Current progress

The earlier proof-of-concept work is retained in `prototype/`. The repository also includes an initial ResNet18 classifier, FastAPI service, and utilities that export class probabilities, entropy, and image embeddings.

The research specifications cover dataset selection, risk targets, action rules, and baseline comparisons. The full quality gate, dual-risk router, threshold calibration, and research evaluation remain to be implemented. Current exports still contain placeholder patient IDs and quality scores.

**Training is planned on university-provided cloud HPC using datasets obtained through Kaggle. Compute time has been requested, and training on that allocation is pending.** The initial ResNet18 pipeline will be adapted for the study; the proposed research backbone is EfficientNet-B0 or ResNet-50.

## Data and training

| Dataset | Planned role |
| --- | --- |
| EyePACS | Primary DR classifier training and internal evaluation. |
| EyeQ or DeepDRiD | Image-quality training and evaluation. |
| APTOS or DDR | External evaluation. |

The proposed data split is:

| Partition | Share | Purpose |
| --- | --- | --- |
| Base-model training | 60% | Train the DR classifier. |
| Base-model validation | 10% | Select the checkpoint and hyperparameters. |
| Router training | 15% | Train risk models on frozen-classifier outputs. |
| Threshold calibration | 5% | Set action thresholds and review budgets. |
| Final internal test | 10% | Evaluate after the models and thresholds are fixed. |

Patient-level separation will be used wherever identifiers are available. The classifier will be frozen before generating router-training outputs, and router-training images will be separate from base-model training images. External datasets will be evaluated separately from these internal partitions.

The structured interface contains image and patient identifiers, true and predicted grades, five class probabilities, entropy, an embedding path, a quality score, and dataset/split metadata.

## Evaluation

We will compare the dual-risk router against maximum-softmax confidence, predictive entropy, MC Dropout, and a generic error predictor at matched review budgets of **10%, 20%, and 30%** of adequate images.

The main outcome is missed referable and Severe/PDR disease among AUTO-GRADE cases at the same review workload. Supporting measures include quality-gate performance, risk-coverage curves, calibration, and external-dataset results. Ablations will test whether embeddings and severity-sensitive risk targets improve routing.

## Next steps

1. Prepare dataset partitions and the university HPC training workflow.
2. Train and freeze the base classifier, then integrate the image-quality gate.
3. Generate structured outputs and train the risk router.
4. Calibrate action thresholds and complete baseline and external evaluations.

This is a research system and has not been clinically validated.
