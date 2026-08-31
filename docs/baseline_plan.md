# Baseline Plan


## Key Focus

At the same amount of expert review, which method leaves the fewest dangerous cases (missed referable or Severe/PDR) wrongly auto-graded?

To keep it fair, every method is compared at the same review budget. We set each method's thresholds (on the calibration split) so they all send the same share of images to review: 10%, then 20%, then 30%. Then we see who catches the most dangerous cases at that same effort.

## Method Comparison

| Method | What it uses | What it tests |
|---|---|---|
| Max softmax | 1 minus the top probability | Is plain confidence enough? |
| Entropy | how spread out the probabilities are | Does the full distribution beat plain confidence? |
| MC Dropout | variance over repeated passes | Does a standard uncertainty method help? |
| Generic error predictor | our model, "is it wrong?" only | Is predicting any error enough? |
| Dual-risk router (ours) | our model, "is it wrong?" plus "is it dangerous?" | Does adding danger help? |

The generic error predictor is the one we most need to beat. It is the same model as ours but without the danger score, so beating it proves the danger idea is what adds value.

## Ablations

1. Probabilities only vs probabilities plus the image embedding.
2. Predicting any error vs predicting how dangerous the error is.
3. A simple yes/no defer decision vs our four-way action decision.

## What we report

Main results:

- Dangerous cases wrongly auto-graded (the primary number).
- How much expert review each method needs.
- How well the quality gate flags bad images.

Supporting results:

- Risk-coverage curve and AURC.
- Failure-prediction AUROC and calibration.
- Results on an external dataset.

## Rules

- Thresholds are set on the calibration split only, never on the test set.
- The test set is reported once, after the method and thresholds are fixed.
- The router only sees predictions from images the base classifier did not train on, so there is no leakage.
- Every method is tested on the same images at the same review budgets, so the only thing that changes is the method.
