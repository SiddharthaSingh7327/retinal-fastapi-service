# Risk and Action Definitions

This writes out, in code-ready form: the any-error label (was the grade wrong), the harm cost (if wrong, how dangerous), the ordered four-action 
rule (what to do with each image)

## What we start from (per image)

For every image, the frozen classifier gives us one record:

```
image_id, patient_id, y_true, y_pred, p0..p4, entropy,
embedding_path, quality_score, dataset, split
```

Two shortcuts we use below:
- `p_severe = p3 + p4` (chance the eye is severe or worse)
- `p_referable = p2 + p3 + p4` (chance the eye needs a doctor)

## 1. Any-error label

Was the classifier's grade wrong? Simple yes or no.

```
e = 1 if y_pred != y_true else 0
```

## 2. Harm cost

If the grade is wrong, how bad is that mistake? Scored 0 to 5.

```
def harm_cost(y_true, y_pred):
    if y_pred == y_true:                       return 0   # correct
    if y_true in (3,4) and y_pred in (0,1):    return 5   # called a severe eye healthy (worst)
    if y_true >= 2 and y_pred < 2:             return 4   # missed an eye that needed a doctor
    if abs(y_true - y_pred) >= 2:              return 3   # off by two grades or more
    return 1                                              # off by one grade, nothing missed
```

Note: these cost values are a starting point. They need an eye doctor to confirm them before we call them final.

## 3. Router outputs

A small model reads the classifier's probabilities and predicts two things:

```
any_error_risk     = P(e = 1)     # how likely the grade is wrong
harmful_error_risk = E[c]         # how dangerous being wrong would be
```

Build the simple version first (probabilities only). Later, add the image embedding as a test to see if it helps.

## 4. The four-action rule

Checked top to bottom. The first line that matches wins.

```
def route(p, quality_score, any_error_risk, harmful_error_risk, T):
    # 1. RE-IMAGE: photo is too poor to grade. Checked first.
    if quality_score < T.adequacy:                     return "RE-IMAGE"

    # 2. PRIORITY REVIEW: looks serious or dangerous.
    #    This is NOT based on low confidence. A confident "severe" still goes here.
    if p_severe >= T.priority_severe or harmful_error_risk >= T.priority_harm:
                                                        return "PRIORITY REVIEW"

    # 3. ROUTINE REVIEW: good photo, but uncertain or likely needs a doctor.
    if p_referable >= T.routine_referable or any_error_risk >= T.routine_anyerror:
                                                        return "ROUTINE REVIEW"

    # 4. AUTO-GRADE: good photo, low risk. Trust the grade.
    return "AUTO-GRADE"
```

The key idea: PRIORITY is triggered by how serious the case is, not by the model being unsure. That is what makes this different from a plain "reject when unsure" system.

## 5. Setting the thresholds

All the cut-offs in `T` are set on the calibration split only, never on the test set. Methods are compared at the same review budget: tune so each sends the same share of images to review (10%, 20%, 30%), then compare.
