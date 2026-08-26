# Dataset Audit

Audit for the "Beyond Abstention" research plan (Section 9 deliverable, due before the
first Fall meeting). Covers the three datasets the plan specifies: EyePACS (primary
DR training data), EyeQ (image-quality labels), APTOS (external test set).

## EyePACS (primary DR classifier training data)

- **Access:** Free. Available via a community-uploaded resized mirror on Kaggle
  ("Resized 2015 & 2019 Diabetic Retinopathy Detection", Wei-Ming Lin). No approval
  process required, unlike some other medical imaging datasets.
- **Images:** 35,126 (matches the known full EyePACS 2015 training set size).
- **Labels:** `trainLabels15.csv`, two columns: `image`, `level` (DR grade, 0-4).
  Same grading scale as APTOS.
- **Patient ID:** Extractable directly from the `image` filename, which follows the
  pattern `<patient_id>_<eye>` (e.g. `10_left`, `10_right`). Confirmed 17,563 unique
  patients across 35,126 images (i.e. essentially every patient has both eyes
  photographed).
- **Leakage risk finding:** the same patient's two eyes can carry *different* DR
  grades (verified example: patient 15's left eye graded 1, right eye graded 2).
  This confirms patient-level splitting is not just a precaution but a real
  requirement -- image-level splitting would let highly correlated images from the
  same patient appear in both train and test.
- **Class balance:**

  | Class | Count | % |
  |---|---|---|
  | 0 (No DR) | 25,810 | 73.5% |
  | 1 (Mild) | 2,443 | 7.0% |
  | 2 (Moderate) | 5,292 | 15.1% |
  | 3 (Severe) | 873 | 2.5% |
  | 4 (Proliferative) | 708 | 2.0% |

  Notably more imbalanced than APTOS -- severe + proliferative combined make up only
  4.5% of the dataset (~1,581 images out of 35,126). This will matter for the
  asymmetric cost protocol in Section 6.5, which specifically weights missed
  severe/proliferative cases heavily -- exactly the classes with the least data here.

## APTOS (external test set)

- **Access:** Free, official Kaggle competition ("aptos2019-blindness-detection").
  Already in use as the primary training set for the current prototype model --
  per the plan, this should be repositioned as the external/transfer test set once
  the EyePACS-trained base classifier exists.
- **Images:** 3,662.
- **Labels:** `train.csv`, columns `id_code`, `diagnosis` (0-4), same scale as EyePACS.
- **Patient ID:** Not confirmed. `id_code` appears to be an image-level identifier
  only; no left/right eye pairing pattern found. Flagged as unresolved -- may need to
  treat each image as its own "patient" for splitting purposes on this dataset, or
  investigate further if patient-level splitting is required here too.
- **Class balance:**

  | Class | Count | % |
  |---|---|---|
  | 0 (No DR) | 1,805 | 49.3% |
  | 1 (Mild) | 370 | 10.1% |
  | 2 (Moderate) | 999 | 27.3% |
  | 3 (Severe) | 193 | 5.3% |
  | 4 (Proliferative) | 295 | 8.1% |

  Less imbalanced than EyePACS, though still majority "No DR."

## EyeQ (image-quality gate training data)

- **Access:** Free, public GitHub repo (`HzFu/EyeQ`). No approval process.
- **Images:** 28,792, labeled across three quality tiers: Good, Usable, Reject.
- **Relationship to EyePACS:** EyeQ is a re-annotated *subset* of EyePACS -- it does
  not ship its own images. Labels are provided as `Label_EyeQ_train.csv` /
  `Label_EyeQ_test.csv`, referencing EyePACS image IDs. Since EyePACS access is
  already confirmed above, this is straightforward to layer on top: match EyeQ's
  labeled image IDs against the EyePACS images already available.
- **Not yet checked:** exact class balance across Good/Usable/Reject, and whether
  the ID format in EyeQ's CSVs matches the `<patient>_<eye>` format found in
  `trainLabels15.csv` directly, or needs reformatting.

## Summary / no blockers

No access blockers across any of the three datasets -- all are freely available,
no approval workflows, no cost. EyeQ's dependency on EyePACS is a sequencing note,
not a blocker, since EyePACS access is already confirmed.

## Open items before EyePACS training begins

1. Confirm EyeQ's exact ID format against EyePACS's `<patient>_<eye>` pattern.
2. Resolve APTOS's patient-ID gap (may not have one -- needs a decision on how to
   handle external-test-set splitting without patient IDs).
3. Full EyeQ class balance check.