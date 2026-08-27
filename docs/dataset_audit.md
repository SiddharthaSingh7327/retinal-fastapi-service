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
- **Patient ID:** Not available. Confirmed -- `train.csv` contains only `id_code`
  and `diagnosis`; no patient identifier or left/right-eye pairing field exists
  anywhere in the file. This is a real gap, not just an unconfirmed detail: per
  Section 5, patient-level splitting is required "whenever patient identifiers...
  are available," and for APTOS it genuinely is not available, so patient-level
  splitting cannot be enforced on this dataset. Splits on APTOS will necessarily be
  image-level. Worth deciding explicitly (see open items) whether that's acceptable
  given APTOS's role as the external/transfer test set rather than training data.
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
- **Images:** 28,792 total per the repo's README (12,543 train + 16,249 test). The
  train split has been independently downloaded and verified below; the test split
  total is still taken from the README and has not yet been independently
  downloaded/verified.
- **Relationship to EyePACS:** EyeQ is a re-annotated *subset* of EyePACS -- it does
  not ship its own images. Labels are provided as `Label_EyeQ_train.csv` /
  `Label_EyeQ_test.csv`, referencing EyePACS image IDs. Since EyePACS access is
  already confirmed above, this is straightforward to layer on top: match EyeQ's
  labeled image IDs against the EyePACS images already available.
- **ID format:** Confirmed. `Label_EyeQ_train.csv` image IDs (e.g.
  `10009_left.jpeg`) follow the exact same `<patient_id>_<eye>` pattern as
  EyePACS's `trainLabels15.csv`, just with the `.jpeg` extension included. Matching
  EyeQ labels to EyePACS images is a straightforward string match once the
  extension is stripped -- no reformatting needed.
- **Class balance (train split, verified):**

  | Quality | Code | Count | % |
  |---|---|---|---|
  | Good | 0 | 8,347 | 66.6% |
  | Usable | 1 | 1,876 | 15.0% |
  | Reject | 2 | 2,320 | 18.5% |

  Note: the EyeQ README lists Usable (train) as 1,896 -- a small 20-image
  discrepancy from the verified count of 1,876. Not investigated further, as it
  doesn't materially change the class-balance picture.

  For the binary framing used in Section 6.3 (adequate vs. reject), Good + Usable =
  "adequate" (10,223 images, 81.5%) vs. Reject (2,320 images, 18.5%) -- a real but
  manageable ~19% imbalance, comparable in scale to what class weighting already
  handled for the DR classifier.

## Summary / no blockers

No access blockers across any of the three datasets -- all are freely available,
no approval workflows, no cost. EyeQ's dependency on EyePACS is a sequencing note,
not a blocker, since EyePACS access is already confirmed.

## Open items

1. ~~Confirm EyeQ's exact ID format against EyePACS's `<patient>_<eye>` pattern.~~
   **Resolved** -- confirmed identical format, `.jpeg` extension strip is the only
   transformation needed.
2. Resolve APTOS's patient-ID gap -- confirmed there is no patient ID field.
   Decision still needed on how to handle external-test-set splitting without
   patient IDs (image-level splitting is the only option; worth documenting
   whether this is acceptable given APTOS's role as external test data rather
   than training data).
3. ~~Full EyeQ class balance check.~~ **Resolved for the train split** (12,543
   images, see table above). The test split (16,249 images) has not yet been
   independently downloaded and verified -- still open if the test split will be
   used anywhere in this project.
