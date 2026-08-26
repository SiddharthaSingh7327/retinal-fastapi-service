"""
src/verify_structured_output.py

Sanity-checks structured_output_prototype.csv for internal consistency --
NOT a model-accuracy check (that's train.py's job). This checks that the
export script correctly recorded what the model actually did.

Checks performed, per row:
  1. p0..p4 sum to ~1.0 (valid probability distribution)
  2. y_pred actually equals argmax(p0..p4) -- catches export bugs
  3. entropy matches what you'd recompute from p0..p4 by hand
  4. the referenced embedding .npy file exists and has the right shape
  5. y_true matches the label in train.csv for that image_id (catches a
     mismatched/misaligned row)

Run:
    python src/verify_structured_output.py
"""

import os
import sys

import numpy as np
import pandas as pd

TOLERANCE = 1e-4


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "outputs", "structured_export", "structured_output_prototype.csv")
    train_csv_path = os.path.join(project_root, "data", "raw", "train.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found -- run export_structured_output.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    train_df = pd.read_csv(train_csv_path).set_index("id_code")["diagnosis"]

    issues = []

    for i, row in df.iterrows():
        probs = np.array([row["p0"], row["p1"], row["p2"], row["p3"], row["p4"]])

        # Check 1: probabilities sum to 1
        prob_sum = probs.sum()
        if abs(prob_sum - 1.0) > TOLERANCE:
            issues.append(f"Row {i} ({row['image_id']}): probs sum to {prob_sum:.6f}, not 1.0")

        # Check 2: y_pred matches argmax of the probabilities
        expected_pred = int(np.argmax(probs))
        if expected_pred != row["y_pred"]:
            issues.append(f"Row {i} ({row['image_id']}): y_pred={row['y_pred']} but argmax(probs)={expected_pred}")

        # Check 3: entropy matches a fresh recomputation
        recomputed_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        if abs(recomputed_entropy - row["entropy"]) > TOLERANCE:
            issues.append(
                f"Row {i} ({row['image_id']}): stored entropy={row['entropy']:.6f}, "
                f"recomputed={recomputed_entropy:.6f}"
            )

        # Check 4: embedding file exists and has the expected shape
        embedding_full_path = os.path.join(project_root, row["embedding_path"])
        if not os.path.exists(embedding_full_path):
            issues.append(f"Row {i} ({row['image_id']}): embedding file missing at {embedding_full_path}")
        else:
            emb = np.load(embedding_full_path)
            if emb.shape != (512,):
                issues.append(f"Row {i} ({row['image_id']}): embedding shape {emb.shape}, expected (512,)")

        # Check 5: y_true matches train.csv's actual label for this image
        if row["image_id"] in train_df.index:
            true_label = int(train_df.loc[row["image_id"]])
            if true_label != row["y_true"]:
                issues.append(
                    f"Row {i} ({row['image_id']}): y_true={row['y_true']} but train.csv says {true_label}"
                )
        else:
            issues.append(f"Row {i} ({row['image_id']}): image_id not found in train.csv at all")

    print(f"Checked {len(df)} rows.\n")
    if issues:
        print(f"FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("PASS: all rows internally consistent -- probabilities valid, "
              "y_pred/entropy correctly derived, embeddings present, labels match train.csv.")


if __name__ == "__main__":
    main()