import pandas as pd

structured_df = pd.read_csv("outputs/structured_export/structured_output_prototype.csv")
quality_df = pd.read_csv("data/processed/quality_scores_aptos.csv")

merged = structured_df.merge(quality_df, on="image_id", how="left", suffixes=("", "_new"))
merged["quality_score"] = merged["quality_score_new"]
merged = merged.drop(columns=["quality_score_new"])

assert len(merged) == len(structured_df), "row count changed"
assert merged["quality_score"].isna().sum() == 0, "some rows didn't get a score"

merged.to_csv("outputs/structured_export/structured_output_prototype.csv", index=False)
print("Done —", len(merged), "rows updated")