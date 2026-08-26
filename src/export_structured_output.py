"""
src/export_structured_output.py

Runs the trained model on a sample of images and exports the structured
record format required by the research plan (Section 6.2): one CSV row per
image, plus a saved embedding file per image.

quality_score and patient_id are placeholders for now -- no quality gate
exists yet, and APTOS doesn't have verified patient IDs.

Run:
    python src/export_structured_output.py --num_images 20
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imgutils import preprocess_for_inference  # noqa: E402


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_frozen_model(model_path: str, device):
    """Loads the model and hooks avgpool so we capture embeddings during normal prediction."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    captured = {}

    def hook(module, input, output):
        captured["embedding"] = output.detach().cpu().numpy().reshape(output.shape[0], -1)

    model.avgpool.register_forward_hook(hook)
    return model, captured


def predict_with_embedding(model, captured, image_rgb: np.ndarray, device) -> dict:
    tensor = transforms.functional.to_tensor(image_rgb)
    tensor = NORMALIZE(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    embedding = captured["embedding"][0]
    y_pred = int(np.argmax(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    return {"y_pred": y_pred, "probs": probs, "entropy": entropy, "embedding": embedding}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--dataset_name", type=str, default="aptos2019")
    parser.add_argument("--split_name", type=str, default="prototype")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "model.pth")
    csv_path = os.path.join(project_root, "data", "raw", "train.csv")
    img_dir = os.path.join(project_root, "data", "raw", "train_images")
    output_dir = os.path.join(project_root, "outputs", "structured_export")
    embedding_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        print(f"ERROR: no trained model found at {model_path}")
        sys.exit(1)

    model, captured = load_frozen_model(model_path, device)

    df = pd.read_csv(csv_path)
    existing_files = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
    df["filename"] = df["id_code"].apply(lambda x: f"{x}.png")
    df = df[df["filename"].isin(existing_files)].reset_index(drop=True)

    if len(df) < args.num_images:
        print(f"WARNING: only {len(df)} local images available, using all of them.")
    sample_df = df.head(args.num_images)

    records = []
    print(f"Running export on {len(sample_df)} images...")

    for i, row in enumerate(sample_df.itertuples(), start=1):
        img_path = os.path.join(img_dir, row.filename)
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        image_array = preprocess_for_inference(image_bytes, size=224)
        result = predict_with_embedding(model, captured, image_array, device)

        embedding_path = os.path.join(embedding_dir, f"{row.id_code}.npy")
        np.save(embedding_path, result["embedding"])

        records.append({
            "image_id": row.id_code,
            "patient_id": row.id_code,  # This is simply a placeholder
            "y_true": int(row.diagnosis),
            "y_pred": result["y_pred"],
            "p0": result["probs"][0],
            "p1": result["probs"][1],
            "p2": result["probs"][2],
            "p3": result["probs"][3],
            "p4": result["probs"][4],
            "entropy": result["entropy"],
            "embedding_path": os.path.relpath(embedding_path, project_root),
            "quality_score": np.nan,  # This is a simply a placeholder
            "dataset": args.dataset_name,
            "split": args.split_name,
        })

        if i % 10 == 0 or i == len(sample_df):
            print(f"  {i}/{len(sample_df)} processed")

    output_df = pd.DataFrame(records)
    csv_out_path = os.path.join(output_dir, "structured_output_prototype.csv")
    output_df.to_csv(csv_out_path, index=False)

    print(f"\nDone. Wrote {len(output_df)} rows to {csv_out_path}")
    print(f"Embeddings saved to {embedding_dir}/")
    print("quality_score and patient_id are placeholders -- see script docstring.")


if __name__ == "__main__":
    main()