import sys
import os
import cv2
import numpy as np
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from imgutils import crop_dark_borders, preprocess_for_inference
def main():
    if len(sys.argv)!=2:
        print("Usage: python check_crop_consistency.py <path_to_image>")
        sys.exit(1)
    img_path=sys.argv[1]

    cv_img= cv2.imread(img_path)
    if cv_img is None:
        print(f"Could not read image at {img_path}")
        sys.exit(1)
    cropped_cv= crop_dark_borders(cv_img)
    resized_cv_bgr= cv2.resize(cropped_cv, (224, 224))
    resized_cv= cv2.cvtColor(resized_cv_bgr, cv2.COLOR_BGR2RGB)
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    resized_pil_as_bgr = preprocess_for_inference(image_bytes, size=224)
    if resized_cv.shape != resized_pil_as_bgr.shape:
        print(f"MISMATCH: shapes differ -- preprocess.py path: {resized_cv.shape}, API path: {resized_pil_as_bgr.shape}")
        sys.exit(1)
    diff= np.abs(resized_cv.astype(int)- resized_pil_as_bgr.astype(int))
    max_diff= diff.max()
    mean_diff= diff.mean()
    print(f"Shapes match: {resized_cv.shape}")
    print(f"Max pixel difference: {max_diff}")
    print(f"Mean pixel difference: {mean_diff:.4f}")
 
    if max_diff <= 2:
        print("PASS: crops are effectively identical (minor diff expected from JPEG/PNG rounding).")
    else:
        print("FAIL: crops diverge more than expected -- investigate preprocess.py vs imgutils.py.")
        sys.exit(1)
if __name__ == "__main__":
    main()