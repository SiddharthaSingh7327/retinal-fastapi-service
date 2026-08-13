import cv2
import numpy as np
import sys
import os

def apply_blur(img: np.ndarray, severity: int=1) -> np.ndarray:
    kernel_size= severity *4+1
    return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

def apply_noise(img: np.ndarray, severity: int=1) -> np.ndarray:
    std_dev= severity*10
    noise= np.random.normal(0, std_dev, img.shape).astype(np.float32)
    noisy= img.astype(np.float32) +noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_brightness_shift(img: np.ndarray, severity: int=1, direction: str= "dark") -> np.ndarray:
    shift_amount= severity *15
    if direction== "dark":
        shift_amount= -shift_amount
    return cv2.convertScaleAbs(img, alpha=1.0, beta= shift_amount)

def apply_compression_artifacts(img: np.ndarray, severity: int=1) -> np.ndarray:
    quality= max(10, 90 - severity*16)
    encode_params= [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded_img= cv2.imencode(".jpg", img, encode_params)
    if not success:
        return img
    decoded_img= cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return decoded_img

DEGRADATION_FUNCTION={
    "blur": apply_blur,
    "noise": apply_noise,
    "brightness_dark": lambda img, severity: apply_brightness_shift(img, severity, direction="dark"),
    "brightness_bright": lambda img, severity: apply_brightness_shift(img, severity, direction="bright"),
    "compression": apply_compression_artifacts,
}

def apply_degradation(img: np.ndarray, degradation_type: str, severity: int =1) -> np.ndarray:
    if degradation_type not in DEGRADATION_FUNCTION:
        raise ValueError(
            f"Unknown degradation type '{degradation_type}'."
            f"Choose from: {list(DEGRADATION_FUNCTION.keys())}"
        )
    return DEGRADATION_FUNCTION[degradation_type](img, severity)

if __name__ == "__main__":
    if len(sys.argv !=2):
        print("usage: python degradation.py <path_to_test_image")
        sys.exit(1)

    img_path= sys.argv[1]
    img= cv2.imread(img_path)
    if img is None:
        print(f"COuld not read image at {img_path}")
        sys.exit(1)

    output_dir= "degradation_test_output"
    os.makedirs(output_dir, exist_ok=True)

    for name, func in DEGRADATION_FUNCTION.items():
        for severity in [1,3,5]:
            degraded= func(img, severity)
            out_path= os.path.join(output_dir, f"{name}_severity{severity}.png")
            cv2.imwrite(out_path, degraded)
            print(f"Saved: {out_path}")

    print(f"\nDone. Check the '{output_dir}' folder to see each degradation applied")