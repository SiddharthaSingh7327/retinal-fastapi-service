import cv2
import numpy as np
from PIL import Image

def crop_dark_borders(img: np.ndarray, tol:int=7) -> np.ndarray:
    if img.ndim ==2:
        mask=img>tol
        if mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim ==3:
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask= gray>tol
        if not mask.any():
            return img
        check_shape= img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape
        if check_shape[0] ==0 or check_shape[1]==0:
            return img
        channels=[
            img[:,:,c][np.ix_(mask.any(1), mask.any(0))] for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
def crop_dark_borders_pill(pil_image):
    rgb_array= np.array(pil_image)
    bgr_array= rgb_array[:,:,::-1].copy()
    cropped_bgr= crop_dark_borders(bgr_array, tol=7)
    cropped_rgb= cropped_bgr[:,:,::-1]
    return Image.fromarray(cropped_rgb)
