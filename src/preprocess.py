import os
import cv2
import numpy as np

def crop_dark_borders(img, tol=7):
    if img.ndim==2:
        mask=img>tol
        return img[np.ix_(mask.any(1), mask.any)]