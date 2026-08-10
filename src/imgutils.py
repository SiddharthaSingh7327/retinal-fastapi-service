'''
imgutils.py

this is a shared image preprocessing utilities which is used by both the offline
training pipline(preprocess.py) and the the live inference API(src/api/app.py)

The reason for this file to exist is as a shared module rather than duplicated code 
is beacuse Fundus (retina) photogtaphs typically have irregular black border around the 
circular eye image. If the crop/resisize steps used to prepare training images dont exactly
match the stpes used at inference time, the model ends up ebing valuated on different image
than the one is was trained on. This hurts the accuracy but does not throw an error. This
file is the single source of truth for that preprocessing so that both parts can never drift 
apart. 
'''
import cv2
import numpy as np
from PIL import Image

def crop_dark_borders(img: np.ndarray, tol:int=7) -> np.ndarray:
    '''
    Crop the black/near black borders common in fundus photogrpahs.

    It works by building a boolean mask of bright enough pixels, then finds the bounding 
    box of that mask and crops to it. This removes the black borders without needing to 
    know their exact size ahead of time.

    Args:
        img: Image as a numpy array, either grayscale or color.
             Color images are expected in BGR order (OpenCV's default), but
             the crop logic itself is color-order agnostic.
        tol: Pixel intensity threshold (0-255). Pixels at or below this value
             are treated as "background" and are candidates for cropping.
    '''
    if img.ndim ==2:
        mask=img>tol
        if not mask.any():
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
def crop_dark_borders_pil(pil_image):
    '''

    this function isnt actually used anywhere right now. It was an
    earlier attempt at cropping images for the API, before we found the
    resizing bug described above and switched to preprocess_for_inference()
    instead. Keeping it here in case it's useful again someday, but it's
    not part of how predictions currently work.
    
    '''
    rgb_array= np.array(pil_image)
    bgr_array= rgb_array[:,:,::-1].copy()
    cropped_bgr= crop_dark_borders(bgr_array, tol=7)
    cropped_rgb= cropped_bgr[:,:,::-1]
    return Image.fromarray(cropped_rgb)
def preprocess_for_inference(image_bytes: bytes, size: int = 224):
    '''
    This is the function the live API calls every time someone uploads a
    photo. It takes the raw uploaded file and gets it ready for the model
    by doing the exact same 3 steps used when the model was trained:
        1. Open the image
        2. Cut off the black border 
        3. Resize it to a fixed size 
    
    The main reason that this function exist is because an earlier version 
    of this project did step 2 correctly but did step 3 using a different tool
    than training used. Two different resizing tools don't always produce
    exactly the same pixels kind of like how two different photo apps
    might shrink a picture slightly differently. That tiny difference was
    enough to measurably hurt the models predictions. This function makes
    sure the exact same resizing method is used every time, for both
    training and live predictions.    
    '''
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr_img is None:
        raise ValueError("Could not decode image bytes")
    cropped_bgr = crop_dark_borders(bgr_img, tol=7)
    resized_bgr = cv2.resize(cropped_bgr, (size, size))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return resized_rgb
