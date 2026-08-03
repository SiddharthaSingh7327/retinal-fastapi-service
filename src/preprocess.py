import os
import cv2
import numpy as np

def crop_dark_borders(img, tol=7):
    if img.ndim==2:
        mask=img>tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim==3:
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask= gray>tol
        check_shape= img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape
        if check_shape[0]==0 or check_shape[1]==0:
            return img
        else:
            img1= img[:,:, 0][np.ix_(mask.any(1), mask.any(0))]
            img2= img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3= img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2,img3], axis=-1)
def preprocess_images(img_size=224):
    scdript_dir= os.path.dirname(os.path.abspath(__file__))
    project_root= os.path.abspath(os.path.join(scdript_dir,".."))
    raw_dir= os.path.join(project_root, "data", "raw", "train_images")
    output_dir= os.path.join(project_root, "data", "processed", "train_images")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        print(f"Error: Raw images folder '{raw_dir}' does not exist")
        return
    image_files= [f for f in os.listdir(raw_dir) if f.endswith('.png')]
    print(f"Preprocessing {len(image_files)} sample images to {img_size}x{img_size}...")
    processed_count=0
    for idx, fname in enumerate(image_files, start=1):
        src_path =os.path.join(raw_dir, fname)
        dst_path= os.path.join(output_dir, fname)
        img= cv2.imread(src_path)
        if img is None:
            continue
        cropped= crop_dark_borders(img)
        resized= cv2.resize(cropped, (img_size, img_size))
        cv2.imwrite(dst_path, resized)
        processed_count+=1
        if idx % 50 ==0 or idx == len(image_files):
            print(f"Processed: {idx}/{len(image_files)} images")
    print(f"Preprocessing finished. Saved {processed_count} images to data/processed/train_images/")
if __name__ == "__main__":
    preprocess_images()
    