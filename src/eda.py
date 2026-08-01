import os
import pandas as pd

def inspect_dataset():
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    projectRoot = os.path.abspath(os.path.join(scriptDir, ".."))
    imageDir = os.path.join(projectRoot, "data", "raw", "train_images")
    csvPath = os.path.join(projectRoot, "data", "raw", "train.csv")

    if not os.path.exists(csvPath):
        print("Error, train.csv not found")
        return
    df= pd.read_csv(csvPath)
    print("Dataset Overview")
    print(f"Total rows in train.csv: {len(df)}")

    download_files= os.listdir(imageDir) if os.path.exists(imageDir) else []
    print(f"Download images in train_images/: {len(download_files)}")
    print("\n Diagnosis Label Breakdown (0= DR, 4=Sever DR)")
    print(df['diagnosis'].value_counts().sort_index())
if __name__=="__main__":
    inspect_dataset()

