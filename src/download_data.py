import os
import pandas as pd
import subprocess

def downloadSampleData(numSamples=250):
	scriptDir = os.path.dirname(os.path.abspath(__file__))
	projectRoot = os.path.abspath(os.path.join(scriptDir, ".."))

	imageDir = os.path.join(projectRoot, "data", "raw", "train_images")
	csvPath = os.path.join(projectRoot, "data", "raw", "train.csv")
	os.makedirs(imageDir, exist_ok=True)
	if not os.path.exists(csvPath):
		print("Error, the file train.csv was not found")
		return
	df= pd.read_csv(csvPath)
	sampleIds= df["id_code"].head(numSamples).tolist()
	print(f"Downloading {len(sampleIds)} sample fundus image..")
	for idx, imageId in enumerate(sampleIds, start=1):
		filename= f"train_images/{imageId}.png"
		cmd= f"kaggle competitions download -c aptos2019-blindness-detection -f {filename} -p {imageDir}/"
		subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		zip_path= f"{imageDir}/{imageId}.png.zip"
		if os.path.exists(zip_path):
			subprocess.run(f'unzip -o -q "{zip_path}" -d "{imageDir}/"', shell=True)
			os.remove(zip_path)
		if idx % 50==0 or idx == numSamples:
			print(f"Progress: {idx}/{numSamples} images processed.")
	print("Sample download complete")
if __name__ == "__main__":
    downloadSampleData(250)