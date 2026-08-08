import io
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imgutils import preprocess_for_inference 

class PredictionResponse(BaseModel):
    filename:str
    predicted_class: int
    diagnosis_label: str
    confidence: float
DIAGNOSIS_MAP={
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}
app= FastAPI(
    title="Retinal Diagnostic API",
    description="Diabetic Retinopathy Classification API using PyTorch and ResNet18",
    version="1.0.0"
)
script_dir= os.path.dirname(os.path.abspath(__file__))
project_root= os.path.abspath(os.path.join(script_dir,"..",".."))
model_path= os.path.join(project_root, "models", "model.pth")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,5)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained weights from: {model_path}")
else:
    print(f"Warning: Model weights not found at {model_path}")
model= model.to(device)
model.eval()

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
@app.get("/")
def read_root():
    return {"status": "ok", "message":"Retinal Diagnostic API Service is running."}
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile= File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        contents = await file.read()
        image_array = preprocess_for_inference(contents, size=224)
        tensor = inference_transform(image_array).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs= model(tensor)
            probabilities= torch.softmax(outputs, dim=1)
            confidence, predicted= torch.max(probabilities,1)
        pred_class= int(predicted.item())
        conf_score= float(confidence.item())
        return PredictionResponse(
            filename=file.filename,
            predicted_class=pred_class,
            diagnosis_label=DIAGNOSIS_MAP.get(pred_class,"Unknown"),
            confidence=round(conf_score,4)
        )
    except HTTPException:  
        raise             
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    