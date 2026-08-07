import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report

class RetinalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df=pd.read_csv(csv_file)
        self.img_dir= img_dir
        self.transform= transform
        existing_files = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
        self.df['filename']= self.df['id_code'].apply(lambda x: f"{x}.png")
        self.df= self.df[self.df['filename'].isin(existing_files)].reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_name=self.df.iloc[idx]['filename']
        img_path= os.path.join(self.img_dir, img_name)
        image= Image.open(img_path).convert('RGB')
        label= int(self.df.iloc[idx]['diagnosis'])
        if self.transform:
            image=self.transform(image)
        return image, label
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels=[],[]
    with torch.np_grid():
        for images, labels in dataloader:
            images= images.to(device)
            outputs= model(images)
            _, predicted= torch.max(outputs,1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
    model.train()
    return all_preds,all_labels
def train_model(epochs=3, batch_size=16, lr=0.001, val_split=0.2, seed=42):
        script_dir= os.path.dirname(os.path.abspath(__file__))
        project_root= os.path.abspath(os.path.join(script_dir, ".."))
        csv_path= os.path.join(project_root, "data","raw","train.csv")
        img_dir= os.path.join(project_root, "data", "processed","train_images")
        output_model_dir= os.path.join(project_root,"models")
        os.makedirs(output_model_dir, exist_ok=True)
        model_save_path= os.path.join(output_model_dir, "model.pth")
        data_transforms= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456,0.406],std=[0.229,0.224,0.225])
        ])
        print("Loading PyTorch Dataset")
        full_dataset = RetinalDataset(csv_file=csv_path, img_dir=img_dir, transform=data_transforms)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Dataset loaded: {len(full_dataset)} samples found")
        val_size= int(len(full_dataset)* val_split)
        train_size= len(full_dataset)- val_size
        generator= torch.Generator().manual_seed(seed)
        train_dataset, val_dataset= random_split(
             full_dataset, [train_size, val_size], generator=generator
        )
        print(f"Split: {train_size} train / {val_size} val")
        train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle= False)
        model= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs= model.fc.in_features
        model.fc= nn.Linear(num_ftrs,5)
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= model.to(device)
        criterion= nn.CrossEntropyLoss()
        optimizer= torch.optim.Adam(model.parameters(), lr=lr)
        print("Starting training process")
        model.train()
        for epoch in range(epochs):
            running_loss=0.0
            correct=0
            total=0
            for images, labels in train_loader:
                images,labels= images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs=model(images)
                loss= criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_loss = running_loss /total
            epoch_acc = correct /total
            val_preds, val_labels= evaluate(model, val_loader, device)
            val_acc= sum(p==1 for p, 1 in zip(val_preds, val_labels)) /max(len(val_labels), 1)
            print(
                 f"Epoch {epoch +1}/{epochs} -",
                 f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - ",
                 f"Val Acc: {val_acc:.4f}"
            )
            #print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        print("\nFinal validation classification report (per-class):")
        final_preds, final_labels= evaluate(model, val_loader, device)
        print(
             classification_report(
                  final_labels,
                  final_preds,
                  labels=[0,1,2,3,4],
                  target_names=[f"Class {i}" for i in range(5)],
                  zero_devision=0,
             )
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"Training finished. Saved model weights to {model_save_path}")

if __name__ == "__main__":
        train_model(epochs=3)
