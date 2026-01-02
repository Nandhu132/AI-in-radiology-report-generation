import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

IMAGE_DIR = r"D:\Lauren_Projects\Ai-assisted radiology report generation\datasets\Images zip\Images zip\images-small"
TRAIN_CSV = r"D:\Lauren_Projects\Ai-assisted radiology report generation\datasets\train.csv"
VAL_CSV   = r"D:\Lauren_Projects\Ai-assisted radiology report generation\datasets\valid.csv"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
MODEL_SAVE_PATH = "densenet121_chestxray.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

NUM_CLASSES = len(LABEL_COLS)
print("Classes:", LABEL_COLS)

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["Image"])
        image = Image.open(img_path).convert("RGB")

        labels = row[LABEL_COLS].values.astype("float32")
        labels = np.nan_to_num(labels, nan=0.0)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = ChestXrayDataset(TRAIN_CSV, IMAGE_DIR, train_transform)
val_dataset   = ChestXrayDataset(VAL_CSV, IMAGE_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.to(DEVICE)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

x, y = train_dataset[0]
print("Sanity check labels -> Min:", y.min().item(), "Max:", y.max().item())


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Val Loss: {avg_val_loss:.4f}"
    )

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved at:", MODEL_SAVE_PATH)
