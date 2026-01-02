import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "densenet121_chestxray.pth"
TEST_IMAGE = "D:/Lauren_Projects/Ai-assisted radiology report generation/datasets/Images zip/Images zip/images-small/00022982_000.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(" Model loaded")

image = Image.open(TEST_IMAGE).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(image)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

print("\n--- Prediction Results ---")
for label, prob in zip(LABEL_COLS, probs):
    print(f"{label:20s}: {prob:.3f}")

THRESHOLD = 0.5
predicted = [LABEL_COLS[i] for i, p in enumerate(probs) if p >= THRESHOLD]

print("\nPredicted Diseases (THRESHOLD = 0.5):")
print(predicted if predicted else "None detected")