import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "D:/Lauren_Projects/Ai-assisted radiology report generation/densenet121_chestxray.pth"
TEST_IMAGE = "D:/Lauren_Projects/Ai-assisted radiology report generation/datasets/Images zip/Images zip/images-small/00001232_002.png"

AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

THRESHOLD = 0.5

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

print("DenseNet-121 model loaded")

image = Image.open(TEST_IMAGE).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    logits = model(image)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

findings = []
confidence_map = {}

for label, prob in zip(LABEL_COLS, probs):
    confidence_map[label] = float(prob)
    if prob >= THRESHOLD:
        findings.append(f"{label} (confidence {prob:.2f})")

if not findings:
    findings.append("No significant abnormality detected")

print("\nDetected Findings:")
for f in findings:
    print("-", f)

PROMPT = f"""
You are an expert radiologist.

Generate a professional chest X-ray radiology report
based ONLY on the findings provided.

Detected Findings:
{', '.join(findings)}

Follow this format strictly:

Study:
Findings:
Impression:
Clinical Notes:

Use formal medical language.
Do NOT add diseases that are not listed.
"""

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
    })
)

result = json.loads(response["body"].read())
report_text = result["content"][0]["text"]

print("\n================ RADIOLOGY REPORT ================\n")
print(report_text)
