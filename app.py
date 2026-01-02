import streamlit as st
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import boto3
import os
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI-Assisted Radiology Report Generation",
    page_icon="🩻",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0e1117; }

.card {
    background: #ffffff;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    color: #0f172a;
}

.card h2 {
    color: #020617;
    font-weight: 700;
}

.card h4 {
    color: #334155;
}

.report-box {
    background: #0b2a44;
    color: #dbeafe;
    padding: 24px;
    border-radius: 16px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()

MODEL_PATH = "densenet121_chestxray.pth"
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

THRESHOLD = 0.5 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)


st.markdown("##  AI-Assisted Radiology Report Generation")
st.caption("Upload chest X-ray → AI detects imaging findings → Generates draft radiology report")
st.divider()

left, center, right = st.columns([1.3, 2.8, 1.8])

with left:
    st.markdown("###  Upload Image")
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )


with center:
    st.markdown("###  Medical Image Viewer")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch")
    else:
        st.info("Upload an X-ray image to view here")

with right:
    st.markdown("###  Detected Imaging Findings")
    st.caption("Identified abnormalities")

    if uploaded_file:
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        confidence_map = dict(zip(LABEL_COLS, probs))

        findings_raw = [
            (label, prob)
            for label, prob in confidence_map.items()
            if prob >= THRESHOLD
        ]

        if findings_raw:
            top_label, top_prob = max(findings_raw, key=lambda x: x[1])

            st.markdown(f"""
            <div class="card">
                <h4>Detected Imaging Finding</h4>
                <h2>{top_label}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
                <h4>Confidence Score</h4>
                <h2>{top_prob*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting image analysis")

st.divider()
st.markdown("###  AI-Generated Radiology Report")
st.caption("Draft report generated from AI-detected imaging findings.")

if uploaded_file:
    findings_text = [
        f"{label} (confidence {prob:.2f})"
        for label, prob in findings_raw
    ]

    if not findings_text:
        findings_text = ["No significant abnormality detected"]

    PROMPT = f"""
You are an expert radiologist.

Generate a draft chest X-ray radiology report
based ONLY on the AI-detected imaging findings provided.

These findings are model predictions and NOT confirmed diagnoses.

Detected Imaging Findings:
{', '.join(findings_text)}

Follow this format strictly:

Study:
Findings:
Impression:
Clinical Notes:

Use formal radiology language.
Do NOT introduce new diseases.
Avoid definitive diagnostic claims.
"""

    with st.spinner("Generating radiology report..."):
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.2,
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": PROMPT}]
                }]
            })
        )

        report_text = json.loads(response["body"].read())["content"][0]["text"]

    st.markdown(
        f"<div class='report-box'>{report_text.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )

    st.download_button(
        " Download Report",
        report_text,
        file_name="radiology_report.txt"
    )
else:
    st.info("AI-generated report will appear after image analysis")
