import streamlit as st
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI-Assisted Radiology Report Generation",
    page_icon="🩻",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
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

# =====================================================
# ENV & CONFIG
# =====================================================
load_dotenv()

MODEL_PATH = "densenet121_chestxray.pth"
HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD DENSENET MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features, len(LABEL_COLS)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# HUGGING FACE CLIENT
# =====================================================
hf_client = InferenceClient(
    model=HF_MODEL_ID,
    token=os.getenv("HF_TOKEN")
)

# =====================================================
# UI
# =====================================================
st.markdown("## 🩻 AI-Assisted Radiology Report Generation")
st.caption("Upload chest X-ray → AI detects findings → Hugging Face LLM generates report")
st.divider()

left, center, right = st.columns([1.3, 2.8, 1.8])

# ---------------- LEFT ----------------
with left:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

# ---------------- CENTER ----------------
with center:
    st.markdown("### 🖼 Medical Image Viewer")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="stretch")
    else:
        st.info("Upload an X-ray image to view here")

# ---------------- RIGHT ----------------
with right:
    st.markdown("### 🧠 Detected Imaging Findings")

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
                <h4>Detected Finding</h4>
                <h2>{top_label}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
                <h4>Confidence</h4>
                <h2>{top_prob * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No significant abnormality detected")

    else:
        st.info("Awaiting image analysis")

# =====================================================
# REPORT GENERATION
# =====================================================
st.divider()
st.markdown("### 📝 AI-Generated Radiology Report")

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
based ONLY on the AI-detected imaging findings below.

These are model predictions and NOT confirmed diagnoses.

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
     response = hf_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert radiologist generating formal chest X-ray reports."
            },
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        max_tokens=500,
        temperature=0.2
    )

    report_text = response.choices[0].message.content.strip()


    st.markdown(
        f"<div class='report-box'>{report_text.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇ Download Report",
        report_text,
        file_name="radiology_report.txt"
    )
else:
    st.info("Upload an image to generate the report")
