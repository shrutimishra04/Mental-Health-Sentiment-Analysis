import torch
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model & tokenizer
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model_domain_adapted")

config = DistilBertConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    local_files_only=True
)


model.to(device)
model.eval()

# -----------------------------
# Labels
# -----------------------------
label_names = [
    "depression",
    "anxiety",
    "stress",
    "suicidal",
    "normal",
    "bipolar",
    "ptsd"
]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Mental Health Sentiment API",
    description="Safety-aware mental health classification using DistilBERT",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (safe for demo)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)


# -----------------------------
# Request schema
# -----------------------------
class TextRequest(BaseModel):
    text: str
    subreddit: str | None = None

# -----------------------------
# Risk level (severity-based)
# -----------------------------
def risk_level(pred):
    if pred == "suicidal":
        return "High"
    if pred in ["depression", "anxiety", "stress"]:
        return "Medium"
    return "Low"


# -----------------------------
# Confidence badge (certainty-based)
# -----------------------------
def confidence_badge(conf, pred):
    if pred == "suicidal":
        if conf >= 0.7:
            return "Medium"
        if conf >= 0.4:
            return "Low"
        return "Low"

    if pred == "depression":
        if conf >= 0.8:
            return "Medium"
        if conf >= 0.5:
            return "Low"
        return "Low"

    if pred in ["anxiety", "stress"]:
        if conf >= 0.8:
            return "High"
        if conf >= 0.5:
            return "Medium"
        return "Low"

    if conf >= 0.8:
        return "High"
    if conf >= 0.5:
        return "Medium"
    return "Low"




# -----------------------------
# Prediction logic (SAME AS NOTEBOOK)
# -----------------------------
def predict_with_safety(text, subreddit=None):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))

    suicidal_prob = probs[3]
    anxiety_prob = probs[1]
    stress_prob = probs[2]

    # Rule 1: direct suicidal probability
    if suicidal_prob >= 0.20:
        label = "suicidal"
        conf = float(suicidal_prob)

    # Rule 2: contextual escalation
    elif subreddit and "suicide" in subreddit.lower():
        if anxiety_prob >= 0.40 or stress_prob >= 0.40:
            label = "suicidal"
            conf = float(max(anxiety_prob, stress_prob))
        else:
            label = label_names[pred]
            conf = float(probs[pred])

    # Map bipolar / ptsd → depression
    elif pred in [5, 6]:
        label = "depression"
        conf = float(probs[pred])

    else:
        label = label_names[pred]
        conf = float(probs[pred])

    return label, conf



# -----------------------------
# API endpoints
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Mental Health Sentiment API is running",
        "disclaimer": "For educational and research purposes only. Not a diagnostic tool."
    }

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/predict")
def predict(input: TextRequest):
    label, conf = predict_with_safety(input.text, input.subreddit)

    return {
        "label": label,
        "confidence": round(conf, 3),
        "confidence_badge": confidence_badge(conf, label),
        "risk": risk_level(label)
    }