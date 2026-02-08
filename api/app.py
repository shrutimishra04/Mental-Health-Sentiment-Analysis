import torch
import numpy as np
import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizerFast, AutoModelForSequenceClassification


# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

ROBERTA_DIR = BASE_DIR / "model" / "roberta_classifier"
BASELINE_MODEL_PATH = BASE_DIR / "model" / "baseline_logreg.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"


# =====================================================
# Device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Label mapping (MUST MATCH TRAINING)
# =====================================================
ID2LABEL = {
    0: "normal",
    1: "stress",
    2: "anxiety",
    3: "depression",
    4: "suicidal",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# =====================================================
# Load RoBERTa
# =====================================================
tokenizer = RobertaTokenizerFast.from_pretrained(
    ROBERTA_DIR,
    local_files_only=True
)

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    ROBERTA_DIR,
    local_files_only=True
)

# Force correct mapping (CRITICAL)
roberta_model.config.id2label = ID2LABEL
roberta_model.config.label2id = LABEL2ID

roberta_model.to(DEVICE)
roberta_model.eval()


# =====================================================
# Load baseline model
# =====================================================
baseline_model = joblib.load(BASELINE_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(
    title="Mental Health Sentiment & Chatbot API",
    description="Hybrid safety-aware mental health classifier with chatbot support",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# Request schema
# =====================================================
class TextRequest(BaseModel):
    text: str


# =====================================================
# Risk logic
# =====================================================
def risk_level(label: str) -> str:
    if label == "suicidal":
        return "High"
    if label in {"depression", "anxiety", "stress"}:
        return "Medium"
    return "Low"


# =====================================================
# Chatbot responses (RULE-BASED)
# =====================================================
def chatbot_response(label: str, risk: str) -> str:
    if risk == "High":
        return (
            "I’m really sorry you’re feeling this way. You’re not alone, and help is available. "
            "If you feel unsafe right now, please contact local emergency services or a suicide "
            "prevention helpline in your country."
        )

    if label == "depression":
        return (
            "I’m sorry you’re going through this. It might help to talk to someone you trust "
            "or consider reaching out to a mental health professional."
        )

    if label == "anxiety":
        return (
            "Feeling anxious can be overwhelming. Slow breathing, grounding exercises, "
            "or taking a short break may help calm your mind."
        )

    if label == "stress":
        return (
            "It sounds like you’re under a lot of pressure. Even small breaks, stretching, "
            "or writing down your thoughts can help reduce stress."
        )

    if label == "normal":
        return (
            "It’s good to hear that you’re feeling okay. If you’d like to share more about "
            "what’s going well or talk about anything else, I’m here."
        )

    return "Thank you for sharing. If you want to talk more, I’m here to listen."


POSITIVE_KEYWORDS = {
    "happy", "content", "calm", "peaceful", "balanced",
    "motivated", "grateful", "relaxed", "enjoyed",
    "looking forward", "excited", "doing well"
}

def positive_guard(text: str) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in POSITIVE_KEYWORDS)


# -----------------------------
# Suicidal intent detection
# -----------------------------
SUICIDAL_INTENT_TERMS = {
    "kill myself",
    "end my life",
    "want to die",
    "suicide",
    "better off dead",
    "no reason to live",
    "can't go on",
    "take my life"
}

def has_suicidal_intent(text: str) -> bool:
    text_l = text.lower()
    return any(term in text_l for term in SUICIDAL_INTENT_TERMS)





# =====================================================
# RoBERTa prediction
# =====================================================
def roberta_predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))

    return ID2LABEL[pred_id], float(probs[pred_id])


# =====================================================
# Baseline prediction
# =====================================================
def baseline_predict(text: str):
    X = vectorizer.transform([text])
    probs = baseline_model.predict_proba(X)[0]

    pred_id = int(np.argmax(probs))
    label = str(baseline_model.classes_[pred_id])
    confidence = float(probs[pred_id])

    return label, confidence



# =====================================================
# Hybrid inference
# =====================================================
def hybrid_predict(text: str):

    # 1️⃣ HARD positive override (must come first)
    if positive_guard(text):
        return "normal", 0.99, "rule_positive_guard"

    label = "normal"
    confidence = 0.0
    model_used = "fallback"

    try:
        r_label, r_conf = roberta_predict(text)

        # 2️⃣ 🚨 SUICIDAL ESCALATION RULE (THIS IS THE FIX)
        if r_label == "suicidal" and not has_suicidal_intent(text):
            r_label = "depression"
            r_conf = min(r_conf, 0.6)

        # 3️⃣ Confidence gate
        if r_conf >= 0.65:
            label = r_label
            confidence = r_conf
            model_used = "roberta"
        else:
            b_label, b_conf = baseline_predict(text)
            label = b_label
            confidence = b_conf
            model_used = "baseline"

    except Exception:
        b_label, b_conf = baseline_predict(text)
        label = b_label
        confidence = b_conf
        model_used = "baseline_fallback"

    return label, confidence, model_used


# =====================================================
# API endpoints
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Mental Health Chatbot API is running",
        "disclaimer": "For educational and research purposes only. Not a clinical tool."
    }


@app.post("/predict")
def predict(request: TextRequest):
    text = request.text.strip()

    label, confidence, model_used = hybrid_predict(text)
    risk = risk_level(label)
    response = chatbot_response(label, risk)

    return {
    "label": str(label),
    "confidence": float(round(confidence, 3)),
    "risk": str(risk),
    "model_used": str(model_used),
    "chatbot_response": str(response)
}

