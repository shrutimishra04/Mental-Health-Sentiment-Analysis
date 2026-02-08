**Mental Health Sentiment Analysis (Hybrid ML + NLP System)**

This project implements an end-to-end mental health text analysis system designed for academic and research purposes. The system classifies user-provided text into mental health–related categories, estimates risk levels, and generates a supportive, rule-based chatbot response.

The focus of this project is not only model accuracy, but also safety-aware deployment, combining machine learning with rule-based safeguards to reduce false escalation in sensitive mental health contexts.

Disclaimer
This project is intended strictly for educational and research use.
It is not a diagnostic or clinical tool.

**Project Objectives**

Analyze mental health–related text data

Classify text into predefined mental health categories

Assign confidence scores and risk levels

Reduce false positives using rule-based safety guards

Provide a simple, supportive chatbot response

Demonstrate a production-style ML + API pipeline

**Mental Health Categories**

The system predicts the following five classes:

normal

stress

anxiety

depression

suicidal

Risk levels are derived from the predicted class and confidence:

High: suicidal

Medium: depression, anxiety, stress

Low: normal

**System Architecture**

The project follows a hybrid inference approach:

Baseline model

TF-IDF + Logistic Regression

Provides stability and fallback predictions

Deep learning model

RoBERTa (domain-adapted on mental health text)

Used only when confidence is sufficiently high

Rule-based safeguards

Positive-language override

Suicidal intent detection

Confidence thresholds

Fallback logic to prevent unsafe escalation

FastAPI backend

Handles inference, risk assignment, and chatbot responses

Frontend (HTML/CSS/JS)

Simple interface for text input and result display

**Folder Structure**
Mental-Health-Sentiment-Analysis/
│
├── api/
│   ├── app.py                     # FastAPI backend (hybrid inference + chatbot)
│   └── model/
│       ├── roberta_classifier/    # Fine-tuned RoBERTa model + tokenizer
│       ├── baseline_logreg.pkl    # Logistic regression baseline model
│       └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│
├── data/
│   ├── raw/                       # Original datasets (not committed)
│   └── processed/                # Cleaned and merged datasets
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_baseline_logreg.ipynb
│   ├── 03_roberta_finetuning.ipynb
│   ├── 04_domain_adaptation_mlm.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   └── merge_reddit_corpus.py
│
├── frontend/
│   └── index.html                 # Web UI
│
├── .gitignore
└── README.md


Large datasets and trained model weights are intentionally excluded from version control.

**Model Training Overview**
**Baseline Model**

TF-IDF features

Logistic Regression classifier

Trained on labeled mental health datasets

Used as a fallback and stability anchor

**RoBERTa Model**

Pretrained roberta-base

Domain adaptation using masked language modeling on Reddit mental health data

Fine-tuned for multi-class mental health classification

Used selectively based on confidence thresholds

**Safety and Rule-Based Logic**

To prevent unsafe or misleading predictions, the system includes:

Positive language guard
Prevents escalation when text clearly expresses well-being

Suicidal intent detection
The suicidal label is only allowed when explicit intent phrases are present

Confidence gating
RoBERTa predictions are accepted only above a minimum confidence threshold

Fallback mechanism
Automatically switches to the baseline model if the deep model is uncertain or fails

These rules are critical for ethical deployment and realistic behavior in mental health applications.

**Backend API**
Start the server

From the api/ directory:

uvicorn app:app --reload


The API runs at:

http://127.0.0.1:8000


Interactive API documentation:

http://127.0.0.1:8000/docs

**Example API Request**

POST /predict

{
  "text": "I feel overwhelmed and exhausted from work lately."
}

**Example Response**
{
  "label": "stress",
  "confidence": 0.63,
  "risk": "Medium",
  "model_used": "roberta",
  "chatbot_response": "It sounds like you’re under a lot of pressure. Even small breaks or slowing down your breathing may help."
}

**Frontend**

Open frontend/index.html in a browser

Enter text and click Analyze

**Displays:**

Predicted label

Confidence score

Risk level

Chatbot support message

**Ethical Considerations**

No personally identifiable information is collected

Uses publicly available datasets

Includes safety disclaimers and conservative escalation rules

Designed for demonstration, not diagnosis

**Author**

Shruti Mishra
B.Tech Graduate | Post Graduate Diploma in Artificial Intelligence
Email: shrutim318@gmail.com

**License**

This project is intended for educational and research purposes only.
Not licensed for clinical, diagnostic, or commercial mental health use.