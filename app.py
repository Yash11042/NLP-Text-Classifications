# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from clean_text import clean_text

# Load model & TF-IDF
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

app = FastAPI(
    title="NLP Text Classification API",
    description="Simple FastAPI service for paragraph classification",
    version="1.0"
)

# Input schema
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]


# Root Endpoint
@app.get("/")
def home():
    return {"message": "NLP Text Classification API is running!"}


# 1️⃣ Single Text Prediction Endpoint
@app.post("/predict")
def predict(data: TextInput):
    cleaned = clean_text(data.text)

    vector = tfidf.transform([cleaned])

    pred = model.predict(vector)[0]

    # Predict probabilities (if available)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0].tolist()
    else:
        probs = "Model does not support probability prediction"

    return {
        "input_text": data.text,
        "cleaned_text": cleaned,
        "predicted_label": str(pred),   # FIXED
        "probabilities": probs
    }


# 2️⃣ Batch Prediction Endpoint (multiple texts)
@app.post("/predict_batch")
def predict_batch(data: BatchInput):

    # Clean all texts
    cleaned = [clean_text(t) for t in data.texts]

    # Vectorize
    vectors = tfidf.transform(cleaned)

    # Predictions
    preds = model.predict(vectors).tolist()

    # Probabilities (if supported)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectors).tolist()
    else:
        probs = "Model does not support probability prediction"

    return {
        "total_texts": len(data.texts),
        "predicted_labels": preds,
        "probabilities": probs
    }
