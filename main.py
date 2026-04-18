import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import HealthResult, ModelInfo, SentimentResult, ReviewRequest, BatchResult, BatchRequest

MODEL_STORE: dict = {}

MODEL_PATH = Path("models/movie_reviews.joblib")


def load_model():
    """Load the trained model"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    raise FileNotFoundError("No trained model found")


def predict_review(text: str, return_probs: bool) -> SentimentResult:
    t0 = time.perf_counter()
    model = MODEL_STORE["model"]

    proba = model.predict_proba([text])[0]
    label_idx = int(np.argmax(proba))

    sentiment = "positive" if label_idx == 1 else "negative"
    confidence = float(proba[label_idx])

    return SentimentResult(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        positive_probability=round(float(proba[1]), 4) if return_probs else None,
        negative_probability=round(float(proba[0]), 4) if return_probs else None,
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MODEL_STORE["model"] = load_model()
        MODEL_STORE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print("Model loaded")
    except (FileNotFoundError, TypeError) as e:
        print(f"Failed to load model: {e}")
        raise
    yield
    MODEL_STORE.clear()
    print("Model cleared")


app = FastAPI(
    title="Movie Reviews Sentiment Analysis API",
    description="Predict positive / negative sentiment for movie reviews using TF-IDF + SGDClassifier pipeline",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
    max_age=3600
)


@app.get("/health", response_model=HealthResult, tags=["System"])
def health():
    return HealthResult(
        status="ok",
        model_loaded="model" in MODEL_STORE,
        version=app.version
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
def model_info():
    """Return metadata about the loaded model"""
    if "model" not in MODEL_STORE:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    model = MODEL_STORE["model"]
    return ModelInfo(
        model_type="sklearn.pipeline.Pipeline",
        vectorizer=type(model.named_steps["preprocess"].named_steps["vectorization"]).__name__,
        classifier=type(model.named_steps["sgd"]).__name__,
        loaded_at=MODEL_STORE["loaded_at"],
        status="ready"
    )


@app.post("/predict", response_model=SentimentResult, tags=["Prediction"])
def predict(request: ReviewRequest):
    """Predict sentiment for a single movie review."""
    if "model" not in MODEL_STORE:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predict_review(request.text, request.return_probabilities)


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """Predict sentiment for up to 100 reviews in one call."""
    if "model" not in MODEL_STORE:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    results = [predict_review(text=text, return_probs=request.return_probabilities) for text in request.reviews]
    pos = sum(1 for result in results if result.sentiment == "positive")

    return BatchResult(
        results=results,
        total_reviews=len(results),
        positive_count=pos,
        negative_count=len(results) - pos,
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
