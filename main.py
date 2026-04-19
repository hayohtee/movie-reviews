"""FastAPI application for movie review sentiment analysis.

This module exposes a REST API that loads a pre-trained scikit-learn
pipeline (TF-IDF + SGDClassifier) and serves sentiment predictions
for individual or batched movie reviews.

Run the server with::

    uvicorn main:app --reload

Endpoints:
    GET  /health         – Liveness / readiness check.
    GET  /model/info     – Metadata about the loaded pipeline.
    POST /predict        – Classify a single review.
    POST /predict/batch  – Classify up to 100 reviews in one call.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import HealthResult, ModelInfo, SentimentResult, ReviewRequest, BatchResult, BatchRequest

MODEL_STORE: dict = {}
"""Runtime store holding the loaded model and its metadata (populated at startup)."""

MODEL_PATH = Path("models/movie_reviews.joblib")
"""Path to the serialised scikit-learn pipeline on disk."""


def load_model():
    """Deserialise the trained pipeline from :data:`MODEL_PATH`.

    Returns:
        sklearn.pipeline.Pipeline: The loaded model pipeline ready for
        inference.

    Raises:
        FileNotFoundError: If the joblib file does not exist at the
            expected path.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    raise FileNotFoundError("No trained model found")


def predict_review(text: str, return_probs: bool) -> SentimentResult:
    """Run inference on a single review string.

    The function delegates to the pipeline stored in :data:`MODEL_STORE`,
    extracts class probabilities and the preprocessed token string, and
    packages everything into a :class:`SentimentResult`.

    Args:
        text: Raw review text to classify.
        return_probs: If ``True``, include per-class probabilities in the
            response; otherwise those fields are set to ``None``.

    Returns:
        SentimentResult: Predicted sentiment, confidence, optional
        probabilities, processing time, and preprocessed text.
    """
    t0 = time.perf_counter()
    model = MODEL_STORE["model"]

    proba = model.predict_proba([text])[0]
    label_idx = int(np.argmax(proba))

    sentiment = "positive" if label_idx == 1 else "negative"
    confidence = float(proba[label_idx])

    preprocessed_text = model.named_steps["preprocess"].named_steps["text_preprocessing"].transform([text])

    return SentimentResult(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        positive_probability=round(float(proba[1]), 4) if return_probs else None,
        negative_probability=round(float(proba[0]), 4) if return_probs else None,
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        preprocessed_text=preprocessed_text[0]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — loads the model on startup and clears it on shutdown.

    On startup the serialised pipeline is loaded into :data:`MODEL_STORE`
    together with a UTC timestamp.  On shutdown the store is cleared to
    free memory.

    Raises:
        FileNotFoundError: Propagated from :func:`load_model` if the
            model file is missing.
    """
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
    """Return the API health status, whether the model is loaded, and the app version."""
    return HealthResult(
        status="ok",
        model_loaded="model" in MODEL_STORE,
        version=app.version
    )


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
def model_info():
    """Return metadata about the loaded model.

    Reports the pipeline type, vectoriser & classifier class names,
    load timestamp, and readiness status.

    Raises:
        HTTPException: 503 if the model has not been loaded.
    """
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
    """Predict sentiment for a single movie review.

    Accepts a JSON body with a ``text`` field (min 5 characters) and an
    optional ``return_probabilities`` flag.

    Raises:
        HTTPException: 503 if the model has not been loaded.
    """
    if "model" not in MODEL_STORE:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predict_review(request.text, request.return_probabilities)


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """Predict sentiment for up to 100 reviews in one call.

    Iterates over the list of review strings, classifies each one, and
    returns per-review results together with aggregate counts and total
    processing time.

    Raises:
        HTTPException: 503 if the model has not been loaded.
    """
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
