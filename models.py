"""Pydantic request and response schemas for the sentiment analysis API.

Every JSON body accepted or returned by the FastAPI endpoints in
:mod:`main` is validated and serialised through one of the models
defined here.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Request body for the ``POST /predict`` endpoint.

    Attributes:
        text: The raw movie review string (minimum 5 characters).
        return_probabilities: When ``True``, the response includes
            per-class probability scores in addition to the label.
    """

    text: str = Field(..., min_length=5, examples=["This movie was absolutely fantastic"])
    return_probabilities: bool = Field(False, description="Return confidence scores alongside the label")


class BatchRequest(BaseModel):
    """Request body for the ``POST /predict/batch`` endpoint.

    Attributes:
        reviews: A list of 1–100 raw review strings to classify.
        return_probabilities: When ``True``, each result includes
            per-class probability scores.
    """

    reviews: list[str] = Field(..., min_length=1, max_length=100, examples=["Great film!", "Worst movie ever."])
    return_probabilities: bool = False


class SentimentResult(BaseModel):
    """Response schema for a single sentiment prediction.

    Attributes:
        sentiment: Predicted label — ``"positive"`` or ``"negative"``.
        confidence: Probability of the predicted class (0–1).
        positive_probability: Class probability for *positive*
            (included only when ``return_probabilities`` was ``True``).
        negative_probability: Class probability for *negative*
            (included only when ``return_probabilities`` was ``True``).
        processing_time_ms: Wall-clock inference time in milliseconds.
        preprocessed_text: The review after the full preprocessing
            pipeline (lowercased, cleaned, lemmatised).
    """

    sentiment: str
    confidence: float
    positive_probability: Optional[float] = None
    negative_probability: Optional[float] = None
    processing_time_ms: float
    preprocessed_text: str


class BatchResult(BaseModel):
    """Response schema for a batch prediction.

    Attributes:
        results: Per-review :class:`SentimentResult` objects.
        total_reviews: Number of reviews processed.
        positive_count: How many reviews were classified as positive.
        negative_count: How many reviews were classified as negative.
        processing_time_ms: Total wall-clock time for the batch in
            milliseconds.
    """

    results: list[SentimentResult]
    total_reviews: int
    positive_count: int
    negative_count: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Response schema for the ``GET /model/info`` endpoint.

    Attributes:
        model_type: Fully-qualified class name of the pipeline
            (e.g. ``sklearn.pipeline.Pipeline``).
        vectorizer: Class name of the vectoriser step.
        classifier: Class name of the classifier step.
        loaded_at: ISO-8601 UTC timestamp of when the model was loaded.
        status: Human-readable readiness indicator (e.g. ``"ready"``).
    """

    model_type: str
    vectorizer: str
    classifier: str
    loaded_at: str
    status: str


class HealthResult(BaseModel):
    """Response schema for the ``GET /health`` endpoint.

    Attributes:
        status: Service status string (``"ok"`` when healthy).
        model_loaded: ``True`` if the ML pipeline is loaded and ready.
        version: Application version from ``FastAPI.version``.
    """

    status: str
    model_loaded: bool
    version: str
