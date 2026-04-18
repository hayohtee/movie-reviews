from pydantic import BaseModel, Field
from typing import Optional


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=5, examples=["This movie was absolutely fantastic"])
    return_probabilities: bool = Field(False, description="Return confidence scores alongside the label")

class BatchRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=100, examples=["Great film!", "Worst movie ever."])
    return_probabilities: bool = False

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    positive_probability: Optional[float] = None
    negative_probability: Optional[float] = None
    processing_time_ms: float

class BatchResult(BaseModel):
    results: list[SentimentResult]
    total_reviews: int
    positive_count: int
    negative_count: int
    processing_time_ms: float

class ModelInfo(BaseModel):
    model_type: str
    vectorizer: str
    classifier: str
    loaded_at: str
    status: str

class HealthResult(BaseModel):
    status: str
    model_loaded: bool