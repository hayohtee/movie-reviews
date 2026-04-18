from models import SentimentResult
from typing import Any
import time

# def predict(model_store: dict[str, Any], text: str, return_probs:bool ) -> SentimentResult:
#     t0 = time.perf_counter()
#     model = model_store["model"]
#
#     proba = model.predict_proba(text)[0]
#     label_idx = int(np)