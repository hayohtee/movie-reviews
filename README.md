# 🎬 Movie Reviews — Sentiment Analysis

A machine learning pipeline that analyzes movie reviews and classifies
them as **positive** or **negative**. The project ships a trained
scikit-learn model behind a FastAPI back-end, along with a minimal
browser-based front-end for interactive use.

---

## Project Layout

```
movie-reviews/
├── main.py                  # FastAPI application (prediction endpoints)
├── models.py                # Pydantic request / response schemas
├── preprocess.py            # Text preprocessing transformer (tokenization, lemmatisation, stop-word removal)
├── helpers.py               # Utility helpers
├── movie_reviews.ipynb      # Jupyter notebook — EDA, training, evaluation
├── pyproject.toml           # Project metadata & dependencies
├── uv.lock                  # Locked dependency versions
├── models/
│   └── movie_reviews.joblib # Serialised trained pipeline (TF-IDF + SGDClassifier)
├── web/
│   └── index.html           # Single-page front-end for the sentiment analyser
└── README.md
```

---

## Model Details

| Component       | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Vectoriser      | `TfidfVectorizer`                                            |
| Classifier      | `SGDClassifier`                                              |
| Preprocessing   | Lowercase → HTML/URL removal → stop-word filtering (keeps negations) → lemmatisation |
| Framework       | scikit-learn 1.8                                             |

### Final Metrics

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.8812 |
| Precision | 0.8813 |
| Recall    | 0.8809 |

> Full training & evaluation steps are documented in
> [`movie_reviews.ipynb`](movie_reviews.ipynb).

---

## Getting Started

### Prerequisites

- **Python ≥ 3.14**
- [**uv**](https://docs.astral.sh/uv/) (recommended) or pip

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

---

## Running the API

Start the FastAPI server with **uvicorn**:

```bash
uvicorn main:app --reload
```

The API will be available at **<http://127.0.0.1:8000>**.

### API Endpoints

| Method | Path             | Description                              |
|--------|------------------|------------------------------------------|
| `GET`  | `/health`        | Health check & model status              |
| `GET`  | `/model/info`    | Metadata about the loaded model          |
| `POST` | `/predict`       | Predict sentiment for a single review    |
| `POST` | `/predict/batch` | Predict sentiment for up to 100 reviews  |

### Example — Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!", "return_probabilities": true}'
```

Interactive API documentation is auto-generated at
**<http://127.0.0.1:8000/docs>** (Swagger UI).

---

## Opening the Web Page

A single-page front-end is included in the `web/` directory. To use it:

1. **Start the API server** (see above).
2. **Open the web page** in your browser:

   ```bash
   # Simply open the file directly
   xdg-open web/index.html       # Linux
   open web/index.html            # macOS
   ```

   Or navigate to `file:///path/to/movie-reviews/web/index.html` in your browser.

3. **Analyse a review** — type or paste a movie review into the text area
   and click **Analyse** (or press `Ctrl + Enter`). The page displays:
   - The predicted sentiment (Positive / Negative)
   - Confidence score
   - Positive & negative probability bars
   - The preprocessed token sequence

> **Note:** The front-end expects the API to be running on
> `http://127.0.0.1:8000`. Make sure the server is started before
> analysing reviews.

---

## Tech Stack

- **ML / Data** — scikit-learn, NLTK, NumPy, Joblib
- **API** — FastAPI, Pydantic, Uvicorn
- **Front-end** — Vanilla HTML / CSS / JavaScript
