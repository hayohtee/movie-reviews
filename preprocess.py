"""Text preprocessing utilities for movie review sentiment analysis.

This module provides functions and a scikit-learn compatible transformer
for cleaning and normalising raw movie review text before vectorisation.

Pipeline steps performed on each review:
    1. Lowercasing
    2. HTML tag removal
    3. URL removal
    4. Special character and digit removal
    5. Stop-word filtering (negations are preserved)
    6. WordNet lemmatisation
"""

import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)


def preprocess_text(text: str) -> str:
    """Clean and normalise a single movie review string.

    The function lowercases the input, strips HTML tags, URLs, special
    characters and digits, removes English stop words (while retaining
    negation words that carry sentiment), and lemmatises the remaining
    tokens using WordNet.

    Args:
        text: Raw review text to preprocess.

    Returns:
        A single space-joined string of cleaned, lemmatised tokens.

    Example:
        >>> preprocess_text("This movie was <b>NOT</b> good at all!")
        'movie not good'
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove special characters and digits
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = text.split()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    # Keep negations as they are important for sentiment
    negations = {"no", "not", "neither", "nor", "never", "none", "nobody", "nothing", "nowhere"}
    stop_words -= negations
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


class TextPreprocessingTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that applies :func:`preprocess_text` to every sample.

    This transformer can be used inside a :class:`sklearn.pipeline.Pipeline`
    so that raw text is automatically cleaned before being passed to a
    vectoriser (e.g. ``TfidfVectorizer``).

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.feature_extraction.text import TfidfVectorizer
        >>> pipe = Pipeline([
        ...     ("preprocess", TextPreprocessingTransformer()),
        ...     ("tfidf", TfidfVectorizer()),
        ... ])
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """No-op fit — this transformer is stateless.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Apply :func:`preprocess_text` to each element in *X*.

        Args:
            X: Iterable of raw review strings.
            y: Ignored.

        Returns:
            numpy.ndarray: Array of preprocessed strings with the same
            length as *X*.
        """
        X_transformed = []
        for text in X:
            X_transformed.append(preprocess_text(text))
        return np.array(X_transformed)
