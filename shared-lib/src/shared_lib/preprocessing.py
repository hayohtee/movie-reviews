import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

def preprocess_text(text: str) -> str:
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
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for text in X:
            X_transformed.append(preprocess_text(text))
        return np.array(X_transformed)

preprocess_pipeline = Pipeline([
    ("preprocess", TextPreprocessingTransformer()),
    ("vectorization", TfidfVectorizer(ngram_range=(1, 2),stop_words="english")),
])