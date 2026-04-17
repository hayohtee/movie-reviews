import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

def preprocess_text(text: str) -> str:
    # Clean the text
    # 1. Convert to lowercase
    # 2. Remove HTML tags
    # 3. Remove URLs
    # 4. Remove special characters and digits
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = text.split()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    # Keep negations
    negations = {"no", "not", "neither", "nor", "never", "none", "nobody", "nothing", "nowhere"}
    stop_words -= negations
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)