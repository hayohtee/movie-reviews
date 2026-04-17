from pathlib import Path
import joblib

DEFAULT_MODEL_DIR = Path("../models")

def save_model(model, model_name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model_name}.joblib"
    with open(path, "wb") as f:
        joblib.dump(model, f, compress=3)
    print(f"Model saved to {path}")

def load_model(name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    path = model_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model {name} not found at {path}")
    with open(path, "rb") as f:
        return joblib.load(f)