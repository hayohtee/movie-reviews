from pathlib import Path
import dill

DEFAULT_MODEL_DIR = Path("../models")

def save_model(model, model_name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model_name}.pkl"
    with open(path, "wb") as f:
        dill.dump(model, f)
    print(f"Model saved to {path}")

def load_model(name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    path = model_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model {name} not found at {path}")
    with open(path, "rb") as f:
        return dill.load(f)