from shared_lib.model_store import load_model

if __name__ == "__main__":
    model = load_model("sgd_classifier")
    print(model.predict("Great film"))