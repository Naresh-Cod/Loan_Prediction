import joblib
import pandas as pd

class CovidInference:
    def __init__(self, model_path="api/covid_model.pkl"):
        self.model = joblib.load(model_path)

    def preprocess(self, data):
        """Preprocess user input data."""
        data["gender"] = 1 if data["gender"] == "Male" else 0
        data["cough"] = 1 if data["cough"] == "Severe" else 0
        return pd.DataFrame([list(data.values())])

    def predict(self, data):
        """Make prediction"""
        processed_data = self.preprocess(data)
        prediction = self.model.predict(processed_data)
        return "Positive" if prediction[0] == 1 else "Negative"
