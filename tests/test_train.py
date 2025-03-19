import sys
import os
import pytest
import joblib
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import model  # Import the trained model

def test_model_training():
    """Test if the model is trained properly and saved."""
    assert model is not None, "Model training failed, model is None!"
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier!"

def test_model_prediction():
    """Test if the trained model makes predictions correctly."""
    test_data = [[1, 0, 50000, 30, 0, 1, 20000, 2]]
    prediction = model.predict(test_data)
    assert prediction is not None, "Model failed to make a prediction"
    assert isinstance(prediction, (list, tuple, set, np.ndarray)), "Prediction is not in the correct format"

def test_model_saving():
    """Test if the model is saved and can be loaded correctly."""
    saved_model_path = "api/model.pkl"
    joblib.dump(model, saved_model_path)
    loaded_model = joblib.load(saved_model_path)
    assert loaded_model is not None, "Model saving or loading failed!"
    assert isinstance(loaded_model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier!"
