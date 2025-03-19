import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Ensure the correct path)
model_path = "api/model.pkl"  # Correct path inside the container
model = joblib.load(model_path)

# Streamlit UI
st.title("Machine Learning Model Predictor")

# Input fields
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
feature5 = st.number_input("Feature 5", value=0.0)
feature6 = st.number_input("Feature 6", value=0.0)
feature7 = st.number_input("Feature 7", value=0.0)
feature8 = st.number_input("Feature 8", value=0.0)

# Predict button
if st.button("Predict"):
    features = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {int(prediction[0])}")
