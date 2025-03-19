# import pandas as pd
# #import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load dataset
# df = pd.read_csv("data/loan_default_dataset.csv")
# print(df.head(3))
# print(df.info()) 
# df = df.drop(columns = ['loan_id']) 
# ####Transform categorical data into numerical data 
# from sklearn.preprocessing import LabelEncoder 
# lb = LabelEncoder()
# df['loan_purpose'] = lb.fit_transform(df['loan_purpose'])


# # Split data
# X = df.drop("loan_status", axis=1)  # Features
# y = df["loan_status"]               # Target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# # Save model
# joblib.dump(model, "api/model.pkl")
# print("Model saved successfully!")



import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 

# Load dataset
df = pd.read_csv("data/loan_default_dataset.csv")
print(df.head(3))
print(df.info()) 

df = df.drop(columns=['loan_id'])  # Drop unwanted column

# Transform categorical data into numerical
lb = LabelEncoder()
df['loan_purpose'] = lb.fit_transform(df['loan_purpose'])

# Split data
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model in the correct directory
joblib.dump(model, "api/model.pkl")
print("Model saved successfully!")
