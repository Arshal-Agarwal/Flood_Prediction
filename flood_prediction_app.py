import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and preprocess the data
df = pd.read_csv("kerala.csv")  # Preload the file by specifying its path

# Convert 'FLOODS' to binary (assumes 'FLOODS' column exists)
df['FLOODS'] = df['FLOODS'].apply(lambda x: 1 if str(x).upper() == 'YES' else 0)

# Select numeric columns for features
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
target = 'FLOODS'
feature_columns = [col for col in numeric_columns if col != target and col != 'YEAR']

X = df[feature_columns]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Flood Prediction System")
st.write(f"Model Accuracy: {accuracy:.2f}")

# Input fields for user data
st.header("Input Monthly Rainfall Data")
user_input = {col: st.number_input(f"{col}", value=0.0) for col in feature_columns}

# Predict button
if st.button("Predict Flood"):
    input_data = pd.DataFrame([user_input])
    prediction = model.predict(input_data)[0]  # Extract the first element
    result = "Flood Likely" if prediction == 1 else "No Flood"
    st.write(f"Prediction: {result}")
