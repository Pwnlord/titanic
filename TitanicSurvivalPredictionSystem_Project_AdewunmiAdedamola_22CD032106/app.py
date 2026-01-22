import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load artifacts
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction System")
st.markdown("Enter passenger details below to predict survival probability.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)

if st.button("Predict Survival"):
    # Preprocess inputs
    sex_encoded = le.transform([sex])[0]
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, fare]], 
                              columns=['pclass', 'sex', 'age', 'sibsp', 'fare'])
    
    # Scaling
    scaled_data = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.success("✅ Prediction: This passenger would have **SURVIVED**.")
    else:
        st.error("❌ Prediction: This passenger would **NOT HAVE SURVIVED**.")