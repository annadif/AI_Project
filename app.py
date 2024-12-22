import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('C:\A  computer eng\projects\Attendance-Management-System-Using-Face-Recognition-main\heart_disease_22\heart_disease_model .pkl')

# Title
st.title("Heart Disease Prediction App")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 1, 100, 50)
    sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 50, 200, 120)
    chol = st.sidebar.slider('Cholesterol Level', 100, 500, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', [0, 1])
    restecg = st.sidebar.slider('Resting ECG Results (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.slider('Slope of Peak Exercise ST Segment (0-2)', 0, 2, 1)


    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 
    }
    return np.array(list(data.values())).reshape(1, -1)

# Get user input
input_features = user_input_features()

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    if prediction[0] == 1:
        st.subheader("The model predicts that the patient is at risk of heart disease.")
    else:
        st.subheader("The model predicts that the patient is NOT at risk of heart disease.")

    st.write(f"Prediction Probability: {prediction_proba[0]}")



