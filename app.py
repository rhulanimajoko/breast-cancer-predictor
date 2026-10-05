import streamlit as st
import joblib
import numpy as np
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
st.title('My Healthcare AI Predictor')
st.write('Enter the values below to get a prediction.')

feature_1 = st.number_input('Mean radius')
feature_2 = st.number_input('Mean texture')
feature_3 = st.number_input('Mean perimeter')
feature_4 = st.number_input('Mean area')
feature_5 = st.number_input('Mean smoothness')
feature_6 = st.number_input('Mean compactness')
feature_7 = st.number_input('Mean concavity')
feature_8 = st.number_input('Mean concave points')
feature_9 = st.number_input('Mean symmetry')
feature_10 = st.number_input('Mean fractal dimension')



if st.button('Predict'):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]])
    scaled_input = scaler.transform(input_data)
    result = model.predict(scaled_input)
  label = "Malignant" if result[0] == 0 else "Benign"
st.success(f"Prediction: {label}")
