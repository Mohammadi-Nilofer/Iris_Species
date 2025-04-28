#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model
with open('Decision_Tree_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature names (same as during model training)
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Streamlit UI
st.title("Iris Species Classification Web App")
st.write("🔍 This app uses a Decision Tree Classifier model to predict Iris species.")

# Taking inputs
SepalLengthCm = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0)
SepalWidthCm = st.number_input("Sepal Width (cm)", min_value=1.0, max_value=4.5)
PetalLengthCm = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0)
PetalWidthCm = st.number_input("Petal Width (cm)", min_value=0.0, max_value=3.0)

# Button to predict
if st.button("Predict Iris Species"):
    # Check for missing values
    if np.isnan(SepalLengthCm) or np.isnan(SepalWidthCm) or        np.isnan(PetalLengthCm) or np.isnan(PetalWidthCm):
        st.warning("⚠️ Please provide values for all features.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame([[
            SepalLengthCm,
            SepalWidthCm,
            PetalLengthCm,
            PetalWidthCm
        ]], columns=feature_names)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Display prediction
        if prediction == 0:
            st.success("🟢 Predicted Iris Class: Setosa")
        elif prediction == 1:
            st.success("🟢 Predicted Iris Class: Versicolor")
        elif prediction == 2:
            st.success("🟢 Predicted Iris Class: Virginica")
        else:
            st.warning("⚠️ Unexpected prediction value. Please check the model.")

