#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('Decision_Tree_Classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

# Feature names used during training
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Streamlit App
st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸", layout="centered")
st.title("🌸 Iris Species Classification App")
st.write("Enter the measurements below to predict the Iris species:")

# Collect input from user
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(
        f"Enter {feature}:", 
        min_value=0.0, 
        step=0.1, 
        format="%.2f"
    )

# Predict button
if st.button("Predict Species"):
    input_df = pd.DataFrame([user_input])

    prediction = clf.predict(input_df)[0]

    # Mapping numeric predictions to Iris classes
    iris_species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    predicted_species = iris_species.get(prediction, "Unknown")
    
    st.success(f"🌟 The predicted Iris species is: **{predicted_species}**")

