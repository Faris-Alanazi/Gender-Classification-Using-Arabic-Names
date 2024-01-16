# Import necessary libraries
import streamlit as st
import pickle
import os
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

# Tokenizer setup (replicating the training script setup)
max_name_length = 17  # Adjust based on your data
vocab_size = 37  # Replace with the actual vocab size used during training

# Load the trained model
model = xgb.XGBClassifier()
model.load_model('saved_models/XGBoost.bin')  # Adjust if you saved with a different filename

# Load the scaler models
scaler_first = joblib.load('saved_models/scaler_models/scaler_first_letter.pkl')
scaler_last = joblib.load('saved_models/scaler_models/scaler_last_letter.pkl')

with open('saved_models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


def create_features(name):
    first_letter = name[0]
    last_letter = name[-1] if len(name) > 1 else name[0]

    first_letter_encoded = np.array([ord(first_letter)])
    last_letter_encoded = np.array([ord(last_letter)])

    # Scale the encoded letters using the loaded scaler models
    first_letter_encoded_scaled = scaler_first.transform(first_letter_encoded.reshape(-1, 1))
    last_letter_encoded_scaled = scaler_last.transform(last_letter_encoded.reshape(-1, 1))

    # Tokenize and pad the name, making sure to pass it as a list
    sequences = tokenizer.texts_to_sequences([name])
    padded_sequences = pad_sequences(sequences, maxlen=max_name_length)

    features = np.concatenate((padded_sequences, first_letter_encoded_scaled, last_letter_encoded_scaled),axis=1)

    return features


input_name = st.text_input("الاسم الأول", "", max_chars=17, placeholder="اكتب الاسم هنا")

if st.button("توقع الجنس"):
    if input_name:  # Check if a name was entered
        # Preprocess the input (replicate training script steps)
        prediction = model.predict(create_features(input_name))
        predicted_class = 'ذكر' if prediction > 0.5 else 'انثى'

        # Display the prediction
        st.success(f"الجنس المتوقع للأسم '{input_name}' بالأغلب انه {predicted_class}")

    else:
        st.error("الرجاء ادخال اسم للبدء")

# # Local file path, constructed based on your directory structure
# model_path = "mlruns/237850698007442871/<run_id>/artifacts/model"
# model = mlflow.pyfunc.load_model(model_path)
