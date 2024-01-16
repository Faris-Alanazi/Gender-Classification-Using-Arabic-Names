# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

# Load the trained model
model = load_model('saved_model/BiLSTM-87_val_acc-91_f_sc.h5')

# Tokenizer setup (replicating the training script setup)
max_name_length = 17  # Adjust based on your data
vocab_size = 27  # Replace with the actual vocab size used during training
tokenizer = Tokenizer(num_words=vocab_size, char_level=True)

input_name = st.text_input("Name", "", max_chars=17, placeholder="Enter Name Here")

if st.button("Predict Gender"):
    if input_name:  # Check if a name was entered
        # Preprocess the input (replicate training script steps)
        sequence = tokenizer.texts_to_sequences([input_name])
        padded_sequence = pad_sequences(sequence, maxlen=max_name_length)

        # Generate the one-hot encoded features for the input name
        one_hot_features = create_one_hot_features(input_name, one_hot_columns)
        
        # Combine the padded sequence with the one-hot encoded features
        padded_sequence_df = pd.DataFrame(padded_sequence, columns=[f'char_{i}' for i in range(padded_sequence.shape[1])])
        full_input_df = pd.concat([padded_sequence_df, one_hot_features], axis=1)
        
        # Ensure the full_input_df matches the expected shape (84 columns in this case)
        if full_input_df.shape[1] != 84:
            st.error('The input does not match the expected shape.')
        else:
            # Make prediction
            prediction = model.predict(full_input_df)
            predicted_class = 'Male' if prediction[0][0] > 0.5 else 'Female'

            # Display the prediction
            st.success(f"Predicted Gender for '{input_name}': {predicted_class}")
    else:
        st.error("Please enter a name for prediction.")

# ...



# # Local file path, constructed based on your directory structure
# model_path = "mlruns/237850698007442871/<run_id>/artifacts/model"
# model = mlflow.pyfunc.load_model(model_path)
