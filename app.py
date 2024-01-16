# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

# Load the trained model
model = load_model('BiLSTM-87_val_acc-91_f_sc.h5')

# Tokenizer setup (replicating the training script setup)
max_name_length = 17  # Adjust based on your data
vocab_size = 27  # Replace with the actual vocab size used during training
tokenizer = Tokenizer(num_words=vocab_size, char_level=True)

# This function creates a one-hot encoded DataFrame for the last and first letters of a name
def create_one_hot_features(name, columns):
    # Create a series with all zeros
    one_hot_encoded = pd.Series(0, index=columns)

    # Set the appropriate last and first letter features to 1
    last_letter_feature = 'last_letter_' + name[-1]
    first_letter_feature = 'first_letter_' + name[0]
    
    if last_letter_feature in one_hot_encoded.index:
        one_hot_encoded[last_letter_feature] = 1
    if first_letter_feature in one_hot_encoded.index:
        one_hot_encoded[first_letter_feature] = 1
    
    # Return the result as a DataFrame
    return one_hot_encoded.to_frame().transpose()

# Columns excluding 'name' and 'sex'
one_hot_columns = ['last_letter_ء', 'last_letter_أ', 'last_letter_ؤ',
       'last_letter_ئ', 'last_letter_ا', 'last_letter_ب', 'last_letter_ة',
       'last_letter_ت', 'last_letter_ث', 'last_letter_ج', 'last_letter_ح',
       'last_letter_خ', 'last_letter_د', 'last_letter_ذ', 'last_letter_ر',
       'last_letter_ز', 'last_letter_س', 'last_letter_ش', 'last_letter_ص',
       'last_letter_ض', 'last_letter_ط', 'last_letter_ظ', 'last_letter_ع',
       'last_letter_غ', 'last_letter_ف', 'last_letter_ق', 'last_letter_ك',
       'last_letter_ل', 'last_letter_م', 'last_letter_ن', 'last_letter_ه',
       'last_letter_و', 'last_letter_ى', 'last_letter_ي', 'first_letter_ء',
       'first_letter_آ', 'first_letter_أ', 'first_letter_ؤ', 'first_letter_إ',
       'first_letter_ا', 'first_letter_ب', 'first_letter_ت', 'first_letter_ث',
       'first_letter_ج', 'first_letter_ح', 'first_letter_خ', 'first_letter_د',
       'first_letter_ذ', 'first_letter_ر', 'first_letter_ز', 'first_letter_س',
       'first_letter_ش', 'first_letter_ص', 'first_letter_ض', 'first_letter_ط',
       'first_letter_ظ', 'first_letter_ع', 'first_letter_غ', 'first_letter_ف',
       'first_letter_ق', 'first_letter_ك', 'first_letter_ل', 'first_letter_م',
       'first_letter_ن', 'first_letter_ه', 'first_letter_و', 'first_letter_ي']  
       # List all your columns for one-hot encoding

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
