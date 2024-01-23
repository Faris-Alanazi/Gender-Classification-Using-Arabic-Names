# python=3.8
# conda env name: gender_pred_env
# Packages
import joblib
import numpy as np
import pandas as pd
import warnings
import mlflow
import os
import optuna
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_score, recall_score, roc_curve, auc, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# Load Data
df = pd.read_pickle('data/dataset_after_preporcessing.pkl')
df.head()
max_name_length = max(df['name'].apply(len))
unique_chars = set(''.join(df['name']))  
vocab_size = len(unique_chars) + 1  

# Tokenizing and padding the 'name' column for LSTM input
tokenizer = Tokenizer(num_words=vocab_size, char_level=True)  # char_level=True for character tokenization
tokenizer.fit_on_texts(df['name'])
sequences = tokenizer.texts_to_sequences(df['name'])
padded_sequences = pad_sequences(sequences, maxlen=max_name_length)
first_letter = df['name'].apply(lambda x: x[0])
last_letter = df['name'].apply(lambda x: x[-1])

first_letter_encoded = np.array([ord(char) for char in first_letter])
last_letter_encoded = np.array([ord(char) for char in last_letter])

max_unicode_value_first = np.max(first_letter_encoded)
max_unicode_value_last = np.max(last_letter_encoded)
max_unicode_value = max(max_unicode_value_first, max_unicode_value_last)
# Load the scaler models
scaler_first = joblib.load('saved_models/scaler_models/scaler_first_letter.pkl')
scaler_last = joblib.load('saved_models/scaler_models/scaler_last_letter.pkl')

# Transform the new data using the loaded scalers
first_letter_encoded_scaled = scaler_first.transform(first_letter_encoded.reshape(-1, 1))
last_letter_encoded_scaled = scaler_last.transform(last_letter_encoded.reshape(-1, 1))
y = df['sex'].values
name_length = df['name_length'].values
X = list(zip(padded_sequences, first_letter_encoded_scaled, last_letter_encoded_scaled, name_length))

# Define the size for the test and validation sets as percentages
test_size_percentage = 0.1
validation_size_percentage = 0.1

# Calculate the actual sizes for the test and validation sets
total_size = test_size_percentage + validation_size_percentage
test_size_actual = test_size_percentage / total_size
validation_size_actual = validation_size_percentage / total_size

train_size_percentage = 1 - total_size

# First split: Separate out the training data and the remaining data
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=total_size, random_state=11)

# Second split: Separate the remaining data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=test_size_actual, random_state=11)

# Print the number of samples in the training, validation, and test sets
print(f"Training set size: {len(X_train)}, Labels: {len(y_train)}")
print(f"Validation set size: {len(X_val)}, Labels: {len(y_val)}")
print(f"Test set size: {len(X_test)}, Labels: {len(y_test)}")

# Unpack the training data into separate arrays for each input
name_train, first_letter_train, last_letter_train, length_train = zip(*X_train)
name_val, first_letter_val, last_letter_val, length_val = zip(*X_val)
name_test, first_letter_test, last_letter_test, length_test = zip(*X_test)

# Convert tuples to numpy arrays
name_train = np.array(name_train)
first_letter_train = np.array(first_letter_train)
last_letter_train = np.array(last_letter_train)
length_train = np.array(length_train)

name_val = np.array(name_val)
first_letter_val = np.array(first_letter_val)
last_letter_val = np.array(last_letter_val)
length_val = np.array(length_val)

name_test = np.array(name_test)
first_letter_test = np.array(first_letter_test)
last_letter_test = np.array(last_letter_test)
length_test = np.array(length_test)

# Reshape the length arrays to have two dimensions
length_train = length_train.reshape(-1, 1)
length_val = length_val.reshape(-1, 1)
length_test = length_test.reshape(-1, 1)

# Concatenate the features
X_train = np.concatenate([name_train, first_letter_train, last_letter_train, length_train], axis=1)
X_val = np.concatenate([name_val, first_letter_val, last_letter_val, length_val], axis=1)
X_test = np.concatenate([name_test, first_letter_test, last_letter_test, length_test], axis=1)

def objective(trial):
    # Define hyperparameters to tune
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 14),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    }

    # Initialize and train the ExtraTreesClassifier
    model = ExtraTreesClassifier(**param)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    
    # We return the validation accuracy for the Optuna study to optimize
    return val_accuracy

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=300)

# Best hyperparameters
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_extratrees_params = study.best_trial.params

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Gender Prediction Models Tracking")

# Start MLflow run
with mlflow.start_run():
    
    # Initialize ExtraTreesClassifier with best hyperparameters
    model = ExtraTreesClassifier(**best_extratrees_params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    
    # Log model parameters
    mlflow.log_params(best_extratrees_params)
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    
    # Confusion matrix logging
    cm = confusion_matrix(y_test, y_test_pred)
    cm_file = "confusion_matrix.txt"
    np.savetxt(cm_file, cm, fmt='%d')
    mlflow.log_artifact(cm_file)
    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log additional information
    mlflow.set_tags({
        "Description": "Optimized ExtraTrees binary classifier",
        "Features": ', '.join(df.columns.tolist()),
        'Number of Features':len(df.columns.tolist()),
        "Encoding" : "Char Level For names | Label Encoding for other features",
        "Model Type": "ExtraTrees"
    })

    # Cleanup: Delete the saved confusion matrix file
    os.remove(cm_file)

# Print all metrics
print('\n-----------------------------------------------------------')
print(f"Train Accuracy: {round(train_accuracy, 3)}")
print(f"Validation Accuracy: {round(val_accuracy, 3)}")
print(f"Test Accuracy: {round(test_accuracy, 3)}")
print("\n---Metrics---\n")
print(f"F1 Score: {round(test_f1, 3)}")
print(f"Precision: {round(test_precision, 3)}")
print(f"Recall: {round(test_recall, 3)}")
print(f"ROC AUC: {round(test_roc_auc, 3)}")
print('-----------------------------------------------------------\n')

if test_accuracy > 0.870 or test_f1 > 0.92:
    xgb_model.save_model(f"saved_models/XGBoost Models/XGBoost_Acc_{round(test_accuracy,3)}_F1_{round(test_f1,3)}_Roc_{round(test_roc_auc,3)}.bin")
    print("Model Saved!")