# python=3.8
# conda env name: gender_pred_env
# Packages
import joblib
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import mlflow
import optuna
from optuna.integration import XGBoostPruningCallback
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_score, recall_score, roc_curve, auc, log_loss
import seaborn as sns
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

# Load Data
df = pd.read_pickle('data/dataset_after_preporcessing.pkl')
df.head()
max_name_length = max(df['name'].apply(len))
unique_chars = set(''.join(df['name']))  
vocab_size = len(unique_chars) + 1  

tokenizer = Tokenizer(num_words=vocab_size, char_level=True) 
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
    # Hyperparameters to be tuned by Optuna
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05, 0.1]),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
        'early_stopping_rounds':trial.suggest_int('early_stopping_rounds', 10, 100)
    }

   
 # Initialize XGBoost with hyperparameters
    xgb_model = xgb.XGBClassifier(**param)

    # Train the model with dynamic early stopping rounds
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  verbose=False, 
                  callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")])

    # Prediction and evaluation
    preds = xgb_model.predict_proba(X_val)[:, 1]
    log_loss_val = log_loss(y_val, preds)

    return log_loss_val

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=79);

# Best hyperparameters
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_params =  study.best_trial.params

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Gender Prediction Models Tracking")

# Start MLflow run
with mlflow.start_run():
    
    # Initialize XGBoost classifier with best hyperparameters
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        **best_params # Unpack best parameters here
    )
 
    # Train the model with early stopping
    xgb_model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Log model parameters and early stopping rounds
    mlflow.log_params(best_params)
    
    # Log train and validation accuracy
    train_accuracy = accuracy_score(y_train, xgb_model.predict(X_train))
    val_accuracy = accuracy_score(y_val, xgb_model.predict(X_val))
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0).reshape(-1)

    # Calculate and log additional metrics
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)

    # Log these metrics in MLflow
    mlflow.log_metrics({
        "test_accuracy": test_accuracy, 
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
    })

    # Predict probabilities for the test set
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # Calculate and log ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metrics({"test_roc_auc": roc_auc})

    # Confusion matrix calculation
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_dict = {f"{i}-{j}": cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])}
    mlflow.set_tag("confusion_matrix", list(cm_dict.values()))

    # Log the model
    mlflow.xgboost.log_model(xgb_model, "model")

    # Log additional information
    mlflow.set_tags({
        "Description": "Optimized XGBoost binary classifier",
        "Features": ', '.join(df.columns.tolist()),
        'Number of Features':len(df.columns.tolist()),
        "Encoding" : "Char Level For names | Label Encoding for other features",
        "Model Type": "XGBoost"
    })

# Print all metrics
print('\n-----------------------------------------------------------')
print(f"Train Accuracy: {round(train_accuracy, 3)}")
print(f"Validation Accuracy: {round(val_accuracy, 3)}")
print(f"Test Accuracy: {round(test_accuracy, 3)}")
print("\n---Metrics---\n")
print(f"F1 Score: {round(f1, 3)}")
print(f"Precision: {round(precision, 3)}")
print(f"Recall: {round(recall, 3)}")
print(f"ROC AUC: {round(roc_auc, 3)}")
print('-----------------------------------------------------------\n')

if test_accuracy > 0.870 or f1 > 0.92:
    xgb_model.save_model(f"saved_models/XGBoost_model_Acc_{round(test_accuracy,3)}_F1_{round(f1,3)}_Roc_{round(roc_auc,3)}.bin")