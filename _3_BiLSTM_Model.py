# python=3.8
# conda env name : gender_pred_env
# Packges
import joblib
import numpy as np
import pandas as pd
import pickle
import optuna
import warnings
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc

# Adjusting warnings
warnings.filterwarnings('ignore')
# Load Data
df = pd.read_pickle('data/dataset_after_preporcessing.pkl')
df.head()
max_name_length = max(df['name'].apply(len))
unique_chars = set(''.join(df['name']))  
vocab_size = len(unique_chars) + 1  

with open('saved_models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
total_features_shape = X_train.shape[1]


def lstm_objective(trial):
    # Hyperparameters to be tuned by Optuna for BiLSTM
    embedding_dim = trial.suggest_int('embedding_dim',64, 512)
    lstm_units = trial.suggest_int('lstm_units', 32, 256)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    l2_lambda = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    # epochs = trial.suggest_int('epochs', 5, 50)
    patience_for_early_stopping = trial.suggest_int('patience_for_early_stopping', 4, 9)

    # Define the model architecture using the hyperparameters
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=total_features_shape))

    model.add(Bidirectional(LSTM(lstm_units,return_sequences=True,kernel_regularizer=l2(l2_lambda))))

    model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(lstm_units//2,kernel_regularizer=l2(l2_lambda))))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Use early stopping as a callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience_for_early_stopping)
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # The objective we want to minimize (validation loss in this case)
    val_loss = np.min(history.history['val_loss'])
    
    # Optionally, you can return additional information to be used later
    trial.set_user_attr('stopped_epoch', len(history.history['loss']))

    return val_loss

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(lstm_objective, n_trials=1)

# Best hyperparameters
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
best_lstm_params = study.best_trial.params

def get_model_summary(model):
    """
    Generates the summary of the given Keras model and returns it as a string.

    Args:
    model (keras.Model): The Keras model to summarize.

    Returns:
    str: The summary of the model.
    """
    model_summary_list = []
    model.summary(print_fn=lambda x: model_summary_list.append(x))
    return '\n'.join(model_summary_list)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Gender Prediction Models Tracking")
# MLflow tracking
with mlflow.start_run():
    # Log model parameters
    
    mlflow.log_params({
        "max_name_length": max_name_length,
        "vocab_size": vocab_size,
        "train_size_percentage": train_size_percentage,
        "test_size_percentage": test_size_percentage,
        "validation_size_percentage": validation_size_percentage
    })
    
    mlflow.log_params(best_lstm_params)

    # Define the model architecture using the hyperparameters
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=best_lstm_params['embedding_dim'], input_length=total_features_shape))

    model.add(Bidirectional(LSTM(best_lstm_params['lstm_units'],return_sequences=True,kernel_regularizer=l2(best_lstm_params['l2_reg']))))

    model.add(Dropout(best_lstm_params['dropout_rate']))

    model.add(Bidirectional(LSTM(best_lstm_params['lstm_units']//2,kernel_regularizer=l2(best_lstm_params['l2_reg']))))

    model.add(Dropout(best_lstm_params['dropout_rate']))

    model.add(Dense(1, activation='sigmoid'))

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=best_lstm_params['patience_for_early_stopping'])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_lstm_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )

    # Log training history
    for epoch in range(len(history.history['accuracy'])):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    # Predictions and additional metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.where(y_pred > 0.5, 1, 0).reshape(-1)
    f1 = f1_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # Calculate AUC (Area Under the ROC Curve)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix calculation
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_dict = {f"{i}-{j}": cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])}

    # Log additional metrics
    mlflow.log_metrics({
    "test_f1": f1,
    "test_precision": precision,
    "test_recall": recall,
    "test_roc_auc": roc_auc,
    "test_loss": test_loss, 
    "test_accuracy": test_accuracy,
    "train_accuracy":train_accuracy,
    "train_loss":train_loss,
    "val_loss":val_loss,
    "val_accuracy":val_accuracy
    })

    # Infer the signature using the combined data and predictions
    signature = infer_signature(X_test, y_pred)

    mlflow.keras.log_model(model, "model", signature=signature)

    # Set additional tags
    mlflow.set_tags({
        "Description": "Character-Level BiLSTM",
        "Encoding": "Character-Level for name | other featuers Coverting them to thier ASCI value then norm",
        "Features": ', '.join(df.columns.tolist()),
        'Number of Features':len(df.columns.tolist()),
        "Model Type": "BiLSTM",
        "model_architecture": get_model_summary(model),
        'confusion_matrix':list(cm_dict.values())
    })
# End MLflow run

print()
print('-----------------------------------------------------------')
print(f"Train Accuracy: {round(train_accuracy, 3)}")
print(f"Val Accuracy: {round(val_accuracy, 3)}")
print(f"Test Accuracy: {round(test_accuracy, 3)}")
print("\n---Metrics---\n")
print(f"F1 Score: {round(f1, 3)}")
print(f"Precision: {round(precision, 3)}")
print(f"Recall: {round(recall, 3)}")
print(f"ROC AUC: {round(roc_auc, 3)}")
print("\n---Loss---\n")
print(f"Train Loss: {round(train_loss, 3)}")
print(f"Val Loss: {round(val_loss, 3)}")
print(f"Test Loss: {round(test_loss, 3)}")
print('-----------------------------------------------------------')

if test_accuracy > 0.870 or f1 > 0.92:
    xgb_model.save_model(f"saved_models/BiLSTM Models/BiLSTM_Acc_{round(test_accuracy,3)}_F1_{round(f1,3)}_Roc_{round(roc_auc,3)}.h5")
    print("model saved!")