import numpy as np
import pandas as pd
import joblib
import pickle
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.under_sampling import RandomUnderSampler

def load_and_prepare_data():
    try:
        # Load preprocessed dataset
        df = pd.read_pickle('data/dataset_after_preporcessing.pkl')

        # Calculate maximum name length and vocabulary size
        max_name_length = max(df['name'].apply(len))
        unique_chars = set(''.join(df['name']))
        vocab_size = len(unique_chars) + 1

        # Load tokenizer and transform names to sequences
        with open('saved_models/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences(df['name'])
        padded_sequences = pad_sequences(sequences, maxlen=max_name_length)

        # Extract and encode first and last letters of names
        first_letter_encoded = np.array([ord(char) for char in df['name'].apply(lambda x: x[0])])
        last_letter_encoded = np.array([ord(char) for char in df['name'].apply(lambda x: x[-1])])

        # Load and apply scaler models
        scaler_first = joblib.load('saved_models/scaler_models/scaler_first_letter.pkl')
        scaler_last = joblib.load('saved_models/scaler_models/scaler_last_letter.pkl')
        first_letter_encoded_scaled = scaler_first.transform(first_letter_encoded.reshape(-1, 1))
        last_letter_encoded_scaled = scaler_last.transform(last_letter_encoded.reshape(-1, 1))

        # Prepare the final dataset
        y = df['sex'].values
        X = np.concatenate([padded_sequences, first_letter_encoded_scaled, last_letter_encoded_scaled], axis=1)

        # Uncomment and use RandomUnderSampler if class imbalance is to be addressed
        # rus = RandomUnderSampler(random_state=42)
        # X, y = rus.fit_resample(X, y)

        # Split the dataset into a train set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

        return X_train, X_test, y_train, y_test, vocab_size, df

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None, None

def create_model(vocab_size, embedding_dim, total_features_shape, lstm_units, l2_lambda, dropout_rate ):

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=total_features_shape))
    model.add(Bidirectional(LSTM(lstm_units,return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units//2))    
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

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

def print_average_metrics(avg_metrics):
    """
    Prints the average metrics in a formatted manner.

    Args:
    avg_metrics (dict): Dictionary containing the average metrics.
    """
    print('\n' + '-' * 59)
    for metric_name, metric_value in avg_metrics.items():
        print(f"Average {metric_name}: {round(metric_value, 3)}")
    print('-' * 59 + '\n')

def evaluate_model(model, X, y):
    test_loss, test_accuracy = model.evaluate(X, y)
    y_pred = model.predict(X).flatten()
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)
    return {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_f1_score": f1_score(y, y_pred_classes),
        "test_precision": precision_score(y, y_pred_classes),
        "test_recall": recall_score(y, y_pred_classes),
        "test_roc_auc": roc_auc_score(y, y_pred)
    }

def hard_voting(models, X):
    """
    Performs hard voting on an ensemble of models.

    Args:
    models (list): List of trained models.
    X (numpy.array): Input features for prediction.

    Returns:
    numpy.array: Hard voting predictions.
    """
    predictions = np.array([model.predict(X) for model in models])
    predictions = (predictions > 0.5).astype(int)
    mode_pred, _ = stats.mode(predictions, axis=0, keepdims=False)
    return mode_pred.flatten()

def soft_voting(models, X):
    """
    Performs soft voting on an ensemble of models.

    Args:
    models (list): List of trained models.
    X (numpy.array): Input features for prediction.

    Returns:
    numpy.array: Soft voting predictions.
    """
    probabilities = np.array([model.predict(X) for model in models])
    avg_probabilities = np.mean(probabilities, axis=0)
    return (avg_probabilities > 0.5).astype(int)

X_train, X_test, y_train, y_test, vocab_size, df = load_and_prepare_data()

# Model Parameters
total_features_shape = X_train.shape[1]
embedding_dim = 512
lstm_units = 128
l2_lambda = 0.001
dropout_rate = 0.5
epochs = 100
batch_size = 16
patience_for_early_stopping = 5
k = 7
validation_split = 0.1
test_size = 0.1
train_size = 1 - (validation_split + test_size)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_for_early_stopping, restore_best_weights=True)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ahh")

kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_metrics = []
models = []  # List to store models from each fold

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "epochs": epochs,
        "batch_size": batch_size,
        "l2_reg": l2_lambda,
        "dropout_rate": dropout_rate,
        "patience_for_early_stopping": patience_for_early_stopping,
        "k-folds": k,
        "validation_size": validation_split,
        "test_size": test_size,
        "train_size": train_size
    })

    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        # Split data for the current fold
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        model = create_model(vocab_size, embedding_dim, total_features_shape, lstm_units, l2_lambda, dropout_rate)

        # Train the model
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test_fold, y_test_fold)

        # Predictions and additional metrics
        y_pred = model.predict(X_test_fold).flatten()
        y_pred_classes = np.where(y_pred > 0.5, 1, 0)

        f1 = f1_score(y_test_fold, y_pred_classes)
        roc_auc = roc_auc_score(y_test_fold, y_pred)
        # Append metrics for each fold
        fold_metrics.append({
            f"fold_{fold+1}": {
                "accuracy": test_accuracy,
                "loss": test_loss,
                "f1_score": f1,
                "precision": precision_score(y_test_fold, y_pred_classes),
                "recall": recall_score(y_test_fold, y_pred_classes),
                "roc_auc": roc_auc
            }
        })

        models.append(model)

        print()
        print("----------------")
        print(f"k-{fold+1} acc {test_accuracy} f1 {f1} roc_auc {roc_auc}")
        print("----------------")

        model.save(f"EL/K-{fold+1}_BiLSTM_Accuracy_{round(test_accuracy, 3)}_F1_{round(f1, 3)}_Roc_{round(roc_auc, 3)}.h5")

        model.reset_states()

    # Extract metric keys from any one of the fold's metrics (assuming all folds have the same metrics)
    metrics_keys = fold_metrics[0]['fold_1'].keys()

    # Initialize dictionaries for sum and count of metrics
    sum_metrics = {metric: 0 for metric in metrics_keys}
    count_metrics = {metric: 0 for metric in metrics_keys}

    # Sum and count metrics for each fold
    for fold_metric in fold_metrics:
        for fold in range(1, k+1):
            fold_key = f'fold_{fold}'
            if fold_key in fold_metric:
                for metric in metrics_keys:
                    sum_metrics[metric] += fold_metric[fold_key][metric]
                    count_metrics[metric] += 1

    # Calculate and log the average metrics
    avg_metrics = {metric: (sum_metrics[metric] / count_metrics[metric]) if count_metrics[metric] > 0 else None for metric in metrics_keys}
    
    # Log the metrics
    mlflow.log_metrics(avg_metrics)

    # Retrain model on the entire training set after cross-validation
    final_model = create_model(vocab_size, embedding_dim, total_features_shape, lstm_units, l2_lambda, dropout_rate)

    final_model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Infer signature and log the final model
    signature = infer_signature(X_train, final_model.predict(X_train))
    mlflow.keras.log_model(final_model, "model", signature=signature)

    # Evaluate and log final model metrics
    final_metrics = evaluate_model(final_model, X_test, y_test)
    mlflow.log_metrics(final_metrics)

    # Set additional tags
    mlflow.set_tags({
        "Description": "Character-Level BiLSTM with k-fold cross validation",
        "Encoding": "Character-Level for name | other features converting them to their ASCII value then normalized",
        "Features": ', '.join(df.columns.tolist()),
        'Number of Features': len(df.columns.tolist()),
        "Model Type": "BiLSTM",
        "model_architecture": get_model_summary(final_model)
    })


print_average_metrics(avg_metrics)

# Use ensemble to make predictions on the test set
hard_votes = hard_voting(models, X_test)
soft_votes = soft_voting(models, X_test)

# Evaluate ensemble performance
hard_voting_accuracy = accuracy_score(y_test, hard_votes)
soft_voting_accuracy = accuracy_score(y_test, soft_votes)

print(f"Hard Voting Accuracy: {hard_voting_accuracy:.3f}")
print(f"Soft Voting Accuracy: {soft_voting_accuracy:.3f}")