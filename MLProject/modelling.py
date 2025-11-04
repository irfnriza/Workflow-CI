"""
Model Bi-LSTM untuk Prediksi Polusi Udara
Menggunakan MLflow Autolog untuk tracking eksperimen
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import os

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Konfigurasi MLflow - Simpan di localhost
# Tracking URI: file:./mlruns untuk penyimpanan lokal
TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Air_Pollution_BiLSTM")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Artifact Location: {mlflow.get_artifact_uri() if mlflow.active_run() else 'No active run'}")

# Enable autolog untuk TensorFlow/Keras
# Autolog akan mencatat: model, metrics, parameters, dan artifacts secara otomatis
mlflow.tensorflow.autolog(
    log_models=True,           # Simpan model
    log_datasets=True,         # Log dataset info
    disable=False,             # Enable autolog
    exclusive=False,           # Allow manual logging
    disable_for_unsupported_versions=False,
    silent=False,              # Show logging messages
    registered_model_name=None # Tidak auto-register ke Model Registry
)

def load_data():
    """
    Load data yang telah dipreprocess
    """
    print("Loading preprocessed data...")
    X_train = np.load('Air Pollution Forecasting_preprocessing/X_train.npy')
    X_test = np.load('Air Pollution Forecasting_preprocessing/X_test.npy')
    y_train = np.load('Air Pollution Forecasting_preprocessing/y_train.npy')
    y_test = np.load('Air Pollution Forecasting_preprocessing/y_test.npy')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def build_bilstm_model(timesteps, features):
    """
    Membangun model Bi-LSTM
    
    Args:
        timesteps: Jumlah timesteps dalam sequence
        features: Jumlah fitur input
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        # Layer 1: Bidirectional LSTM pertama
        Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), 
                     input_shape=(timesteps, features)),
        Dropout(0.2),
        
        # Layer 2: Bidirectional LSTM kedua
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        Dropout(0.2),
        
        # Layer 3: Fully connected layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(1)  # Untuk regression
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model dan return metrics
    """
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2
    }
    
    return metrics, y_pred

def plot_predictions(y_true, y_pred, save_path='predictions_plot.png'):
    """
    Plot perbandingan prediksi vs actual
    """
    plt.figure(figsize=(15, 5))
    
    # Plot pertama: Time series comparison (sample 500 data points)
    plt.subplot(1, 2, 1)
    sample_size = min(500, len(y_true))
    plt.plot(y_true[:sample_size], label='Actual', alpha=0.7)
    plt.plot(y_pred[:sample_size], label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Pollution Value')
    plt.title('Actual vs Predicted (First 500 samples)')
    plt.legend()
    plt.grid(True)
    
    # Plot kedua: Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot: Actual vs Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    """
    Main function untuk training model
    """
    print("=" * 60)
    print("BI-LSTM MODEL TRAINING WITH MLFLOW")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Get dimensions
    timesteps = X_train.shape[1]
    features = X_train.shape[2]
    
    # Start MLflow run
    with mlflow.start_run(run_name="BiLSTM_Model"):
        print("\n" + "=" * 60)
        print("Building Bi-LSTM Model...")
        print("=" * 60)
        
        # Build model
        model = build_bilstm_model(timesteps, features)
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Log additional parameters
        mlflow.log_param("model_type", "Bidirectional LSTM")
        mlflow.log_param("timesteps", timesteps)
        mlflow.log_param("features", features)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 50)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create checkpoint directory
        os.makedirs('models', exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            'models/best_bilstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("Training Model...")
        print("=" * 60)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("Evaluating Model...")
        print("=" * 60)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Print metrics
        print("\nTest Metrics:")
        print(f"  MSE:  {metrics['test_mse']:.4f}")
        print(f"  RMSE: {metrics['test_rmse']:.4f}")
        print(f"  MAE:  {metrics['test_mae']:.4f}")
        print(f"  R2:   {metrics['test_r2']:.4f}")
        
        # Log additional metrics
        mlflow.log_metrics(metrics)
        
        # Create and log prediction plot
        print("\nCreating prediction plots...")
        plot_path = plot_predictions(y_test, y_pred)
        mlflow.log_artifact(plot_path)
        
        # Log best model sebagai artifact
        mlflow.log_artifact('models/best_bilstm_model.h5')
        
        # Get run info
        run = mlflow.active_run()
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print("\n📊 MLflow Tracking Information:")
        print(f"  Experiment: Air_Pollution_BiLSTM")
        print(f"  Run ID: {run_id}")
        print(f"  Artifact URI: {artifact_uri}")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
        print("\n🌐 Cara Melihat Hasil di MLflow UI:")
        print("  1. Aktifkan conda environment:")
        print("     conda activate MSML")
        print("  2. Jalankan MLflow UI:")
        print("     mlflow ui")
        print("     ATAU: mlflow ui --host 127.0.0.1 --port 5000")
        print("  3. Buka browser:")
        print("     http://localhost:5000")
        print("     ATAU: http://127.0.0.1:5000")
        print("=" * 60)

if __name__ == "__main__":
    main()
