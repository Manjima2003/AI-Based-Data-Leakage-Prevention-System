import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import json  # <-- Import JSON to save thresholds

# Define file paths
BASE_DIR = r"C:\Users\Sofiya Raj\Desktop\aidlpsystem\CODE"
PROCESSED_FILE_PATH = os.path.join(BASE_DIR, "processed_cloudtrail.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
AUTOENCODER_MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_model.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "thresholds.json")  # <-- Save thresholds

# Load processed dataset
df = pd.read_csv(PROCESSED_FILE_PATH)

# Ensure all features are numeric
features = df.drop(columns=['errorCode', 'errorMessage'], errors='ignore')

# Normalize the dataset
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save the scaler for later use in detection
joblib.dump(scaler, SCALER_PATH)

# Split into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(scaled_features, test_size=0.2, random_state=42)

# Reshape for LSTM (Samples, Timesteps, Features)
X_train_LSTM = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_LSTM = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ==============================
# 🚀 LSTM Model for Anomaly Detection
# ==============================
lstm_model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
    keras.layers.LSTM(32, return_sequences=False),
    keras.layers.Dense(X_train.shape[1], activation="sigmoid")
])

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_history = lstm_model.fit(
    X_train_LSTM, X_train, 
    epochs=10, batch_size=32, 
    validation_data=(X_test_LSTM, X_test),
    verbose=1
)

# Save the trained LSTM model
lstm_model.save(LSTM_MODEL_PATH)

# Evaluate LSTM Model
train_loss, train_mae = lstm_model.evaluate(X_train_LSTM, X_train, verbose=0)
test_loss, test_mae = lstm_model.evaluate(X_test_LSTM, X_test, verbose=0)

print(f"📊 LSTM Model Metrics:")
print(f"Train Loss (MSE): {train_loss:.4f}, Train MAE: {train_mae:.4f}")
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# ==============================
# 🚀 Autoencoder Model for Anomaly Detection
# ==============================
autoencoder = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(X_train.shape[1], activation="sigmoid")
])

autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
autoencoder_history = autoencoder.fit(
    X_train, X_train, 
    epochs=10, batch_size=32, 
    validation_data=(X_test, X_test),
    verbose=1
)

# Save the trained Autoencoder model
autoencoder.save(AUTOENCODER_MODEL_PATH)

# Evaluate Autoencoder Model
train_loss_auto, train_mae_auto = autoencoder.evaluate(X_train, X_train, verbose=0)
test_loss_auto, test_mae_auto = autoencoder.evaluate(X_test, X_test, verbose=0)

print(f"\n📊 Autoencoder Model Metrics:")
print(f"Train Loss (MSE): {train_loss_auto:.4f}, Train MAE: {train_mae_auto:.4f}")
print(f"Test Loss (MSE): {test_loss_auto:.4f}, Test MAE: {test_mae_auto:.4f}")

# ==============================
# 🔥 Calculate & Save Anomaly Thresholds
# ==============================
print("\n🔹 Calculating Anomaly Detection Thresholds...")

# Compute reconstruction errors
lstm_train_errors = np.mean(np.abs(lstm_model.predict(X_train_LSTM) - X_train), axis=1)
autoencoder_train_errors = np.mean(np.abs(autoencoder.predict(X_train) - X_train), axis=1)

# Set threshold as the 95th percentile of training errors
lstm_threshold = np.percentile(lstm_train_errors, 95)
autoencoder_threshold = np.percentile(autoencoder_train_errors, 95)

# Save thresholds to JSON file
thresholds = {"lstm": float(lstm_threshold), "autoencoder": float(autoencoder_threshold)}
with open(THRESHOLD_PATH, "w") as f:
    json.dump(thresholds, f)

print(f"✅ Thresholds saved successfully! LSTM: {lstm_threshold:.4f}, Autoencoder: {autoencoder_threshold:.4f}")

# ==============================
# 📈 Plot Training & Validation Loss Curves
# ==============================
plt.figure(figsize=(12, 5))

# LSTM Loss Plot
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history["loss"], label="Train Loss (MSE)")
plt.plot(lstm_history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("LSTM Model Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()

# Autoencoder Loss Plot
plt.subplot(1, 2, 2)
plt.plot(autoencoder_history.history["loss"], label="Train Loss (MSE)")
plt.plot(autoencoder_history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("Autoencoder Model Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()

plt.tight_layout()
plt.show()

print("✅ Training completed. Models and thresholds saved successfully!")
