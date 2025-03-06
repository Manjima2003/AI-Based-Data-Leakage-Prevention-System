import pandas as pd
import numpy as np
import torch
from lstmtrain import LSTMAnomaly
from autoencodetrain import Autoencoder

# ✅ Load Preprocessed Data in Small Batches
chunk_size = 5000  # Process in smaller chunks to avoid memory issues
print("🚀 Loading Data in Chunks...")
data_chunks = pd.read_csv("processed_aws_logs.csv", chunksize=chunk_size)

# ✅ Load Trained LSTM Model
print("🚀 Loading LSTM Model...")
lstm_model = LSTMAnomaly(input_size=18)  # Adjust input size based on dataset
lstm_model.load_state_dict(torch.load("models/lstm_model.pth", weights_only=True))
lstm_model.eval()

# ✅ Load Trained Autoencoder Model
print("🚀 Loading Autoencoder Model...")
autoencoder = Autoencoder(input_dim=18)  # Adjust input size based on dataset
autoencoder.load_state_dict(torch.load("models/autoencoder_model.pth", weights_only=True))
autoencoder.eval()

# ✅ Initialize Empty Lists to Store Loss Values
lstm_losses = []
autoencoder_losses = []

# ✅ Process Each Chunk Separately
print("🚀 Processing Data in Chunks...")
for i, chunk in enumerate(data_chunks):
    print(f"➡ Processing Chunk {i+1}...")  # Debugging: Show Progress

    data = chunk.values.astype(np.float32)
    data = data.reshape(data.shape[0], 1, data.shape[1])  # Reshape for LSTM

    # ✅ Convert to Torch Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # ✅ Compute LSTM Reconstruction Loss
    with torch.no_grad():
        lstm_reconstructed = lstm_model(data_tensor).detach().numpy()
        lstm_mse_loss = np.mean(np.power(data - lstm_reconstructed, 2), axis=(1, 2))
        lstm_losses.extend(lstm_mse_loss)

    # ✅ Compute Autoencoder Reconstruction Loss
    data_flat = data.reshape(data.shape[0], -1)  # Flatten for Autoencoder
    data_tensor_flat = torch.tensor(data_flat, dtype=torch.float32)

    with torch.no_grad():
        autoencoder_reconstructed = autoencoder(data_tensor_flat).detach().numpy()
        autoencoder_mse_loss = np.mean(np.power(data_flat - autoencoder_reconstructed, 2), axis=1)
        autoencoder_losses.extend(autoencoder_mse_loss)

print("✅ All Chunks Processed! Calculating Thresholds...")

# ✅ Convert Loss Lists to NumPy Arrays
lstm_losses = np.array(lstm_losses)
autoencoder_losses = np.array(autoencoder_losses)

# ✅ Set Anomaly Thresholds (Mean + 3 Standard Deviations)
lstm_threshold = np.mean(lstm_losses) + 3 * np.std(lstm_losses)
autoencoder_threshold = np.mean(autoencoder_losses) + 3 * np.std(autoencoder_losses)

# ✅ Save Thresholds
with open("models/lstm_anomaly_threshold.txt", "w") as f:
    f.write(str(lstm_threshold))
with open("models/autoencoder_anomaly_threshold.txt", "w") as f:
    f.write(str(autoencoder_threshold))

print(f" LSTM Anomaly Detection Threshold: {lstm_threshold}")
print(f" Autoencoder Anomaly Detection Threshold: {autoencoder_threshold}")
print(" Threshold Calculation Complete! 🎉")
