import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ✅ Ensure 'models/' directory exists
os.makedirs("models", exist_ok=True)

# ✅ Load Preprocessed Data
df = pd.read_csv("processed_aws_logs.csv")
data = df.values.astype(np.float32)

# ✅ Train-Test Split (90% Train, 10% Test)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# ✅ Convert to PyTorch tensors
train_tensor = torch.tensor(train_data)
test_tensor = torch.tensor(test_data)

# ✅ DataLoader
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

# ✅ Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)  # Reconstruct input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ Initialize Model
input_dim = train_data.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error for anomaly detection
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train Model
for epoch in range(20):
    optimizer.zero_grad()
    output = model(train_tensor)
    loss = criterion(output, train_tensor)  # Compare reconstructed vs. original data
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ✅ Save Model
torch.save(model.state_dict(), "models/autoencoder_model.pth")
print(" Autoencoder Model Training Complete & Saved!")
