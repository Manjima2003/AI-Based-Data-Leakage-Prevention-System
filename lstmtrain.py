import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#  Ensure 'models/' directory exists
os.makedirs("models", exist_ok=True)

#  Load Preprocessed Data
df = pd.read_csv("processed_aws_logs.csv")
data = df.values.astype(np.float32)
data = data.reshape(data.shape[0], 1, data.shape[1])  # Reshape for LSTM

# Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_tensor = torch.tensor(train_data)
test_tensor = torch.tensor(test_data)

# DataLoader
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True)

# Define LSTM Model
class LSTMAnomaly(nn.Module):
    def __init__(self, input_size):
        super(LSTMAnomaly, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, input_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Initialize Model
model = LSTMAnomaly(input_size=train_data.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
for epoch in range(20):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets[:, -1, :])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save Model (Ensure 'models/' directory exists)
torch.save(model.state_dict(), "models/lstm_model.pth")
print(" LSTM Model Training Complete & Saved!")
