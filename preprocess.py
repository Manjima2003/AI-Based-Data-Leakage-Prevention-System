import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#  Load AWS CloudTrail Logs
df = pd.read_csv("C:/Users/Sofiya Raj/aidlpsystem/aidlpsystem/cloudtriallog.csv")

#  Convert eventTime to UNIX timestamp
df["eventTime"] = pd.to_datetime(df["eventTime"], errors='coerce')  # Handle invalid dates
df["eventTime"] = df["eventTime"].astype("int64") // 10**9  # Convert to seconds

#  Identify Non-Numeric Columns
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("Non-Numeric Columns:", non_numeric_cols)

#  Encode Categorical Columns Using Label Encoding
label_enc = LabelEncoder()
for col in non_numeric_cols:
    df[col] = label_enc.fit_transform(df[col].astype(str))  # Convert to string & encode

#  Scale numerical values
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

#  Save Preprocessed Data
df.to_csv("processed_aws_logs.csv", index=False)

print(" Preprocessing Complete: AWS Logs Ready for LSTM & Autoencoder Training!")
