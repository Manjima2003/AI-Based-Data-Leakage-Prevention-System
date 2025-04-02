import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Drop unnecessary columns (UUIDs, account IDs, and other irrelevant fields)
    drop_cols = ['eventID', 'userIdentityaccountId', 'userIdentityprincipalId', 
                 'userIdentityarn', 'userIdentityaccessKeyId', 'sourceIPAddress']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Convert eventTime to Unix timestamp safely
    if 'eventTime' in df.columns:
        df['eventTime'] = pd.to_datetime(df['eventTime'], errors='coerce')
        df['eventTime'] = df['eventTime'].view(np.int64) // 10**9  # Convert to seconds

    # Encode categorical columns
    categorical_cols = ['eventName', 'eventSource', 'userAgent', 'awsRegion', 
                        'userIdentitytype', 'eventType', 'userIdentityuserName', 
                        'errorCode', 'errorMessage', 'requestParametersinstanceType']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)  # Convert to string before encoding
            df[col] = LabelEncoder().fit_transform(df[col])

    # Normalize numerical columns
    numerical_cols = ['eventTime', 'eventVersion']
    scaler = MinMaxScaler()

    for col in numerical_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    # Fill any missing values with 0
    df.fillna(0, inplace=True)

    return df

# File path (Ensure the path is correct and the file exists)
file_path = r"C:\Users\Sofiya Raj\Desktop\aidlpsystem\CODE\cloudtrail_sampled.csv"

# Preprocess the dataset
processed_df = preprocess_data(file_path)

# Print the first few rows of the processed data
print(processed_df.head())

# Save processed data (optional)
processed_df.to_csv("processed_cloudtrail.csv", index=False)
