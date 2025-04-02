import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import streamlit as st

# Streamlit Web Interface
def main():
    st.title("AI-Based Data Leak Detection System")
    uploaded_file = st.file_uploader("Upload Log File", type=["csv"])
    
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        new_df.fillna('Unknown', inplace=True)

        # Load encoder and scaler
        encoder = pickle.load(open("encoder.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))

        # Ensure feature consistency
        expected_features = scaler.feature_names_in_  # Get expected feature names from scaler
        missing_cols = [col for col in expected_features if col not in new_df.columns]
        extra_cols = [col for col in new_df.columns if col not in expected_features]

        # Drop extra columns
        new_df = new_df.drop(columns=extra_cols, errors='ignore')

        # Add missing columns with default values
        for col in missing_cols:
            new_df[col] = 0  # Default value (change if needed)

        # Reorder columns to match training data
        new_df = new_df[expected_features]

        # Handle unseen labels in categorical features
        for col in new_df.select_dtypes(include=['object']).columns:
            new_df[col] = new_df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
            encoder.classes_ = np.append(encoder.classes_, 'Unknown')  # Ensure 'Unknown' exists in encoder
            new_df[col] = encoder.transform(new_df[col])

        new_scaled = scaler.transform(new_df)

        # Load trained model
        autoencoder = tf.keras.models.load_model("data_leak_detector.h5")
        recon = autoencoder.predict(new_scaled)

        # Compute Mean Squared Error (MSE)
        mse = np.mean(np.power(new_scaled - recon, 2), axis=1)

        # Define anomaly detection threshold
        thresh = np.percentile(mse, 95)  # Adjust threshold if needed

        # Classify as Leak or Normal
        results = ['Leak' if e > thresh else 'Normal' for e in mse]
        new_df['Anomaly Detection'] = results

        # Display results
        st.write(new_df)
        
        leaks = new_df[new_df['Anomaly Detection'] == 'Leak']
        if not leaks.empty:
            st.error("🚨 Potential Data Leak Detected!")
            for _, row in leaks.iterrows():
                st.write(row.to_dict())
        else:
            st.success("✅ No Data Leaks Detected")

# Fix incorrect `if` statement
if __name__ == "__main__":
    main()
# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import pickle
# import json

# # Load trained model and preprocessing tools
# autoencoder = tf.keras.models.load_model("data_leak_detector.h5")
# scaler = pickle.load(open("scaler.pkl", "rb"))
# encoder = pickle.load(open("encoder.pkl", "rb"))

# st.set_page_config(page_title="Data Leak Prevention System")

# st.title("AI-Based Data Leak Prevention System")

# # File uploader for React frontend
# uploaded_file = st.file_uploader("Upload Log File (CSV)", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     df.fillna("Unknown", inplace=True)

#     # Feature consistency
#     expected_features = scaler.feature_names_in_
#     missing_cols = [col for col in expected_features if col not in df.columns]
#     extra_cols = [col for col in df.columns if col not in expected_features]

#     # Drop extra columns
#     df = df.drop(columns=extra_cols, errors='ignore')

#     # Add missing columns
#     for col in missing_cols:
#         df[col] = 0

#     # Reorder columns
#     df = df[expected_features]

#     # Encode categorical features
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")
#         encoder.classes_ = np.append(encoder.classes_, "Unknown")
#         df[col] = encoder.transform(df[col])

#     # Scale the data
#     scaled_data = scaler.transform(df)

#     # Get model predictions
#     reconstructions = autoencoder.predict(scaled_data)
#     mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)

#     # Set anomaly detection threshold
#     threshold = np.percentile(mse, 95)

#     # Classify as Leak or Normal
#     df["Anomaly Detection"] = ["Leak" if e > threshold else "Normal" for e in mse]

#     # Convert results to JSON format
#     result_json = df.to_json(orient="records")

#     # Return JSON response to frontend
#     st.write(result_json)
