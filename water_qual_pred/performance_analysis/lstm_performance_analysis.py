import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATA
# Make sure your CSV is in the same folder as this script
csv_path = '../fetching_data_from_open_meteo/final_water_quality_log.csv'
df = pd.read_csv(csv_path)

# 2. SELECT FEATURES
# These must be the same 6 features used during training
features = ['air_temp', 'humidity', 'rain', 'water_temp', 'do', 'ph']
data = df[features].values

# 3. SCALE DATA
# LSTMs are sensitive to scale; we must normalize to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. CREATE SEQUENCES (Sliding Window)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        # Target: Next hour's Water Temp, DO, and pH (indices 3, 4, 5)
        y.append(data[i + seq_length, 3:]) 
    return np.array(X), np.array(y)

SEQ_HOURS = 24 
X, y = create_sequences(scaled_data, SEQ_HOURS)

# 5. SPLIT TO GET X_TEST
# We take the last 15% of the data as the "Test Set" to simulate real-world performance
split_index = int(0.85 * len(X))
X_test = X[split_index:]
y_test = y[split_index:]

# 6. LOAD YOUR TRAINED MODEL
try:
    model = tf.keras.models.load_model('../models/lstm_model/water_qual_param.keras')
    print(" Model loaded successfully!")
except:
    print(" Error: 'water_qual_param.keras' not found in this folder.")

# 7. GENERATE PREDICTIONS
y_pred = model.predict(X_test)

# 8. CALCULATE METRICS
params = ["Water Temp", "Dissolved Oxygen", "pH Level"]
metrics_list = []


print(" PERFORMANCE METRICS ANALYSIS")


# for i in range(len(params)):
#     # Raw metrics on the scaled data
#     mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
#     rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
#     r2 = r2_score(y_test[:, i], y_pred[:, i])
    
#     # Accuracy calculation (100 - Mean Absolute Percentage Error)
#     # Adding a small epsilon to avoid division by zero
#     mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-10))) * 100
    
#     metrics_list.append({
#         "Parameter": params[i],
#         "MAE": round(mae, 4),
#         "RMSE": round(rmse, 4),
#         "R2 Score": round(r2, 4),
#         "Accuracy": f"{round(100 - mape, 2)}%"
#     })

# ... (previous code above)

for i in range(len(params)):
    # Keep these standard metrics
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    
    # --- PASTE THE NEW LOGIC HERE ---
    # Calculate accuracy based on the range of data to avoid the 'pH zero-division' error
    data_range = np.max(y_test[:, i]) - np.min(y_test[:, i])
    
    # Check to prevent division by zero if data_range is somehow 0
    if data_range == 0:
        stable_accuracy = 100.0
    else:
        stable_accuracy = (1 - (mae / data_range)) * 100

    metrics_list.append({
        "Parameter": params[i],
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
        "Accuracy": f"{round(stable_accuracy, 2)}%" 
    })
    # --- END OF NEW LOGIC ---

# ... (rest of the code to print the results)

# 9. OUTPUT RESULTS
results_df = pd.DataFrame(metrics_list)
print(results_df.to_string(index=False))
