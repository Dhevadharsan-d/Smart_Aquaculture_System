#lstm model trainig with combined dataset of 4 districts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. CONFIGURATION & LOCATIONS
# Distributed data from 5 diverse climatic zones in Tamil Nadu
LOCATIONS = {
    "Chennai": {"lat": 13.08, "lon": 80.27},
    "Coimbatore": {"lat": 11.01, "lon": 76.95},
    "Kanyakumari": {"lat": 8.08, "lon": 77.53},
    "Krishnagiri": {"lat": 12.51, "lon": 78.21},
    "Trichy": {"lat": 10.79, "lon": 78.70}
}

PAST_DAYS = 365          # 1 year of data per location
SEQ_HOURS = 24           # 24-hour sliding window for LSTM
FEATURES = ["air_temp", "humidity", "rain", "water_temp", "do", "ph"]
TARGETS = ["water_temp", "do", "ph"]

# 2. DATA EXTRACTION & BIOLOGICAL FORMULA INTEGRATION
def fetch_and_process_all_districts():
    all_data = []
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=PAST_DAYS)
    
    global_id = 1
    
    for city, coord in LOCATIONS.items():
        print(f"Fetching data for {city}...")
        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={coord['lat']}&longitude={coord['lon']}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&hourly=temperature_2m,relativehumidity_2m,precipitation&timezone=UTC")
        
        data = requests.get(url).json()
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "air_temp": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relativehumidity_2m"],
            "rain": data["hourly"]["precipitation"],
            "location": city
        })

        # Apply Biological Formulas
        wt, do_val, ph_val = [], [], []
        prev_tw = None
        
        for _, r in df.iterrows():
            # Water Temp logic
            tw = r.air_temp - 1 if prev_tw is None else 0.7 * prev_tw + 0.3 * (r.air_temp - 1)
            # DO logic
            d = (14.6 - 0.2 * tw) * (1 + (r.humidity - 50) / 500)
            d = np.clip(d, 2, 14)
            # pH logic (Diurnal Cycle)
            p = 7.4 + 0.3 * math.sin(2 * math.pi * r.timestamp.hour / 24) - 0.01 * (tw - 26) - 0.08 * math.log1p(r.rain)
            p = np.clip(p, 6.5, 9.0)

            wt.append(round(tw, 2)); do_val.append(round(d, 2)); ph_val.append(round(p, 2))
            prev_tw = tw

        df["water_temp"], df["do"], df["ph"] = wt, do_val, ph_val
        all_data.append(df)
    
    master_df = pd.concat(all_data).reset_index(drop=True)
    master_df.insert(0, 'id', range(1, len(master_df) + 1)) # Proper ID injection
    return master_df

# 3. DATA PREPARATION FOR LSTM
df = fetch_and_process_all_districts()
df.to_csv("tn_distributed_aquaculture.csv", index=False)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[FEATURES])

X, y = [], []
for i in range(len(scaled_data) - SEQ_HOURS):
    X.append(scaled_data[i:i + SEQ_HOURS])
    y.append(scaled_data[i + SEQ_HOURS][3:]) # Target: Water Temp, DO, pH

X, y = np.array(X), np.array(y)

# 4. DISTRIBUTED MODEL TRAINING
model = Sequential([
    LSTM(64, input_shape=(SEQ_HOURS, len(FEATURES)), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(3) # Predicting 3 parameters
])

model.compile(optimizer="adam", loss="mse")
history = model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, 
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# 5. SAVE MODEL FOR FLUTTER
model.save("water_lstm_distributed.keras")
