
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import os

# # 1. Configuration (Local VS Code)
# LATITUDE = 10.98267
# LONGITUDE = 76.97678
# PAST_DAYS = 365 
# CSV_PATH = 'water_qual_param.csv'

# # 2. Fetching Function
# def fetch_hourly_weather(lat, lon, past_days):
#     # Using now() instead of utcnow() for modern Python versions
#     end_date = datetime.now().date()
#     start_date = end_date - timedelta(days=past_days)

#     url = (
#         "https://archive-api.open-meteo.com/v1/archive?"
#         f"latitude={lat}&longitude={lon}"
#         f"&start_date={start_date}&end_date={end_date}"
#         "&hourly=temperature_2m,relativehumidity_2m,precipitation"
#         "&timezone=UTC"
#     )

#     print(f"📡 Requesting data from: {start_date} to {end_date}...")
#     data = requests.get(url).json()

#     df = pd.DataFrame({
#         "datetime": pd.to_datetime(data["hourly"]["time"]),
#         "air_temp": data["hourly"]["temperature_2m"],
#         "humidity": data["hourly"]["relativehumidity_2m"],
#         "rain": data["hourly"]["precipitation"]
#     })

#     return df

# # 3. Execute and Save
# print("🚀 Starting Data Fetch...")
# df_weather = fetch_hourly_weather(LATITUDE, LONGITUDE, PAST_DAYS)

# # Save to CSV in your VS Code folder
# df_weather.to_csv(CSV_PATH, index=False)

# # 4. Final Verification Prints
# if os.path.exists(CSV_PATH):
#     print(f"✅ Dataset successfully saved to: {os.path.abspath(CSV_PATH)}")
#     print("\n📊 Preview of Fetched Data:")
#     print(df_weather.head())  # Wrapped in print() for VS Code
# else:
#     print("❌ Error: File was not saved.")


import pandas as pd
import numpy as np
import math
import os



# 1. Configuration
INPUT_CSV = 'water_qual_param.csv'  # The file you just fetched from Open-Meteo
OUTPUT_CSV = 'final_water_quality_log.csv'

# 2. Logic to calculate water quality (from your notebook)
def compute_final_params(df):
    wt, do, ph = [], [], []
    prev_tw = None

    for _, r in df.iterrows():
        # --- Water temperature calculation ---
        # Logic: 0.7 * previous_temp + 0.3 * (current_air_temp - 1)
        tw = r.air_temp - 1 if prev_tw is None else 0.7 * prev_tw + 0.3 * (r.air_temp - 1)

        # --- Dissolved Oxygen calculation ---
        # Logic: Formula based on water temp and humidity
        d = (14.6 - 0.2 * tw) * (1 + (r.humidity - 50) / 500)
        d = np.clip(d, 2, 14)

        # --- pH calculation ---
        # Logic: Diurnal cycle based on hour, temp, and rain
        hour = pd.to_datetime(r.datetime).hour
        p = (
            7.4
            + 0.3 * math.sin(2 * math.pi * hour / 24)
            - 0.01 * (tw - 26)
            - 0.08 * math.log1p(r.rain)
            + np.random.normal(0, 0.1)
        )
        p = np.clip(p, 6.5, 9.0)

        wt.append(round(tw, 2))
        do.append(round(d, 2))
        ph.append(round(p, 2))
        prev_tw = tw

    # Create the final structured dataframe
    final_df = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "timestamp": df["datetime"],
        "water_temp": wt,
        "do": do,
        "ph": ph
    })
    
    return final_df

# 3. Process and Save
if os.path.exists(INPUT_CSV):
    print(f"📖 Reading raw weather data from {INPUT_CSV}...")
    raw_df = pd.read_csv(INPUT_CSV)
    
    print("🧪 Calculating Water Quality Parameters...")
    processed_df = compute_final_params(raw_df)
    
    # Save only the ID, Timestamp, and 3 parameters
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Success! Final parameters saved to: {os.path.abspath(OUTPUT_CSV)}")
    
    # Show preview
    print("\n📊 Preview of Final Log:")
    print(processed_df.head())
else:
    print(f"❌ Error: {INPUT_CSV} not found. Run the fetcher script first.")
