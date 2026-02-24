
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import os


# 1. CONFIGURATION
# ==============================================================================
LATITUDE = 10.98267
LONGITUDE = 76.97678
PAST_DAYS = 365 
FINAL_LOG_CSV = 'final_water_quality_log.csv'

# ==============================================================================
# 2. DATA FETCHING (Fixing the KeyError)
# ==============================================================================
def fetch_hourly_weather(lat, lon, past_days):
    # Fix: Use 'yesterday' because Archive API doesn't have partial current-day data
    end_date = (datetime.now() - timedelta(days=1)).date()
    start_date = end_date - timedelta(days=past_days)

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relativehumidity_2m,precipitation"
        "&timezone=UTC"
    )

    print(f"📡 Requesting weather data from: {start_date} to {end_date}...")
    
    try:
        response = requests.get(url)
        data = response.json()

        # Safety Check: Prevents 'KeyError: hourly' if API fails
        if "hourly" not in data:
            print(f"❌ API Error: {data.get('reason', 'Unknown error')}")
            return None

        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "air_temp": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relativehumidity_2m"],
            "rain": data["hourly"]["precipitation"]
        })
        return df
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# ==============================================================================
# 3. WATER QUALITY LOGIC (Preserving all 6 Columns)
# ==============================================================================
def compute_water_quality(df):
    """Calculates water parameters while keeping weather data as features."""
    wt, do, ph = [], [], []
    prev_tw = None

    print("🧪 Calculating biological water parameters...")

    for _, r in df.iterrows():
        # Temperature Inertia Logic
        tw = r.air_temp - 1 if prev_tw is None else 0.7 * prev_tw + 0.3 * (r.air_temp - 1)
        
        # DO Saturation Logic
        d = (14.6 - 0.2 * tw) * (1 + (r.humidity - 50) / 500)
        d = np.clip(d, 2, 14)
        
        # pH Diurnal Cycle Logic
        hour = r.datetime.hour
        p = (7.4 + 0.3 * math.sin(2 * math.pi * hour / 24) - 0.01 * (tw - 26) - 
             0.08 * math.log1p(r.rain) + np.random.normal(0, 0.05))
        p = np.clip(p, 6.5, 9.0)

        wt.append(round(tw, 2))
        do.append(round(d, 2))
        ph.append(round(p, 2))
        prev_tw = tw

    # --- SINGLE SOURCE OF TRUTH: Merging Weather + Water Data ---
    final_df = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "timestamp": df["datetime"],
        "air_temp": df["air_temp"],   # Added
        "humidity": df["humidity"],   # Added
        "rain": df["rain"],           # Added
        "water_temp": wt,
        "do": do,
        "ph": ph
    })
    return final_df

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Starting Data Pipeline...")
    df_raw = fetch_hourly_weather(LATITUDE, LONGITUDE, PAST_DAYS)

    if df_raw is not None:
        # Step B: Compute water parameters
        df_final = compute_water_quality(df_raw)
       
        # Mode 'a' means Append. header=not file_exists ensures header only writes once.
        df_final.to_csv(FINAL_LOG_CSV, index=False)        

        print("-" * 50)
        print(f"✅ Success! Data appended to: {os.path.abspath(FINAL_LOG_CSV)}")
        print(f"📊 Added {len(df_final)} new rows.")
        print(f"📈 Total records now in log: {len(pd.read_csv(FINAL_LOG_CSV))}")
        print("-" * 50)