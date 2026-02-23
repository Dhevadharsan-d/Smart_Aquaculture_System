
import pandas as pd
import numpy as np
import math
import os


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
