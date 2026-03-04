
    
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from twilio.rest import Client

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
MODEL_PATH = r'../models/lstm_model/water_qual_param.keras' 
DATA_PATH = r'../fetching_data_from_open_meteo/final_water_quality_log.csv' 
SEQ_HOURS = 24 
FEATURES = ["air_temp", "humidity", "rain", "water_temp", "do", "ph"]

# Twilio Credentials
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
TWILIO_PHONE = '+19134210609'
FARMER_PHONE = '+917305392527'

def check_safety_thresholds(prediction):
    """
    Detailed fluctuation detection for specific parameters.
    Reference: [1] Rupali P., et al. (2025) - IoT-enabled real-time monitoring.
    """
    temp, do, ph = prediction[0], prediction[1], prediction[2]
    alerts = []
    
    # 1. Dissolved Oxygen (DO) 
    if do < 4.0:
        alerts.append(f"CRITICAL: Very Low Oxygen ({do:.2f} mg/L). Start Aerators!")
    elif do < 5.5:
        alerts.append(f"WARNING: Low Oxygen ({do:.2f} mg/L).")
    elif do > 12.0:
        alerts.append(f"WARNING: High Oxygen ({do:.2f} mg/L).")

    # 2. pH Level
    if ph < 6.5:
        alerts.append(f"CRITICAL: Acidic Water ({ph:.2f}). Add lime.")
    elif ph > 9.0:
        alerts.append(f"CRITICAL: Highly Alkaline ({ph:.2f}). Ammonia Risk!")
    elif ph > 8.5:
        alerts.append(f"WARNING: High pH ({ph:.2f}).")

    # 3. Water Temperature
    if temp < 50.0:
        alerts.append(f"ALERT: Low Temp ({temp:.2f}°C). Metabolism slow.")
    elif temp > 32.0:
        alerts.append(f"ALERT: High Temp ({temp:.2f}°C). Stress risk.")
    
    return alerts

def send_sms_alert(message):
    if not TWILIO_SID or 'YOUR' in TWILIO_SID:
        print("Skipping SMS: Credentials missing.")
        return

    client = Client(TWILIO_SID, TWILIO_TOKEN)
    try:
        client.messages.create(
            from_=TWILIO_PHONE,
            to=FARMER_PHONE,
            body=f"Pond Alert: {message}"
        )
        print(" SMS Alert sent to farmer!")
    except Exception as e:
        print(f" SMS Failed: {e}")

def run_prediction_on_log():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)):
        print(" Error: Missing files.")
        return

    model = load_model(MODEL_PATH)
    full_df = pd.read_csv(DATA_PATH)
    
    if len(full_df) < SEQ_HOURS:
        print(f" Error: Need {SEQ_HOURS} rows.")
        return

    df = full_df.tail(SEQ_HOURS).copy()
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES]) 
    scaled_input = scaler.transform(df[FEATURES].values)
    input_reshaped = np.expand_dims(scaled_input, axis=0)

    prediction_scaled = model.predict(input_reshaped, verbose=0)
    dummy_out = np.zeros((1, len(FEATURES)))
    dummy_out[0, 3:] = prediction_scaled[0]
    final_values = scaler.inverse_transform(dummy_out)[0, 3:]

    # --- Detailed Console Fluctuations ---
    p_temp, p_do, p_ph = final_values[0], final_values[1], final_values[2]
    print(f"\n [FORECAST] Temp: {p_temp:.2f}°C | DO: {p_do:.2f} mg/L | pH: {p_ph:.2f}")

    danger_alerts = check_safety_thresholds(final_values)
    
    if danger_alerts:
        print(f" FLUCTUATION DETECTED in {len(danger_alerts)} parameters:")
        for alert in danger_alerts:
            print(f"    {alert}") # Tells you exactly which parameter and what the issue is
        
        send_sms_alert(" | ".join(danger_alerts))
    else:
        print(" Status: Normal")

if __name__ == "__main__":
    run_prediction_on_log()