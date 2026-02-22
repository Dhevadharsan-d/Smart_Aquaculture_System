# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import os
# from twilio.rest import Client

# # Hide TensorFlow startup logs for a cleaner terminal
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# # ==============================================================================
# # 1. CONFIGURATION
# # ==============================================================================
# MODEL_PATH = r'../models/lstm_model/water_qual_param.keras' 
# DATA_PATH = r'../fetching_data_from_open_meteo/final_water_quality_log.csv' 

# SEQ_HOURS = 24
# FEATURES = ["air_temp", "humidity", "rain", "water_temp", "do", "ph"]

# # Twilio Credentials (Fill these in for SMS to work)
# TWILIO_SID = 'YOUR_ACCOUNT_SID'
# TWILIO_TOKEN = 'YOUR_AUTH_TOKEN'
# TWILIO_PHONE = '+1234567890'
# FARMER_PHONE = '+917305392527'

# # ==============================================================================
# # 2. CORE LOGIC
# # ==============================================================================

# def check_safety_thresholds(prediction):
#     temp, do, ph = prediction[0], prediction[1], prediction[2]
#     alerts = []

#     if do < 5.0:
#         alerts.append(f"🚨 CRITICAL: Low Oxygen ({do:.2f} mg/L). Turn on aerators!")
#     if ph > 8.5:
#         alerts.append(f"⚠️ WARNING: High pH ({ph:.2f}). Check ammonia levels.")
#     elif ph < 6.5:
#         alerts.append(f"⚠️ WARNING: Low pH ({ph:.2f}).")
#     if temp > 32.0:
#         alerts.append(f"🌡️ ALERT: High Temp ({temp:.2f}°C).")

#     return alerts

# def send_sms_alert(message):
#     # Only try to send if credentials are provided
#     if 'YOUR' in TWILIO_SID:
#         print("💡 Skipping SMS: Twilio credentials not yet configured.")
#         return

#     client = Client(TWILIO_SID, TWILIO_TOKEN)
#     try:
#         client.messages.create(
#             from_=TWILIO_PHONE,
#             to=FARMER_PHONE,
#             body=f"🌊 Pond Alert: {message}"
#         )
#         print("📲 SMS Alert sent to farmer!")
#     except Exception as e:
#         print(f"❌ Failed to send SMS: {e}")

# def run_prediction_on_log():
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ Error: Model not found at {os.path.abspath(MODEL_PATH)}")
#         return
#     if not os.path.exists(DATA_PATH):
#         print(f"❌ Error: CSV not found at {os.path.abspath(DATA_PATH)}")
#         return

#     # A. Load Model and Data
#     print("🧠 Loading LSTM Model...")
#     model = load_model(MODEL_PATH)
#     df = pd.read_csv(DATA_PATH)

#     # B. Validate Structure and Sequence
#     if len(df) < SEQ_HOURS:
#         print(f"❌ Error: Need at least 24 hours of data.")
#         return

#     # C. Scaling and Prediction
#     scaler = MinMaxScaler()
#     scaler.fit(df[FEATURES]) 
#     recent_data = df[FEATURES].tail(SEQ_HOURS).values
#     scaled_input = scaler.transform(recent_data)
#     input_reshaped = np.expand_dims(scaled_input, axis=0)

#     print("🚀 Running prediction for the next hour...")
#     prediction_scaled = model.predict(input_reshaped, verbose=0)

#     # D. Inverse Transform
#     dummy_out = np.zeros((1, len(FEATURES)))
#     dummy_out[0, 3:] = prediction_scaled[0]
#     final_values = scaler.inverse_transform(dummy_out)[0, 3:]

#     # E. Output Results
#     print("-" * 50)
#     print(f"📊 WATER QUALITY FORECAST (NEXT HOUR):")
#     print("-" * 50)
#     print(f"🌡️ Water Temp: {final_values[0]:.2f} °C")
#     print(f"💧 Diss. Oxygen: {final_values[1]:.2f} mg/L")
#     print(f"🧪 pH Level:     {final_values[2]:.2f}")
#     print("-" * 50)

#     # ==============================================================================
#     # 3. INTEGRATED SAFETY GATE (THE CRITICAL FIX)
#     # ==============================================================================
#     danger_alerts = check_safety_thresholds(final_values)
    
#     if danger_alerts:
#         print(f"🚨 FOUND {len(danger_alerts)} SAFETY ISSUES:")
#         for alert in danger_alerts:
#             print(f" -> {alert}")
#             send_sms_alert(alert) # Automatically sends SMS for each alert
#     else:
#         print("✅ Forecast looks safe. No alerts necessary.")

# if __name__ == "__main__":
#     run_prediction_on_log()


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from twilio.rest import Client
from datetime import datetime, timedelta

# Hide TensorFlow startup logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_PATH = r'../models/lstm_model/water_qual_param.keras' 
DATA_PATH = r'../fetching_data_from_open_meteo/final_water_quality_log.csv' 

SEQ_HOURS = 24  # Your LSTM lookback window
FEATURES = ["air_temp", "humidity", "rain", "water_temp", "do", "ph"]

# Twilio Credentials
TWILIO_SID = 'YOUR_ACCOUNT_SID'
TWILIO_TOKEN = 'YOUR_AUTH_TOKEN'
TWILIO_PHONE = '+1234567890'
FARMER_PHONE = '+917305392527'

# ==============================================================================
# 2. CORE LOGIC
# ==============================================================================

def check_safety_thresholds(prediction):
    temp, do, ph = prediction[0], prediction[1], prediction[2]
    alerts = []

    if do < 5.0:
        alerts.append(f"🚨 CRITICAL: Low Oxygen ({do:.2f} mg/L). Turn on aerators!")
    if ph > 8.5:
        alerts.append(f"⚠️ WARNING: High pH ({ph:.2f}). Check ammonia levels.")
    elif ph < 6.5:
        alerts.append(f"⚠️ WARNING: Low pH ({ph:.2f}).")
    if temp > 32.0:
        alerts.append(f"🌡️ ALERT: High Temp ({temp:.2f}°C).")

    return alerts

def send_sms_alert(message):
    if 'YOUR' in TWILIO_SID:
        print("💡 Skipping SMS: Twilio credentials not yet configured.")
        return

    client = Client(TWILIO_SID, TWILIO_TOKEN)
    try:
        client.messages.create(
            from_=TWILIO_PHONE,
            to=FARMER_PHONE,
            body=f"🌊 Pond Alert: {message}"
        )
        print("📲 SMS Alert sent to farmer!")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def run_prediction_on_log():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {os.path.abspath(MODEL_PATH)}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: CSV not found at {os.path.abspath(DATA_PATH)}")
        return

    # A. Load Model and latest SEQ_HOURS data
    print("🧠 Loading LSTM Model...")
    model = load_model(MODEL_PATH)
    
    # MODIFIED: Load only the last SEQ_HOURS for prediction efficiency
    full_df = pd.read_csv(DATA_PATH)
    
    if len(full_df) < SEQ_HOURS:
        print(f"❌ Error: Need at least {SEQ_HOURS} hours of data. Current rows: {len(full_df)}")
        return

    df = full_df.tail(SEQ_HOURS).copy()

    # B. Calculate Next Hour Timestamp
    # Convert string timestamp to datetime, add 1 hour, and format
    last_ts = pd.to_datetime(df['timestamp'].iloc[-1])
    next_hour_ts = last_ts + timedelta(hours=1)
    # Adding +5:30 to next_hour_ts if your API data is in UTC but you want IST display
    next_hour_ist = next_hour_ts + timedelta(hours=5, minutes=30)
    display_time = next_hour_ist.strftime('%Y-%m-%d %I:00 %p')

    # C. Scaling and Prediction
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES]) 
    recent_data = df[FEATURES].values
    scaled_input = scaler.transform(recent_data)
    input_reshaped = np.expand_dims(scaled_input, axis=0)

    print(f"🚀 Running prediction for: {display_time}")
    prediction_scaled = model.predict(input_reshaped, verbose=0)

    # D. Inverse Transform
    dummy_out = np.zeros((1, len(FEATURES)))
    dummy_out[0, 3:] = prediction_scaled[0]
    final_values = scaler.inverse_transform(dummy_out)[0, 3:]

    # E. Output Results
    print("-" * 50)
    print(f"📊 WATER QUALITY FORECAST FOR: {display_time}")
    print("-" * 50)
    print(f"🌡️ Water Temp: {final_values[0]:.2f} °C")
    print(f"💧 Diss. Oxygen: {final_values[1]:.2f} mg/L")
    print(f"🧪 pH Level:     {final_values[2]:.2f}")
    print("-" * 50)

    # ==============================================================================
    # 3. INTEGRATED SAFETY GATE
    # ==============================================================================
    danger_alerts = check_safety_thresholds(final_values)
    
    if danger_alerts:
        print(f"🚨 FOUND {len(danger_alerts)} SAFETY ISSUES:")
        for alert in danger_alerts:
            print(f" -> {alert}")
            send_sms_alert(alert)
    else:
        print("✅ Forecast looks safe. No alerts necessary.")

if __name__ == "__main__":
    run_prediction_on_log()