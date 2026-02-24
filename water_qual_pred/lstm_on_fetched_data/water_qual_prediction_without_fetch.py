
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from twilio.rest import Client
from datetime import datetime, timedelta

# Hide TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
MODEL_PATH = r'../models/lstm_model/water_qual_param.keras' 
DATA_PATH = r'../fetching_data_from_open_meteo/final_water_quality_log.csv' 
ALERT_LOG = 'last_alert_timestamp.txt'  # File to track last alert time

SEQ_HOURS = 24 
FEATURES = ["air_temp", "humidity", "rain", "water_temp", "do", "ph"]

# Twilio Credentials (Using environment variables is the "Kanna" way)
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
TWILIO_PHONE = '+19134210609'
FARMER_PHONE = '+917305392527'

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def is_cooldown_active(minutes=30):
    """Prevents sending too many SMS alerts in a short window."""
    if not os.path.exists(ALERT_LOG):
        return False
    try:
        with open(ALERT_LOG, 'r') as f:
            last_time_str = f.read().strip()
            last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')
            return datetime.now() < last_time + timedelta(minutes=minutes)
    except:
        return False

def update_alert_timer():
    """Saves the current timestamp to the alert log."""
    with open(ALERT_LOG, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

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
        print("💡 Skipping SMS: Twilio credentials not configured.")
        return
    
    if is_cooldown_active(minutes=30):
        print("⏳ Alert skipped: Cooldown active (30 min limit).")
        return

    client = Client(TWILIO_SID, TWILIO_TOKEN)
    try:
        client.messages.create(
            from_=TWILIO_PHONE,
            to=FARMER_PHONE,
            body=f"🌊 Pond Alert: {message}"
        )
        print("📲 SMS Alert sent to farmer!")
        update_alert_timer()
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

# ==============================================================================
# 3. MAIN PREDICTION BLOCK
# ==============================================================================

def run_prediction_on_log():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)):
        print("❌ Error: Model or CSV file missing.")
        return

    print("🧠 Loading LSTM Model...")
    model = load_model(MODEL_PATH)
    full_df = pd.read_csv(DATA_PATH)
    
    if len(full_df) < SEQ_HOURS:
        print(f"❌ Error: Need {SEQ_HOURS} rows of data.")
        return

    df = full_df.tail(SEQ_HOURS).copy()

    # Scaling and Prediction
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES]) 
    scaled_input = scaler.transform(df[FEATURES].values)
    input_reshaped = np.expand_dims(scaled_input, axis=0)

    prediction_scaled = model.predict(input_reshaped, verbose=0)

    # Inverse Transform
    dummy_out = np.zeros((1, len(FEATURES)))
    dummy_out[0, 3:] = prediction_scaled[0]
    final_values = scaler.inverse_transform(dummy_out)[0, 3:]

    # Integrated Safety Gate
    danger_alerts = check_safety_thresholds(final_values)
    
    if danger_alerts:
        print(f"🚨 FOUND {len(danger_alerts)} SAFETY ISSUES.")
        # Combine all alerts into one SMS to save credits
        combined_msg = " | ".join(danger_alerts)
        send_sms_alert(combined_msg)
    else:
        print("✅ Forecast looks safe. No alerts necessary.")

if __name__ == "__main__":
    run_prediction_on_log()