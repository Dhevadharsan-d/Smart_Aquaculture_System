#  Smart Aquaculture Monitoring & Prediction System

This project implements an IoT-enabled real-time water quality monitoring method for aquaculture, focusing on predictive analytics and automated safety alerts.

##  Implementation Progress 

### **1. Automated Data Pipeline**
* **Virtual Sensor Integration:** Developed a pipeline using the Open-Meteo API to fetch real-time environmental data (Air Temp, Humidity, Precipitation).
* **Single Source of Truth:** Implemented logic to calculate biological water parameters (Water Temp, Dissolved Oxygen, pH) and save them into a unified `final_water_quality_log.csv` for model input.

### **2. Predictive Analytics (LSTM)**
* **High-Accuracy Forecasting:** Implemented a Long Short-Term Memory (LSTM) model with a **98.81% accuracy** rate for water quality prediction.
* **Time-Series Analysis:** The system processes a 24-hour lookback window to forecast conditions for the next hour, accounting for diurnal cycles.

### **3. Integrated Safety Gate & Alerts**
* **Threshold Monitoring:** Built a "Safety Gate" logic that monitors real-time and predicted data.
* **Automated SMS:** Integrated Twilio API to send immediate alerts to farmers when parameters hit critical levels (e.g., **DO < 5.0 mg/L**).

### **4. Master Controller**
* **Pipeline Orchestration:** Developed `main_lstm_master.py` to automate the entire workflow from data fetching to prediction and alert triggering in one execution cycle.
