import time
import os
import subprocess
import sys
from datetime import datetime

# Path Configuration (Absolute paths are safer for automation)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FETCH_SCRIPT = os.path.join(BASE_DIR, "..", "fetching_data_from_open_meteo", "converting_air_to_water_qual_param.py")
PREDICT_SCRIPT = os.path.join(BASE_DIR, "..", "lstm_on_fetched_data", "water_qual_prediction_without_fetch.py")

def run_stage(script_path):
    abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_path)
    try:
        # Run the script and wait for it to finish
        subprocess.run([sys.executable, abs_path], check=True, cwd=script_dir)
        return True
    except Exception as e:
        print(f"❌ Error in {os.path.basename(script_path)}: {e}")
        return False

def start_automated_system():
    print("🌊 STARTING 10-MINUTE AUTOMATED MONITORING...")
    
    try:
        while True:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{now}] 🔄 Starting new cycle...")

            # Stage 1: Fetch & Store (Now overwriting CSV with fresh 10-min updates)
            if run_stage(FETCH_SCRIPT):
                # Stage 2: Predict & SMS (Safety Gate logic runs here)
                run_stage(PREDICT_SCRIPT)
            
            print(f"\n🕒 Cycle complete. Sleeping for 2 minutes...")
            time.sleep(120)  # 60 seconds = 1 minutes
            
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")

if __name__ == "__main__":
    start_automated_system()