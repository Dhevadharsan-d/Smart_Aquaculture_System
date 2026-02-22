import os
import subprocess
import sys
from datetime import datetime

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# BASE_DIR is: D:\BE. CSE\Mini Project\sm_aq\master_for_lstm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# We use '..' to go up to the 'sm_aq' folder, then down into the specific module folders
FETCH_SCRIPT = os.path.join(BASE_DIR, "..", "fetching_data_from_open_meteo", "converting_air_to_water_qual_param.py")
PREDICT_SCRIPT = os.path.join(BASE_DIR, "..", "lstm_on_fetched_data", "water_qual_prediction_without_fetch.py")

# ==============================================================================
# 2. AUTOMATION ENGINE
# ==============================================================================
def run_stage(script_path):
    """Executes a pipeline stage and ensures paths are absolute for safety."""
    script_name = os.path.basename(script_path)
    # Convert to absolute path to avoid any 'File Not Found' errors
    abs_path = os.path.abspath(script_path)
    
    print(f"\n▶️  Starting Stage: {script_name}")
    
    if not os.path.exists(abs_path):
        print(f"❌ Error: Cannot find script at {abs_path}")
        return False
    
    try:
        # We set the 'cwd' (current working directory) to the script's folder
        # This ensures the scripts can find their local CSV files
        script_dir = os.path.dirname(abs_path)
        result = subprocess.run([sys.executable, abs_path], check=True, cwd=script_dir)
        
        if result.returncode == 0:
            print(f"✅ Stage {script_name} Completed.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed for {script_name}: {e}")
        return False

def run_master_pipeline():
    print("=" * 60)
    print(f"🌊 SMART AQUACULTURE MASTER CONTROLLER")
    print(f"🕒 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # STEP 1: Fetch Weather & Calculate Water Parameters
    # This creates the 'Single Source of Truth' file
    if run_stage(FETCH_SCRIPT):
        
        # STEP 2: Run LSTM Prediction
        # Uses the 6-feature log to predict the next hour
        run_stage(PREDICT_SCRIPT)
        
    else:
        print("\n🛑 Pipeline Halted: Initial data stage failed.")

    print("\n" + "=" * 60)
    print("🏁 Full Pipeline Cycle Finished.")
    print("=" * 60)

# ==============================================================================
# 3. RUN
# ==============================================================================
if __name__ == "__main__":
    run_master_pipeline()