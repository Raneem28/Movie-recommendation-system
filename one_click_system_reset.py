import os
import subprocess
import time
import pandas as pd

def step(msg):
    print(f"\n👉 {msg}")
    time.sleep(1)

def run_fix():
    step("Running Data Repair (fix_enriched_data.py)...")
    try:
        subprocess.run(["python", "fix_enriched_data.py"], check=True)
        
        # Verify
        df = pd.read_csv('ml-32m-split/movies_enriched.csv')
        if 'year' in df.columns and 'avg_rating' in df.columns:
            print("✅ Data Verification Passed: 'year' and 'avg_rating' columns exist.")
        else:
            print(f"❌ Data Verification Failed. Columns: {list(df.columns)}")
            exit(1)
            
    except subprocess.CalledProcessError:
        print("❌ Data repair script failed.")
        exit(1)

def run_training():
    step("Retraining Classifier with REPAIRED data...")
    # We use subprocess to run it, but we need to ensure it uses the new file
    # and doesn't wait for input.
    # The current script has interactive loops.
    # For now, we rely on the file being fixed.
    # The user logic in main.py loads the file.
    
    print("ℹ️  Skipping auto-retrain (interactive script).")
    print("ℹ️  The important part is that 'movies_enriched.csv' is now FIXED.")
    print("ℹ️  The Backend reads this file directly.")

def main():
    print("="*60)
    print("🔄 ONE-CLICK SYSTEM RESET")
    print("="*60)
    
    run_fix()
    
    step("Ready to launch!")
    print("\n✅ SYSTEM REPAIRED.")
    print("1. Restart your Backend (Terminal 1) -> python -m uvicorn api.main:app ...")
    print("2. The backend will now see the 'year' and 'avg_rating' columns.")
    print("3. Enjoy the fix!")

if __name__ == "__main__":
    main()
