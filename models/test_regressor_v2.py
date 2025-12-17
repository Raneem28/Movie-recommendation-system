import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.rating_regressor import load_regressor

def verify():
    print("🧪 Verifying Regressor Upgrade...")
    
    try:
        # Load
        model, features, actor_scores, global_avg, lookup_df = load_regressor()
        
        if model is None:
            print("❌ Failed to load model (None returned).")
            return
            
        # Check Features
        print(f"✅ Model Loaded. Features: {len(features)}")
        if 'cast_rating_potential' in features:
            print("✅ Feature 'cast_rating_potential' FOUND.")
        else:
            print("❌ Feature 'cast_rating_potential' MISSING.")
            
        # Check Actor Scores
        if actor_scores and isinstance(actor_scores, dict):
            count = len(actor_scores)
            print(f"✅ Actor Scores Dictionary Found. Contains {count} actors.")
            if count > 1000:
                print("   (Target Encoding successfully learned from full dataset)")
        else:
            print("❌ Actor Scores missing or invalid.")
            
        # Check Global Avg
        print(f"✅ Global Rating Average: {global_avg}")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify()
