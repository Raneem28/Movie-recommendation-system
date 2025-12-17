import sys
import os
import pandas as pd
from sklearn.metrics import r2_score

# Add project root to path
sys.path.append(os.getcwd())

from models.rating_regressor import load_data, prepare_features, load_regressor

def verify_score():
    print("🧪 Verifying Regressor Performance...")
    
    # 1. Load Model
    model, feature_cols, actor_scores, global_avg, _ = load_regressor()
    if model is None:
        print("❌ Model not found!")
        return
        
    print(f"✅ Model loaded. Features: {len(feature_cols)}")
    print(f"✅ Actor Scores: {len(actor_scores)} found.")
    
    # 2. Load Data
    movies, _, test_ratings = load_data()
    
    # 3. Prepare Test Data (using loaded scores)
    print("   Preparing Test Data...")
    test_df, _, _, _ = prepare_features(movies, test_ratings, "TEST", actor_scores, global_avg)
    
    # 4. Align Columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
            
    X_test = test_df[feature_cols]
    y_test = test_df['avg_rating']
    
    # 5. Score
    print("   Predicting...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Final R2 Score: {r2:.4f}")
    
    if r2 > 0.30:
        print("✅ SUCCESS: Target Encoding significantly improved accuracy!")
    else:
        print("⚠️ WARNING: R2 Score is still low.")

if __name__ == "__main__":
    verify_score()
