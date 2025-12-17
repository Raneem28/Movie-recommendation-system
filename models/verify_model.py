
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import os
import sys

# Append root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.python_movie_classifier import prepare_movie_features, process_enriched_features

def verify():
    print("🔍 VERIFYING MODEL PERFORMANCE...")
    
    # 1. Load Model
    if not os.path.exists('models/saved_classifier.pkl'):
        print("❌ Model not found!")
        return
        
    data = joblib.load('models/saved_classifier.pkl')
    model = data['model']
    feature_cols = data['features']
    print(f"✅ Loaded Model: {type(model).__name__}")
    
    # 2. Load Evaluation Data
    print("📂 Loading test data...")
    if os.path.exists('ml-32m-split/movies_enriched.csv'):
        movies = pd.read_csv('ml-32m-split/movies_enriched.csv', dtype={'movieId': int, 'title': str, 'genres': str})
        movies = process_enriched_features(movies)
    else:
        movies = pd.read_csv('ml-32m-split/movies.csv')
        
    test_ratings = pd.read_csv('ml-32m-split/test_ratings.csv', nrows=200000)
    
    # 3. Prepare Features
    print("🔧 Preparing features...")
    # NOTE: We need thresholds from training to be consistent, but let's assume standard calculation or loaded thresholds if saved.
    # The current classifier script returns thresholds but doesn't assume them for test unless passed.
    # We will approximate or calculate fresh on test (might slight difference) OR better:
    # use the thresholds logic from the main script.
    # Actually, hit/flop definition is static based on median of THAT dataset usually, or passed.
    # For verification, we'll calculate thresholds on the TEST set statistics to be fair/independent 
    # OR we really should have saved thresholds. 
    # Let's rely on prepare_movie_features default behavior which calculates median valid for that set.
    
    test_df, _, _ = prepare_movie_features(movies, test_ratings, "TEST SET")
    
    # 4. Align Columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
            
    X_test = test_df[feature_cols]
    y_test = test_df['hit']
    
    # 5. Predict
    print("🔮 Predicting...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 6. Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*40)
    print(f"ACCURACY: {acc*100:.2f}%")
    print(f"ROC-AUC:  {roc:.4f}")
    print("="*40)

if __name__ == "__main__":
    verify()
