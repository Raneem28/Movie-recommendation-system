import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add path to import api modules
sys.path.append(os.getcwd())

from api.features import prepare_regressor_features
from models.rating_regressor import predict_new_movie_regression

def verify():
    print("🔍 DIAGNOSTIC MODE: Verifying Model Parity")
    
    # 1. Load Model
    path = "models/saved_regressor.pkl"
    if not os.path.exists(path):
        print("❌ Model not found at", path)
        return

    data = joblib.load(path)
    model = data['model']
    feature_cols = data['features']
    print(f"✅ Model loaded. Expecting {len(feature_cols)} features.")
    
    # 2. Define Input (Sikandar)
    movie_input = {
        "title": "Sikandar",
        "year": 2025,
        "runtime": 133,
        "genres": ["Action", "Drama"], # Assuming these based on typical Salman Khan movies? Or just Action.
        # User terminal showed: Cast: Salman Khan, Rashmika Mandanna, Sathyaraj, Sharman Joshi, Prateik Smita Patil
        "cast": ["Salman Khan", "Rashmika Mandanna", "Sathyaraj", "Sharman Joshi", "Prateik Smita Patil"]
    }
    print(f"\n🎥 Input Movie: {movie_input}")

    # 3. Method A: API Logic
    print("\n--- Method A: API Logic (prepare_regressor_features) ---")
    
    # API Main passes: top_actors=models.get('top_actors', []) -> likely []
    # expected_cols = feature_cols
    
    api_feats = prepare_regressor_features(
        year=movie_input['year'],
        runtime=movie_input['runtime'],
        genres=movie_input['genres'],
        cast=movie_input['cast'],
        top_actors_list=[], # Simulating missing top_actors in pkl
        expected_cols=feature_cols
    )
    
    full_vector_a = api_feats[0]
    pred_a = model.predict(api_feats)[0]
    print(f"\n🔮 API Prediction (Method A): {pred_a}")
    
    # 4. Method B: Terminal Logic Simulation
    print("\n--- Method B: Terminal Logic Simulation ---")
    features_b = {col: 0 for col in feature_cols}
    features_b['year'] = movie_input['year']
    features_b['runtime'] = movie_input['runtime']
    
    # Stars
    for actor in movie_input['cast']:
        col_name = f"star_{actor.replace(' ', '_')}"
        if col_name in features_b:
            features_b[col_name] = 1
            
    # Genres (Simulation of robust matching)
    for g in movie_input['genres']:
        # Try both formats
        if f"genre_{g}" in features_b:
            features_b[f"genre_{g}"] = 1
        elif g in features_b:
            features_b[g] = 1
            
    df_b = pd.DataFrame([features_b])
    df_b = df_b[feature_cols] # Ensure order
    full_vector_b = df_b.values[0]
    pred_b = model.predict(df_b)[0]
    print(f"🔮 Terminal Prediction (Method B): {pred_b}")

    # 5. Compare
    print("\n--- Comparison ---")
    if np.array_equal(full_vector_a, full_vector_b):
        print("✅ Feature Vectors are match.")
    else:
        print("❌ Feature Vectors are DIFFERENT!")
        diff_count = 0
        for i in range(len(feature_cols)):
            v_a = full_vector_a[i]
            v_b = full_vector_b[i]
            if v_a != v_b:
                print(f"   Mismatch at [{feature_cols[i]}]: API={v_a} vs Terminal={v_b}")
                diff_count += 1
                if diff_count > 10: break


if __name__ == "__main__":
    verify()
