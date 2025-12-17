import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
sys.path.append(os.getcwd())

from models.rating_regressor import load_data, prepare_features

def test_linear_baseline():
    print("🧪 Testing Linear Baseline (Single Feature)...")
    
    # 1. Load Data
    movies, train_ratings, test_ratings = load_data()
    
    # 2. Prepare Features
    print("   Preparing Features...")
    train_df, _, actor_scores, global_avg = prepare_features(movies, train_ratings, "TRAIN")
    test_df, _, _, _ = prepare_features(movies, test_ratings, "TEST", actor_scores, global_avg)
    
    # 3. Select Single Feature
    feat = 'cast_rating_potential'
    
    X_train = train_df[[feat]].fillna(global_avg)
    y_train = train_df['avg_rating']
    
    X_test = test_df[[feat]].fillna(global_avg)
    y_test = test_df['avg_rating']
    
    # 4. Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    target_var = np.var(y_test)
    
    with open('linear_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Linear Regression Results (Feature: {feat})\n")
        f.write(f"Coefficient: {lr.coef_[0]:.4f}\n")
        f.write(f"Intercept:   {lr.intercept_:.4f}\n")
        f.write(f"MSE:         {mse:.4f}\n")
        f.write(f"Target Var:  {target_var:.4f}\n")
        f.write(f"R2 Score:    {r2:.4f}\n")
        
    print("✅ Results written to linear_results.txt")

if __name__ == "__main__":
    test_linear_baseline()
