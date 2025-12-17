import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.getcwd())
from models.rating_regressor import load_data, prepare_features

def test_multivariate():
    print("🧪 Testing Multivariate Linear Baseline (Cast + Year)...")
    
    movies, train_ratings, test_ratings = load_data()
    train_df, _, actor_scores, global_avg = prepare_features(movies, train_ratings, "TRAIN")
    test_df, _, _, _ = prepare_features(movies, test_ratings, "TEST", actor_scores, global_avg)
    
    features = ['cast_rating_potential', 'year']
    
    # Fill NAs
    for df in [train_df, test_df]:
        df['cast_rating_potential'] = df['cast_rating_potential'].fillna(global_avg)
        df['year'] = df['year'].fillna(2000)
        
    X_train = train_df[features]
    y_train = train_df['avg_rating']
    
    X_test = test_df[features]
    y_test = test_df['avg_rating']
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    with open('linear_v2_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Linear Regression V2 (Features: {features})\n")
        f.write(f"Coefficients: {lr.coef_}\n")
        f.write(f"Intercept:    {lr.intercept_:.4f}\n")
        f.write(f"MSE:          {mse:.4f}\n")
        f.write(f"R2 Score:     {r2:.4f}\n")
        
    print(f"✅ V2 Results written. R2: {r2:.4f}")

if __name__ == "__main__":
    test_multivariate()
