
import pandas as pd
import joblib
import os
import sys

# Add project root to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Try to import classifier class definitions in case they are needed for unpickling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def debug_importance():
    print("DEBUG: Starting analysis...")
    
    model_path = os.path.join(os.path.dirname(__file__), 'saved_classifier.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    try:
        data = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model = data['model']
    feature_cols = data['features']
    print(f"Model Type: {type(model).__name__}")
    
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("❌ Model does not have feature_importances_ attribute.")
        return

    # 1. Raw Importance
    raw_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    
    print("\n--- TOP 10 RAW FEATURES ---")
    print(raw_importance.sort_values('importance', ascending=False).head(10))
    
    # 2. Aggregation Logic
    special_cols = ['avg_rating', 'year', 'runtime', 'cast_size']
    agg_data = {'Genre': 0.0, 'Stars': 0.0, 'Rest': 0.0}
    
    # Initialize specials
    for col in special_cols:
        matched = raw_importance[raw_importance['feature'] == col]
        if not matched.empty:
            agg_data[col] = matched['importance'].values[0]

    for idx, row in raw_importance.iterrows():
        f = row['feature']
        imp = row['importance']
        
        if f in special_cols:
            continue
        elif f.startswith('star_'):
            agg_data['Stars'] += imp
        elif str(f) in ['0', '1']: # Check for wierd column names
             pass
        else:
            # Assume Genre (anything else)
            agg_data['Genre'] += imp
            
    print("\n--- AGGREGATED CATEGORIES ---")
    for cat, val in agg_data.items():
        print(f"{cat}: {val:.4f}")

if __name__ == "__main__":
    debug_importance()
