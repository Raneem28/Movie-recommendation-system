import joblib
import os

def inspect():
    path = "models/saved_regressor.pkl"
    if not os.path.exists(path):
        print("Model not found")
        return

    data = joblib.load(path)
    features = data['features']
    
    print(f"Total Features: {len(features)}")
    
    print("\n--- Genre Columns ---")
    genres = [f for f in features if 'cn' in f.lower() or 'ma' in f.lower() or 'Act' in f or 'Gen' in f]
    # Just print first 20 that look like genres or start with 'genre'
    for f in features:
        if 'genre' in f.lower() or f in ['Action', 'Drama', 'Comedy']:
            print(f"  '{f}'")
            
    print("\n--- Salman Khan Columns ---")
    salman = [f for f in features if 'Salman' in f]
    for f in salman:
        print(f"  '{f}'")
        
    print("\n--- Sample Features (First 10) ---")
    print(features[:10])

if __name__ == "__main__":
    inspect()
