"""
🎬 ENTERTAINMENT ML PROJECT - REGRESSION TASK
Goal: Predict the EXACT average rating of a movie (0.0 - 5.0)
Dataset: MovieLens 32M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data_enrichment.tmdb_client import search_movies_list, fetch_movie_details, extract_metadata
except ImportError:
    print("⚠️ Warning: TMDB Client not found. External search will be disabled.")
    def search_movies_list(*args): return []
    def fetch_movie_details(*args): return {}
    def extract_metadata(*args): return {}

warnings.filterwarnings('ignore')

# Global store for Top 30 Actors (Consistency)
TOP_30_ACTORS = []

def process_enriched_features(movies_df):
    """
    Process runtime, cast size, and top actors from enriched data.
    (Duplicated from classifier for independence, or could be shared utils)
    """
    global TOP_30_ACTORS
    print("✨ Processing Enriched Features for Regression...")

    df = movies_df.copy()
    
    # 1. Runtime
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(95.0)

    # 2. Cast Processing
    df['cast'] = df['cast'].fillna('').astype(str)
    
    def get_cast_list(x):
        if not x or x.lower() == 'nan': return []
        return [c.strip() for c in x.split('|') if c.strip()]

    df['cast_list'] = df['cast'].apply(get_cast_list)
    df['cast_size'] = df['cast_list'].apply(len)

    df['cast_size'] = df['cast_list'].apply(len)

    # 3. Clean Cast
    # 3. Top Actors Analysis (Re-enabled for Compatibility)
    if not TOP_30_ACTORS:
        all_actors = [actor for sublist in df['cast_list'] for actor in sublist]
        from collections import Counter
        actor_counts = Counter(all_actors)
        TOP_30_ACTORS = [actor for actor, count in actor_counts.most_common(1000)] # Extended to 1000 for high accuracy
        print(f"🌟 Top 1000 Actors identified for Regressor: {TOP_30_ACTORS[:5]}...")

    # 4. One-hot encode top actors
    print("   -> Generating Star Actor columns...")
    for actor in TOP_30_ACTORS:
        col_name = f"star_{actor.replace(' ', '_')}"
        df[col_name] = df['cast_list'].apply(lambda x: 1 if actor in x else 0)

    # Drop intermediate cast_list to save memory, keep 'cast' string for display
    # Actually, keep cast_list? No, prepare_features drops it.
    
    return df

# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_data():
    """Load movies and ratings data"""
    print("="*60)
    print("MOVIE RATING PREDICTOR (REGRESSION)")
    print("="*60)
    
    try:
        print("Loading data from 'ml-32m-split' folder...")
        if os.path.exists('ml-32m-split/movies_enriched.csv'):
            print("Loading Enriched Movies...")
            movies = pd.read_csv('ml-32m-split/movies_enriched.csv', dtype={'movieId': int, 'title': str, 'genres': str})
            movies = process_enriched_features(movies)
        else:
            print("⚠️ Enriched movies not found. Using standard.")
            movies = pd.read_csv('ml-32m-split/movies.csv')
        
        # Load Ratings (Limit to prevent memory issues, but enough for good training)
        print("   Loading training ratings (1,000,000 rows)...")
        train_ratings = pd.read_csv('ml-32m-split/train_ratings.csv', 
                                  dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int}, 
                                  nrows=1000000)
        
        print("   Loading test ratings (200,000 rows)...")
        test_ratings = pd.read_csv('ml-32m-split/test_ratings.csv', 
                                 dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int},
                                 nrows=200000)
        
        return movies, train_ratings, test_ratings

    except FileNotFoundError:
        print("❌ Error: 'ml-32m-split' not found. Please run the setup/split script first.")
        return None, None, None

# ============================================
# STEP 2: FEATURE ENGINEERING
# ============================================

def prepare_features(movies_df, ratings_df, dataset_name="Dataset"):
    """
    Create movie-level features from ratings.
    Target Variable: 'avg_rating'
    Predictors: 'year', 'runtime', 'cast_size', 'Genres', 'Top Actors'
    """
    print(f"\nSTEP 2: Engineering Features for {dataset_name}...")
    
    # 1. Aggregate ratings by movie to get the Target (avg_rating) and Popularity (count)
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    
    # 2. Extract Year from Title
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_df['year'] = movies_df['year'].fillna(movies_df['year'].median())
    
    # 3. Merge Metadata with Stats
    df = movies_df.merge(movie_stats, on='movieId', how='inner')
    
    # Fill missing enriched
    if 'runtime' in df.columns: df['runtime'] = df['runtime'].fillna(95.0)
    if 'cast_size' in df.columns: df['cast_size'] = df['cast_size'].fillna(0)

    # 4. Handle Star Columns (One-Hot)
    # They should already be in movies_df if process_enriched_features does its job.
    # We just need to ensure they are preserved and NaNs filled.
    star_cols = [c for c in df.columns if c.startswith('star_')]
    if star_cols:
         df[star_cols] = df[star_cols].fillna(0)
    
    # 5. One-Hot Encode Genres
    genre_dummies = df['genres'].str.get_dummies(sep='|')
    df = pd.concat([df, genre_dummies], axis=1)
    
    # Drop raw list to save memory
    if 'cast_list' in df.columns:
        df = df.drop(columns=['cast_list'])
    
    print(f"✅ Processed {len(df)} movies.")
    # Return df and cols. No actor_scores needed.
    return df, genre_dummies.columns.tolist()

# ============================================
# STEP 3: TRAIN REGRESSOR
# ============================================


def train_regressor(train_df, test_df, genre_cols):
    """Train Gradient Boosting Regressor (Hyper-Tuned)"""
    print("\nSTEP 3: Training Regression Model...")
    
    # Feature Selection
    # REMOVED: 'rating_count' (to avoid cheating/leakage for new movies)
    feature_cols = ['year'] + genre_cols
    
    # Add dynamic enriched features
    extra = ['runtime', 'cast_size']
    for f in extra:
        if f in train_df.columns:
            feature_cols.append(f)

    # Add Top 100 Actor columns (One-Hot)
    star_cols = [c for c in train_df.columns if c.startswith('star_')]
    if star_cols:
        print(f"   Using {len(star_cols)} Top Actor features")
        feature_cols.extend(star_cols)
            
    print(f"   Using {len(feature_cols)} features (Year, Runtime, Cast Size, {len(star_cols)} Stars, Genres)")
    
    X_train = train_df[feature_cols]
    y_train = train_df['avg_rating']
    
    X_test = test_df[feature_cols]
    y_test = test_df['avg_rating']
    
    print(f"📦 Training on {len(X_train)} movies")
    print(f"\n🌲 Training Gradient Boosting Regressor (1000-Actor Enhanced Mode)...")
    
    model = GradientBoostingRegressor(
        n_estimators=2000,       # High capacity for large feature set
        learning_rate=0.01,      # Fine-grained learning
        max_depth=6,             # Balanced depth to prevent overfitting
        subsample=0.8,           # Stochastic
        max_features='sqrt',     # Critical for high-dimensional data
        min_samples_leaf=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    print("✅ Model trained successfully!")
    
    return model, X_test, y_test, y_pred, feature_cols

import joblib
import os

# ============================================
# STEP 4: EVALUATE & SAVE
# ============================================

def save_model(model, feature_cols, lookup_df=None, filename='models/saved_regressor.pkl'):
    """Save regressor model, features, and actor scoring data"""
    os.makedirs('models', exist_ok=True)
    data = {
        'model': model, 
        'features': feature_cols
    }
    
    if lookup_df is not None:
        # Keep essential columns
        keep = ['title', 'year', 'avg_rating', 'rating_count', 'genres']
        valid_cols = [c for c in keep if c in lookup_df.columns]
        data['lookup'] = lookup_df[valid_cols]
        
    joblib.dump(data, filename)
    print(f"💾 Regressor Model saved to {filename}")

def load_regressor(filename='models/saved_regressor.pkl'):
    """Load regressor model"""
    if not os.path.exists(filename):
        return None, None, None, None, None
    data = joblib.load(filename)
    # Return model, features, lookup (None for actor_scores/global_avg place holders to keep signature compatible if needed, but better to clean)
    # Actually, main expects 5 values. Let's return None for the scores.
    return data['model'], data['features'], None, None, data.get('lookup')

def evaluate_model(y_test, y_pred):
    """Calculate RMSE, MAE, R2"""
    print("\n📊 STEP 4: Model Evaluation")
    print("-" * 30)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📉 RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"   (On average, predictions are off by {rmse:.3f} stars)")
    print(f"📉 MAE  (Mean Absolute Error):    {mae:.3f}")
    print(f"📈 R² Score (Variance Explained): {r2:.3f}")
    
    return rmse, r2

# ============================================
# STEP 5: INTERACTIVE PREDICTION
# ============================================

# ============================================
# STEP 5: INTERACTIVE PREDICTION
# ============================================

def predict_new_movie_regression(model, feature_cols, year, genres, runtime=95, cast_list=[], rating_count=0):
    """
    Predict rating using One-Hot Encoded features.
    """
    # 1. Build Feature Dictionary initialized to 0
    features = {col: 0 for col in feature_cols}
    
    # 2. Set Attributes
    features['year'] = year
    if 'rating_count' in feature_cols: features['rating_count'] = rating_count
    if 'runtime' in features: features['runtime'] = runtime if runtime else 95.0
    if 'cast_size' in features: features['cast_size'] = len(cast_list)
        
    # 3. Handle Star Columns (One-Hot)
    # Check which top actors are in the provided cast_list
    # We infer valid star cols from feature_cols
    star_features = [c for c in feature_cols if c.startswith('star_')]
    
    for actor in cast_list:
        col_name = f"star_{actor.replace(' ', '_')}"
        if col_name in features:
            features[col_name] = 1
        
    # 4. Set Genres
    for g in genres:
        if g in feature_cols:
            features[g] = 1
            
    # 5. Predict
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_cols] # Ensure strict order
    
    prediction = model.predict(input_df)[0]
    return prediction

def interactive_prediction_loop(model, feature_cols, lookup_df):
    print("\n" + "="*60)
    print("🎮 INTERACTIVE REGRESSION MODE (Disambiguation Enabled)")
    print("="*60)
    print("Search for a movie logic: Local -> TMDB -> Manual")
    
    while True:
        print("\n------------------------------------------------")
        query = input("👉 Enter Movie Name or 'manual' (or 'exit'): ").strip()
        
        if query.lower() == 'exit':
            break
            
        if query.lower() == 'manual':
            # --- MANUAL ENTRY MODE ---
            try:
                year = float(input("📅 Year (e.g. 1995): "))
                genres_in = input("🎭 Genres (e.g. Action, Comedy): ")
                genres = [g.strip() for g in genres_in.split(',')]
                runtime = float(input("⏱️ Runtime (minutes, default 95): ") or 95)
                cast_in = input("👥 Top 3 Actors (comma-sep): ")
                cast_list = [c.strip() for c in cast_in.split(',')] if cast_in else []
                
                pred = predict_new_movie_regression(model, feature_cols, year, genres, runtime, cast_list)
                print(f"\n🌟 PREDICTED RATING: {pred:.2f} / 5.0")
            except ValueError:
                print("⚠️ Invalid input.")
            continue
            
        # --- SEARCH LOGIC ---
        # 1. Local Search
        matches = lookup_df[lookup_df['title'].str.contains(query, case=False, na=False)]
        
        chosen_movie = None
        is_tmdb = False
        
        # Display Local Options
        if not matches.empty:
            matches = matches.sort_values('rating_count', ascending=False).head(5)
            print(f"\n✅ Found {len(matches)} matches in Local DB:")
            options = []
            print(f"{'#':<4} {'Title':<40} {'Year':<6} {'Rating':<10}")
            print("-" * 70)
            for i, (idx, row) in enumerate(matches.iterrows(), 1):
                title = (row['title'][:38] + '..') if len(row['title']) > 38 else row['title']
                print(f"{i:<4} {title:<40} {int(row['year']):<6} {row['avg_rating']:.1f} ({int(row['rating_count'])})")
                options.append(row)
                
            sel = input(f"\n👉 Select movie number (1-{len(options)}) or Press Enter to search TMDB: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(options):
                chosen_movie = options[int(sel)-1]
                
        # 2. TMDB Search (If not chosen locally)
        if chosen_movie is None:
            print(f"🌐 Searching TMDB Database for '{query}'...")
            tmdb_results = search_movies_list(query, limit=5)
            
            if tmdb_results:
                print(f"\n✅ Found {len(tmdb_results)} matches on TMDB:")
                print(f"{'#':<4} {'Title':<40} {'Year':<6}")
                print("-" * 60)
                for i, m in enumerate(tmdb_results, 1):
                    title = m['title']
                    year = m.get('release_date', 'N/A')[:4]
                    print(f"{i:<4} {title:<40} {year:<6}")
                    
                sel = input(f"\n👉 Select TMDB movie number (1-{len(tmdb_results)}) or Enter to skip: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(tmdb_results):
                    chosen_movie = tmdb_results[int(sel)-1]
                    # Normalize keys for downstream use
                    chosen_movie['release_year'] = int(chosen_movie.get('release_date', '2000')[:4]) if chosen_movie.get('release_date') else 2000
                    is_tmdb = True
            
        if chosen_movie is None:
            print("❌ No valid selection made.")
            continue
            
        # --- PREDICTION ---
        print(f"\n🎬 Selected: {chosen_movie['title']} ({chosen_movie.get('year') or chosen_movie.get('release_year', 'N/A')})")
        
        if is_tmdb:
            # Fetch full details
            details = fetch_movie_details(chosen_movie['id'])
            runtime, cast, genres = extract_metadata(details)
            print(f"   Fetched Metadata: {runtime} mins, {len(cast)} actors")
            
            pred = predict_new_movie_regression(
                model, feature_cols,
                year=chosen_movie['release_year'],
                genres=genres,
                runtime=runtime,
                cast_list=cast
            )
            print(f"   🤖 MODEL PREDICTION: {pred:.2f} / 5.0")
            
        else:
            # Local DB prediction
            # Use columns directly if possible, or recalculate if features missing
            # Simple approach: use the pre-processed columns in the row
            # But the row might miss some One-Hot cols if it was loaded from raw.
            # Best to use predict_new_movie logic to recreate vector from raw features in row?
            # Or just predict(row) if it has all columns.
            # Let's try predict(row) assuming training data has full schema.
            
            # Since we passed Train/Test DF as lookup, it has all columns!
            try:
                # Ensure all columns exist (fill 0)
                row_df = pd.DataFrame([chosen_movie])
                for c in feature_cols:
                    if c not in row_df.columns:
                         row_df[c] = 0
                row_df = row_df[feature_cols]
                
                pred = model.predict(row_df)[0]
                print(f"   🤖 MODEL PREDICTION: {pred:.2f} / 5.0")
                print(f"   (Real Rating was: {chosen_movie['avg_rating']:.2f})")
                print(f"   (Error: {pred - chosen_movie['avg_rating']:.2f})")
            except Exception as e:
                print(f"Prediction Error: {e}")

def match_genres(row, feature_cols):
    """Helper not strictly needed if we have 'genres' string, but good for robustness"""
    return []

if __name__ == "__main__":
    # 0. Check for existing model
    model_loaded = False
    model = None
    feature_cols = None
    
    if os.path.exists('models/saved_regressor.pkl'):
        try:
            choice = input("\n💾 Found existing trained regressor. Load it? (y/n) [y]: ").strip().lower()
            if choice in ['', 'y', 'yes']:
                print("🔄 Loading saved regressor...")
                # Updated unpacking (only 5 items returned, but we expect model, features, None, None, lookup)
                model, feature_cols, _, _, lookup_df = load_regressor()
                if model:
                    print("✅ Model loaded successfully!")
                    model_loaded = True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")

    # 1. Load & Prepare (Only if not loaded)
    if not model_loaded:
        movies, train_ratings, test_ratings = load_data()
        
        # 2. Prepare Match
        # old sig: prepare_features(movies, ratings, name, actor_scores, global)
        # new sig: prepare_features(movies, ratings, name)
        train_df, genre_cols = prepare_features(movies, train_ratings, "TRAIN")
        
        # Prepare Test
        test_df, _ = prepare_features(movies, test_ratings, "TEST")
        
        # Align columns
        for col in genre_cols:
            if col not in test_df.columns:
                test_df[col] = 0
                
        # 3. Train
        # train_regressor returns: model, X_test, y_test, y_pred, feature_cols
        model, X_test, y_test, y_pred, feature_cols = train_regressor(train_df, test_df, genre_cols)
        
        # 4. Evaluate
        evaluate_model(y_test, y_pred)
        
        # Create lookup for interactive loop
        lookup_df = pd.concat([train_df, test_df])
        
        # Save Model (Persist everything)
        save_model(model, feature_cols, lookup_df)
        
    else:
        # If loaded, lookup_df comes from load
        if lookup_df is None:
             print("⚠️ Lookup data missing in cache. Interactive mode limited.")
             lookup_df = pd.DataFrame(columns=['title', 'year', 'avg_rating', 'rating_count', 'genres'])

    # 5. Interactive Mode
    if model:
        # Ensure lookup_df exists
        if 'lookup_df' not in locals() or lookup_df is None:
             lookup_df = pd.DataFrame(columns=['title', 'year', 'avg_rating', 'rating_count'])

        interactive_prediction_loop(model, feature_cols, lookup_df)
