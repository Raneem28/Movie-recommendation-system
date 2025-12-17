"""
🎬 ENTERTAINMENT ML PROJECT - CLASSIFICATION TASK
Goal: Predict if a movie will be a HIT or FLOP
Dataset: MovieLens 32M (with proper train/test split)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import sys

# Add project root to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_enrichment.tmdb_client import search_movie, search_movies_list, fetch_movie_details, extract_metadata
from models.rating_regressor import load_regressor, predict_new_movie_regression

warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD SPLIT DATASETS
# ============================================

# ============================================
# STEP 1: LOAD SPLIT DATASETS
# ============================================

# (Moved to __main__ to prevent execution on import)

# Global store for Top 30 Actors (to ensure consistency between training/prediction)
TOP_30_ACTORS = []

def process_enriched_features(movies_df):
    """
    Process runtime, cast size, and top actors from enriched data.
    """
    global TOP_30_ACTORS
    
    print("✨ Processing Enriched Features (Runtime, Cast...)...")

    # Work on a copy to avoid SettingWithCopy warnings
    df = movies_df.copy()

    # 1. Runtime
    # Convert to numeric and fill missing with median (approx 90-100 min)
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(95.0)

    # 2. Cast Processing
    # Ensure cast is string
    df['cast'] = df['cast'].fillna('').astype(str)
    
    # helper to split
    def get_cast_list(x):
        if not x or x.lower() == 'nan': return []
        return [c.strip() for c in x.split('|') if c.strip()]

    # Create temporary list column
    df['cast_list'] = df['cast'].apply(get_cast_list)
    
    # Feature: Cast Size
    df['cast_size'] = df['cast_list'].apply(len)

    # 3. Top Actors
    # If not yet defined (first run), calculate them from this dataset
    if not TOP_30_ACTORS:
        all_actors = [actor for sublist in df['cast_list'] for actor in sublist]
        from collections import Counter
        actor_counts = Counter(all_actors)
        # Take top 500 most frequent actors (Massive expansion for Accuracy)
        TOP_30_ACTORS = [actor for actor, count in actor_counts.most_common(500)]
        print(f"🌟 Top 500 Actors identified: {TOP_30_ACTORS[:5]}... and {len(TOP_30_ACTORS)-5} others")

    # 4. One-hot encode top actors
    print("   -> Generating Star Actor columns...")
    for actor in TOP_30_ACTORS:
        # Create column star_ActorName (sanitize name for column)
        col_name = f"star_{actor.replace(' ', '_')}"
        # Set to 1 if actor is in the movie's cast list
        df[col_name] = df['cast_list'].apply(lambda x: 1 if actor in x else 0)

    # Drop intermediate cast_list to save memory, keep 'cast' string for display
    df = df.drop(columns=['cast_list'])
    
    return df


# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================

def prepare_movie_features(movies_df, ratings_df, dataset_name="", thresholds=None):
    """
    Engineer features for classification
    
    Features we'll create:
    - avg_rating: Average rating of the movie
    - rating_count: Number of ratings (popularity)
    - genre_features: One-hot encoding of genres
    - hit_label: 1 if HIT, 0 if FLOP
    """
    
    print(f"\n🔧 STEP 2: Engineering Features for {dataset_name}...")
    
    # Aggregate ratings per movie
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count', 'std']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'rating_std']
    
    # 3. Extract Year from Title
    # Regex to find (1995) pattern
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies_df['year'] = movies_df['year'].fillna(movies_df['year'].median()) # Handle missing years
    
    # Merge with movie metadata
    df = movies_df.merge(movie_stats, on='movieId', how='inner')
    
    # Fill missing std with 0
    df['rating_std'] = df['rating_std'].fillna(0)
    
    # Fill missing enriched features if any (for safety)
    if 'runtime' in df.columns:
        df['runtime'] = df['runtime'].fillna(95.0)
    if 'cast_size' in df.columns:
        df['cast_size'] = df['cast_size'].fillna(0)

    # Verify if we have star columns
    star_cols = [c for c in df.columns if c.startswith('star_')]
    if star_cols:
        print(f"   ℹ️ Included {len(star_cols)} Top Actor features")
    
    # Create HIT/FLOP label
    # Definition: HIT = avg_rating >= 4.0 AND rating_count >= 50
    # (You can adjust these thresholds based on your data distribution)
    
    if thresholds:
        rating_p75 = thresholds['rating_p75']
        count_p75 = thresholds['count_p75']
        print(f"   ℹ️ Using provided thresholds: Rating >= {rating_p75:.2f}, Count >= {count_p75:.0f}")
    else:
        # Relaxed thresholds to Median (50th percentile) to make the model more balanced/optimistic
        rating_p75 = movie_stats['avg_rating'].quantile(0.50)
        count_p75 = movie_stats['rating_count'].quantile(0.50)
        print(f"   ℹ️ Calculated new thresholds (Median): Rating >= {rating_p75:.2f}, Count >= {count_p75:.0f}")

    thresholds = {'rating_p75': rating_p75, 'count_p75': count_p75}

    df['hit'] = ((df['avg_rating'] >= rating_p75) &
             (df['rating_count'] >= count_p75)).astype(int)

    
    
    # One-hot encode genres
    # Genres are pipe-separated like "Action|Adventure|Sci-Fi"
    genre_dummies = df['genres'].str.get_dummies(sep='|')
    df = pd.concat([df, genre_dummies], axis=1)
    
    print(f"✅ Created {len(genre_dummies.columns)} genre features")
    print(f"✅ Total movies: {len(df):,}")
    print(f"✅ HIT movies: {df['hit'].sum():,} ({df['hit'].mean()*100:.1f}%)")
    print(f"✅ FLOP movies: {(1-df['hit']).sum():,} ({(1-df['hit']).mean()*100:.1f}%)")
    
    return df, genre_dummies.columns.tolist(), thresholds


# ============================================
# STEP 3: TRAIN CLASSIFICATION MODEL
# ============================================

def train_hit_classifier(train_df, test_df, genre_columns):
    """
    Train Random Forest Classifier using pre-split train/test data
    """
    
    print("\n🤖 STEP 3: Training Classification Model...")
    
    
    # Select features
    # RE-ADDED: 'avg_rating' (User Request)
    # EXCLUDED: 'rating_count' (To avoid popularity leakage)
    # Predictors: Avg Rating + Year + Genres
    # Predictors: Avg Rating + Year + Genres + NEW Enriched Features
    feature_cols = ['avg_rating', 'year'] + genre_columns
    
    # Add simple enriched features if they exist
    extra_features = ['runtime', 'cast_size']
    # Add star features dynamically
    star_cols = [c for c in train_df.columns if c.startswith('star_')]
    
    # Only add if they are present
    for f in extra_features + star_cols:
        if f in train_df.columns:
            feature_cols.append(f)
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['hit']
    
    # Prepare testing data
    X_test = test_df[feature_cols]
    y_test = test_df['hit']
    
    print(f"📦 Training set: {len(X_train):,} movies")
    print(f"📦 Test set: {len(X_test):,} movies")
    
    # Train Random Forest
    print("\n🌲 Training Random Forest Classifier...")
    print("\n🌲 Training Stochastic Gradient Boosting Classifier (Accurate & Balanced)...")
    model = GradientBoostingClassifier(
        n_estimators=1000,         # Very High Capacity for >90% Accuracy
        learning_rate=0.05,        # Fine-grained learning
        max_depth=9,               # Deep trees
        subsample=0.8,             # Stochastic (prevents overfitting)
        max_features='sqrt',       # CRITICAL: Forces model to use attributes other than avg_rating (Fixes Plot)
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("✅ Model trained successfully!\n")
    
    return model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_cols


import joblib

# ============================================
# STEP 4: EVALUATE MODEL
# ============================================

def save_model(model, feature_cols, lookup_df=None, filename='models/saved_classifier.pkl'):
    """Save model, feature columns, and lookup data"""
    os.makedirs('models', exist_ok=True)
    data = {'model': model, 'features': feature_cols}
    if lookup_df is not None:
        # Keep only necessary columns for web lookup to save space
        keep_cols = ['title', 'avg_rating', 'rating_count', 'year', 'genres']
        # check if columns exist
        actual_cols = [c for c in keep_cols if c in lookup_df.columns]
        data['lookup'] = lookup_df[actual_cols]
        
    joblib.dump(data, filename)
    print(f"💾 Model & Lookup Data saved to {filename}")

def load_model(filename='models/saved_classifier.pkl'):
    """Load model and feature columns"""
    if not os.path.exists(filename):
        return None, None, None
    data = joblib.load(filename)
    # Return model, features, and lookup (if exists)
    return data['model'], data['features'], data.get('lookup')

def evaluate_classifier(y_test, y_pred, y_pred_proba):
    """
    Comprehensive evaluation with metrics and visualizations
    """
    
    print("📊 STEP 4: Model Evaluation\n")
    print("="*50)
    
    # Classification Report
    print("📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['FLOP', 'HIT']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n🎯 CONFUSION MATRIX:")
    print(f"True Negatives (Correct FLOPs): {cm[0,0]:,}")
    print(f"False Positives (Predicted HIT, Actually FLOP): {cm[0,1]:,}")
    print(f"False Negatives (Predicted FLOP, Actually HIT): {cm[1,0]:,}")
    print(f"True Positives (Correct HITs): {cm[1,1]:,}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
    
    # Additional metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision_hit = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall_hit = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    print(f"\n📈 ADDITIONAL METRICS:")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"HIT Precision: {precision_hit*100:.2f}%")
    print(f"HIT Recall: {recall_hit*100:.2f}%")
    
    return cm, roc_auc


def plot_evaluation(cm, y_test, y_pred_proba):
    """
    Create visualization plots
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['FLOP', 'HIT'])
    axes[0].set_yticklabels(['FLOP', 'HIT'])
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    print("\n💾 Visualization saved as 'classification_results.png'")
    # plt.show() # Disabled for headless environment


def plot_feature_importance(model, feature_cols):
    """
    Plot Top 5 most important features (Aggregating Genres and Stars)
    """
    
    # 1. Get raw importance
    raw_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    })

    # 2. Aggregate Categories
    # We will sum importance for 'Genres' and 'Stars'
    
    # Identify genre columns (all assume they are not starting with star_ or special keywords)
    # A safer way is to check against known lists or patterns
    # For this project: 'star_' prefix => Stars. 
    # 'avg_rating', 'year', 'runtime', 'cast_size' => Specifics.
    # Everything else => Genres.
    
    special_cols = ['avg_rating', 'year', 'runtime', 'cast_size']
    
    agg_data = {'Genre': 0.0, 'Stars': 0.0}
    
    # Initialize specials
    for col in special_cols:
        matched = raw_importance[raw_importance['feature'] == col]
        if not matched.empty:
            agg_data[col] = matched['importance'].values[0]

    # Aggregate rest
    for idx, row in raw_importance.iterrows():
        f = row['feature']
        imp = row['importance']
        
        if f in special_cols:
            continue
        elif f.startswith('star_'):
            agg_data['Stars'] += imp
        else:
            # Assume Genre
            agg_data['Genre'] += imp
            
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(list(agg_data.items()), columns=['feature', 'importance'])
    plot_df = plot_df.sort_values('importance', ascending=False).head(5)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='importance', y='feature', palette='viridis')
    plt.title('Top 5 Feature Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Aggregate Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Aggregated Feature importance saved as 'feature_importance.png'")
    # plt.show() # Disabled for headless environment


# STEP 5: PREDICT ON NEW MOVIES
# ============================================

def predict_new_movie(model, feature_cols, avg_rating=None, rating_std=None, genres=None, year=None, runtime=None, cast_list=None):
    """
    Predict if a new movie will be a HIT or FLOP
    """    
    # Create input feature vector
    features = {col: 0 for col in feature_cols}
    
    # Set Numeric Features
    if 'avg_rating' in features:
        features['avg_rating'] = avg_rating if avg_rating is not None else 0.0
        
    if 'year' in features:
        features['year'] = year if year is not None else 2000
    
    if 'runtime' in features:
        features['runtime'] = runtime if runtime is not None else 95.0 # Default ~1h 35m
        
    if 'cast_size' in features:
        features['cast_size'] = len(cast_list) if cast_list else 0
    
    # Set Star Features
    # Check which top actors are in the provided cast_list
    global TOP_30_ACTORS
    if cast_list and TOP_30_ACTORS:
        for actor in TOP_30_ACTORS:
            col_name = f"star_{actor.replace(' ', '_')}"
            if col_name in features and actor in cast_list:
                features[col_name] = 1
    
    # Set Genres
        
    # Set Genres
    if genres:
        for genre in genres:
            if genre in features:
                features[genre] = 1
                
    # Create DataFrame (single row)
    X_new = pd.DataFrame([features])
    
    # Order columns exactly as model expects
    X_new = X_new[feature_cols]

    # Debug: Show the exact final vector
    print("\n[DEBUG] Final feature vector:")
    print(X_new.T)

    # Predict
    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]

    result = "🔥 HIT" if prediction == 1 else "❌ FLOP"
    confidence = proba[1] if prediction == 1 else proba[0]

    print("\n🎬 Prediction:", result)
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print(f"🎯 HIT Probability: {proba[1]*100:.2f}%")
    print(f"🎯 FLOP Probability: {proba[0]*100:.2f}%")

    return prediction, proba


def interactive_prediction_loop(model, feature_cols, lookup_df=None):
    """
    Continuous loop for user to enter movie details and get predictions
    
    Args:
        model: Trained classifier
        feature_cols: List of feature names
        lookup_df: Optional dataframe containing 'title', 'avg_rating', 'rating_count', 'genres', 'year'
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE PREDICTION MODE")
    print("="*60)
    # Load Rating Regressor for Pre-Release Simulation
    print("🤖 Loading Rating Regressor for Pre-Release Prediction...")
    regressor_model, regressor_cols, actor_scores, global_avg, _ = load_regressor('models/saved_regressor.pkl')
    if regressor_model:
        print("✅ Regressor Loaded Successfully (Simulating Pre-Release Mode)")
    else:
        print("⚠️ Warning: Regressor model not found. Using Manual/Average.")

    while True:
        print("\n------------------------------------------------")
        print("OPTIONS:")
        print("1. Type a Movie Name (Simulates Pre-Release Prediction)")
        print("2. Type 'manual' to enter raw metadata")
        
        user_choice = input("\n👉 Enter Movie Name or 'manual': ").strip()
        
        if user_choice.lower() == 'exit': break
        
        avg_rating = 0.0
        rating_count = 0 
        year = 2000
        genres = []
        found_movie = False

        # PATH A: MOVIE LOOKUP (Existing Database)
        if user_choice.lower() != 'manual' and lookup_df is not None:
             # Search for movie
             mask = lookup_df['title'].str.contains(user_choice, case=False, na=False)
             matches = lookup_df[mask]
             
             if len(matches) > 0:
                 # Sort by popularity to show most relevant first
                 unique_matches = matches.sort_values('rating_count', ascending=False).drop_duplicates('title')
                 # Take top 10 unique options
                 results_to_show = unique_matches.head(10).reset_index(drop=True)
                 
                 print(f"\n✅ Found {len(results_to_show)} matches in Local DB:")
                 
                 # 1. DISPLAY OPTIONS
                 print(f"{'#':<4} {'Title':<40} {'Year':<6} {'Rating':<15}")
                 print("-" * 70)
                 for idx, movie in results_to_show.iterrows():
                     print(f"{idx+1:<4} {movie['title'][:38]:<40} {int(movie['year']):<6} {movie['avg_rating']:.1f} ({movie['rating_count']:.0f})")
                     
                 # 2. SELECT OPTION
                 try:
                     sel = input("\n👉 Select movie number (or press Enter to search TMDB): ").strip()
                     if not sel: raise ValueError("Skip")
                     selection_idx = int(sel) - 1
                     if selection_idx < 0 or selection_idx >= len(results_to_show): raise ValueError("Invalid")
                     
                     # 3. USE SELECTED MOVIE
                     selected_movie = results_to_show.iloc[selection_idx]
                     print("\n" + "="*40)
                     print(f"🎬 Selected: {selected_movie['title']} ({int(selected_movie['year'])})")
                     
                     genres_list = str(selected_movie['genres']).split('|')
                     cast_str = str(selected_movie.get('cast', ''))
                     current_cast = [c.strip() for c in cast_str.split('|') if c.strip()]
                     runtime = selected_movie.get('runtime', 95.0)

                     # PRE-RELEASE PREDICTION LOGIC (As before)
                     predicted_rating = 3.0 # Default
                     
                     if regressor_model:
                         # Construct features for Regressor
                         reg_feats = {col: 0 for col in regressor_cols}
                         reg_feats['year'] = selected_movie['year']
                         if 'runtime' in reg_feats: reg_feats['runtime'] = runtime
                         if 'cast_size' in reg_feats: reg_feats['cast_size'] = len(current_cast)
                         
                         for actor in current_cast:
                             col_name = f"star_{actor.replace(' ', '_')}"
                             if col_name in reg_feats: reg_feats[col_name] = 1
                         for g in genres_list:
                             if g in reg_feats: reg_feats[g] = 1
                             
                         import pandas as pd
                         X_reg = pd.DataFrame([reg_feats])[regressor_cols]
                         predicted_rating = regressor_model.predict(X_reg)[0]
                         print(f"🔮 Pre-Release Model Estimated Rating: {predicted_rating:.2f} / 5.0")
                         print(f"   (Real Rating was: {selected_movie['avg_rating']:.2f})")
                     else:
                         predicted_rating = selected_movie['avg_rating']

                     # Now Predict HIT/FLOP using Predicted Rating
                     predict_new_movie(model, feature_cols,
                                    avg_rating=predicted_rating,
                                    rating_std=0.0,
                                    genres=genres_list,
                                    year=selected_movie['year'],
                                    runtime=runtime,
                                    cast_list=current_cast)
                     
                     found_movie = True 
                 
                 except (ValueError, IndexError):
                     print("⏩ Skipping Local selection...")
             
             if not found_movie:
                 print(f"\n🌐 Searching TMDB Database for '{user_choice}'...")
                 
                 tmdb_results = search_movies_list(user_choice, limit=5)
                 
                 if tmdb_results:
                     print(f"\n✅ Found {len(tmdb_results)} matches on TMDB:")
                     print(f"{'#':<4} {'Title':<40} {'Year':<6}")
                     print("-" * 60)
                     
                     for idx, res in enumerate(tmdb_results):
                         year_str = res.get('release_date', 'N/A')[:4]
                         print(f"{idx+1:<4} {res['title'][:38]:<40} {year_str:<6}")
                         
                     try:
                         sel = input("\n👉 Select movie number (or press Enter for Manual): ").strip()
                         if sel:
                             selection_idx = int(sel) - 1
                             if 0 <= selection_idx < len(tmdb_results):
                                 tmdb_result = tmdb_results[selection_idx]
                                 print(f"\n⏳ Fetching details for '{tmdb_result['title']}'...")
                                 
                                 details = fetch_movie_details(tmdb_result['id'])
                                 runtime, cast, genres = extract_metadata(details)
                                 year_tmdb = int(tmdb_result['release_date'][:4]) if tmdb_result.get('release_date') else 2024
                                 
                                 print(f"   Runtime: {runtime} min")
                                 print(f"   Cast: {', '.join(cast[:5])}")
                                 
                                 # AUTOMATIC RATING PREDICTION
                                 if regressor_model:
                                     reg_feats = {col: 0 for col in regressor_cols}
                                     reg_feats['year'] = year_tmdb
                                     if 'runtime' in reg_feats: reg_feats['runtime'] = runtime
                                     if 'cast_size' in reg_feats: reg_feats['cast_size'] = len(cast)
                                     
                                     for actor in cast:
                                         col_name = f"star_{actor.replace(' ', '_')}"
                                         if col_name in reg_feats: reg_feats[col_name] = 1
                                     for g in genres:
                                         if g in reg_feats: reg_feats[g] = 1
                                         
                                     import pandas as pd
                                     X_reg = pd.DataFrame([reg_feats])[regressor_cols]
                                     predicted_rating = regressor_model.predict(X_reg)[0]
                                     print(f"🔮 Model Predicts Rating: {predicted_rating:.2f} / 5.0")
                                     avg_rating = predicted_rating
                                 else:
                                     avg_rating = 3.0 # Fallback
                                     
                                 predict_new_movie(model, feature_cols,
                                                 avg_rating=avg_rating,
                                                 rating_std=0.0,
                                                 genres=genres,
                                                 year=year_tmdb,
                                                 runtime=runtime,
                                                 cast_list=cast)
                                 
                                 found_movie = True
                     except ValueError:
                         pass

                 if not found_movie:
                     print("❌ Match skipped or not found. Switching to MANUAL entry.")

        # PATH B: MANUAL ENTRY
        if user_choice.lower() == 'manual' or (not found_movie and lookup_df is not None):
            try:
                # Get Rating
                rating_input = input("🌟 Enter Average Rating (0.0 - 5.0): ")
                if rating_input.lower() == 'exit': break
                avg_rating = float(rating_input)
                
                # Get Count
                count_input = input("👥 Enter Number of Ratings: ")
                if count_input.lower() == 'exit': break
                rating_count = int(count_input)
                
                # Get Year
                year_input = input("📅 Enter Release Year (e.g. 1995): ")
                if year_input.lower() == 'exit': break
                year = int(year_input)
                
                # Get Genres
                print("🎭 Genres (comma-separated, e.g. Action, Comedy, Sci-Fi)")
                genre_input = input("   Enter Genres: ")
                if genre_input.lower() == 'exit': break
                genres = [g.strip() for g in genre_input.split(',')]
                
            except ValueError:
                print("⚠️ Invalid input! Please enter numbers.")
                continue

        # Predict (Only if we didn't find a movie via lookup)
        if not found_movie:
            predict_new_movie(model, feature_cols,
                            avg_rating=avg_rating,
                            rating_std=0.0, # Defaulting to 0 seems reasonable for quick test
                            genres=genres,
                            year=year,
                            runtime=95.0, # Default
                            cast_list=[]) # Default empty

    print("\n👋 Exiting Interactive Mode.")
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    print("="*60)
    print("🎬 MOVIE HIT/FLOP CLASSIFICATION SYSTEM")
    print("Using Pre-Split Train/Test Datasets")
    print("="*60)
    
    # 0. CHECK FOR EXISTING MODEL
    model_loaded = False
    model = None
    if os.path.exists('models/saved_classifier.pkl'):
        try:
            choice = input("\n💾 Found existing trained model. Load it? (y/n) [y]: ").strip().lower()
            if choice in ['', 'y', 'yes']:
                print("🔄 Loading saved model...")
                model, feat_cols, lookup_df = load_model()
                if model:
                    print("✅ Model loaded successfully!")
                    model_loaded = True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")

    # 1. LOAD DATA (Only if not loaded)
    if not model_loaded:
        # Load pre-split data with explicit types to prevent pandas/numpy errors
        try:
            print("📂 Loading data from 'ml-32m-split' folder...")
            
            # Load ENRICHED movies if available
            if os.path.exists('ml-32m-split/movies_enriched.csv'):
                print("✨ Loading ENRICHED movie metadata (Runtime, Cast)...")
                movies = pd.read_csv('ml-32m-split/movies_enriched.csv', dtype={'movieId': int, 'title': str, 'genres': str})
                # Process enriched features immediately
                movies = process_enriched_features(movies)
            else:
                print("⚠️ Enriched data not found, falling back to standard movies.csv")
                movies = pd.read_csv('ml-32m-split/movies.csv', dtype={'movieId': int, 'title': str, 'genres': str})

            print("⚠️ NOTE: Loading 1,000,000 ratings to prevent MemoryError (Sufficient for pattern learning)")
            # Load partial data for memory efficiency
            train_ratings = pd.read_csv('ml-32m-split/train_ratings.csv', 
                                      dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int}, 
                                      nrows=1000000)
            test_ratings = pd.read_csv('ml-32m-split/test_ratings.csv', 
                                     dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int},
                                     nrows=200000)
        except Exception as e:
            print(f"⚠️ Error loading split files: {e}")
            print("🔄 Falling back to regenerating split from raw 'ml-32m/ratings.csv'...")
            
            # Load Raw Data with limit
            movies = pd.read_csv('ml-32m/movies.csv', dtype={'movieId': int, 'title': str, 'genres': str})
            ratings = pd.read_csv('ml-32m/ratings.csv', 
                                dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int},
                                nrows=1200000)
            
            # Create Split
            unique_users = ratings['userId'].unique()
            mask = np.random.rand(len(unique_users)) < 0.8
            train_users = unique_users[mask]
            test_users = unique_users[~mask]
            
            train_ratings = ratings[ratings['userId'].isin(train_users)]
            test_ratings = ratings[ratings['userId'].isin(test_users)]
            
            print(f"✅ Re-split complete: Train={len(train_ratings):,}, Test={len(test_ratings):,}")
    
    # Process training and testing data separately
    if not model_loaded:
        train_df, genre_cols, train_thresholds = prepare_movie_features(movies, train_ratings, "TRAINING SET")
        test_df, _, _ = prepare_movie_features(movies, test_ratings, "TESTING SET", thresholds=train_thresholds)
        
        # Ensure both datasets have the same genre columns
        # Add missing genre columns with 0 values
        all_genres = set(genre_cols)
        for genre in all_genres:
            if genre not in test_df.columns:
                test_df[genre] = 0
        
        # Train model
        model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feat_cols = train_hit_classifier(
            train_df, test_df, genre_cols
        )
        
        # Create valid lookup dataframe (combine train and test to search entire library)
        print("\n📚 preparing movie lookup database...")
        lookup_df = pd.concat([train_df, test_df])
        
        # Save Model for Web App (WITH Lookup Data)
        save_model(model, feat_cols, lookup_df)

        # Evaluate
        cm, roc_auc = evaluate_classifier(y_test, y_pred, y_pred_proba)
        plot_evaluation(cm, y_test, y_pred_proba)
        plot_feature_importance(model, feat_cols)
        
        print("\n" + "="*60)
        print("✅ CLASSIFICATION COMPLETE!")
        print("="*60)
        print(f"📊 Model Performance: ROC-AUC = {roc_auc:.4f}")
        print(f"📁 Results saved as PNG files in current directory")
        print(f"\n🎓 This model was trained on {len(train_ratings):,} ratings")
        print(f"   and tested on {len(test_ratings):,} UNSEEN ratings")
            
        # Create valid lookup dataframe (combine train and test to search entire library)
        print("\n📚 preparing movie lookup database...")
        lookup_df = pd.concat([train_df, test_df])
    
    # Enter Interactive Mode
    interactive_prediction_loop(model,feat_cols, lookup_df)