from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.neighbors import NearestNeighbors

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Model Logic
from models.python_movie_classifier import predict_new_movie
from models.collaborative_recommender import recommend_people_also_watched, recommend_best_movies_by_actor as recommend_by_actor

app = Flask(__name__)

# Global storage for models
MODELS = {
    'classifier': None,
    'classifier_feats': None,
    'classifier_lookup': None,
    'regressor': None,  # NEW
    'regressor_feats': None, # NEW
    'recommender_matrix': None,
    'recommender_mapper': None,
    'recommender_movies': None,
    'knn_model': None,
    'enriched_data': None
}

def load_models():
    print("⏳ Loading Models for Web App...")
    
    # 1. Load Classifier
    if os.path.exists('models/saved_classifier.pkl'):
        data = joblib.load('models/saved_classifier.pkl')
        MODELS['classifier'] = data['model']
        MODELS['classifier_feats'] = data['features']
        MODELS['classifier_lookup'] = data.get('lookup') # Load lookup df
        print("✅ Classifier Loaded")
        
    # 2. Load Regressor (Rating Predictor)
    if os.path.exists('models/saved_regressor.pkl'):
        data = joblib.load('models/saved_regressor.pkl')
        MODELS['regressor'] = data['model']
        MODELS['regressor_feats'] = data['features']
        print("✅ Regressor Loaded")

    # 3. Load Recommender Data
    if os.path.exists('models/saved_recommender.pkl'):
        data = joblib.load('models/saved_recommender.pkl')
        MODELS['recommender_matrix'] = data['matrix']
        MODELS['recommender_mapper'] = data['mapper']
        MODELS['recommender_movies'] = data['movies']
        
        # We need to re-fit the KNN because we only saved the matrix
        print("   🧠 Fitting KNN on the fly...")
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        knn.fit(DATA_MATRIX := data['matrix'])
        MODELS['knn_model'] = knn
        print("✅ Recommender Loaded")
        
    # 4. Load Enriched Data for Actor Search
    if os.path.exists('ml-32m-split/movies_enriched.csv'):
        MODELS['enriched_data'] = pd.read_csv('ml-32m-split/movies_enriched.csv')
        print("✅ Enriched Data Loaded")

# Load immediately on start
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/lookup', methods=['GET'])
def lookup_movie():
    """API endpoint to search for movie details"""
    query = request.args.get('query', '').lower()
    if not query or MODELS['classifier_lookup'] is None:
        return jsonify({'error': 'No query or data loaded'})
    
    df = MODELS['classifier_lookup']
    # Case insensitive search
    try:
        matches = df[df['title'].str.contains(query, case=False, na=False)]
        matches = matches.sort_values('rating_count', ascending=False).head(5)
        
        results = []
        for _, row in matches.iterrows():
            results.append({
                'title': row['title'],
                'year': int(row['year']),
                'avg_rating': round(float(row['avg_rating']), 2),
                'rating_count': int(row['rating_count']),
                'runtime': float(row['runtime']) if 'runtime' in row and not pd.isna(row['runtime']) else 90.0,
                'cast_size': int(row['cast_size']) if 'cast_size' in row and not pd.isna(row['cast_size']) else 0,
                'genres': str(row['genres']).split('|') # Convert piped string to list
            })
        return jsonify(results)
    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify([])

@app.route('/api/predict_rating', methods=['POST'])
def predict_rating():
    """API to predict expected rating using Regressor"""
    if not MODELS['regressor']:
        return jsonify({'error': 'Regressor model not loaded'})
        
    try:
        data = request.json
        year = float(data.get('year', 2024))
        # rating_count (popularity) is a feature in regressor, use default if new
        rating_count = int(data.get('rating_count', 100)) 
        genres = data.get('genres', [])
        
        # Build Vector
        features = {col: 0 for col in MODELS['regressor_feats']}
        features['year'] = year
        
        if 'rating_count' in features:
            features['rating_count'] = rating_count
            
        for g in genres:
            if g in features:
                features[g] = 1
                
        # Predict
        input_df = pd.DataFrame([features])
        input_df = input_df[MODELS['regressor_feats']] # Ensure order
        
        prediction = MODELS['regressor'].predict(input_df)[0]
        return jsonify({'predicted_rating': round(prediction, 2)})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    prediction = None
    confidence = 0
    
    if request.method == 'POST':
        if not MODELS['classifier']:
            return render_template('classifier.html', error="Model not loaded. Please run training script.")
            
        # Get Form Data
        try:
            avg_rating = float(request.form.get('avg_rating', 0))
            # rating_count is NOT used for prediction, only log/debug
            year = int(request.form.get('year', 2000))
            runtime = float(request.form.get('runtime', 90.0))
            cast_size = int(request.form.get('cast_size', 5))
            
            # simple genre handling
            genres = request.form.getlist('genres') 
            
            # Run Prediction
            pred_class, probs = predict_new_movie(
                MODELS['classifier'], 
                MODELS['classifier_feats'], 
                avg_rating=avg_rating, 
                year=year,
                runtime=runtime,
                cast_size=cast_size,
                genres=genres
            )
            
            prediction = "HIT" if pred_class == 1 else "FLOP"
            confidence = round((probs[1] if pred_class == 1 else probs[0]) * 100, 1)
            
        except Exception as e:
            return render_template('classifier.html', error=str(e))
    
    # Get all trained genres for the checkbox list
    # Filter features to find only genres (exclude avg_rating, year, runtime, cast_size)
    all_genres = []
    if MODELS['classifier_feats']:
        all_genres = [c for c in MODELS['classifier_feats'] if c not in ['avg_rating', 'year', 'runtime', 'cast_size'] and not c.startswith('star_')]
        all_genres.sort()
            
    return render_template('classifier.html', prediction=prediction, confidence=confidence, all_genres=all_genres)

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    recommendations = []
    search_type = 'movie'
    query = ''
    
    if request.method == 'POST':
        query = request.form.get('query')
        search_type = request.form.get('type')
        
        if search_type == 'movie':
            if MODELS['knn_model']:
                recommendations = recommend_people_also_watched(
                    query, 
                    MODELS['recommender_movies'], 
                    MODELS['recommender_matrix'], 
                    MODELS['recommender_mapper'], 
                    MODELS['knn_model']
                )
        elif search_type == 'actor':
            if MODELS['enriched_data'] is not None:
                recommendations = recommend_by_actor(query, MODELS['enriched_data'])
                
    return render_template('recommender.html', recommendations=recommendations, query=query, type=search_type)

@app.route('/insights')
def insights():
    # List images in static/images
    images = []
    if os.path.exists('static/images'):
        images = [f for f in os.listdir('static/images') if f.endswith('.png')]
    return render_template('insights.html', images=images)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
