"""
Complete FastAPI Backend for Movie ML Project.
Handles Predictions, Recommendations, and Movie Data Lookup.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Ensure root directory is in path for local module imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import local modules
from api.features import prepare_classifier_features, prepare_regressor_features, calculate_cast_size
from data_enrichment.tmdb_client import search_movies_list, fetch_movie_details, extract_metadata
from models.collaborative_recommender import recommend_people_also_watched, recommend_best_movies_by_actor as recommend_by_actor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Models (Pydantic) ---

class MovieInput(BaseModel):
    title: str = "Unknown"
    year: int
    runtime: int
    genres: List[str]
    cast: List[str] = []
    cast_size: Optional[int] = None # Allow manual input
    user_rating: Optional[float] = None  # Optional override for manual testing

class ClassificationResponse(BaseModel):
    prediction: str  # "Hit" or "Flop"
    confidence: float
    probability: float

class RegressionResponse(BaseModel):
    predicted_rating: float
    confidence_interval: Optional[List[float]] = None

class RecommendationInput(BaseModel):
    movie_titles: list[str]

class RecommendationResponse(BaseModel):
    recommendations: list[dict]

# --- Global State ---
models: Dict[str, Any] = {}
data: Dict[str, Any] = {}

# --- Helper to load Association Rules ---
def load_association_rules():
    path = 'models/association_rules.pkl'
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading association rules: {e}")
            return None
    return None

# --- Lifecycle Manager ---

def load_models():
    """Load all ML models and data artifacts at startup"""
    try:
        logger.info("loading models...")
        
        # 1. Load Classifier
        if os.path.exists('models/saved_classifier.pkl'):
            clf_data = joblib.load('models/saved_classifier.pkl')
            models['classifier'] = clf_data['model']
            models['classifier_features'] = clf_data.get('feature_names', clf_data.get('features'))
            # Try to get actors list if saved, else infer
            models['classifier_actors'] = clf_data.get('top_actors', [])
            logger.info("✓ Classifier loaded")
        else:
            logger.warning("⚠️ Classifier model not found at models/saved_classifier.pkl")

        # 2. Load Regressor
        if os.path.exists('models/saved_regressor.pkl'):
            reg_data = joblib.load('models/saved_regressor.pkl')
            models['regressor'] = reg_data['model']
            models['regressor_features'] = reg_data['features']
            logger.info("✓ Regressor loaded")
        else:
            logger.warning("⚠️ Regressor model not found at models/saved_regressor.pkl")

        # 3. Load Recommender System & Enriched Data (Unified Loading)
        try:
            from models.collaborative_recommender import load_data
            # load_data returns: movies, matrix_sparse, movie_idx_to_id, enriched
            movies, matrix, mapper, enriched = load_data()
            
            # Recommender Data
            data['movies_df'] = movies
            data['movie_matrix'] = matrix
            data['matrix_titles'] = mapper
            
            # Fit KNN logic is inside load_data? No, it's in interactive_loop. 
            # We need to fit it here or load a saved one.
            # load_data creates the matrix. We should fit KNN here.
            
            from sklearn.neighbors import NearestNeighbors
            logger.info("   Fitting KNN model for Recommender...")
            knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
            knn.fit(matrix)
            data['knn_model'] = knn
            
            # Enriched Data (Now includes 'avg_rating' from load_data logic!)
            data['enriched_movies'] = enriched
            logger.info(f"✓ Recommender Data & Enriched DB loaded ({len(enriched)} records)")
            
            # 4. Load Association Rules
            data['association_rules'] = load_association_rules()
            if data['association_rules'] is not None:
                logger.info(f"✓ Association Rules loaded ({len(data['association_rules'])} rules)")
            
        except Exception as e:
            logger.error(f"❌ Error loading Recommender Data: {e}")
            # Fallback to empty if critical failure, but this is essential
            pass
            
    except Exception as e:
        logger.error(f"FATAL: Error loading models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown
    models.clear()
    data.clear()

app = FastAPI(title="Movie ML API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Web Config ---
app.mount("/static", StaticFiles(directory=os.path.join(root_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(root_dir, "templates"))

# --- Web Routes ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/classifier", response_class=HTMLResponse, include_in_schema=False)
async def read_classifier(request: Request):
    # Pass genres for the dropdown/checkboxes
    all_genres = []
    if 'classifier_features' in models:
        all_genres = [c for c in models['classifier_features'] if c not in ['avg_rating', 'year', 'runtime', 'cast_size'] and not c.startswith('star_')]
        all_genres.sort()
    return templates.TemplateResponse("classifier.html", {"request": request, "all_genres": all_genres})

@app.get("/recommender", response_class=HTMLResponse, include_in_schema=False)
async def read_recommender(request: Request):
    return templates.TemplateResponse("recommender.html", {"request": request})

@app.get("/insights", response_class=HTMLResponse, include_in_schema=False)
async def read_insights(request: Request):
    images = []
    static_img_dir = os.path.join(root_dir, "static/images")
    if os.path.exists(static_img_dir):
        images = [f for f in os.listdir(static_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return templates.TemplateResponse("insights.html", {"request": request, "images": images})


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/predict/hit-flop", response_model=ClassificationResponse, tags=["Predictions"])
async def predict_hit_flop(movie: MovieInput):
    """
    Predict if a movie will be a Hit or Flop.
    """
    if 'classifier' not in models:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    # 1. Determine Rating to use (User Override > Predicted > Default)
    used_rating = movie.user_rating
    if used_rating is None:
        # If no user override, predict it internally using Regressor
        if 'regressor' in models:
            reg_feats = prepare_regressor_features(
                year=movie.year,
                runtime=movie.runtime,
                genres=movie.genres,
                cast=movie.cast,
                expected_cols=models.get('regressor_features')
            )
            used_rating = float(models['regressor'].predict(reg_feats)[0])
            used_rating = max(0.0, min(5.0, used_rating)) # Clip
        else:
            used_rating = 3.0 # Fallback average
    
    # 2. Prepare Features for Classifier
    c_size = movie.cast_size if movie.cast_size is not None else calculate_cast_size(movie.cast)
    
    feats = prepare_classifier_features(
        year=movie.year,
        runtime=movie.runtime,
        genres=movie.genres,
        avg_rating=used_rating,
        cast_size=c_size,
        cast=movie.cast,
        expected_cols=models.get('classifier_features')
    )
    
    # 3. Predict
    clf = models['classifier']
    prob = float(clf.predict_proba(feats)[0][1]) # Probability of class 1 (Hit)
    prediction = "Hit" if prob > 0.5 else "Flop"
    confidence = prob if prediction == "Hit" else (1 - prob)
    
    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "probability": prob
    }

@app.post("/predict/rating", response_model=RegressionResponse, tags=["Predictions"])
async def predict_rating(movie: MovieInput):
    """
    Predict the star rating (0-5).
    """
    if 'regressor' not in models:
        raise HTTPException(status_code=503, detail="Regressor not loaded")
        
    logger.info(f"Rating Req: {movie.title} | Genres: {movie.genres} | Cast: {movie.cast}")
    
    feats = prepare_regressor_features(
        year=movie.year,
        runtime=movie.runtime,
        genres=movie.genres,
        cast=movie.cast,
        expected_cols=models.get('regressor_features')
    )
    
    val = float(models['regressor'].predict(feats)[0])
    return {"predicted_rating": round(max(0.0, min(5.0, val)), 1)}


class SearchQuery(BaseModel):
    query: str
    type: str = "movie" # 'movie', 'actor'
    selection: Optional[Dict] = None # For Disambiguation: {source, id, title, metadata}

@app.post("/search/candidates", tags=["Search"])
async def search_candidates(item: SearchQuery):
    """
    Search for movie candidates (Local + TMDB) for disambiguation.
    """
    if 'movies_df' not in data:
         raise HTTPException(status_code=503, detail="Movie data not loaded")
    
    # Import here to avoid circular dep issues early on
    from models.collaborative_recommender import find_movie_candidates
    
    candidates = find_movie_candidates(item.query, data['movies_df'])
    return {"candidates": candidates}

@app.post("/recommend", tags=["Recommendations"])
async def get_recommendations(item: SearchQuery):
    """
    Get recommendations based on Movie Title (with optional Selection) or Actor Name.
    """
    results = []
    
    if item.type == 'movie':
        if 'knn_model' not in data:
             raise HTTPException(status_code=503, detail="Recommender model not loaded")
             
        # Use the imported function
        # recommend_people_also_watched(movie_title, movies, matrix_sparse, movie_idx_to_id, model_knn)
        # Note: 'movies' arg in function expects the DF, 'matrix_sparse' the matrix...
        rec_list = recommend_people_also_watched(
            item.query, 
            data['movies_df'], 
            data['movie_matrix'], 
            data['matrix_titles'], 
            data['knn_model'],
            interactive=False # Don't block for input
        )
        
        # Handle "FALLBACK" return from function which returns [FLAG, data]
        if rec_list and isinstance(rec_list, list) and len(rec_list) > 0 and rec_list[0] == "FALLBACK_TO_CONTENT":
            # rec_list[1] contains the selection data needed for content recommender
            # We need to call content recommender!
            # Import: from models.collaborative_recommender import recommend_similar_content
            # Wait, verify import first.
            from models.collaborative_recommender import recommend_similar_content
            if 'enriched_movies' in data:
                 # Pass selection data (Local ID or TMDB Metadata)
                 content_recs = recommend_similar_content(rec_list[1], data['enriched_movies'], interactive=False)
                 results = content_recs
            else:
                 results = []
        else:
            results = rec_list
            
    elif item.type == 'actor':
        if 'enriched_movies' in data:
            results = recommend_by_actor(item.query, data['enriched_movies'])
            
    # Post-process: Ensure consistency (Titles -> Dicts)
    # If results is list of strings, enrich it
    final_results = []
    if results and isinstance(results, list):
        if results and isinstance(results[0], str):
            # It's a list of titles
            df = data.get('enriched_movies')
            for title in results:
                rating = 0.0
                if df is not None:
                     match = df[df['title'] == title]
                     if not match.empty:
                         rating = float(match.iloc[0].get('avg_rating', 0.0))
                final_results.append({"title": title, "avg_rating": round(rating, 1)})
        else:
            # Already dicts
            final_results = results
            
    return {"results": final_results}

@app.get("/movies/search/{query}", tags=["Movies"])
async def search_for_movie(query: str):
    """
    Flexible search:
    1. Local DB (fuzzy/partial match)
    2. TMDB (external API)
    Returns: List of candidates for user selection.
    """
    results = []
    
    # 1. Local Search
    df = data.get('enriched_movies')
    if df is not None:
        # Simple content match
        mask = df['title'].str.contains(query, case=False, na=False)
        local_matches = df[mask].head(5)
        
        # Format
        for _, row in local_matches.iterrows():
            results.append({
                "source": "Local DB",
                "title": row.get('title', 'Unknown'),
                "year": int(row.get('year')) if pd.notna(row.get('year')) else None,
                "rating": round(row.get('avg_rating', 0), 1),
                "count": int(row.get('rating_count', 0))
            })
            
    # 2. TMDB Search (Append to results)
    try:
        tmdb_hits = search_movies_list(query, limit=5)
        for hit in tmdb_hits:
            hit_year = int(hit['release_date'][:4]) if hit.get('release_date') else None
            
            # Simple deduplication
            is_dup = False
            for r in results:
                if r['title'].lower() == hit['title'].lower():
                    # If years are close, assume duplicate
                    if r['year'] and hit_year and abs(r['year'] - hit_year) <= 1:
                        is_dup = True
                        break
            
            if not is_dup:
                results.append({
                    "source": "TMDB",
                    "title": hit['title'],
                    "year": hit_year,
                    "rating": hit.get('vote_average', 0),
                    "count": hit.get('vote_count', 0)
                })
    except Exception as e:
        logger.error(f"TMDB Search failed: {e}")
        
    return results

@app.get("/movies/lookup/{title}", tags=["Movies"])
async def lookup_movie_details(title: str, year: Optional[int] = None):
    """
    Get full details for population (Runtime, Cast, Genres).
    Logic:
    1. Try TMDB List Search -> Filter by Year (if provided).
    2. Fallback to Local DB Strict Match.
    3. Fallback to Local DB Fuzzy Match.
    """
    
    # 1. TMDB (Primary source for metadata richness, especially for new movies)
    # We use list-search + year filter for robustness
    try:
        candidates = search_movies_list(title, limit=10)
        best_tmdb = None
        
        if candidates and year:
            for cand in candidates:
                y_str = cand.get('release_date', '')[:4]
                if y_str.isdigit():
                    if abs(int(y_str) - year) <= 1:
                        best_tmdb = cand
                        break
        elif candidates:
            best_tmdb = candidates[0] # Default to top hit if no year
            
        if best_tmdb:
            details = fetch_movie_details(best_tmdb['id'])
            runtime, cast, genres = extract_metadata(details)
            return {
                "title": best_tmdb['title'],
                "year": int(best_tmdb.get('release_date', '2000')[:4]) if best_tmdb.get('release_date') else 2000,
                "runtime": runtime,
                "genres": genres,
                "cast": cast,
                "rating": best_tmdb.get('vote_average', 0)
            }
    except Exception as e:
        logger.error(f"TMDB Lookup Error: {e}")

    # 2. Local DB Fallback
    df = data.get('enriched_movies')
    if df is not None:
        # Title Match
        matches = df[df['title'].str.lower() == title.lower()]
        
        if year and not matches.empty:
            matches = matches[
                (matches['year'] >= year - 1) & 
                (matches['year'] <= year + 1)
            ]
        
        if not matches.empty:
            # Sort by popularity
            if 'rating_count' in matches.columns:
                 matches = matches.sort_values('rating_count', ascending=False)
            
            row = matches.iloc[0]
            return {
                "title": row['title'],
                "year": int(row['year']),
                "runtime": int(row.get('runtime', 90)),
                "genres": str(row.get('genres', '')).split('|'),
                "cast": str(row.get('cast', '')).split('|'),
                "rating": float(row.get('avg_rating', 0))
            }

    raise HTTPException(status_code=404, detail="Movie not found")

@app.get("/insights/associations", tags=["Insights"])
async def get_association_insights():
    """
    Get Association Rules (e.g., If User likes Action -> Also likes Adventure)
    """
    rules = data.get('association_rules')
    if rules is None:
         # Try to load on demand if not loaded at startup
         rules = load_association_rules()
         if rules is None:
            raise HTTPException(status_code=404, detail="Association Rules not available. Run models/association_rules.py first.")
            
    # Convert DataFrame to JSON-friendly format
    result = []
    for _, row in rules.iterrows():
        result.append({
            "antecedents": row['antecedents'], # List
            "consequents": row['consequents'], # List
            "support": round(row['support'], 3),
            "confidence": round(row['confidence'], 3),
            "lift": round(row['lift'], 3)
        })
    return {"rules": result}

@app.get("/insights/forecast", tags=["Insights"])
async def get_forecast_insights():
    """
    Get Time Series Forecast data.
    Note: For this project, we primarily render static plots, but this endpoint
    could return the raw trend data used for plotting.
    """
    # Simply running the analysis script logic to get raw numbers if needed,
    # or returning a static message pointing to the generated image.
    # Let's return the simplified trend data if we can, or just status.
    
    return {
        "message": "Time Series Analysis is processed offline.",
        "assets": ["/static/images/time_series_enriched.png"],
        "description": "Analysis of Movie Popularity, Runtime, and Cast Size over 100 years."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn api.main:app --reload