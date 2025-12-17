import os
import requests
from flask import Flask, render_template, request, flash, redirect, url_for
import logging

app = Flask(__name__)
app.secret_key = 'super_secret_cute_key'

# Configure Backend URL (Docker service name or localhost)
# Default to 127.0.0.1:8000 for local dev stability
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000')

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Home page dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page (Hit/Flop + Rating) with Toggle Support"""
    prediction = None
    form_data = {}
    
    if request.method == 'POST':
        mode = request.form.get('mode', 'manual')
        try:
            payload = {}
            
            if mode == 'search':
                # 1. Look up movie features first
                title_query = request.form.get('search_title')
                logger.info(f"Looking up movie: {title_query}")
                
                lookup_resp = requests.get(f"{API_URL}/movies/lookup/{title_query}")
                if lookup_resp.status_code != 200:
                    flash(f"Movie '{title_query}' not found in database. Try manual entry!", "error")
                    return render_template('predict.html', prediction=None)
                
                movie_data = lookup_resp.json()
                # Populate payload with looked-up data
                payload = {
                    "title": movie_data['title'],
                    "year": movie_data['year'],
                    "runtime": movie_data['runtime'],
                    "genres": movie_data['genres'],
                    "cast": movie_data['cast']
                }
                # Keep data to fill form in UI
                form_data = payload
                
            else:
                # Manual Entry
                payload = {
                    "title": request.form.get('title'),
                    "year": int(request.form.get('year')),
                    "runtime": int(request.form.get('runtime')),
                    "genres": request.form.getlist('genres'),
                    "cast": [c.strip() for c in request.form.get('cast', '').split(',') if c.strip()],
                    "user_rating": float(request.form.get('user_rating')) if request.form.get('user_rating') else None
                }
                form_data = payload

            # 2. Get Hit/Flop Prediction
            logger.info(f"Sending prediction request to {API_URL}")
            hit_flop_resp = requests.post(f"{API_URL}/predict/hit-flop", json=payload)
            
            # 3. Get Rating Prediction
            rating_resp = requests.post(f"{API_URL}/predict/rating", json=payload)
            
            if hit_flop_resp.status_code == 200:
                result = hit_flop_resp.json()
                
                rating_val = "N/A"
                if rating_resp.status_code == 200:
                    rating_val = rating_resp.json().get('predicted_rating')
                    if isinstance(rating_val, float):
                        rating_val = round(rating_val, 1)

                prediction = {
                    "title": payload['title'],
                    "class": result['prediction'],
                    "confidence": f"{result['confidence']:.1%}",
                    "rating": rating_val,
                    "probabilities": result['probabilities']
                }
            else:
                flash(f"Prediction API Error: {hit_flop_resp.text}", "error")
                
        except requests.exceptions.ConnectionError:
            flash("Cannot connect to backend API. Is it running on port 8000?", "error")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            flash(f"An error occurred: {str(e)}", "error")
            
    return render_template('predict.html', prediction=prediction, form_data=form_data)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Recommendation page"""
    results = []
    search_type = 'movie'
    query = ''
    
    if request.method == 'POST':
        query = request.form.get('query')
        search_type = request.form.get('type')
        
        try:
            if search_type == 'movie':
                # Similar Movies
                payload = {"movie_title": query, "top_n": 6}
                resp = requests.post(f"{API_URL}/recommend/similar", json=payload)
                if resp.status_code == 200:
                    results = resp.json().get('recommendations', [])
                elif resp.status_code == 404:
                     flash(f"Movie '{query}' not found.", "warning")
                else:
                     flash("Error fetching recommendations.", "error")
            else:
                # Actor Search
                resp = requests.get(f"{API_URL}/recommend/actor/{query}?top_n=6")
                if resp.status_code == 200:
                    results = resp.json().get('top_movies', [])
                elif resp.status_code == 404:
                     flash(f"Actor '{query}' not found.", "warning")
                else:
                     flash("Error fetching actor movies.", "error")
                    
        except requests.exceptions.ConnectionError:
             flash("Backend API unavailable.", "error")
        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            
    return render_template('recommend.html', results=results, query=query, type=search_type)

@app.route('/insights')
def insights():
    """Insights gallery page with specific descriptions"""
    
    # Map filenames to descriptions
    insight_map = {
        'clusters.png': "Visualizing how movies group together based on their features.",
        'feature_importance.png': "Which features (Cast, Genre, Year) matter most for success?",
        'movie_embeddings_dr.png': "A 2D map of the entire movie universe.",
        'time_series_enriched.png': "How movie trends have changed over the decades."
    }
    
    images = []
    img_dir = os.path.join(app.static_folder, 'images')
    
    if os.path.exists(img_dir):
        # Scan directory, only include if in our map (or all if we want generic)
        # Choosing to show all present, with custom desc if available
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                desc = insight_map.get(f, "A detailed analysis visualization.")
                images.append({'file': f, 'desc': desc})
                
    return render_template('insights.html', images=images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
