# collaborative_recommender.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process
import os
import sys

# Ensure we can import from data_enrichment
sys.path.append(os.getcwd())
from data_enrichment.tmdb_client import search_movies_list, fetch_movie_details, extract_metadata

def load_data():
    print("📂 Loading data for Collaborative Filtering...")
    
    # --- OPTIMIZATION: Try loading pre-calculated model first ---
    # This assumes 'saved_recommender.pkl' contains: matrix, mapper, and minimal movies df
    try:
        from models.collaborative_recommender import load_recommender
        matrix, mapper, movies_minimal = load_recommender()
        
        # Load Enriched Data (Always needed for details/content-recs)
        enriched_path = 'ml-32m-split/movies_enriched.csv'
        if os.path.exists(enriched_path):
            enriched = pd.read_csv(enriched_path)
        else:
            enriched = pd.DataFrame(columns=['title', 'cast'])
            print("⚠️ Enriched data not found.")

        if matrix is not None:
             print("   ✓ Loaded pre-computed matrix from 'saved_recommender.pkl'. Skipping raw data processing.")
             
             # We still need enriched stats (avg_rating) merged if they aren't in the saved movies df
             # The saved 'movies' is often just ID/Title. 
             # Let's ensure 'enriched' has the stats we need.
             # In the CSV path below, we calculate stats from 'train_ratings.csv'.
             # If we load from PKL, we might skip calculating stats if we don't have ratings loaded.
             # Ideally, 'enriched_movies.csv' SHOULD have these stats already saved during the last run.
             # Let's check if 'avg_rating' is in enriched.
             if 'avg_rating' not in enriched.columns:
                 # If we don't have ratings loaded, we can't calculate them.
                 # We assume enriched data is sufficient or we accept 0.0 ratings in this fast-path.
                 enriched['avg_rating'] = 0.0
             
             # Also need 'movies' dataframe. The PKL one is minimal. 
             # If we need more, we might need to read movies.csv (fast)
             # But the PKL one usually suffices for the Matrix mapping.
             
             return movies_minimal, matrix, mapper, enriched
    except Exception as e:
        print(f"   ⚠️ Could not load saved model ({e}). Falling back to raw data processing...")

    # --- FALLBACK: Original Raw Data Processing ---
    
    # 1. Load Movies
    movies = pd.read_csv('ml-32m-split/movies.csv')
    
    # 2. Load Ratings (Sparse)
    print("   Loading ratings (this takes a moment)...")
    ratings = pd.read_csv('ml-32m-split/train_ratings.csv', usecols=['userId', 'movieId', 'rating'])
    
    # 3. Create Sparse Matrix (Efficiently)
    # Map IDs to consecutive integers
    # We need to map movieId to 0..N for the matrix column
    print("   Mapping IDs to matrix indices...")
    
    # Create mappers
    user_ids = ratings['userId'].astype('category')
    movie_ids = ratings['movieId'].astype('category')
    
    # Create the matrix
    # Rows: Movies (so we can find similar movies), Cols: Users
    # Transposing logic relative to pivot(index=movie, col=user)
    
    # matrix[movie_idx, user_idx] = rating
    row_ind = movie_ids.cat.codes
    col_ind = user_ids.cat.codes
    data_vals = ratings['rating'].values
    
    n_movies = len(movie_ids.cat.categories)
    n_users = len(user_ids.cat.categories)
    
    print(f"   Building sparse matrix ({n_movies} movies x {n_users} users)...")
    matrix_sparse = csr_matrix((data_vals, (row_ind, col_ind)), shape=(n_movies, n_users))
    
    # Store mapping to look up title from matrix index
    # matrix index i -> movie_ids.cat.categories[i] -> movies.csv
    movie_idx_to_id = dict(enumerate(movie_ids.cat.categories))
    
    # 4. Load Enriched for Actor Search & Content Recs
    enriched_path = 'ml-32m-split/movies_enriched.csv'
    if os.path.exists(enriched_path):
        enriched = pd.read_csv(enriched_path)
    else:
        enriched = pd.DataFrame(columns=['title', 'cast'])

    # 5. Merge Avg Rating (Important for "Best Movies" sorting)
    # Re-use the ratings calculation logic to get movie stats
    movie_counts = ratings['movieId'].value_counts()
    movie_sums = ratings.groupby('movieId')['rating'].sum()
    movie_avgs = movie_sums / movie_counts
    
    stats_df = pd.DataFrame({'avg_rating': movie_avgs, 'rating_count': movie_counts})
    enriched = enriched.merge(stats_df, left_on='movieId', right_index=True, how='left')
    enriched['avg_rating'] = enriched['avg_rating'].fillna(0)
    
    # 6. Pre-process for Content Similarity (Soup)
    # Minimal enrichment for performance
    if 'cast' not in enriched.columns: enriched['cast'] = ''
    if 'tmdb_genres' not in enriched.columns: enriched['tmdb_genres'] = ''
    
    return movies, matrix_sparse, movie_idx_to_id, enriched

import joblib

def save_recommender(matrix_sparse, movie_idx_to_id, movies_df, filename='models/saved_recommender.pkl'):
    os.makedirs('models', exist_ok=True)
    # Convert movies_df to simple dict or smaller DF to save space? 
    # Actually for 80k movies it's fine.
    # We don't save the KNN model object because it's huge? No, KNN is small, matrix is big.
    # We save the data needed to recreate the recommender.
    # Actually, saving the FITTED model is best.
    data = {
        'matrix': matrix_sparse,
        'mapper': movie_idx_to_id,
        'movies': movies_df[['movieId', 'title']] # minimal cols
    }
    joblib.dump(data, filename)
    print(f"💾 Recommender data saved to {filename}")

def load_recommender(filename='models/saved_recommender.pkl'):
    if not os.path.exists(filename):
        return None, None, None
    data = joblib.load(filename)
    return data['matrix'], data['mapper'], data['movies']

def recommend_people_also_watched(movie_title, movies, matrix_sparse, movie_idx_to_id, model_knn, interactive=True):
    # 1. Interactive Selection (or Pre-Selection)
    if isinstance(movie_title, (list, tuple)):
        selection = movie_title
    else:
        selection = select_movie_interactively(movie_title, movies, interactive=interactive)
        
    if not selection:
        return []
    
    # Check Mode
    if selection[0] == 'TMDB':
        print("⚠️ TMDB movie not in local User Rating DB. Switching to Content Filtering...")
        return ["FALLBACK_TO_CONTENT", selection]
        
    # Local Mode
    _, idx, found_title, target_id = selection
    
    # print(f"   Selected: {found_title}")
    
    # 3. Check Rating Count (Quality Control)
    movie_id_to_idx = {v: k for k, v in movie_idx_to_id.items()}
    if target_id not in movie_id_to_idx:
        print("⚠️ This movie has no ratings. Switching to Content Search.")
        return ["FALLBACK_TO_CONTENT", selection]
        
    # Check popularity in matrix (row nnz)
    query_idx = movie_id_to_idx[target_id]
    user_votes = matrix_sparse[query_idx].getnnz()
    
    # AGGRESSIVE Fallback
    # If a movie has fewer than 100 ratings, user behavior is too noisy.
    # We switch to Actor/Genre matching which is far more reliable for niche/Indian movies.
    if user_votes < 100:
        if interactive: print(f"⚠️ Only {user_votes} users watched this. CF is unreliable. Switching to Content Filtering...")
        return ["FALLBACK_TO_CONTENT", selection]
        
    # 4. Find Neighbours
    distances, indices = model_knn.kneighbors(matrix_sparse[query_idx], n_neighbors=7)
    
    results = []
    for i in range(1, len(distances.flatten())):
        neighbor_matrix_idx = indices.flatten()[i]
        neighbor_id = movie_idx_to_id[neighbor_matrix_idx]
        title_matches = movies[movies['movieId'] == neighbor_id]['title']
        if not title_matches.empty:
            results.append(title_matches.values[0])
            
    return results

def find_movie_candidates(query, movies_df, limit=5):
    """
    Search for candidates (Local + TMDB) for the disambiguation UI.
    Returns list of dicts: {'source': 'LOCAL'/'TMDB', 'id': ..., 'title': ..., 'year': ...}
    """
    candidates = []
    
    # 1. Search Local
    mask = movies_df['title'].str.contains(query, case=False, regex=False, na=False)
    matches_df = movies_df[mask].head(limit)
    
    # Fuzzy if low matches
    if len(matches_df) < 3:
        fuzzy = process.extract(query, movies_df['title'], limit=limit)
        fuzzy_indices = [x[2] for x in fuzzy if x[1] > 60]
        fuzzy_matches = movies_df.loc[fuzzy_indices]
        matches_df = pd.concat([matches_df, fuzzy_matches]).drop_duplicates().head(limit)
        
    for idx, row in matches_df.iterrows():
        # Extract Year
        import re
        m = re.search(r'\((\d{4})\)', str(row['title']))
        year = m.group(1) if m else "N/A"
        candidates.append({
            'source': 'LOCAL',
            'id': int(row.get('movieId')), # Ensure native int
            'title': row['title'],
            'year': year,
            'metadata': {'idx': int(idx)} # Needed for internal lookup
        })
        
    # 2. TMDB
    tmdb_results = search_movies_list(query, limit=5)
    if tmdb_results:
        for m in tmdb_results:
             candidates.append({
                'source': 'TMDB',
                'id': m['id'],
                'title': m.get('title'),
                'year': m.get('release_date', 'N/A')[:4],
                'metadata': m 
             })
             
    return candidates

def select_movie_interactively(query, movies_df, interactive=True):
    """
    Search for a movie (Local -> TMDB).
    Returns tuple: ('LOCAL'/'TMDB', data)
    """
    # 1. Search Local
    mask = movies_df['title'].str.contains(query, case=False, regex=False, na=False)
    matches = movies_df[mask].head(10)
    
    if len(matches) < 3:
        fuzzy = process.extract(query, movies_df['title'], limit=5)
        fuzzy_indices = [x[2] for x in fuzzy if x[1] > 60]
        fuzzy_matches = movies_df.loc[fuzzy_indices]
        matches = pd.concat([matches, fuzzy_matches]).drop_duplicates().head(10)
        
    # --- NON-INTERACTIVE MODE (API) ---
    if not interactive:
        if not matches.empty:
            # Return Top Match
            row = matches.iloc[0]
            return ('LOCAL', row.name, row['title'], row.get('movieId'))
        
        # If no local, try TMDB automatically
        tmdb_results = search_movies_list(query, limit=1)
        if tmdb_results:
            m = tmdb_results[0]
            # Fetch details
            details = fetch_movie_details(m['id'])
            runtime, cast, tmdb_genres = extract_metadata(details)
            metadata = {
                'title': m['title'],
                'year': m.get('release_date', '')[:4],
                'cast': "|".join(cast),
                'tmdb_genres': "|".join(tmdb_genres),         
                'avg_rating': m.get('vote_average', 0)
            }
            return ('TMDB', metadata)
        return None

    # --- INTERACTIVE MODE (CLI) ---
    # 2. Display Local Options
    print(f"\n✅ Found {len(matches)} local matches:")
    print(f"{'#':<4} {'Title (Local Database)':<50} {'Year':<10}")
    print("-" * 70)
    
    options = []
    for i, (idx, row) in enumerate(matches.iterrows(), 1):
        import re
        m = re.search(r'\((\d{4})\)', str(row['title']))
        year = m.group(1) if m else "N/A"
        print(f"{i:<4} {row['title'][:48]:<50} {year:<10}")
        options.append(('LOCAL', idx, row['title'], row.get('movieId')))
        
    print(f"{len(options)+1:<4} 🌐 Search TMDB (Online)...")
    
    # 3. User Selection
    while True:
        try:
            sel = input(f"\n👉 Select movie (1-{len(options)+1}) or 0 to cancel: ").strip()
            if sel == '0': return None
            val = int(sel)
            
            # Local Selection
            if 1 <= val <= len(options):
                return options[val-1]
                
            # TMDB Selection
            if val == len(options) + 1:
                print(f"   🌐 Searching TMDB for '{query}'...")
                tmdb_results = search_movies_list(query)
                if not tmdb_results:
                    print("   ❌ No results on TMDB.")
                    return None
                    
                print(f"\n✅ Found on TMDB:")
                tmdb_options = []
                for j, m in enumerate(tmdb_results, 1):
                    title = m.get('title', 'Unknown')
                    date = m.get('release_date', 'N/A')[:4]
                    print(f"{j:<4} {title[:48]:<50} {date:<10}")
                    tmdb_options.append(m)
                    
                sub_sel = input(f"\n👉 Select TMDB movie (1-{len(tmdb_options)}): ").strip()
                if sub_sel.isdigit() and 1 <= int(sub_sel) <= len(tmdb_options):
                    idx = int(sub_sel) - 1
                    movie_data = tmdb_options[idx]
                    
                    # Fetch Full Details (Cast/Genre)
                    print("   📥 Downloading metadata...")
                    details = fetch_movie_details(movie_data['id'])
                    runtime, cast, tmdb_genres = extract_metadata(details)
                    
                    metadata = {
                        'title': movie_data['title'],
                        'year': movie_data.get('release_date', '')[:4],
                        'cast': "|".join(cast),
                        'tmdb_genres': "|".join(tmdb_genres),         'avg_rating': movie_data.get('vote_average', 0)
                    }
                    return ('TMDB', metadata)
                    
        except ValueError:
            pass
        print("Invalid selection.")

## UPDATED FUNCTIONS TO USE SELECTION

def recommend_similar_content(movie_title, enriched_data, interactive=True):
    # 1. Interactive Selection (or use Pre-Selection)
    if isinstance(movie_title, tuple) or isinstance(movie_title, list):
        selection = movie_title
    else:
        selection = select_movie_interactively(movie_title, enriched_data, interactive=interactive)
        
    if not selection:
        return []
        
    mode, data1, data2, data3 = (None, None, None, None)
    
    # Unpack based on mode
    if selection[0] == 'LOCAL':
        idx = selection[1]
        found_title = selection[2]
        movie_id = selection[3]
        
        # Robust Lookup by ID
        target_row = enriched_data[enriched_data['movieId'] == movie_id]
        if not target_row.empty:
             target = target_row.iloc[0]
        else:
             # Fallback to Index if ID fails (unlikely)
             target = enriched_data.loc[idx]
             
        print(f"\n   Using Content Source (Local): {found_title}")
    elif selection[0] == 'TMDB':
        metadata = selection[1]
        target = metadata
        idx = -1 # Marker for "New"
        print(f"\n   Using Content Source (TMDB): {metadata['title']}")
    
    # 2. Extract Features (Robustly)
    # Combine TMDB genres and MovieLens genres
    def get_genre_set(row):
        g1 = set(str(row.get('tmdb_genres', '')).split('|')) if pd.notna(row.get('tmdb_genres')) else set()
        g2 = set(str(row.get('genres', '')).split('|')) if pd.notna(row.get('genres')) else set()
        return g1 | g2 - {''} # Union and remove empty
        
    target_genres = get_genre_set(target)
    target_cast = set(str(target['cast']).split('|')) if pd.notna(target['cast']) else set()
    
    print(f"   Features: {len(target_genres)} Genres, {len(target_cast)} Iterating...")

    # Extract Year from title (assuming format "Title (YYYY)")
    import re
    def get_year(t):
        if isinstance(t, str) and re.search(r'\((\d{4})\)', t):
             return int(re.search(r'\((\d{4})\)', t).group(1))
        # Handle simple YYYY string for TMDB
        if isinstance(t, str) and t.isdigit() and len(t) == 4:
            return int(t)
        return 0
        
    # Handle Year difference between Local (Title) and TMDB (Year key)
    if 'year' in target and str(target['year']).isdigit():
        target_year = int(target['year'])
    else:
        target_year = get_year(target.get('title', ''))
    
    # Create copies for scoring
    candidates = enriched_data.copy()
    
    def calculate_score(row):
        # Skip self
        if row.name == idx: return -1
        
        # Genre Overlap (Combined)
        row_genres = get_genre_set(row)
        if not target_genres:
             genre_sim = 0
        else:
             intersection = len(target_genres & row_genres)
             union = len(target_genres | row_genres)
             genre_sim = intersection / union if union > 0 else 0
             
        # Cast Overlap (Weighted Heavily)
        row_cast = set(str(row['cast']).split('|')) if pd.notna(row['cast']) else set()
        if not target_cast:
            cast_sim = 0
        else:
            intersection = len(target_cast & row_cast)
            # Boost: 1.0 full point per shared actor (up to 3.0 max?)
            # Or just raw intersection count?
            # Let's use intersection count * 1.5
            cast_sim = intersection * 1.5
            
        # Year Proximity (Decay)
        row_year = get_year(row['title'])
        if target_year and row_year:
            diff = abs(target_year - row_year)
            # Equation: 1 / (1 + diff/5)
            year_sim = 1 / (1 + diff/5)
        else:
            year_sim = 0
            
        # Total Score
        # Weights: Cast (Very High), Genre (Medium), Year (Low)
        return (genre_sim * 1.0) + (cast_sim * 3.0) + (year_sim * 0.5)

    candidates['similarity'] = candidates.apply(calculate_score, axis=1)
    
    # 4. Sort and Return Top 5
    # User requested MINIMAL rating impact for content match.
    # We use rating only as a tiny tie-breaker (e.g. + 0.01 * Rating)
    # This ensures that if Cast/Genre score is identical, the better movie wins,
    # but a worse movie with BETTER cast/genre match will ALWAYS win.
    candidates['final_score'] = candidates['similarity'] + (candidates['avg_rating'].fillna(0) * 0.01)
    
    results = candidates.sort_values('final_score', ascending=False).head(5)
    
    # Debug print top match details
    # top = results.iloc[0]
    # print(f"   Top Match Debug: {top['title']} - Sim: {top['similarity']:.2f}")
    
    return results[['title', 'similarity', 'avg_rating']].to_dict('records')

def recommend_best_movies_by_actor(actor_name, enriched_data):
    """
    Search for an actor and return their Top 5 highest rated movies.
    """
    # print(f"\n🎭 Searching for movies starring '{actor_name}'...")
    if enriched_data.empty or 'cast' not in enriched_data.columns:
         return []

    # Filter
    matches = enriched_data[enriched_data['cast'].str.contains(actor_name, case=False, na=False)].copy()
    
    if matches.empty:
        return []
        
    # Sort by Rating
    matches = matches.sort_values('avg_rating', ascending=False)
    
    # Return Top 5 tuples (Title, Rating, Year)
    # Extract year if possible from title
    return matches.head(5)[['title', 'avg_rating']].to_dict('records')

def interactive_loop():
    movies, matrix_sparse, movie_idx_to_id, enriched = load_data()
    
    # Save for App
    save_recommender(matrix_sparse, movie_idx_to_id, movies)
    
    # Fit KNN Model
    print("🧠 Training KNN model for Collaborative Filtering...")
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(matrix_sparse)
    
    while True:
        print("\n========================================")
        print("🎬 RECOMENDER SYSTEM V2")
        print("1. People Also Watched (Collaborative - Users who liked X also liked Y)")
        print("2. Content Similarity (Find match by Genre, Actor, Time Period)")
        print("3. Actor Filmography (Top 5 Best Rated Movies by an Actor)")
        print("4. Exit")
        choice = input("👉 Select mode (1-4): ").strip()
        
        if choice == '1':
            query = input("   Enter movie title: ")
            print("   🔍 Searching User Patterns...")
            res = recommend_people_also_watched(query, movies, matrix_sparse, movie_idx_to_id, model_knn)
            
            # Handle Fallback
            if res and isinstance(res, list) and res[0] == "FALLBACK_TO_CONTENT":
                selection_data = res[1]
                # Extract title for display
                display_title = "Unknown"
                if selection_data[0] == 'LOCAL': display_title = selection_data[2]
                elif selection_data[0] == 'TMDB': display_title = selection_data[1]['title']
                
                print(f"   🔄 Automatically keeping you in the loop: Analyzing metadata for '{display_title}'...")
                res_content = recommend_similar_content(selection_data, enriched)
                
                print(f"\n✅ Users haven't rated this much, but here are similar movies by Cast/Genre:")
                for r in res_content:
                    print(f"   - {r['title']} (Rating: {r['avg_rating']:.1f})")
            else:
                print(f"\n✅ Users who watched '{query}' also watched:")
                for r in res: print(f"   - {r}")
            
        elif choice == '2':
            query = input("   Enter movie title to match style: ")
            print("   🔍 Analyzing Metadata (Genre, Cast, Year)...")
            res = recommend_similar_content(query, enriched)
            print(f"\n✅ Movies similar to '{query}' (Content-Based):")
            for r in res:
                print(f"   - {r['title']} (Rating: {r['avg_rating']:.1f})")

        elif choice == '3':
            query = input("   Enter actor name: ")
            print(f"   🔍 Finding best movies for {query}...")
            res = recommend_best_movies_by_actor(query, enriched)
            if not res:
                print("   ⚠️ User not found or no enriched data.")
            else:
                print(f"\n✅ Top 5 Best Rated Movies for '{query}':")
                for r in res:
                    print(f"   - {r['title']} (Rating: {r['avg_rating']:.1f})")
                    
        elif choice == '4':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    interactive_loop()
