import os
import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import joblib

# Ensure path context
sys.path.append(os.getcwd())

def create_lite_model():
    print("🚀 Creating Lite Model for Cloud Deployment...")
    
    # 1. Load Data (Movies + Ratings)
    # We need to construct the matrix from scratch for the subset
    print("   Loading raw data...")
    try:
        movies = pd.read_csv('ml-32m-split/movies.csv')
        ratings = pd.read_csv('ml-32m-split/train_ratings.csv', usecols=['userId', 'movieId', 'rating'])
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # 2. Identify Top 10,000 Movies
    print("   Identifying Top 10k Popular Movies...")
    movie_counts = ratings['movieId'].value_counts()
    top_movie_ids = movie_counts.head(10000).index.tolist()
    
    # Filter Ratings to only these movies
    ratings_lite = ratings[ratings['movieId'].isin(top_movie_ids)].copy()
    movies_lite = movies[movies['movieId'].isin(top_movie_ids)].copy()
    
    print(f"   Reduced Matrix: {len(movies_lite)} movies, {len(ratings_lite)} ratings")
    
    # 3. Create Matrix
    print("   Building Sparse Matrix...")
    user_ids = ratings_lite['userId'].astype('category')
    movie_ids = ratings_lite['movieId'].astype('category')
    
    row_ind = movie_ids.cat.codes
    col_ind = user_ids.cat.codes
    data_vals = ratings_lite['rating'].values
    
    n_movies = len(movie_ids.cat.categories)
    n_users = len(user_ids.cat.categories)
    
    matrix_lite = csr_matrix((data_vals, (row_ind, col_ind)), shape=(n_movies, n_users))
    
    # Mapper
    movie_idx_to_id = dict(enumerate(movie_ids.cat.categories))
    
    # 4. Save
    output_path = 'models/saved_recommender_lite.pkl'
    data = {
        'matrix': matrix_lite,
        'mapper': movie_idx_to_id,
        'movies': movies_lite[['movieId', 'title']]
    }
    
    joblib.dump(data, output_path)
    print(f"   ✅ Saved Lite Model to {output_path}")
    print(f"   File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_lite_model()
