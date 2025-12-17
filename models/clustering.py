# clustering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_hybrid_data():
    print("📂 Loading Hybrid Dataset (MovieLens + TMDB)...")
    
    # 1. Load Full MovieLens
    movies = pd.read_csv('ml-32m-split/movies.csv')
    
    # 2. Extract Year
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    
    # 3. Load Rating Stats
    if os.path.exists('ml-32m-split/train_ratings.csv'):
        # Optimize loading
        ratings = pd.read_csv('ml-32m-split/train_ratings.csv', usecols=['movieId', 'rating'])
        stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
        stats.columns = ['movieId', 'avg_rating', 'rating_count']
        movies = movies.merge(stats, on='movieId', how='left')
    else:
        movies['avg_rating'] = 3.0
        movies['rating_count'] = 0
        
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['rating_count'] = movies['rating_count'].fillna(0)
    
    # 4. Load TMDB Enriched Data
    enriched_path = 'ml-32m-split/movies_enriched.csv'
    if os.path.exists(enriched_path):
        print(f"   Merging with enriched TMDB data from '{enriched_path}'...")
        enriched = pd.read_csv(enriched_path, usecols=['movieId', 'runtime', 'cast', 'tmdb_genres'])
        movies = movies.merge(enriched, on='movieId', how='left')
        
        # Fill missing numeric
        movies['runtime'] = movies['runtime'].fillna(movies['runtime'].median())
        
        # Process Cast Size
        movies['cast'] = movies['cast'].fillna('').astype(str)
        movies['cast_size'] = movies['cast'].apply(lambda x: len(x.split('|')) if x and x != 'nan' else 0)
        
        # Merge Genres (Union of ML and TMDB)
        movies['tmdb_genres'] = movies['tmdb_genres'].fillna('')
        movies['all_genres'] = movies['genres'] + '|' + movies['tmdb_genres']
        
        movies['is_enriched'] = movies['runtime'].notna().astype(int)
    else:
        print("⚠️ Enriched data not found. Using basics.")
        movies['runtime'] = 90.0
        movies['cast_size'] = 0
        movies['all_genres'] = movies['genres']
        movies['is_enriched'] = 0

    return movies

def perform_clustering(movies):
    print(f"🧠 Engineering features for {len(movies)} movies...")
    
    # Feature 1: Unified Genres
    # Remove duplicates in pipe strings? get_dummies handles overlap automatically (A|B + B|C -> A=1, B=1, C=1)
    genres_dummies = movies['all_genres'].str.get_dummies(sep='|')
    # Filter rare genres to reduce noise? (Optional, but K-Means struggles with high dim. PCA helps.)
    
    # Feature 2: Numerical Stats (Added Runtime, Cast Size)
    movies['log_count'] = np.log1p(movies['rating_count'])
    
    # Normalize Year (1900-2025 -> 0-1)
    
    numeric_features = movies[['year', 'avg_rating', 'log_count', 'runtime', 'cast_size']]
    
    # Combine
    X = pd.concat([numeric_features, genres_dummies], axis=1)
    X = X.fillna(0)
    
    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    n_clusters = 8
    print(f"🔄 Running K-Means (K={n_clusters})...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    movies['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualization (PCA)
    print("🎨 Generating 2D visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    movies['pca_1'] = X_pca[:, 0]
    movies['pca_2'] = X_pca[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='pca_1', y='pca_2', 
        hue='cluster', 
        palette='viridis',
        data=movies,
        s=15, alpha=0.6,
        legend='full'
    )
    plt.title(f"Hybrid Movie Clusters (ML + TMDB Features)\nN={len(movies)}")
    plt.savefig('clusters.png', dpi=300)
    print("✅ Saved plot to 'clusters.png'")
    
    # Interpret Clusters
    print("\n🔍 Interpreting Clusters (Top Examples):")
    for i in range(n_clusters):
        cluster_data = movies[movies['cluster'] == i]
        size = len(cluster_data)
        avg_rtg = cluster_data['avg_rating'].mean()
        avg_rt = cluster_data['runtime'].mean()
        
        # Top Genre
        genre_counts = genres_dummies.loc[cluster_data.index].sum().sort_values(ascending=False)
        top_genre = genre_counts.index[0] if not genre_counts.empty else "N/A"
        
        print(f"\n📁 Cluster {i}: {top_genre}")
        print(f"   Stats: Size={size:,} | Rating={avg_rtg:.1f} | Runtime={avg_rt:.0f} min")
        
        top_examples = cluster_data.sort_values('rating_count', ascending=False).head(4)['title'].tolist()
        print(f"   Hits: {', '.join(top_examples)}")

    return movies

if __name__ == "__main__":
    df = load_hybrid_data()
    clustered_df = perform_clustering(df)
    clustered_df.to_csv('movies_clustered.csv', index=False)
