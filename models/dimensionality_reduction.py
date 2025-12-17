# dimensionality_reduction.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prep_data():
    print("📂 Loading data...")
    # 1. Load Enriched Movies
    enriched_path = 'ml-32m-split/movies_enriched.csv'
    if not os.path.exists(enriched_path):
        print(f"❌ '{enriched_path}' not found. Please run enrich_data.py first.")
        return None
    
    movies = pd.read_csv(enriched_path)
    
    # 2. Extract Year
    # "Toy Story (1995)" -> 1995
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())

    # 3. Calculate Ratings Features (Avg Rating, Count)
    print("   Calculating rating stats...")
    ratings_path = 'ml-32m-split/train_ratings.csv'
    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path, usecols=['movieId', 'rating'])
        stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
        stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Merge
        movies = movies.merge(stats, on='movieId', how='left')
    else:
        print("⚠️ Ratings file not found. Simulating data for demo.")
        movies['avg_rating'] = 3.0
        movies['rating_count'] = 0

    # Fill NaNs
    movies['avg_rating'] = movies['avg_rating'].fillna(3.0)
    movies['rating_count'] = movies['rating_count'].fillna(0)
    movies['runtime'] = movies['runtime'].fillna(movies['runtime'].median())
    
    return movies

def run_dr(movies):
    print("🧠 Performing Dimensionality Reduction...")
    
    # Log-transform rating_count to handle extreme values (0 to 75000+)
    movies['log_rating_count'] = np.log1p(movies['rating_count'])
    
    # Extract and one-hot encode genres for PCA
    # This gives PCA categorical features to work with
    print("   Preparing genre features...")
    
    # Combine MovieLens and TMDB genres
    def get_all_genres(row):
        g1 = str(row.get('genres', '')).split('|') if pd.notna(row.get('genres')) else []
        g2 = str(row.get('tmdb_genres', '')).split('|') if pd.notna(row.get('tmdb_genres')) else []
        return list(set(g1 + g2) - {''})
    
    movies['all_genres'] = movies.apply(get_all_genres, axis=1)
    
    # Get top genres (appearing in at least 100 movies)
    from collections import Counter
    all_genre_list = [g for genres in movies['all_genres'] for g in genres]
    genre_counts = Counter(all_genre_list)
    top_genres = [g for g, count in genre_counts.most_common(20) if count >= 100]
    
    print(f"   Using top {len(top_genres)} genres: {', '.join(top_genres[:5])}...")
    
    # One-hot encode genres
    for genre in top_genres:
        movies[f'genre_{genre}'] = movies['all_genres'].apply(lambda x: 1 if genre in x else 0)
    
    # Combine numeric and genre features
    numeric_features = ['avg_rating', 'log_rating_count', 'runtime', 'year']
    genre_features = [f'genre_{g}' for g in top_genres]
    features = numeric_features + genre_features
    
    print(f"   Total features for PCA: {len(features)} ({len(numeric_features)} numeric + {len(genre_features)} genre)")
    
    X = movies[features]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. PCA (Fast - run on full dataset)
    print("   Running PCA on full dataset...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    movies['pca_1'] = X_pca[:, 0]
    movies['pca_2'] = X_pca[:, 1]
    
    print(f"   PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%} = {sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. t-SNE (Slow - sample for performance)
    # t-SNE has O(n²) complexity, so we sample for visualization
    MAX_TSNE_SAMPLES = 3000
    
    if len(movies) > MAX_TSNE_SAMPLES:
        print(f"   ⚡ Sampling {MAX_TSNE_SAMPLES} movies for t-SNE (dataset has {len(movies)} total)...")
        # Stratified sampling: prioritize movies with more ratings
        # Sort by rating_count and take top N/2 + random N/2
        movies_sorted = movies.sort_values('rating_count', ascending=False)
        top_rated = movies_sorted.head(MAX_TSNE_SAMPLES // 2)
        remaining = movies_sorted.iloc[MAX_TSNE_SAMPLES // 2:]
        random_sample = remaining.sample(n=min(MAX_TSNE_SAMPLES // 2, len(remaining)), random_state=42)
        sample_movies = pd.concat([top_rated, random_sample])
        
        # Get indices for sampling
        sample_indices = sample_movies.index
        X_tsne_input = X_scaled[sample_indices]
    else:
        print(f"   Running t-SNE on full dataset ({len(movies)} movies)...")
        sample_indices = movies.index
        X_tsne_input = X_scaled
    
    print("   Running t-SNE (this may take 30-60 seconds)...")
    perp = min(30, len(sample_indices) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=300, verbose=1)
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    # Initialize with NaN
    movies['tsne_1'] = np.nan
    movies['tsne_2'] = np.nan
    
    # Fill in t-SNE results for sampled movies
    movies.loc[sample_indices, 'tsne_1'] = X_tsne[:, 0]
    movies.loc[sample_indices, 'tsne_2'] = X_tsne[:, 1]
    
    return movies

def plot_results(movies):
    print("🎨 Plotting results...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot PCA
    scatter1 = ax1.scatter(
        movies['pca_1'], 
        movies['pca_2'],
        c=movies['avg_rating'],
        s=np.log1p(movies['rating_count']) * 3,  # Size based on log of rating count
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3
    )
    ax1.set_xlabel('First Principal Component', fontsize=11)
    ax1.set_ylabel('Second Principal Component', fontsize=11)
    ax1.set_title('PCA: Movie Embeddings (Full Dataset)\nShows linear relationships between features', 
                  fontsize=12, fontweight='bold')
    
    # Add colorbar for PCA
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Average Rating', fontsize=10)
    
    # Plot t-SNE (only sampled movies)
    movies_tsne = movies.dropna(subset=['tsne_1', 'tsne_2'])
    scatter2 = ax2.scatter(
        movies_tsne['tsne_1'],
        movies_tsne['tsne_2'],
        c=movies_tsne['avg_rating'],
        s=np.log1p(movies_tsne['rating_count']) * 3,
        cmap='plasma',  # Different colormap for distinction
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3
    )
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax2.set_title(f't-SNE: Movie Embeddings ({len(movies_tsne):,} sampled)\nReveals non-linear clusters and patterns', 
                  fontsize=12, fontweight='bold')
    
    # Add colorbar for t-SNE
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Average Rating', fontsize=10)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    fig.text(0.5, 0.02, 
             'Point size represents popularity (log of rating count) | Color represents average rating',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_file = 'movie_embeddings_dr.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to '{output_file}'")
    plt.close()

if __name__ == "__main__":
    df = load_and_prep_data()
    if df is not None and not df.empty:
        df_processed = run_dr(df)
        plot_results(df_processed)
