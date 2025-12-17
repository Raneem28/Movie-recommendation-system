# time_series_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_trends():
    print("📂 Loading data for Enhanced Time Series Analysis...")
    
    # 1. Load Movies & Ratings (Base)
    movies = pd.read_csv('ml-32m-split/movies.csv')
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    movies = movies[(movies['year'] >= 1920) & (movies['year'] <= 2025)] # Filter relevant era
    
    # 2. Load Ratings Count (Popularity)
    if os.path.exists('ml-32m-split/train_ratings.csv'):
        print("   Loading ratings...")
        ratings = pd.read_csv('ml-32m-split/train_ratings.csv', usecols=['movieId'])
        pop = ratings['movieId'].value_counts().reset_index()
        pop.columns = ['movieId', 'rating_count']
        movies = movies.merge(pop, on='movieId', how='left')
    else:
        movies['rating_count'] = 0
        
    # 3. Load TMDB Enriched Data (Runtime, Cast)
    enriched_path = 'ml-32m-split/movies_enriched.csv'
    if os.path.exists(enriched_path):
        print("   Loading enriched TMDB data...")
        enriched = pd.read_csv(enriched_path, usecols=['movieId', 'runtime', 'cast'])
        
        # Calculate Cast Size
        enriched['cast'] = enriched['cast'].fillna('').astype(str)
        enriched['cast_size'] = enriched['cast'].apply(lambda x: len(x.split('|')) if x and x != 'nan' else 0)
        
        movies = movies.merge(enriched, on='movieId', how='inner') # Inner join to analyze only what we have data for
    else:
        print("⚠️ Enriched data missing. Cannot analyze Runtime/Cast trends.")
        return

    # 4. Group by Year
    print(f"📊 Analyzing {len(movies)} movies...")
    trends = movies.groupby('year').agg({
        'rating_count': 'sum',
        'runtime': 'mean',
        'cast_size': 'mean',
        'movieId': 'count' # Number of movies released
    }).reset_index()
    
    # Smooth lines for better visuals
    trends['runtime_smooth'] = trends['runtime'].rolling(window=5, min_periods=1).mean()
    trends['cast_smooth'] = trends['cast_size'].rolling(window=5, min_periods=1).mean()

    # 5. Plotting
    print("🎨 Generating Enhanced Time Series Plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    sns.set_style("whitegrid")
    
    # Plot 1: Popularity (Total Ratings)
    sns.lineplot(data=trends, x='year', y='rating_count', ax=axes[0], color='#3498db', linewidth=2)
    axes[0].set_title('Movie Popularity (Total Ratings per Year)', fontsize=14)
    axes[0].set_ylabel('Total Ratings')
    
    # Plot 2: Average Runtime
    sns.lineplot(data=trends, x='year', y='runtime', ax=axes[1], color='#e74c3c', alpha=0.3, label='Raw')
    sns.lineplot(data=trends, x='year', y='runtime_smooth', ax=axes[1], color='#c0392b', linewidth=2, label='5-Year Avg')
    axes[1].set_title('Evolution of Movie Length (Average Runtime)', fontsize=14)
    axes[1].set_ylabel('Minutes')
    axes[1].legend()
    
    # Plot 3: Cast Size
    sns.lineplot(data=trends, x='year', y='cast_size', ax=axes[2], color='#2ecc71', alpha=0.3, label='Raw')
    sns.lineplot(data=trends, x='year', y='cast_smooth', ax=axes[2], color='#27ae60', linewidth=2, label='5-Year Avg')
    axes[2].set_title('Evolution of Ensemble Scale (Average Cast Size)', fontsize=14)
    axes[2].set_ylabel('Actors')
    axes[2].set_xlabel('Release Year', fontsize=12)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('time_series_enriched.png', dpi=300)
    print(f"✅ Saved plot to 'time_series_enriched.png'")

if __name__ == "__main__":
    analyze_trends()
