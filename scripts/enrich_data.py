# enrich_data.py
import pandas as pd
import sys
import os
import time
import csv
import re
import concurrent.futures
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_enrichment.tmdb_client import search_movie, fetch_movie_details, extract_metadata

OUTPUT_FILE = 'ml-32m-split/movies_enriched.csv'
CSV_LOCK = threading.Lock()
COUNTER_LOCK = threading.Lock()
GLOBAL_COUNT = 0
MAX_WORKERS = 50  # MAX SPEED (Warning: High API usage)

def load_existing_enriched_ids():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    
    try:
        # Read just the movieId column
        df = pd.read_csv(OUTPUT_FILE, usecols=['movieId'])
        return set(df['movieId'].unique())
    except Exception:
        return set()

def initialize_file_if_needed():
    # If file doesn't exist, create it with header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['movieId', 'title', 'genres', 'runtime', 'cast', 'tmdb_genres'])
        print(f"📄 Created new file: {OUTPUT_FILE}")

def save_single_row(row_data):
    """Thread-safe save"""
    with CSV_LOCK:
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                row_data['movieId'], 
                row_data['title'], 
                row_data['genres'], 
                row_data['runtime'], 
                row_data['cast'], 
                row_data['tmdb_genres']
            ])

def process_movie(row):
    """Worker function for a single movie"""
    title = row['title']
    movie_id = row['movieId']
    
    # 1. Clean Title & Year
    year_match = re.search(r'\((\d{4})\)', title)
    year = year_match.group(1) if year_match else None
    clean_title = re.sub(r'\(\d{4}\).*', '', title).strip()
    
    try:
        # 2. Search
        search_result = search_movie(clean_title, year=year)
        
        if search_result:
            tmdb_id = search_result['id']
            details = fetch_movie_details(tmdb_id)
            runtime, cast, tmdb_genres = extract_metadata(details)
            
            # 3. Prepare Data
            row_data = {
                'movieId': movie_id,
                'title': title,
                'genres': row['genres'],
                'runtime': runtime,
                'cast': "|".join(cast),
                'tmdb_genres': "|".join(tmdb_genres)
            }
            
            # 4. Save immediately (thread-safe)
            save_single_row(row_data)
            
            with COUNTER_LOCK:
                global GLOBAL_COUNT
                GLOBAL_COUNT += 1
                c = GLOBAL_COUNT
                
            print(f"   ✅ [{c}] Enriched: {title}")
            return True
            
        else:
            # Not found
            # print(f"   ⚠️ Not Found: {title}")
            return False
            
    except Exception as e:
        # print(f"   ❌ Error {title}: {e}")
        return False

def enrich_all_movies():
    print(f"🚀 Starting PARALLEL Enrichment (Workers={MAX_WORKERS})...")
    
    # 1. Load All Movies
    print("📂 Loading full movie list...")
    movies = pd.read_csv('ml-32m-split/movies.csv')
    
    # Optional: Prioritize by popularity to ensure useful ones get done first
    if os.path.exists('ml-32m-split/train_ratings.csv'):
        print("   Prioritizing by popularity...")
        ratings = pd.read_csv('ml-32m-split/train_ratings.csv', usecols=['movieId'])
        popularity = ratings['movieId'].value_counts()
        movies['popularity'] = movies['movieId'].map(popularity).fillna(0)
        movies = movies.sort_values('popularity', ascending=False)
    
    # 2. Check what's already done
    done_ids = load_existing_enriched_ids()
    print(f"   Found {len(done_ids)} movies already enriched.")
    
    # 3. Filter TODO list
    movies_to_do = movies[~movies['movieId'].isin(done_ids)]
    print(f"🎯 Movies remaining to enrich: {len(movies_to_do)}")
    
    initialize_file_if_needed()
    
    # Initialize global counter
    global GLOBAL_COUNT
    GLOBAL_COUNT = len(done_ids)
    
    print("\n🌐 Starting Parallel Fetch Loop...")
    
    # 4. Parallel Execution
    rows_to_process = [row for _, row in movies_to_do.iterrows()]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map returns an iterator, but we just need execution. 
        # We can use submit to get futures if we want progress counting.
        futures = {executor.submit(process_movie, row): row for row in rows_to_process}
        
        # Monitor progress
        completed = 0
        total = len(rows_to_process)
        
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                print(f"   📊 Progress: {completed}/{total} movies checked...")

    print("\n✅ Enrichment Run Complete!")

if __name__ == "__main__":
    enrich_all_movies()
