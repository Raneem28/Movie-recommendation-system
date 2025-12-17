"""
Feature preparation utilities for Movie ML Project API.
This module is the 'Source of Truth' for converting a movie's raw data
(from TMDB or manual input) into the exact feature vector expected by the trained models.

It robustly handles:
1. Feature Parity: Dynamic detection of column names (e.g. 'Action' vs 'genre_Action').
2. Missing Data: Safe defaults for runtime, year, etc.
3. Model Variations: Adapts to whatever columns were present during training.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Default lists (fallback only, usually inferred from model)
TOP_GENRES_DEFAULT = [
    'Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 
    'Horror', 'Crime', 'Documentary', 'Adventure', 'Sci-Fi',
    'Mystery', 'Fantasy', 'Animation', 'Family', 'War',
    'Music', 'Western', 'History', 'Musical', 'Sport'
]

TOP_ACTORS_COUNT = 1000 # Standard clamp max

def prepare_classifier_features(
    year: int,
    runtime: int,
    genres: List[str],
    avg_rating: float,
    cast_size: int,
    cast: List[str],
    top_actors_list: List[str] = None,
    expected_cols: List[str] = None
) -> np.ndarray:
    """
    Prepare features for the Hit/Flop Classifier.
    """
    # Initialize basic features
    features = {
        'year': year,
        'runtime': runtime,
        'avg_rating': avg_rating,
        'cast_size': cast_size
    }
    
    # --- Smart Feature Construction ---
    
    # 1. Genres
    # We check expected_cols to see how the model wants genres named via "Smart Logic"
    if expected_cols:
        matched_genres = 0
        for known_genre in TOP_GENRES_DEFAULT:
            # Try 'genre_Action' format
            col_name = f"genre_{known_genre}"
            if col_name not in expected_cols:
                # Try raw 'Action' format
                if known_genre in expected_cols:
                    col_name = known_genre
                else:
                    # Genre not tracked by model
                    continue
            
            # Set 1 if movie has this genre, else 0
            features[col_name] = 1 if known_genre in genres else 0
            if known_genre in genres: matched_genres += 1
    else:
        # Legacy/Fallback behavior
        for g in TOP_GENRES_DEFAULT:
            features[f"genre_{g}"] = 1 if g in genres else 0

    # 2. Actors (Stars)
    # Strategy: Iterate through expected columns and find "star_" ones
    if expected_cols:
        for col in expected_cols:
            if col.startswith("star_"):
                # Decode actor name: 'star_Tom_Hanks' -> 'Tom Hanks'
                # Note: The training script usually replaces space with underscore
                actor_name = col[5:].replace('_', ' ')
                
                # Check if this actor is in the movie's cast
                if actor_name in cast:
                    features[col] = 1
                else:
                    features[col] = 0
    elif top_actors_list:
        # Fallback if expected_cols missing but we have the top actors list
        for actor in top_actors_list[:TOP_ACTORS_COUNT]:
            features[f"star_{actor}"] = 1 if actor in cast else 0
            
    # --- Final DataFrame & Reindexing ---
    df = pd.DataFrame([features])
    
    if expected_cols:
        # Crucial: Reindex forces the DataFrame to match the model's structure exactly.
        # 1. Fills missing known columns with 0.
        # 2. Drops extra columns not in the model.
        # 3. Reorders columns to match training order.
        df = df.reindex(columns=expected_cols, fill_value=0)
    
    return df.values


def prepare_regressor_features(
    year: int,
    runtime: int,
    genres: List[str],
    cast: List[str],
    top_actors_list: List[str] = None,
    expected_cols: List[str] = None
) -> np.ndarray:
    """
    Prepare features for the Rating Regressor.
    """
    features = {
        'year': year,
        'runtime': runtime,
    }
    
    # --- Smart Feature Construction ---
    
    # 1. Genres
    if expected_cols:
        for known_genre in TOP_GENRES_DEFAULT:
            # Try 'genre_Action' vs 'Action'
            col_name = f"genre_{known_genre}"
            if col_name not in expected_cols:
                if known_genre in expected_cols:
                    col_name = known_genre
                else:
                    continue
            
            features[col_name] = 1 if known_genre in genres else 0
    else:
        for g in TOP_GENRES_DEFAULT:
            features[f"genre_{g}"] = 1 if g in genres else 0
            
    # 2. Actors (Stars)
    if expected_cols:
        for col in expected_cols:
            if col.startswith("star_"):
                actor_name = col[5:].replace('_', ' ')
                if actor_name in cast:
                    features[col] = 1
                else:
                    features[col] = 0
    elif top_actors_list:
        for actor in top_actors_list[:TOP_ACTORS_COUNT]:
            features[f"star_{actor}"] = 1 if actor in cast else 0
            
    # --- Final DataFrame ---
    df = pd.DataFrame([features])
    
    if expected_cols:
        df = df.reindex(columns=expected_cols, fill_value=0)
        
    return df.values

def calculate_cast_size(cast: List[str]) -> int:
    return len(cast) if cast else 0
