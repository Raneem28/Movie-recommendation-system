"""
📊 DATASET TRAIN-TEST SPLIT SCRIPT
Purpose: Split MovieLens dataset into training and testing sets
Dataset: MovieLens 32M
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("="*70)
print("📊 MOVIELENS DATASET SPLITTER")
print("="*70)

# ============================================
# STEP 1: LOAD ORIGINAL DATA
# ============================================

print("\n🔄 STEP 1: Loading original dataset...")

movies = pd.read_csv('ml-32m/movies.csv')
ratings = pd.read_csv('ml-32m/ratings.csv')

print(f"✅ Loaded {len(movies):,} movies")
print(f"✅ Loaded {len(ratings):,} ratings")
print(f"✅ Total users: {ratings['userId'].nunique():,}")

# Display basic statistics
print(f"\n📊 Dataset Statistics:")
print(f"   - Movies: {len(movies):,}")
print(f"   - Ratings: {len(ratings):,}")
print(f"   - Users: {ratings['userId'].nunique():,}")
print(f"   - Avg ratings per movie: {len(ratings)/len(movies):.1f}")
print(f"   - Avg ratings per user: {len(ratings)/ratings['userId'].nunique():.1f}")
print(f"   - Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")

# ============================================
# STEP 2: SPLIT RATINGS (80-20)
# ============================================

print("\n🔪 STEP 2: Splitting ratings into Train (80%) and Test (20%)...")

# Split ratings
train_ratings, test_ratings = train_test_split(
    ratings, 
    test_size=0.2, 
    random_state=42,
    shuffle=True  # Shuffle to ensure random distribution
)

print(f"✅ Training set: {len(train_ratings):,} ratings ({len(train_ratings)/len(ratings)*100:.1f}%)")
print(f"✅ Testing set: {len(test_ratings):,} ratings ({len(test_ratings)/len(ratings)*100:.1f}%)")

# ============================================
# STEP 3: VERIFY SPLIT QUALITY
# ============================================

print("\n🔍 STEP 3: Verifying split quality...")

# Check rating distribution
train_avg = train_ratings['rating'].mean()
test_avg = test_ratings['rating'].mean()
print(f"   - Training avg rating: {train_avg:.3f}")
print(f"   - Testing avg rating: {test_avg:.3f}")
print(f"   - Difference: {abs(train_avg - test_avg):.3f} ✅")

# Check user distribution
train_users = train_ratings['userId'].nunique()
test_users = test_ratings['userId'].nunique()
overlap_users = len(set(train_ratings['userId']) & set(test_ratings['userId']))

print(f"   - Users in training: {train_users:,}")
print(f"   - Users in testing: {test_users:,}")
print(f"   - Users in both: {overlap_users:,} ({overlap_users/ratings['userId'].nunique()*100:.1f}%)")

# Check movie distribution
train_movies = train_ratings['movieId'].nunique()
test_movies = test_ratings['movieId'].nunique()
overlap_movies = len(set(train_ratings['movieId']) & set(test_ratings['movieId']))

print(f"   - Movies in training: {train_movies:,}")
print(f"   - Movies in testing: {test_movies:,}")
print(f"   - Movies in both: {overlap_movies:,} ({overlap_movies/movies['movieId'].nunique()*100:.1f}%)")

# ============================================
# STEP 4: CREATE OUTPUT DIRECTORY
# ============================================

print("\n📁 STEP 4: Creating output directory...")

output_dir = 'ml-32m-split'
os.makedirs(output_dir, exist_ok=True)
print(f"✅ Directory created: {output_dir}/")

# ============================================
# STEP 5: SAVE SPLIT DATASETS
# ============================================

print("\n💾 STEP 5: Saving split datasets...")

# Save movies (same for both train and test)
movies.to_csv(f'{output_dir}/movies.csv', index=False)
print(f"✅ Saved: {output_dir}/movies.csv ({len(movies):,} rows)")

# Save training ratings
train_ratings.to_csv(f'{output_dir}/train_ratings.csv', index=False)
print(f"✅ Saved: {output_dir}/train_ratings.csv ({len(train_ratings):,} rows)")

# Save testing ratings
test_ratings.to_csv(f'{output_dir}/test_ratings.csv', index=False)
print(f"✅ Saved: {output_dir}/test_ratings.csv ({len(test_ratings):,} rows)")

