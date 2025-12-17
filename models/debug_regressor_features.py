import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from models.rating_regressor import load_data, prepare_features, get_actor_scores

def debug_pipeline():
    print("DEBUG: Analyzing Regressor Feature Engineering...")
    
    # 1. Load Data
    movies, train_ratings, test_ratings = load_data()
    
    # 2. Run Feature Engineering (Train)
    train_df, genre_cols, actor_scores, global_avg = prepare_features(movies, train_ratings, "TRAIN")
    
    # Redirect critical output to detailed log file
    with open('internal_debug_log.txt', 'w', encoding='utf-8') as f:
        f.write("DEBUG ANALYSIS REPORT\n")
        f.write("=====================\n")
        
        # 3. Inspect Actor Scores
        f.write(f"\nACTOR SCORES:\n")
        f.write(f"   Count: {len(actor_scores)}\n")
        f.write(f"   Shape of dict: {list(actor_scores.items())[:5]}\n")
        
        # 4. Inspect Cast Potential Column
        col = 'cast_rating_potential'
        if col not in train_df.columns:
            f.write(f"CRITICAL: Column '{col}' is MISSING from train_df!\n")
            return
            
        vals = train_df[col]
        f.write(f"\n'{col}' STATISTICS:\n")
        f.write(f"   Mean: {vals.mean():.4f}\n")
        f.write(f"   Std : {vals.std():.4f}\n")
        f.write(f"   Min : {vals.min():.4f}\n")
        f.write(f"   Max : {vals.max():.4f}\n")
        
        # Check for Constant Values
        n_global = (vals == global_avg).sum()
        pct_global = (n_global / len(vals)) * 100
        f.write(f"   Rows == Global Avg ({global_avg:.4f}): {n_global} ({pct_global:.2f}%)\n")
        
        if pct_global > 90:
            f.write("WARNING: Feature is practically constant! Target Encoding failed.\n")
            
        # 5. Inspect Correlation
        corr = train_df[[col, 'avg_rating']].corr().iloc[0,1]
        f.write(f"\nCorrelation with Target (avg_rating): {corr:.4f}\n")
        
        # 6. Sample Rows
        f.write("\nSAMPLE ROWS (Feature vs Target):\n")
        sample = train_df[['title', 'cast_rating_potential', 'avg_rating']].head(10)
        f.write(sample.to_string())
        
        # ---------------------------------------------------------
        # PART 2: TEST SET ANALYSIS
        # ---------------------------------------------------------
        f.write("\n\n---------------------\nTEST SET ANALYSIS\n---------------------\n")
        
        # Process Test Set using Learned Scores
        test_df, _, _, _ = prepare_features(movies, test_ratings, "TEST", actor_scores, global_avg)
        
        if col not in test_df.columns:
            f.write(f"CRITICAL: Column '{col}' is MISSING from test_df!\n")
            return
            
        t_vals = test_df[col]
        f.write(f"\n'{col}' (TEST) STATISTICS:\n")
        f.write(f"   Mean: {t_vals.mean():.4f}\n")
        f.write(f"   Std : {t_vals.std():.4f}\n")
        
        # Check for Constant Values in Test
        n_global_t = (t_vals == global_avg).sum()
        pct_global_t = (n_global_t / len(t_vals)) * 100
        f.write(f"   Rows == Global Avg: {n_global_t} ({pct_global_t:.2f}%)\n")
        
        # Test Correlation
        t_corr = test_df[[col, 'avg_rating']].corr().iloc[0,1]
        f.write(f"\nCorrelation with Target (TEST): {t_corr:.4f}\n")
        
        # Overlap Check
        f.write("\nGENERALIZATION CHECK:\n")
        # Check how many test movies had at least one scored actor
        def has_scored_actor(row):
            if 'cast_list' not in movies.columns: return False
            # This check is hard because we don't have cast_list in test_df anymore
            # But if value != global_avg, it had a score.
            return row[col] != global_avg
            
        scored_count = (t_vals != global_avg).sum()
        f.write(f"   Test Movies with Scored Cast: {scored_count} / {len(t_vals)} ({(scored_count/len(t_vals))*100:.2f}%)\n")
        
    print("✅ Debug log written to internal_debug_log.txt")

if __name__ == "__main__":
    debug_pipeline()
