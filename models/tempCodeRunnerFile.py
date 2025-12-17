import glob
import pandas as pd

all_chunks = glob.glob("temp/movie_stats_chunk_*.csv")
dfs = [pd.read_csv(f) for f in all_chunks]

# Combine all chunks and compute final statistics
final_stats = pd.concat(dfs)
final_stats = final_stats.groupby("movie_id").agg({"mean":"mean", "count":"sum"}).reset_index()
final_stats.rename(columns={"mean":"avg_rating","count":"num_ratings"}, inplace=True)
final_stats.to_csv("data/processed/movie_stats.csv", index=False)
