# models/association_rules.py
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AssociationRules")

def run_association_analysis(min_support=0.05, min_confidence=0.2):
    """
    Run Apriori algo to find associations between genres.
    """
    logger.info("loading data for Association Rules...")
    
    # 1. Load Movies with Genres
    movies_path = 'ml-32m-split/movies.csv'
    if not os.path.exists(movies_path):
        logger.error(f"File not found: {movies_path}")
        return
        
    movies = pd.read_csv(movies_path)
    
    # 2. Prepare Transactions
    # Each movie has a set of genres. We treat each movie as a transaction and genres as items.
    # This tells us: "If a movie is Genre A, it is likely also Genre B"
    logger.info("Preparing transactions...")
    movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
    transactions = movies['genres_list'].tolist()
    
    # 3. One-Hot Encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 4. Apriori
    logger.info(f"Running Apriori (Min Sup: {min_support})...")
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return
        
    # 5. Association Rules
    logger.info(f"Generating Rules (Min Conf: {min_confidence})...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort by Lift (strength of association)
    rules = rules.sort_values('lift', ascending=False)
    
    # Save results
    output_path = 'models/association_rules.pkl'
    # We save a simplified version of rules for the API
    simple_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(50)
    # Convert sets to strings for JSON serialization later
    simple_rules['antecedents'] = simple_rules['antecedents'].apply(lambda x: list(x))
    simple_rules['consequents'] = simple_rules['consequents'].apply(lambda x: list(x))
    
    joblib.dump(simple_rules, output_path)
    logger.info(f"✅ Saved {len(rules)} rules to '{output_path}'")
    
    # Print examples
    print("\n🔍 Top Association Rules (Genre Correlations):")
    for idx, row in rules.head(5).iterrows():
        ant = list(row['antecedents'])[0]
        con = list(row['consequents'])[0]
        print(f"   If movie is '{ant}' -> It is likely '{con}' (Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

if __name__ == "__main__":
    run_association_analysis()
