
import sys
import os
from fastapi.testclient import TestClient

# Add root to path
sys.path.append(os.getcwd())

from api.main import app
client = TestClient(app)

def test_disambiguation_flow():
    print("Testing Disambiguation Flow...")
    
    # Step 1: Search Candidates
    query = "Sikandar"
    print(f"1. Searching candidates for '{query}'...")
    res = client.post("/search/candidates", json={"query": query})
    assert res.status_code == 200
    candidates = res.json()["candidates"]
    print(f"   Found {len(candidates)} candidates.")
    
    # Find a TMDB one
    tmdb_cand = next((c for c in candidates if c['source'] == 'TMDB'), None)
    if tmdb_cand:
        print(f"   Selecting TMDB Candidate: {tmdb_cand['title']} ({tmdb_cand['year']})")
        
        # Step 2: Recommend using Selection
        payload = {
            "query": query, 
            "type": "movie",
            "selection": tmdb_cand
        }
        res_rec = client.post("/recommend", json=payload)
        assert res_rec.status_code == 200
        recs = res_rec.json()["results"]
        print(f"   ✅ Recommendations received: {len(recs)}")
        if recs: print(f"      Top: {recs[0]}")
    else:
        print("   ⚠️ No TMDB candidate found to test flow.")

if __name__ == "__main__":
    test_disambiguation_flow()
