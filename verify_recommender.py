
import sys
import os
from fastapi.testclient import TestClient

# Add root to path
sys.path.append(os.getcwd())

from api.main import app
client = TestClient(app)

def test_recommend_tmdb_fallback():
    print("Testing /recommend with TMDB movie (e.g. recent movie)...")
    # "Sikandar" is likely not in local DB, should trigger TMDB search + Content Fallback
    payload = {"query": "Sikandar", "type": "movie"}
    response = client.post("/recommend", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.text}")
    else:
        data = response.json()
        print(f"✅ Status 200. Results found: {len(data['results'])}")
        if len(data['results']) > 0:
            print(f"   Sample: {data['results'][0]}")

if __name__ == "__main__":
    test_recommend_tmdb_fallback()
