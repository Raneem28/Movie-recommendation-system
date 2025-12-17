
import sys
import os
from fastapi.testclient import TestClient

# Add root to path
sys.path.append(os.getcwd())

from api.main import app
client = TestClient(app)

def test_recommend_enrichment():
    print("Testing /recommend enrichment (Collaborative)...")
    # Use a popular movie that definitely triggers collaborative (no fallback)
    payload = {"query": "Inception", "type": "movie"}
    response = client.post("/recommend", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.text}")
    else:
        data = response.json()
        print(f"✅ Status 200. Results found: {len(data['results'])}")
        for r in data['results']:
            print(f"   item: {r}")
            # Check structure
            if isinstance(r, dict) and 'avg_rating' in r and r['avg_rating'] != 0.0:
                print("     ✅ Has rating!")
            else:
                print("     ⚠️ Missing rating or zero.")

if __name__ == "__main__":
    test_recommend_enrichment()
