import sys
import os
from fastapi.testclient import TestClient

# Add root to path
sys.path.append(os.getcwd())

# Import app
try:
    from api.main import app
except ImportError as e:
    print(f"FAILED to import app: {e}")
    sys.exit(1)

client = TestClient(app)

def test_home():
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200
    assert "MovieAI" in response.text
    print("✅ Home Page OK")

def test_classifier_page():
    print("Testing GET /classifier ...")
    response = client.get("/classifier")
    assert response.status_code == 200
    assert "Hit Predictor" in response.text
    print("✅ Classifier Page OK")

def test_recommender_page():
    print("Testing GET /recommender ...")
    response = client.get("/recommender")
    assert response.status_code == 200
    assert "Smart Recommender" in response.text
    print("✅ Recommender Page OK")

def test_prediction_api():
    print("Testing POST /predict/hit-flop ...")
    payload = {
        "title": "Test Movie",
        "year": 2025,
        "runtime": 120,
        "genres": ["Action", "Sci-Fi"],
        "cast_size": 20,
        "user_rating": 8.0
    }
    response = client.post("/predict/hit-flop", json=payload)
    if response.status_code != 200:
        print(f"❌ Prediction Failed: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    print(f"✅ Prediction OK: {data['prediction']} ({data['confidence']}%)")

def test_recommend_api():
    print("Testing POST /recommend ...")
    # Using a known movie usually? Or strict search?
    # Recommendation logic might fail if models not loaded properly
    # api/main try-excepts model loading, so app starts even if models missing.
    # We'll check if it 200s even if empty results.
    payload = {"query": "Iron Man", "type": "movie"}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    print(f"✅ Recommend API OK (Results: {len(data['results'])})")

if __name__ == "__main__":
    try:
        test_home()
        test_classifier_page()
        test_recommender_page()
        test_prediction_api()
        test_recommend_api()
        print("\n🎉 ALL TESTS PASSED!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)
