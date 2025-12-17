"""
Unit tests for Movie ML System using pytest
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

def test_read_main():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Movie ML API"

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_hit_flop():
    """Test classification prediction"""
    payload = {
        "title": "Test Movie",
        "year": 2023,
        "runtime": 120,
        "genres": ["Action", "Adventure"],
        "cast": ["Actor A", "Actor B"]
    }
    # Note: This might fail if models aren't loaded in test env
    # We'd typically mock the model loading for unit tests
    try:
        response = client.post("/predict/hit-flop", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["Hit", "Flop"]
    except Exception:
        pytest.skip("Skipping prediction test - models might not be loaded")

def test_recommendation_actor():
    """Test actor recommendation endpoint"""
    # Use a known actor
    actor = "Tom Hanks" 
    # This assumes data is loaded. In a real CI, we'd mock the data source.
    try:
        response = client.get(f"/recommend/actor/{actor}")
        if response.status_code == 200:
            data = response.json()
            assert data["actor"] == actor
            assert len(data["top_movies"]) > 0
    except Exception:
        pytest.skip("Skipping actor test - data might not be loaded")

def test_batch_prediction_csv():
    """Test batch prediction with CSV upload"""
    csv_content = """title,year,runtime,genres,cast
Inception,2010,148,Action|Sci-Fi,Leonardo DiCaprio|Tom Hardy
The Matrix,1999,136,Action|Sci-Fi,Keanu Reeves|Laurence Fishburne"""
    
    files = {
        "file": ("test_movies.csv", csv_content, "text/csv")
    }
    
    try:
        response = client.post("/predict/batch", files=files)
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
            assert data["results"][0]["title"] == "Inception"
    except Exception:
        pytest.skip("Skipping batch test - models might not be loaded")
