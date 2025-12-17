"""
Final System Verification Script
Run this to verify that your MLOps Movie Project is fully operational.
"""
import requests
import sys
import json

BASE_URL = "http://localhost:8000"

def print_pass(msg):
    print(f"✅ PASS: {msg}")

def print_fail(msg):
    print(f"❌ FAIL: {msg}")
    sys.exit(1)

def verify_system():
    print("🚀 Starting Final System Verification...\n")
    
    # 1. Check Health Endpoint
    try:
        print(f"Testing connectivity to {BASE_URL}...")
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            data = resp.json()
            print_pass(f"API is Online! Status: {data['status']}")
            print(f"   Using models: {list(data.get('models_loaded', {}).keys())}")
        else:
            print_fail(f"API returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print_fail("Could not connect to API. Is Docker running? (Try 'docker-compose up')")

    # 2. Test Prediction (Hit/Flop)
    try:
        payload = {
            "title": "Test Blockbuster",
            "year": 2023,
            "runtime": 150,
            "genres": ["Action", "Sci-Fi"],
            "cast": ["Tom Cruise", "Zendaya"]
        }
        print(f"\nTesting Prediction Model with: {payload['title']}...")
        resp = requests.post(f"{BASE_URL}/predict/hit-flop", json=payload)
        
        if resp.status_code == 200:
            result = resp.json()
            print_pass(f"Prediction successful! Result: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print_fail(f"Prediction failed: {resp.text}")
            
    except Exception as e:
        print_fail(f"Prediction test error: {e}")

    # 3. Test Batch Prediction (New Feature)
    try:
        print("\nTesting Batch Upload (New Feature)...")
        csv_data = "title,year,runtime,genres,cast\nMatrix 4,2021,148,Sci-Fi,Keanu Reeves"
        files = {'file': ('test.csv', csv_data, 'text/csv')}
        
        resp = requests.post(f"{BASE_URL}/predict/batch", files=files)
        if resp.status_code == 200:
            batch_res = resp.json()
            if len(batch_res['results']) > 0:
                print_pass("Batch prediction processing works!")
            else:
                print_fail("Batch response empty")
        else:
            print_fail(f"Batch upload failed: {resp.status_code}")

    except Exception as e:
        print_fail(f"Batch test error: {e}")

    print("\n" + "="*40)
    print("🎉 SYSTEM STATUS: 100% OPERATIONAL")
    print("="*40)
    print("You can verify this by checking Docker Desktop logs.")

if __name__ == "__main__":
    verify_system()
