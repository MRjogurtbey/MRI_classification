"""
Quick API test script
Run this after starting the API server with: python api.py
"""

import requests
import sys
from pathlib import Path

def test_api():
    """Test FastAPI endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("="*70)
    print("FASTAPI ENDPOINT TEST")
    print("="*70)
    
    # Test 1: Root endpoint
    print("\n[1] Testing root endpoint (GET /)...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("\n   -> API sunucusu çalışıyor mu? Başlatmak için: python api.py")
        return
    
    # Test 2: Health check
    print("\n[2] Testing health endpoint (GET /health)...")
    try:
        response = requests.get(f"{base_url}/health")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Model loaded: {data.get('model_loaded')}")
        print(f"   Device: {data.get('device')}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: Get classes
    print("\n[3] Testing classes endpoint (GET /classes)...")
    try:
        response = requests.get(f"{base_url}/classes")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Classes: {list(data.keys())}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 4: Prediction
    print("\n[4] Testing prediction endpoint (POST /predict)...")
    
    # Find a test image
    test_image_paths = [
        "Testing/notumor/1107.jpg",
        "Testing/glioma/1.jpg",
        "Testing/meningioma/1.jpg",
        "Testing/pituitary/1.jpg"
    ]
    
    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image = path
            break
    
    if test_image:
        print(f"   Using test image: {test_image}")
        try:
            with open(test_image, "rb") as f:
                files = {"file": (Path(test_image).name, f, "image/jpeg")}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {response.status_code}")
                print(f"   Predicted: {data['predicted_class']}")
                print(f"   Confidence: {data['confidence']*100:.2f}%")
                print(f"   Probabilities:")
                for cls, prob in data['probabilities'].items():
                    print(f"      {cls}: {prob*100:.2f}%")
            else:
                print(f"   ERROR: {response.status_code}")
                print(f"   {response.text}")
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print("   WARNING: Test image not found!")
        print(f"   Tried: {test_image_paths}")
    
    # Test 5: TTA prediction
    print("\n[5] Testing TTA prediction (POST /predict?use_tta=true)...")
    if test_image:
        try:
            with open(test_image, "rb") as f:
                files = {"file": (Path(test_image).name, f, "image/jpeg")}
                response = requests.post(
                    f"{base_url}/predict?use_tta=true",
                    files=files
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {response.status_code}")
                print(f"   Predicted: {data['predicted_class']}")
                print(f"   Confidence: {data['confidence']*100:.2f}%")
            else:
                print(f"   ERROR: {response.status_code}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n" + "="*70)
    print("SWAGGER UI: http://localhost:8000/docs")
    print("="*70)

if __name__ == "__main__":
    test_api()
