"""
LIVE API DEMO - Juri icin canli gosterim
Sunumda projektorle gostermek icin
"""

import requests
from pathlib import Path
import sys

def live_demo():
    """Canli API demo - Juri onunde calistirilacak"""
    
    print("=" * 80)
    print("NEUROBRIDGE AI - LIVE API DEMO")
    print("=" * 80)
    print()
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health Check
    print("[1] API HEALTH CHECK")
    print(f"    URL: {base_url}/health")
    print()
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"    [OK] Status: {data['status']}")
            print(f"    [OK] Model Loaded: {data['model_loaded']}")
            print(f"    [OK] Device: {data['device']}")
            print(f"    [OK] Model Path: {data['model_path']}")
        else:
            print(f"    [ERROR] Error: {response.status_code}")
            return
    except Exception as e:
        print(f"    [ERROR] API sunucusu calismiyor!")
        print(f"    [ERROR] Hata: {e}")
        print()
        print("    -> Baslat: python api.py")
        return
    
    print()
    
    # Test 2: Prediction
    print("[2] CANLI MRI TAHMINI")
    
    # Find test image
    test_images = list(Path("Testing").rglob("*.jpg"))[:5] if Path("Testing").exists() else []
    
    if not test_images:
        print("    [ERROR] Test gorseli bulunamadi!")
        return
    
    test_image = test_images[0]
    print(f"    Gorsel: {test_image}")
    print(f"    URL: {base_url}/predict")
    print()
    
    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image.name, f, "image/jpeg")}
            response = requests.post(f"{base_url}/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"    [OK] Response Status: 200 OK")
            print()
            print(f"    SONUC:")
            print(f"      - Predicted Class: {data['predicted_class']}")
            print(f"      - Confidence: {data['confidence']*100:.2f}%")
            print(f"      - Device: {data['device_used']}")
            print(f"      - Description: {data['description']}")
            print()
            print(f"    OLASILIKLAR:")
            for cls, prob in data['probabilities'].items():
                bar = "#" * int(prob * 50)
                print(f"      - {cls:12s}: {prob*100:6.2f}% {bar}")
        else:
            print(f"    [ERROR] Error: {response.status_code}")
            print(f"    {response.text}")
    except Exception as e:
        print(f"    [ERROR] Hata: {e}")
    
    print()
    print("=" * 80)
    print("SWAGGER UI: http://localhost:8000/docs")
    print("API DOCS: http://localhost:8000/redoc")
    print("=" * 80)

if __name__ == "__main__":
    live_demo()
