"""
Setup Test Script
Kurulumun doğru yapılıp yapılmadığını kontrol eder
"""

import sys
from pathlib import Path

def check_python_version():
    """Python versiyonunu kontrol et"""
    version = sys.version_info
    print(f"🐍 Python Versiyonu: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8 veya üzeri gerekli!")
        return False
    else:
        print("   ✅ Python versiyonu uygun")
        return True

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    print("\n📦 Bağımlılıklar Kontrolü:")
    
    required_packages = [
        'torch',
        'torchvision',
        'streamlit',
        'numpy',
        'pandas',
        'PIL',
        'cv2',
        'monai',
        'matplotlib',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
                print(f"   ✅ {package} (Pillow)")
            elif package == 'cv2':
                __import__('cv2')
                print(f"   ✅ {package} (opencv-python)")
            else:
                __import__(package)
                print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - EKSIK!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Eksik paketler: {', '.join(missing_packages)}")
        print("\n💡 Kurmak için:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Tüm bağımlılıklar yüklü")
        return True

def check_project_structure():
    """Proje yapısını kontrol et"""
    print("\n📁 Proje Yapısı Kontrolü:")
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'README.md',
        'utils/__init__.py',
        'utils/preprocessing.py',
        'utils/inference.py',
        'utils/gradcam.py'
    ]
    
    required_dirs = [
        'utils',
        'models'
    ]
    
    project_root = Path(__file__).parent
    all_good = True
    
    # Dosyaları kontrol et
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - EKSIK!")
            all_good = False
    
    # Klasörleri kontrol et
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - EKSIK!")
            all_good = False
    
    return all_good

def check_model_files():
    """Model dosyalarını kontrol et"""
    print("\n🤖 Model Dosyaları:")
    
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    
    if not models_dir.exists():
        print("   ❌ models/ klasörü bulunamadı!")
        return False
    
    model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.pth'))
    
    if model_files:
        print(f"   ✅ {len(model_files)} model dosyası bulundu:")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"      - {model_file.name} ({size_mb:.2f} MB)")
        return True
    else:
        print("   ⚠️  Model dosyası bulunamadı!")
        print("   💡 Eğitilmiş modelinizi models/ klasörüne yerleştirin")
        return False

def check_gpu():
    """GPU kullanılabilirliğini kontrol et"""
    print("\n🎮 GPU Kontrolü:")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ CUDA kullanılabilir")
            print(f"   🎮 GPU: {gpu_name}")
            print(f"   📊 GPU Sayısı: {gpu_count}")
            return True
        else:
            print("   ⚠️  CUDA kullanılamıyor - CPU modunda çalışacak")
            print("   💡 GPU kullanmak için CUDA ve uyumlu PyTorch yükleyin")
            return False
    except ImportError:
        print("   ❌ PyTorch yüklü değil")
        return False

def main():
    """Ana test fonksiyonu"""
    print("=" * 70)
    print("🧠 NeuroBridge AI - Setup Test")
    print("SuHack 2026 - NeuroBridge AI Hackathon")
    print("=" * 70)
    
    results = {
        'Python': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Structure': check_project_structure(),
        'Model Files': check_model_files(),
        'GPU': check_gpu()
    }
    
    print("\n" + "=" * 70)
    print("📊 ÖZET")
    print("=" * 70)
    
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    # Model dosyası opsiyonel
    critical_checks = ['Python', 'Dependencies', 'Project Structure']
    critical_passed = all(results[check] for check in critical_checks)
    
    print("\n" + "=" * 70)
    
    if critical_passed:
        print("✅ KURULUM BAŞARILI!")
        print("\n🚀 Uygulamayı başlatmak için:")
        print("   python run.py")
        print("   VEYA")
        print("   streamlit run app.py")
        
        if not results['Model Files']:
            print("\n⚠️  DİKKAT: Model dosyası yok!")
            print("   Uygulamayı çalıştırabilirsiniz ancak tahmin yapmak için")
            print("   models/ klasörüne eğitilmiş model eklemelisiniz.")
    else:
        print("❌ KURULUM SORUNLU!")
        print("\n💡 Lütfen yukarıdaki hataları düzeltin ve tekrar deneyin.")
        print("   pip install -r requirements.txt")
    
    print("=" * 70)
    
    return 0 if critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
