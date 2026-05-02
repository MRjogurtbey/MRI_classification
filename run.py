"""
NeuroBridge AI - Quick Start Script
Streamlit uygulamasını başlatır
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Streamlit uygulamasını başlat"""
    print("=" * 60)
    print("🧠 NeuroBridge AI - MRI Sınıflandırma Sistemi")
    print("SuHack 2026 - NeuroBridge AI Hackathon")
    print("=" * 60)
    print()
    
    # Proje kök dizini
    project_root = Path(__file__).parent
    app_path = project_root / "app.py"
    
    if not app_path.exists():
        print("❌ app.py bulunamadı!")
        sys.exit(1)
    
    print("🚀 Streamlit uygulaması başlatılıyor...")
    print(f"📂 Konum: {app_path}")
    print()
    print("💡 Tarayıcınız otomatik olarak açılacaktır.")
    print("💡 Kapatmak için Ctrl+C tuşlarına basın.")
    print()
    print("-" * 60)
    
    try:
        # Streamlit çalıştır
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--theme.base=light"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Uygulama kapatıldı.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Hata: {e}")
        print("\n💡 Streamlit kurulu mu? Kurmak için:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
