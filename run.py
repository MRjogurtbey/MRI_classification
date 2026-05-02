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
    print("NeuroBridge AI - MRI Siniflandirma Sistemi")
    print("SuHack 2026 - NeuroBridge AI Hackathon")
    print("=" * 60)
    print()
    
    # Proje kök dizini
    project_root = Path(__file__).parent
    app_path = project_root / "app.py"
    
    if not app_path.exists():
        print("[HATA] app.py bulunamadi!")
        sys.exit(1)
    
    print("[BASLATILIYOR] Streamlit uygulamasi baslatiliyor...")
    print(f"[KONUM] {app_path}")
    print()
    print("[BILGI] Tarayiciniz otomatik olarak acilacaktir.")
    print("[BILGI] Kapatmak icin Ctrl+C tushlarina basin.")
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
        print("\n\n[KAPATILDI] Uygulama kapatildi.")
    except subprocess.CalledProcessError as e:
        print(f"\n[HATA] {e}")
        print("\n[BILGI] Streamlit kurulu mu? Kurmak icin:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"\n[HATA] Beklenmeyen hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
