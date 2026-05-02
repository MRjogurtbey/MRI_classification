@echo off
echo ========================================
echo NeuroBridge AI - Ngrok Deployment
echo ========================================
echo.
echo Bu script Streamlit ve API'yi internete acar
echo.
echo ADIMLAR:
echo 1. Ngrok hesabi olustur: https://ngrok.com
echo 2. Ngrok indir ve PATH'e ekle
echo 3. ngrok config add-authtoken YOUR_TOKEN
echo.
pause

echo.
echo [1] Streamlit baslatiiliyor...
start "Streamlit" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && streamlit run app.py"

timeout /t 10

echo.
echo [2] Ngrok tunnel aciliyor (Streamlit)...
start "Ngrok-Streamlit" cmd /k "ngrok http 8501"

echo.
echo ========================================
echo HAZIR!
echo ========================================
echo.
echo Ngrok penceresinde gorunen URL'i kopyala
echo Ornek: https://abc123.ngrok.io
echo.
echo Bu linki juriye gonder!
echo.
pause
