@echo off
echo ========================================
echo Hugging Face Deployment Script
echo ========================================
echo.
echo ONCE:
echo 1. https://huggingface.co/join - Hesap ac
echo 2. https://huggingface.co/spaces - New Space olustur
echo    - Name: neurobridge-ai
echo    - SDK: Streamlit
echo    - Public
echo.
set /p HF_USERNAME="Hugging Face username'inizi girin: "
echo.

cd ..
echo Klonlaniyor...
git clone https://huggingface.co/spaces/%HF_USERNAME%/neurobridge-ai
cd neurobridge-ai

echo Dosyalar kopyalaniyor...
xcopy /Y ..\MRI_classification\app.py .
xcopy /Y ..\MRI_classification\config.py .
xcopy /Y ..\MRI_classification\requirements_hf.txt .\requirements.txt
xcopy /Y ..\MRI_classification\README_HF.md .\README.md
xcopy /E /I /Y ..\MRI_classification\utils .\utils
xcopy /E /I /Y ..\MRI_classification\checkpoints .\checkpoints

echo Git commit...
git add .
git commit -m "Initial deployment"
git push

echo.
echo ========================================
echo HAZIR!
echo ========================================
echo.
echo Link: https://huggingface.co/spaces/%HF_USERNAME%/neurobridge-ai
echo.
pause
