@echo off
echo ========================================
echo NeuroBridge AI - FastAPI Server
echo ========================================
echo.
echo Starting API server...
echo Swagger UI: http://localhost:8000/docs
echo.

call .venv\Scripts\activate.bat
python api.py
