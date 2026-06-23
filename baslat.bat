@echo off
title MEXC Multi-Agent Trading System
color 0A

echo ============================================
echo   MEXC Multi-Agent Trading System
echo   Tum sunucular baslatiliyor...
echo ============================================
echo.

cd /d "%~dp0"

echo [1/4] Eski sunucular durduruluyor...
taskkill /FI "WINDOWTITLE eq MEXC Backend*" /F 2>nul
taskkill /FI "WINDOWTITLE eq MEXC Frontend*" /F 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Bagimliliklar kontrol ediliyor...
pip install -q fastapi uvicorn httpx python-dotenv numpy pandas 2>nul

echo [3/4] Backend baslatiliyor (port 8000)...
cd backend
start "MEXC Backend" /MIN python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
cd ..

echo    Backend baslatildi, 6 saniye bekleniyor...
timeout /t 6 /nobreak >nul

echo [4/4] Frontend baslatiliyor (port 5173)...
cd frontend
start "MEXC Frontend" /MIN node node_modules\vite\bin\vite.js --host
cd ..

echo.
echo ============================================
echo   Tum sunucular baslatildi!
echo.
echo   Dashboard:  http://localhost:5173
echo   API Docs:   http://localhost:8000/docs
echo.
echo   Kapatmak icin durdur.bat calistirin.
echo ============================================
echo.
timeout /t 5 /nobreak >nul
