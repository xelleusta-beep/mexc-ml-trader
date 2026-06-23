@echo off
title MEXC Sunuculari Durduruluyor
color 0C

echo Tum MEXC sunuculari durduruluyor...

taskkill /FI "WINDOWTITLE eq MEXC Backend*" /F 2>nul
taskkill /FI "WINDOWTITLE eq MEXC Frontend*" /F 2>nul
taskkill /FI "WINDOWTITLE eq MEXC*" /F 2>nul

echo.
echo Sunucular durduruldu.
pause
