@echo off
chcp 65001 >nul
title Enterprise AI Assistant - FastAPI

echo ========================================
echo   Enterprise AI Assistant - FastAPI
echo   Concurrency: uvicorn --workers 2
echo ========================================
echo.

echo [1/3] Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo     Ollama not running, starting...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo     Ollama is running
)

echo [2/3] Changing to project directory...
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Activate your environment first, e.g.: conda activate ^<your-env^>
    pause
    exit /b 1
)

echo [3/3] Starting FastAPI (2 workers)...
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4" ^| findstr /V "169.254" ^| findstr /V "192.168.88" ^| findstr /V "172.28"') do set IP=%%a
set IP=%IP: =%
echo ========================================
echo   Local:    http://localhost:8001
echo   Network:  http://%IP%:8001
echo   API Doc:  http://localhost:8001/docs
echo ========================================
echo.

python -m uvicorn app.api:app --host 0.0.0.0 --port 8001 --workers 2

pause
