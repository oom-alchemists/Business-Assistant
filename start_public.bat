@echo off
chcp 65001 >nul
title Enterprise AI Assistant - Public Tunnel

echo ========================================
echo   Enterprise AI Assistant - Public Link
echo ========================================
echo.

echo [1/4] Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo     Ollama not running, starting...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo     Ollama is running
)

echo [2/4] Changing to project directory...
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Activate your environment first, e.g.: conda activate ^<your-env^>
    pause
    exit /b 1
)

echo [3/4] Starting FastAPI on port 8001...
start "FastAPI-Server" python -m uvicorn app.api:app --host 0.0.0.0 --port 8001 --workers 2

echo     Waiting for FastAPI to be ready...
for /l %%i in (1,1,30) do (
    powershell -Command "try{$r=Invoke-RestMethod -Uri 'http://127.0.0.1:8001/api/health' -TimeoutSec 3; if($r.status -eq 'ok'){exit 0}}catch{}" >nul 2>nul
    if not errorlevel 1 goto :ready
    timeout /t 2 /nobreak >nul
)
echo [WARN] FastAPI did not become ready in time, continuing anyway...
:ready

echo [4/4] Starting public tunnel (cpolar)...
echo.
echo ========================================
echo   Please copy the "Forwarding" https URL
echo   shown by cpolar below, e.g.
echo     https://xxxxxx.cpolar.top
echo   Share it with anyone to open the chat.
echo ========================================
echo.

where cpolar >nul 2>nul
if errorlevel 1 (
    echo [WARN] cpolar not found in PATH.
    echo        Install from https://www.cpolar.com
    echo        Then run: cpolar authtoken ^<your-token^>
    echo        And re-run this script.
    echo.
    echo        Alternatives: natapp / ngrok (edit this script).
    pause
    exit /b 1
)

cpolar http 8001

pause
