@echo off
chcp 65001 >nul
title Enterprise AI Assistant - Start All
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Activate your environment first, e.g.: conda activate ^<your-env^>
    pause
    exit /b 1
)

python scripts\start_all.py
pause
