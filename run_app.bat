@echo off
title Heart Disease Prediction Web App - CardioCare
echo ============================================================
echo   Starting Heart Disease Prediction Web App (CardioCare AI)
echo ============================================================
echo.
echo Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python.
    pause
    exit /b 1
)

echo.
echo Launching Streamlit application...
python -m streamlit run app.py
pause
