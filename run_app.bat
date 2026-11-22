@echo off
setlocal
title PhishGuard AI - Phishing Email Detection System
color 0B

echo =======================================================================
echo          PHISHGUARD AI - PHISHING EMAIL DETECTION SYSTEM
echo =======================================================================
echo.

:: Detect Python executable
set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    ) else if exist "%ProgramFiles%\Python313\python.exe" (
        set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
    ) else if exist "py.exe" (
        set "PYTHON_EXE=py"
    )
)

echo [INFO] Using Python interpreter: %PYTHON_EXE%
echo [INFO] Verifying model artifacts...

if not exist "models\phishing_detector_model.joblib" (
    echo [INFO] Trained model artifact not found. Initiating model training...
    "%PYTHON_EXE%" train_model.py
    if %errorlevel% neq 0 (
        echo [ERROR] Model training failed. Exiting.
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Launching Streamlit Cybersecurity Dashboard...
echo [INFO] Access the web UI in your browser at http://localhost:8501
echo.

"%PYTHON_EXE%" -m streamlit run app.py

pause
