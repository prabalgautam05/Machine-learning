@echo off
REM Streamlit App Setup and Launch Script for Windows

echo.
echo ========================================
echo   EDA & Preprocessing Streamlit App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [✓] Python found
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo Please ensure Python is properly installed
    pause
    exit /b 1
)

echo [✓] pip found
echo.

REM Install requirements
echo [→] Installing required packages...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)

echo [✓] All packages installed successfully
echo.

REM Launch Streamlit app
echo [→] Launching Streamlit app...
echo.
streamlit run app.py

pause
