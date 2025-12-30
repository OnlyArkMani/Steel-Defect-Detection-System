@echo off
REM Steel Defect Detection - Environment Setup Script
REM This script creates a virtual environment and installs all dependencies

echo.
echo ========================================
echo STEEL DEFECT DETECTION - ENVIRONMENT SETUP
echo ========================================
echo.

REM Navigate to project directory
cd /d C:\Projects\CV_SDT

echo [1/4] Creating virtual environment...
python -m venv venv
echo ✓ Virtual environment created!
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated!
echo.

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip
echo ✓ Pip upgraded!
echo.

echo [4/4] Installing dependencies...
echo This may take 5-10 minutes depending on your internet speed...
echo.
pip install -r requirements.txt
echo.
echo ✓ All dependencies installed!
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo To activate the virtual environment in the future, run:
echo    venv\Scripts\activate
echo.
echo To verify installation, run:
echo    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo.
pause