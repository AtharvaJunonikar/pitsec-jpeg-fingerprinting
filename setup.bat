@echo off
REM PITSEC JPEG Fingerprinting - Automated Setup Script (Windows)
REM Usage: setup.bat
REM This script creates the virtual environment and installs all dependencies

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   PITSEC JPEG Fingerprinting - Environment Setup
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo + Found Python %PYTHON_VERSION%
echo.

REM Check if venv already exists
if exist "venv\" (
    echo W Virtual environment already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing old venv...
        rmdir /s /q venv
    ) else (
        echo Using existing venv. Skipping creation.
        echo Activating existing environment...
        call venv\Scripts\activate.bat
        echo + Venv activated
        echo.
        echo Checking for dependency updates...
        pip install -r requirements.txt
        echo + Dependencies installed/updated
        echo.
        echo ================================================================
        echo   Setup complete! Your environment is ready.
        echo ================================================================
        pause
        exit /b 0
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo + Virtual environment created
echo.

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo + Venv activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo + pip upgraded
echo.

REM Install requirements
echo Installing dependencies from requirements.txt...
echo (This may take a few minutes...)
pip install -r requirements.txt
if errorlevel 1 (
    echo X Installation failed. Please check the output above.
    pause
    exit /b 1
)
echo + All dependencies installed
echo.

REM Verify installation
echo Verifying installation...
python -c "import numpy, cv2, skimage, sklearn, jpeglib; print('+ All core packages verified')"
if errorlevel 1 (
    echo X Verification failed. Some packages may not have installed correctly.
    pause
    exit /b 1
)
echo.

REM Create necessary folders
echo Creating project directories...
if not exist "data\alaska_tif" mkdir data\alaska_tif
if not exist "data\compressed" mkdir data\compressed
if not exist "output" mkdir output
if not exist "src" mkdir src
echo + Folders created
echo.

echo ================================================================
echo   + Setup Complete!
echo ================================================================
echo.
echo Your environment is ready. Next steps:
echo.
echo 1. Activate the virtual environment in future sessions:
echo    venv\Scripts\activate.bat
echo.
echo 2. Add your ALASKA TIFF files to:
echo    data\alaska_tif\
echo.
echo 3. Run the exploration script:
echo    python src\test_jpeg_version.py
echo.
echo 4. Run the production pipeline:
echo    python src\bulk_classify.py
echo.
echo For more details, see SETUP.md and README.md
echo.
pause
