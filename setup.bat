@echo off
REM ============================================================
REM RiskSense AI - Quick Setup Script
REM Run this first after cloning the repository
REM ============================================================

echo.
echo  ============================================================
echo   RISKSENSE AI - Quick Setup
echo  ============================================================
echo.

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
echo   [OK] Python found

REM Create Virtual Environment
echo.
echo [2/6] Creating virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo   [OK] Virtual environment created
) else (
    echo   [OK] Virtual environment already exists
)

REM Activate venv and install dependencies
echo.
echo [3/6] Installing dependencies into .venv...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo   [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo   [OK] Dependencies installed in isolation

REM Check data files
echo.
echo [4/6] Checking data files...
if exist "data\raw\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv" (
    echo   [OK] Lending Club data found
) else (
    echo   [WARNING] Data file not found. 
    echo   Please download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club
    echo   Extract to: data\raw\
)

REM Configure VS Code
echo.
echo [5/6] Configuring VS Code...
if not exist ".vscode" mkdir ".vscode"
if not exist ".vscode\settings.json" (
    echo { > .vscode\settings.json
    echo     "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe", >> .vscode\settings.json
    echo     "python.terminal.activateEnvironment": true >> .vscode\settings.json
    echo } >> .vscode\settings.json
    echo   [OK] VS Code configured to use .venv
) else (
    echo   [OK] VS Code settings already exist
)

REM Train quick model
echo.
echo [6/6] Training quick model (10K sample)...
python -c "from src import train; results = train.train_pipeline(sample_n=10000); print(f'Champion AUC: {results[\"champion\"][\"metrics\"][\"auc_roc\"]:.4f}')"
if errorlevel 1 (
    echo   [WARNING] Training failed - check data files
) else (
    echo   [OK] Model trained successfully
)

echo.
echo  ============================================================
echo   SETUP COMPLETE!
echo  ============================================================
echo.
echo  Next steps:
echo    1. Run 'launcher.bat' to access all features
echo    2. Or use these commands directly:
echo       - Full training:  python -m src.train
echo       - Start API:      uvicorn api.main:app --reload
echo       - Run tests:      pytest tests/ -v
echo       - Open notebook:  jupyter notebook notebooks/
echo.
pause
