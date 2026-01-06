@echo off
REM ============================================================
REM RiskSense AI - Launcher Script
REM Bank-Grade ML Risk Intelligence Platform
REM ============================================================

title RiskSense AI

REM Activate Virtual Environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo.
echo  ============================================================
echo   RISKSENSE AI - Bank-Grade ML Risk Intelligence Platform
echo  ============================================================
echo.
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Running in global environment. Run setup.bat first.
) else (
    echo [INFO] Running in virtual env: %VIRTUAL_ENV%
)
echo.

:MENU
echo  Select an option:
echo.
echo  [1] Install dependencies (first-time setup)
echo  [2] Train models (full pipeline)
echo  [3] Train models (quick - 10K sample)
echo  [4] Start API server
echo  [5] Run tests
echo  [6] Open Jupyter notebook
echo  [7] Check system status
echo  [0] Exit
echo.
set /p choice="Enter your choice: "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto TRAIN_FULL
if "%choice%"=="3" goto TRAIN_QUICK
if "%choice%"=="4" goto API
if "%choice%"=="5" goto TEST
if "%choice%"=="6" goto NOTEBOOK
if "%choice%"=="7" goto STATUS
if "%choice%"=="0" goto END

echo Invalid choice. Please try again.
goto MENU

:INSTALL
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Dependencies installed successfully!
pause
goto MENU

:TRAIN_FULL
echo.
echo Training models on full dataset (1.37M records)...
echo This may take 5-10 minutes...
python -m src.train
echo.
echo Training complete!
pause
goto MENU

:TRAIN_QUICK
echo.
echo Training models on sample dataset (10K records)...
python -c "from src import train; train.train_pipeline(sample_n=10000)"
echo.
echo Quick training complete!
pause
goto MENU

:API
echo.
echo Starting FastAPI server...
echo API will be available at: http://localhost:8000
echo Swagger docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
goto MENU

:TEST
echo.
echo Running tests...
python -m pytest tests/ -v
echo.
pause
goto MENU

:NOTEBOOK
echo.
echo Opening Jupyter notebook...
echo Navigate to notebooks/risksense_demo.ipynb
python -m jupyter notebook notebooks/
goto MENU

:STATUS
echo.
echo  ============================================================
echo   SYSTEM STATUS
echo  ============================================================
echo.
echo Checking Python...
python --version
echo.
echo Checking data files...
if exist "data\raw\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv" (
    echo   [OK] Lending Club data found
) else (
    echo   [MISSING] Lending Club data not found
)
echo.
echo Checking trained models...
if exist "models\champion_xgb.joblib" (
    echo   [OK] Champion model found
) else (
    echo   [MISSING] Model not trained - run option [2] or [3]
)
echo.
pause
goto MENU

:END
echo.
echo Goodbye!
exit /b 0
