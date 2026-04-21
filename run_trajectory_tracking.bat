@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 main_trajectory_tracking_fixed.py
) else (
    python main_trajectory_tracking_fixed.py
)

if %errorlevel% neq 0 (
    echo.
    echo Failed to launch the app.
    echo Ensure Python is installed and dependencies are available:
    echo   pip install -r requirements.txt
    pause
)

endlocal