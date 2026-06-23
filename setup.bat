@echo off
cd /d "%~dp0"
echo Creating virtual environment (one time)...
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
echo Installing dependencies, this can take a few minutes...
pip install -r requirements.txt
echo.
echo Setup complete. Now double-click run_demo.bat
pause
