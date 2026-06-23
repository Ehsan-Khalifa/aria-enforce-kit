@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate ( call .venv\Scripts\activate )
echo Starting ARIA-Enforce... a browser tab will open shortly.
python -m aria_enforce.app
pause
