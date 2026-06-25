@echo off
REM ZeroLeakX one-click launcher (Windows)
cd /d "%~dp0"
echo Installing dependencies (first run only)...
python -m pip install -r requirements.txt
echo.
echo Starting ZeroLeakX on http://127.0.0.1:8000
start "" http://127.0.0.1:8000
python server.py
