@echo off
cd /d "%~dp0"
where python >nul 2>&1 || (echo Python not found. & pause & exit /b 1)
python -c "import flask" 2>nul || pip install flask
python ui\app.py %*
