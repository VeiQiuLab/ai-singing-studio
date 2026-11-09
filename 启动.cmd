@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
where pythonw.exe >nul 2>nul
if errorlevel 1 goto console
start "" pythonw.exe "%~dp0gui.py"
exit /b
:console
python.exe "%~dp0gui.py"
if errorlevel 1 pause
