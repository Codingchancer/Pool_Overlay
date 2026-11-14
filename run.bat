@echo off
rem Start the Pool Practice Overlay; runs setup.bat first when needed.
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "import cv2, mss, numpy, PySide6" >nul 2>&1
if errorlevel 1 (
    call setup.bat
    if errorlevel 1 exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py"
