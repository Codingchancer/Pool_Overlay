@echo off
rem Pool Practice Overlay - one-time setup.
rem Builds a private .venv with the newest installed Python 3.11+ (3.15 first),
rem installs the requirements, and adds a desktop shortcut.
setlocal
cd /d "%~dp0"

rem A .venv copied from another PC points at that PC's Python. Rebuild it then.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo The existing .venv was made for a Python that is not on this PC. Rebuilding it.
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    set "PYTHON="
    for %%V in (3.15 3.14 3.13 3.12 3.11) do (
        if not defined PYTHON (
            py -%%V -c "import sys" >nul 2>&1 && set "PYTHON=py -%%V"
        )
    )
    if not defined PYTHON (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "PYTHON=python"
    )
    call :make_venv
    if errorlevel 1 exit /b 1
)

echo Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installing the requirements failed. See the messages above.
    pause
    exit /b 1
)

echo Creating the desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'Pool Practice Overlay.lnk'));" ^
  "$s.TargetPath = '%~dp0.venv\Scripts\pythonw.exe';" ^
  "$s.Arguments = '\"%~dp0main.py\"';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.Description = 'Pool Practice Overlay (visual practice aid)';" ^
  "$s.Save()"

".venv\Scripts\python.exe" -c "import sys; print('Ready. Python', sys.version.split()[0])"
echo Start it with the desktop shortcut or run.bat.
pause
exit /b 0

:make_venv
if not defined PYTHON (
    echo Python 3.11 or newer was not found. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" or keep the "py" launcher, then run setup.bat again.
    pause
    exit /b 1
)
echo Creating .venv with: %PYTHON%
%PYTHON% -m venv .venv
if errorlevel 1 (
    echo Creating the virtual environment failed.
    pause
    exit /b 1
)
exit /b 0
