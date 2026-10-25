@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run the setup commands in README.md first.
  pause
  exit /b 1
)
echo LoanLens: http://127.0.0.1:5000
".venv\Scripts\python.exe" launch.py
pause
