@echo off
setlocal

set "PROJECT_DIR=%~dp0.."
set "PYTHON_BIN=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON_BIN%" (
  echo Project virtual environment not found. Run: py -3.11 -m venv .venv
  echo Then run: .venv\Scripts\pip install -e .[dev]
  exit /b 1
)

cd /d "%PROJECT_DIR%"

if /I "%~1"=="--check" (
  "%PYTHON_BIN%" -c "import streamlit; import probstat_tutor; print('Windows launcher check passed')"
  if errorlevel 1 exit /b 1
  exit /b 0
)

"%PYTHON_BIN%" -m streamlit run app.py --server.address 127.0.0.1
exit /b %ERRORLEVEL%
