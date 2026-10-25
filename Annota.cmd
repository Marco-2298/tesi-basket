@echo off
cd /d "%~dp0"
"%~dp0.venv-eval\Scripts\python.exe" "%~dp0scripts\annotate.py"
if errorlevel 1 pause
