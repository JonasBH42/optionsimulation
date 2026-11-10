@echo off
cd /d "%~dp0"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PYTHONPATH=%~dp0src"
call "%~dp0.venv\Scripts\activate.bat"
"%~dp0.venv\Scripts\python.exe" -m qqq_sim %*
