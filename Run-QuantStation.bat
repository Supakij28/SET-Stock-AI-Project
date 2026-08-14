@echo off
TITLE Quant Strategy Station
echo ---------------------------------------------------
echo 🚀 Starting Quant Strategy Station...
echo ---------------------------------------------------
cd /d "%~dp0"
".venv 313\Scripts\streamlit.exe" run "stock_dashboard.py"
pause
