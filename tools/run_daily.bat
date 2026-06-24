@echo off
REM StrikeOut Bot — daily K projections to Discord
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\harri\Documents\StrikeOut Bot\mlb-k-predictor"
if not exist "tools\out" mkdir "tools\out"
".venv\Scripts\python.exe" tools\daily_picks.py >> "tools\out\daily.log" 2>&1
