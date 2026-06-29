@echo off
REM StrikeOut Bot — daily K projections to Discord
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\harri\Documents\StrikeOut Bot\mlb-k-predictor"
if not exist "tools\out" mkdir "tools\out"
echo ==== %DATE% %TIME% ==== >> "tools\out\daily.log"
REM grade yesterday's picks first, then post today's
".venv\Scripts\python.exe" tools\results.py     >> "tools\out\daily.log" 2>&1
".venv\Scripts\python.exe" tools\daily_picks.py >> "tools\out\daily.log" 2>&1
