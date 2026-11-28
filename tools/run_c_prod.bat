@echo off
cd /d "D:\VSCode project\ai+"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo === c_prod sweep started %date% %time% ===
"D:\VSCode project\ai+\.venv\Scripts\python.exe" -m research.tracks.c_prod --folds 6 --seeds 42,7
echo rc=%ERRORLEVEL%
echo === c_prod sweep finished %date% %time% ===
