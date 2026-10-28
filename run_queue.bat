@echo off
rem Starts the Revision-3 experiment queue with Git Bash (not WSL) and keeps Windows awake.
rem Finished jobs are skipped, so this is safe to run again after an interruption.
cd /d "%~dp0"
echo. > results_v3\QUEUE_RUNNING
start "X-UAD keep awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\_session_runs\keep_awake.ps1"
echo Queue running. Progress: results_v3\queue.log   (close this window to stop)
"C:\Program Files\Git\bin\bash.exe" scripts/_session_runs/v3_queue.sh
echo Queue finished.
pause
