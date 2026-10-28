@echo off
rem Starts the X-UAD website on this computer and opens it in the browser.
rem Keep this window open while you use the site; close it to stop.
cd /d "%~dp0"
echo.
echo   X-UAD website: http://localhost:8000
echo   Keep this window open while presenting. Close it to stop the site.
echo.
start "" "http://localhost:8000/"
python -m http.server 8000 --bind 127.0.0.1
if errorlevel 1 (
  echo.
  echo   Could not start. Is port 8000 already in use, or is Python missing?
  pause
)
