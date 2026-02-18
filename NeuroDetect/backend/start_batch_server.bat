@echo off
REM Start Batch Processing API Server

echo Starting NeuroDetect Batch API Server...
echo.

cd /d "%~dp0.."

REM Activate conda environment
call conda activate fraud-detection

REM Start the FastAPI server
python server/batch_api.py

pause
