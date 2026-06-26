@echo off
REM LTX-2 Inference Server startup script

REM Check if .env exists
if not exist .env (
    echo Error: .env file not found
    echo Please copy .env.example to .env and configure your model paths
    exit /b 1
)

REM Check if required directories exist
if not exist outputs mkdir outputs
if not exist temp mkdir temp

REM Run the server
echo Starting LTX-2 Inference Server...
echo API docs will be available at http://localhost:8000/docs

python main.py %*
