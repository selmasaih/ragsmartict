# Launch backend (FastAPI) and frontend (Vite) together for local development.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\dev.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$backend = Start-Process -PassThru -WorkingDirectory $root powershell `
    -ArgumentList "-NoExit", "-Command", "python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000"

Write-Host "Starting Vite frontend on http://localhost:5173 ..." -ForegroundColor Cyan
$frontendDir = Join-Path $root "frontend"
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "Installing frontend dependencies ..." -ForegroundColor Yellow
    Push-Location $frontendDir; npm install; Pop-Location
}
$frontend = Start-Process -PassThru -WorkingDirectory $frontendDir powershell `
    -ArgumentList "-NoExit", "-Command", "npm run dev"

Write-Host ""
Write-Host "Backend PID: $($backend.Id)  Frontend PID: $($frontend.Id)" -ForegroundColor Green
Write-Host "Close the spawned windows (or Stop-Process the PIDs above) to stop the servers." -ForegroundColor Green
