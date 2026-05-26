param(
    [int]$Port = 8000,
    [switch]$Reload = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir  = Join-Path $ProjectRoot "backend"
$PersistDir  = Join-Path $ProjectRoot "data"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MEXC ML Trader - Local Development" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Persistence directory
if (-not (Test-Path $PersistDir)) {
    New-Item -ItemType Directory -Path $PersistDir | Out-Null
    Write-Host "  [OK] Persistence directory created" -ForegroundColor Green
}

# 2. Virtual environment
$VenvDir = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path (Join-Path $VenvDir "Scripts" "Activate.ps1"))) {
    Write-Host "  [..] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvDir
    if (-not $?) { Write-Host "  [FAIL] Could not create venv" -ForegroundColor Red; exit 1 }
    Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
}

# 3. Activate venv and install deps
& (Join-Path $VenvDir "Scripts" "Activate.ps1")

Write-Host "  [..] Installing dependencies..." -ForegroundColor Yellow
pip install -r (Join-Path $BackendDir "requirements.txt") 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    pip install --no-cache-dir -r (Join-Path $BackendDir "requirements.txt") 2>&1 | Out-Null
}
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

# 4. Check lightgbm (optional on Windows)
try {
    python -c "import lightgbm" 2>$null
    Write-Host "  [OK] lightgbm available" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] lightgbm not found - sklearn/numpy fallback will be used" -ForegroundColor Yellow
}

# 5. Set env vars and start
$reloadFlag = if ($Reload) { "--reload" } else { "" }
$env:PERSIST_DIR = $PersistDir
$env:PORT = $Port

Write-Host ""
Write-Host "-----------------------------------------" -ForegroundColor Cyan
Write-Host "  Starting server on http://localhost:$Port" -ForegroundColor Cyan
Write-Host "  Dashboard: http://localhost:$Port/" -ForegroundColor White
Write-Host "  ML Details: http://localhost:$Port/ml-details" -ForegroundColor White
Write-Host "  Health: http://localhost:$Port/health" -ForegroundColor White
Write-Host "  Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Cyan
Write-Host ""

Set-Location -LiteralPath $BackendDir
uvicorn main:app --host 0.0.0.0 --port $Port $reloadFlag
