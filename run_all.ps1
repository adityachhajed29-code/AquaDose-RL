# ==============================================================================
# AquaDose-RL: End-to-End Build, Test, Train & Benchmark Pipeline (PowerShell)
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " AquaDose-RL: Automated Pipeline Execution" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Environment Verification
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Gray
    & ".\.venv\Scripts\Activate.ps1"
}

# 2. Dependency Installation
Write-Host "[2/4] Installing / verifying dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

# 3. Unit Test Verification
Write-Host "[3/4] Running unit test suite with pytest..." -ForegroundColor Yellow
pytest tests/ -v

# 4. Quick Training Run
Write-Host "[4/4] Executing quick PPO training run (8,192 timesteps)..." -ForegroundColor Yellow
python experiments/train.py --total-timesteps 8192 --eval-freq 2048 --seed 42

# 5. Benchmark Evaluation & Plot Generation
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Generating benchmark evaluation and publication-ready plots..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
python experiments/evaluate_and_plot.py --steps 1000 --seed 2024

Write-Host "======================================================================" -ForegroundColor Green
Write-Host " Pipeline Complete!" -ForegroundColor Green
Write-Host " Plots generated in: experiments/plots/" -ForegroundColor Green
Write-Host " Logs recorded in:  experiments/logs/" -ForegroundColor Green
Write-Host " Models saved in:   experiments/checkpoints/" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
