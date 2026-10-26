#!/usr/bin/env bash
# ==============================================================================
# AquaDose-RL: End-to-End Build, Test, Train & Benchmark Pipeline
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo " AquaDose-RL: Automated Pipeline Execution"
echo "======================================================================"

# 1. Environment Verification / Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv || python -m venv .venv
fi

# Activate virtual environment if script exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# 2. Dependency Installation
echo "[2/4] Installing / verifying dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# 3. Unit Test Verification
echo "[3/4] Running unit test suite with pytest..."
pytest tests/ -v

# 4. Quick Training Run
echo "[4/4] Executing quick PPO training run (8,192 timesteps)..."
python experiments/train.py --total-timesteps 8192 --eval-freq 2048 --seed 42

# 5. Benchmark Evaluation & Plot Generation
echo "======================================================================"
echo " Generating benchmark evaluation and publication-ready plots..."
echo "======================================================================"
python experiments/evaluate_and_plot.py --steps 1000 --seed 2024

echo "======================================================================"
echo " Pipeline Complete!"
echo " Plots generated in: experiments/plots/"
echo " Logs recorded in:  experiments/logs/"
echo " Models saved in:   experiments/checkpoints/"
echo "======================================================================"
