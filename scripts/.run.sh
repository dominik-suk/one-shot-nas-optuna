#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

uv sync

export CUDA_VISIBLE_DEVICES="0"

PYTHON=".venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT"

export MPLBACKEND=Agg

LOG_DIR="logs/run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +'%Y-%m-%d')_run.log"


run_pipeline() {
    echo "Supernet Training"
    $PYTHON scripts/supernet_do_complete_train_run.py

    echo "SPOS NAS"
    $PYTHON scripts/spos_do_nas_run.py

    echo "SPOS Random Search NAS"
    $PYTHON scripts/spos_do_random_search_run.py

    echo "Baseline NAS"
    $PYTHON scripts/baseline_do_nas_run.py

    echo "Benchmarking and Analysis"
    $PYTHON scripts/perform_benchmarking_and_analysis.py

    echo "Rank Correlation Analysis"
    $PYTHON scripts/conduct_rank_correlation_analysis.py
}

run_pipeline 2>&1 | stdbuf -oL -eL tee -a "$LOG_FILE"
