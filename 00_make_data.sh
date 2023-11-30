#!/bin/bash

# env setting
. ./env.sh || exit 1
set -euo pipefail

mkdir -p logs
# python -u -B ./data/download_data.py > logs/download_data.log 2>&1
python -u -B ./prepare_data/preprocess_data.py > logs/preprocess_data.log 2>&1

# sbatch --gres=gpu:1 -p medium -C "cu118" ./00_make_data.sh
