#!/bin/bash

# env setting
. ./env.sh || exit 1
set -euo pipefail

mkdir -p logs
python -u -B ./main.py > logs/main.log 2>&1

# sbatch --gres=gpu:1 -p medium -C "cu118" ./01_train.sh
