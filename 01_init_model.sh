#!/bin/bash

# env setting
. ./env.sh || exit 1
set -euo pipefail

mkdir -p logs
python -u -B ./model/init_model.py > logs/init_model.log 2>&1

# sbatch --gres=gpu:1 -p medium -C "cu118" ./00_make_data.sh
