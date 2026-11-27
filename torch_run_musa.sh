#!/bin/bash
# M3: training launcher for Moore Threads S5000 (MUSA) pod.
# Uses the image's torch_musa; deps come from the pylibs overlay (see pod_bootstrap.sh).
#
# Usage (inside a GPU pod with mthreads.com/gpu):
#   bash torch_run_musa.sh [NPROC] [EXTRA_ARGS...]
#   e.g.: bash torch_run_musa.sh 8 --batch_size 512 --train_epochs 500
#   smoke: bash torch_run_musa.sh 1 --batch_size 8 --train_epochs 1 --num_workers 2
set -euo pipefail
WORK=${WORK:-${WORK:-/data/dp-nuplan}}
REPO=$WORK/repo
NPROC=${1:-8}; shift || true

export PYTHONPATH="$WORK/pylibs:$REPO:$REPO/third_party/nuplan-devkit:${PYTHONPATH:-}"
# MUSA conventions carried over from the v4.1 verified profile
export MCCL_IB_DISABLE=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/runs"

STAMP=$(date +%Y%m%d_%H%M)
RUN_DIR=$WORK/runs/train_$STAMP
mkdir -p "$RUN_DIR"

cd "$REPO"
# NOTE: no sudo here (upstream used sudo; pod runs as root/image user already)
python3 -m torch.distributed.run --nnodes 1 --nproc-per-node "$NPROC" --standalone train_predictor.py \
    --name "dp-mini-musa-$STAMP" \
    --device musa \
    --train_set "$WORK/cache/mini_train" \
    --train_set_list "$WORK/cache/mini_train/diffusion_planner_training.json" \
    --save_dir "$RUN_DIR" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
echo "training done -> $RUN_DIR"
