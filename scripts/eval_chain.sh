#!/bin/bash
# Batch-eval chain: self-trained checkpoint -> official checkpoint, both on mini_eval (musa).
# Usage: SELF_CKPT=<path> SELF_ARGS=<path> bash scripts/eval_chain.sh
set -uo pipefail
WORK=${WORK:-/data/dp-nuplan}
HERE=$(cd "$(dirname "$0")" && pwd)
SELF_CKPT=${SELF_CKPT:?set SELF_CKPT to the self-trained checkpoint path}
SELF_ARGS=${SELF_ARGS:?set SELF_ARGS to the args.json saved alongside the training run}
OFFICIAL_CKPT=${OFFICIAL_CKPT:-$WORK/checkpoints/model.pth}
OFFICIAL_ARGS=${OFFICIAL_ARGS:-$WORK/checkpoints/args.json}

echo "=== [1/2] self-trained checkpoint, mini_eval (musa) ==="
FILTER=mini_eval DEVICE=musa CKPT=$SELF_CKPT ARGS=$SELF_ARGS bash "$HERE/sim_eval.sh"
echo "=== [2/2] official checkpoint, mini_eval (musa) ==="
FILTER=mini_eval DEVICE=musa CKPT=$OFFICIAL_CKPT ARGS=$OFFICIAL_ARGS bash "$HERE/sim_eval.sh"
echo "=== EVAL CHAIN DONE ==="
