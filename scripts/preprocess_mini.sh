#!/bin/bash
# M2: mini split -> training .npz (run INSIDE pod, multi-process).
set -euo pipefail
WORK=${WORK:-${WORK:-/data/dp-nuplan}}
REPO=$WORK/repo
export PYTHONPATH=$WORK/pylibs:$REPO:$REPO/third_party/nuplan-devkit:${PYTHONPATH:-}

LOG_LIST=$WORK/mini_logs.json
python "$REPO/scripts/make_mini_log_list.py" "$WORK/data/nuplan/nuplan-v1.1/splits/mini" "$LOG_LIST"

mkdir -p "$WORK/cache/mini_train" "$WORK/runs"
cd "$REPO"
python data_process.py \
    --data_path "$WORK/data/nuplan/nuplan-v1.1/splits/mini" \
    --map_path  "$WORK/data/nuplan/maps" \
    --save_path "$WORK/cache/mini_train" \
    --log_list  "$LOG_LIST" \
    --total_scenarios 1000000 \
    --num_proc  "$(nproc)" \
    2>&1 | tee "$WORK/runs/preprocess_$(date +%Y%m%d_%H%M).log"
cp diffusion_planner_training.json "$WORK/cache/mini_train/" 2>/dev/null || true
echo "npz count: $(ls "$WORK/cache/mini_train" | grep -c '\.npz$')"
echo "failed:    $(wc -l < "$WORK/cache/mini_train/failed_tokens.txt" 2>/dev/null || echo 0)"
