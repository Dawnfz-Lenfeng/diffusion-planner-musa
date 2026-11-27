#!/bin/bash
# Delivery-repo smoke: download -> preprocess -> train -> eval, using the
# sanitized delivery code against existing data on the pod.
set -uo pipefail
export WORK=/mnt_moer/dp-nuplan
export REPO=/mnt_moer/delivery-repo
export PYTHONPATH=$WORK/pylibs:$REPO:$REPO/third_party/nuplan-devkit:${PYTHONPATH:-}
export MCCL_IB_DISABLE=1 PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd $REPO
# repo 相对路径的数据指向既有数据目录（S1 即刻校验跳过）
ln -sfn $WORK/data $REPO/data

echo "=== S1: download (existing files -> size/idempotent pass) ==="
bash scripts/download_data.sh "$WORK"; echo "S1 rc=$?"

echo "=== S2: preprocess (idempotent, skips existing npz) ==="
WORK=$WORK bash scripts/preprocess_mini.sh 2>&1 | tail -5; echo "S2 rc=$?"

echo "=== S3: train smoke (musa, 1 proc, 256 samples, 1 epoch) ==="
mkdir -p $WORK/runs/delivery_train_smoke
python train_predictor.py --name delivery-smoke --device musa --ddp false \
    --batch_size 8 --train_epochs 1 --warm_up_epoch 1 --num_workers 2 \
    --train_set $WORK/cache/mini_train \
    --train_set_list $WORK/cache/mini_train/smoke_train.json \
    --save_dir $WORK/runs/delivery_train_smoke 2>&1 | tail -4
echo "S3 rc=$?"

echo "=== S4: eval smoke (mini_one 14 scenarios, cpu planner + metrics) ==="
bash scripts/sim_smoke.sh 2>&1 | tail -6
echo "S4 rc=$?"

EXP=$WORK/exp
D=$(ls -dt $EXP/exp/simulation/closed_loop_nonreactive_agents/dp_mini/closed_loop_nonreactive_agents/smoke_* 2>/dev/null | head -1)
M=$(ls -dt $EXP/exp/simulation/closed_loop_nonreactive_agents/dp_mini/closed_loop_nonreactive_agents/metric_* 2>/dev/null | head -1)
echo "=== SCORES ==="
python scripts/read_scores.py "$M" "$D/runner_report.parquet" 2>/dev/null | tail -6
echo "=== DELIVERY SMOKE DONE ==="
