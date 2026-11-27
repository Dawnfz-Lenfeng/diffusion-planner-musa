#!/bin/bash
# M4: batch closed-loop eval on mini. Two-stage: simulation -> metric computation.
# Usage:
#   FILTER=mini_eval CKPT=$WORK/runs/train_main/.../checkpoint_xxx.pth DEVICE=musa \
#     bash scripts/sim_eval.sh
# Env:
#   FILTER        scenario_filter yaml name under diffusion_planner/config/scenario_filter (default mini_eval)
#   CKPT          checkpoint path (default: official checkpoints/model.pth)
#   ARGS          matching args.json (default: checkpoints/args.json; for self-trained use
#                 the args.json saved next to the training run)
#   DEVICE        planner inference device: cpu | musa (default musa on GPU pods)
#   NPROC_SIM     ray threads (default $(nproc))
set -uo pipefail
WORK=${WORK:-${WORK:-/data/dp-nuplan}}
REPO=$WORK/repo
FILTER=${FILTER:-mini_eval}
DEVICE=${DEVICE:-musa}
CKPT=${CKPT:-$WORK/checkpoints/model.pth}
ARGS=${ARGS:-$WORK/checkpoints/args.json}
CHALLENGE=closed_loop_nonreactive_agents

export PYTHONPATH=$REPO:$REPO/third_party/nuplan-devkit:$WORK/pylibs:${PYTHONPATH:-}
export NUPLAN_DATA_ROOT="$WORK/data/nuplan"
export NUPLAN_MAPS_ROOT="$WORK/data/nuplan/maps"
export NUPLAN_EXP_ROOT="$WORK/exp"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" | sed 's/\.pth//')
STAMP=$(date +%Y%m%d_%H%M)
OUT_UID=dp_mini/$CHALLENGE/eval_${FILTER}_${TAG}_$STAMP

echo "== stage 1: simulation ($FILTER, ckpt=$CKPT, device=$DEVICE) =="
python "$REPO/third_party/nuplan-devkit/nuplan/planning/script/run_simulation.py" \
    +simulation=$CHALLENGE \
    planner=diffusion_planner \
    planner.diffusion_planner.config.args_file="$ARGS" \
    planner.diffusion_planner.ckpt_path="$CKPT" \
    planner.diffusion_planner.device=$DEVICE \
    scenario_builder=nuplan \
    scenario_filter=$FILTER \
    experiment_uid=$OUT_UID \
    verbose=true \
    worker=sequential \
    distributed_mode=SINGLE_NODE \
    number_of_gpus_allocated_per_simulation=0.0 \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    2>&1 | tail -6
echo "sim rc=$?"

RUN_DIR=$NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/dp_mini/$CHALLENGE/$(basename $OUT_UID)
echo "== stage 2: metrics from $RUN_DIR =="
WORK=$WORK FILTER=$FILTER bash "$REPO/scripts/run_metric_on.sh" "$RUN_DIR"
echo "RESULT_DIR=$RUN_DIR"
