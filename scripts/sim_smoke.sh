#!/bin/bash
# M1: closed-loop simulation smoke on mini, official checkpoint, CPU planner.
# Requires: pod_bootstrap.sh done, make_mini_filters.py executed (mini_one.yaml present).
set -uo pipefail
WORK=${WORK:-${WORK:-/data/dp-nuplan}}
REPO=$WORK/repo
# nuplan must resolve from the devkit SOURCE tree (pip --target copy lacks package
# data: the yaml configs hydra needs via pkg://nuplan.*)
export PYTHONPATH=$REPO:$REPO/third_party/nuplan-devkit:$WORK/pylibs:${PYTHONPATH:-}
export NUPLAN_DATA_ROOT="$WORK/data/nuplan"
export NUPLAN_MAPS_ROOT="$WORK/data/nuplan/maps"
export NUPLAN_EXP_ROOT="$WORK/exp"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# On many-core machines torch's default intra-op thread pool makes small-model CPU
# inference very slow; cap threads for the planner process.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$NUPLAN_EXP_ROOT" "$WORK/runs"

[ -f "$REPO/diffusion_planner/config/scenario_filter/mini_one.yaml" ] || {
  echo "mini_one.yaml missing -> run scripts/make_mini_filters.py first"; exit 1; }

CHALLENGE=closed_loop_nonreactive_agents
STAMP=$(date +%Y%m%d_%H%M)
python "$REPO/third_party/nuplan-devkit/nuplan/planning/script/run_simulation.py" \
    +simulation=$CHALLENGE \
    planner=diffusion_planner \
    planner.diffusion_planner.config.args_file="$WORK/checkpoints/args.json" \
    planner.diffusion_planner.ckpt_path="$WORK/checkpoints/model.pth" \
    planner.diffusion_planner.device=cpu \
    scenario_builder=nuplan \
    scenario_filter=mini_one \
    experiment_uid=dp_mini/$CHALLENGE/smoke_$STAMP \
    verbose=true \
    worker=sequential \
    distributed_mode=SINGLE_NODE \
    number_of_gpus_allocated_per_simulation=0.0 \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    2>&1 | tee "$WORK/runs/sim_smoke_$STAMP.log"
echo "sim rc=$?"
echo "=== stage 2: metrics from simulation logs ==="
RUN_DIR=$NUPLAN_EXP_ROOT/exp/simulation/$CHALLENGE/dp_mini/$CHALLENGE/smoke_$STAMP
bash "$REPO/scripts/run_metric_on.sh" "$RUN_DIR"
