#!/bin/bash
# Stage 2 of eval: compute metrics from a finished simulation run's simulation_log,
# then aggregate. Usage: bash scripts/run_metric_on.sh <simulation_run_dir>
set -uo pipefail
WORK=${WORK:-${WORK:-/data/dp-nuplan}}
REPO=$WORK/repo
RUN_DIR=${1:?simulation log main dir}
FILTER=${FILTER:-mini_one}
export PYTHONPATH=$REPO:$REPO/third_party/nuplan-devkit:$WORK/pylibs:${PYTHONPATH:-}
export NUPLAN_DATA_ROOT="$WORK/data/nuplan"
export NUPLAN_MAPS_ROOT="$WORK/data/nuplan/maps"
export NUPLAN_EXP_ROOT="$WORK/exp"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

CHALLENGE=closed_loop_nonreactive_agents
echo "simulation_log root: $RUN_DIR (builder appends simulation_log/<planner>/...)"

STAMP=$(date +%Y%m%d_%H%M)
python "$REPO/third_party/nuplan-devkit/nuplan/planning/script/run_metric.py" \
    +simulation=$CHALLENGE \
    scenario_builder=nuplan \
    scenario_filter=$FILTER \
    experiment_uid=dp_mini/$CHALLENGE/metric_$STAMP \
    simulation_log_main_path="$RUN_DIR" \
    worker=sequential \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    2>&1 | tail -30
echo "metric rc=$?"
