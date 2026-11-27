#!/usr/bin/env python3
"""Pick N scenario tokens from the mini split (by type) and emit scenario_filter yamls.

Produces, under diffusion_planner/config/scenario_filter/:
  mini_one.yaml   single scenario (simulation smoke, M1)
  mini_eval.yaml  mini baseline eval set (M4)

Run inside the pod with devkit importable (PYTHONPATH per preprocess_mini.sh).
"""
import os
import sys

REPO = os.environ.get("REPO", "${WORK:-/data/dp-nuplan}/repo")

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

WORK = os.environ.get("WORK", "${WORK:-/data/dp-nuplan}")
DATA = f"{WORK}/data/nuplan/nuplan-v1.1/splits/mini"
MAPS = f"{WORK}/data/nuplan/maps"
OUT = f"{REPO}/diffusion_planner/config/scenario_filter"

TYPES_14 = [
    "starting_left_turn", "starting_right_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "stopping_with_lead", "high_lateral_acceleration", "high_magnitude_speed",
    "low_magnitude_speed", "traversing_pickup_dropoff",
    "waiting_for_pedestrian_to_cross", "behind_long_vehicle",
    "stationary_in_traffic", "near_multiple_vehicles", "changing_lane",
    "following_lane_with_lead",
]


def get_filter(num_per_type, limit_total, tokens=None):
    # signature matches upstream data_process.get_filter_parameters order
    return ScenarioFilter(
        scenario_types=TYPES_14,        # 14 official eval types
        scenario_tokens=tokens,         # None -> discover by type
        log_names=None,                 # None -> all logs in builder (builder = mini split only)
        map_names=None,
        num_scenarios_per_type=num_per_type,
        limit_total_scenarios=limit_total,
        timestamp_threshold_s=15,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )


def discover(builder, num_per_type, limit_total):
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(get_filter(num_per_type, limit_total), worker)
    out = []
    for s in scenarios:
        out.append((s.token, s.scenario_type))
    return out


def yaml_body(tokens, num_per_type):
    token_lines = "\n".join(f'  - "{t}"' for t in tokens)
    return f"""_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: "all"

scenario_types: # the 14 official eval types
{TYPES_14_YAML}

scenario_tokens:
{token_lines}

log_names: null
map_names: null

num_scenarios_per_type: {num_per_type}
limit_total_scenarios: null
timestamp_threshold_s: 15
ego_displacement_minimum_m: null
ego_start_speed_threshold: null
ego_stop_speed_threshold: null
speed_noise_tolerance: null

expand_scenarios: false
remove_invalid_goals: true
shuffle: false
"""


TYPES_14_YAML = "\n".join(f"  - {t}" for t in TYPES_14)


def main():
    os.makedirs(OUT, exist_ok=True)
    builder = NuPlanScenarioBuilder(DATA, MAPS, None, None, "nuplan-maps-v1.0")
    # 1 scenario per type -> smoke set (M1)
    smoke = discover(builder, 1, None)
    tokens1 = [t for t, _ in smoke]
    with open(f"{OUT}/mini_one.yaml", "w") as f:
        f.write(yaml_body(tokens1, 1))
    print(f"mini_one.yaml: {len(tokens1)} scenarios")
    for t, ty in smoke:
        print(f"  {t}  {ty}")

    # baseline eval set (M4): up to 20 per type
    eval_set = discover(builder, 20, None)
    tokens2 = sorted({t for t, _ in eval_set})
    with open(f"{OUT}/mini_eval.yaml", "w") as f:
        f.write(yaml_body(tokens2, 20))
    print(f"mini_eval.yaml: {len(tokens2)} scenarios")


if __name__ == "__main__":
    main()
