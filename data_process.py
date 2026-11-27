# Diffusion-Planner data_process.py — mini-split-ready version
# Based on upstream ZhengYinan-AIR/Diffusion-Planner@main data_process.py.
# Changes vs upstream:
#   1. --log_list: path to the log-name JSON (upstream hardcoded './nuplan_train.json').
#      For nuPlan mini, generate one with scripts/make_mini_log_list.py.
#   2. --num_proc: worker processes for the parallelized DataProcessor (patches/data_processor_parallel.py).
#   3. Reads the replacement DataProcessor from our patched module.

import os
import argparse
import json

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

# our patched processor (parallel + fault tolerant), same class name as upstream
from diffusion_planner.data_process.data_processor import DataProcessor  # noqa: F401  (ensures pkg import path)


def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num)
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data (dir containing .db files; for mini: .../splits/mini)')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data (dir containing nuplan-maps-v1.0.json)')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data (.npz per scenario)')
    parser.add_argument('--log_list', default='./nuplan_train.json', type=str, help='JSON list of log names to keep; upstream default ./nuplan_train.json')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)

    parser.add_argument('--num_proc', type=int, default=0, help='worker processes for data extraction (0 = cpu_count)')

    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    with open(args.log_list, "r", encoding="utf-8") as file:
        log_names = json.load(file)
    print(f"log_list={args.log_list}: {len(log_names)} logs")

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter
    processor = DataProcessor(args)
    processor.work(scenarios)

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(sorted(npz_files), json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")
