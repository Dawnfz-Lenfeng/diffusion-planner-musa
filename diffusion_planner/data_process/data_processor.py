# Diffusion-Planner data_processor.py — parallelized version (mini/full)
# Based on upstream ZhengYinan-AIR/Diffusion-Planner@main data_processor.py.
# Changes vs upstream:
#   1. work() scenario loop parallelized with joblib (issue #69: serial took ~600h for 1M scenarios)
#   2. per-scenario try/except -> failures logged to <save_dir>/failed_tokens.txt, loop continues
#   3. idempotent: skips scenarios whose .npz already exists (resume after crash)
#   4. num_proc configurable via config.num_proc (argparse --num_proc), default = os.cpu_count()
#   5. save_to_disk: atomic write (tmp file + rename) so a crash never leaves half-written npz

import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process,
    sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list,
    agent_future_process,
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import (
    get_ego_past_array_from_scenario,
    get_ego_future_array_from_scenario,
    calculate_additional_ego_states,
)
from diffusion_planner.data_process.utils import convert_to_model_inputs


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100  # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']
        self._max_elements = {
            'LANE': config.lane_num,
            'LEFT_BOUNDARY': config.lane_num,
            'RIGHT_BOUNDARY': config.lane_num,
            'ROUTE_LANES': config.route_num,
        }
        self._max_points = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.route_len,
        }
        self._num_proc = int(getattr(config, "num_proc", 0)) or os.cpu_count() or 4

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None  # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents,
                               static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit,
                                 lane_route, self._map_features, self._max_elements, self._max_points)

        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data

    # ------------------------------------------------------------------
    # Single-scenario processing extracted from upstream work() loop body.
    # Runs in worker processes; all state it needs comes from self/scenario.
    # ------------------------------------------------------------------
    def process_scenario(self, scenario, save_dir):
        map_name = scenario._map_name
        token = scenario.token
        out_file = os.path.join(save_dir, f"{map_name}_{token}.npz")
        if os.path.exists(out_file):
            return token  # idempotent resume

        map_api = scenario.map_api

        ''' ego & agents past '''
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
                                    dtype=np.float64)
        ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
            scenario, self.num_past_poses, self.past_time_horizon)

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses)
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)

        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents,
                               static_objects, static_objects_types, self.num_static, self.max_ped_bike,
                               anchor_ego_state)

        ''' Map '''
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data)

        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit,
                                 lane_route, self._map_features, self._max_elements, self._max_points)

        ''' ego & agents future '''
        ego_agent_future = get_ego_future_array_from_scenario(
            scenario, ego_state, self.num_future_poses, self.future_time_horizon)

        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses)
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list,
                                                      self.num_agents, neighbor_indices)

        ''' ego current '''
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # gather data
        data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future, "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
        data.update(vector_map)

        self.save_to_disk(save_dir, data, out_file)
        return token

    # Use for data preprocess
    def work(self, scenarios):
        save_dir = self._save_dir
        failed_path = os.path.join(save_dir, "failed_tokens.txt")
        done = 0

        def _one(scenario):
            try:
                return self.process_scenario(scenario, save_dir)
            except Exception as e:
                msg = f"{scenario.token}\t{scenario._map_name}\t{type(e).__name__}: {e}"
                with open(failed_path, "a") as f:
                    f.write(msg + "\n")
                return None

        n_proc = min(self._num_proc, max(1, len(scenarios)))
        print(f"DataProcessor.work: {len(scenarios)} scenarios, n_proc={n_proc}")
        for _ in tqdm(Parallel(n_jobs=n_proc, verbose=5, batch_size=4)(
                delayed(_one)(s) for s in scenarios)):
            done += 1

        n_failed = (open(failed_path).read().count("\n")) if os.path.exists(failed_path) else 0
        print(f"work() finished: submitted={done}, failed(log)={n_failed}")

    def save_to_disk(self, dir, data, out_file=None):
        if out_file is None:
            out_file = os.path.join(dir, f"{data['map_name']}_{data['token']}.npz")
        tmp_file = out_file + ".tmp.npz"
        np.savez(tmp_file, **data)
        os.replace(tmp_file, out_file)  # atomic on POSIX
