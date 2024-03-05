from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper

import networkx as nx

from collections import defaultdict
from typing import Tuple, Dict, List, Set

def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]

class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)
        self.timestep = 0

        global_state = get_globalstate(obs)
        graph = ObservationHelper.get_multidigraph(global_state)
        self.path_matrix = PathMatrix(graph)
        self.grouped_routes = GroupedRoutes(obs)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def increment_timestep(self) -> None:
        self.timestep += 1

    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        self.increment_timestep()
        return self._action_helper.sample_valid_actions(obs)


class GroupedRoutes:
    def __init__(self, obs: Dict) -> None:
        global_state = get_globalstate(obs)
        self.groups = {
            plane_type: self._get_subgroup_dict(route)
            for plane_type, route in global_state["route_map"].items()
        }
        self.aircrafts = self._get_aircrafts_in_groups(obs)

    def _get_aircrafts_in_groups(self, obs: Dict) -> Dict[int, Dict[int, Set[str]]]:
        aircrafts_in_groups = {
            plane_type: {
                subgroup: set() for subgroup in subgroup_dict} 
                for plane_type, subgroup_dict in self.groups.items()
            }
        for aircraft, data in obs.items():
            plane_type = data["plane_type"]
            for group_key, group_nodes in self.groups[plane_type].items():
                if data["current_airport"] in group_nodes:
                    aircrafts_in_groups[plane_type][group_key].add(aircraft)
                    break
        return aircrafts_in_groups    

    def _get_subgroup_dict(self, route: nx.graph) -> Dict[int, Set[int]]:
        return {count: group for count, group in enumerate(self._get_subgroups(route))}
    
    def _get_subgroups(self, route: nx.graph) -> List[Set[int]]:
        return [c for c in nx.weakly_connected_components(route) if len(c) > 1]
    

class PathMatrix:
    def __init__(self, graph: nx.graph) -> None:
        self.graph: nx.graph = graph
        self._matrix: Dict[Tuple[int, int], List[int]] = {}

    def _compute_shortest_path(self, orig: int, dest: int, from_to: Tuple[int, int]):
        path = nx.shortest_path(self.graph, orig, dest, weight="cost")
        self._matrix[from_to] = path
        for sub_ind in range(1, len(path)-1):
            sub_from_to = (path[sub_ind], dest)
            self._matrix[sub_from_to] = path[sub_ind:len(path)]

    def get_path(self, orig: int, dest: int):
        from_to = (orig, dest)
        if from_to not in self._matrix:
            self._compute_shortest_path(orig, dest, from_to)       
        return self._matrix[from_to]