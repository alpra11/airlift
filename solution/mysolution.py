from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper

import networkx as nx

from typing import Tuple, Dict, List

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

        global_state = next(iter(obs.values()))["globalstate"]
        graph = ObservationHelper.get_multidigraph(global_state)
        self.path_matrix = PathMatrix(graph)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def increment_timestep(self) -> None:
        self.timestep += 1

    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        self.increment_timestep()
        return self._action_helper.sample_valid_actions(obs)

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