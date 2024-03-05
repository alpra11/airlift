from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper
from airlift.envs.airport import NOAIRPORT_ID

import networkx as nx

from typing import Tuple, Dict, List, Set, Optional, ItemsView

def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]

def get_processing_time(globalstate: Dict) -> Dict:
    if len(globalstate["scenario_info"]) != 1:
        print("ERROR!!!!! Number of scenarios in scenario_info != 1")
    return globalstate["scenario_info"][0].processing_time

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

        globalstate = get_globalstate(obs)
        graph = ObservationHelper.get_multidigraph(globalstate)
        self.path_matrix = PathMatrix(graph, globalstate)
        self.grouped_paths = DictGroupedPaths(obs)
        self.cargo_estimate: Dict[int, CargoEstimate] = {
            c.id: CargoEstimate(c.id, c.earliest_pickup_time, c.location) 
            for c in globalstate["active_cargo"]
        }
        self.agents: Dict[str, Agent] = {a: Agent(a) for a in obs}

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def increment_timestep(self) -> None:
        self.timestep += 1

    def update_cargo_estimate(self, globalstate: Dict) -> Dict:
        for c in globalstate["event_new_cargo"]:
            self.cargo_estimate[c.id] = CargoEstimate(c.id, c.earliest_pickup_time, c.location)
    
    def get_sorted_cargo(self, globalstate: Dict) -> List:
        return sorted(
            globalstate["active_cargo"], 
            key=lambda x: 
            x.soft_deadline - self.path_matrix.get_path(
                x.location if x.location!=0 else self.cargo_estimate[x.id].estimated_destination, 
                x.destination
            ).time
        )

    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        globalstate = get_globalstate(obs)
        self.update_cargo_estimate(globalstate)

        for _ in self.get_sorted_cargo(globalstate):
            pass
            
        self.increment_timestep()

        return self._action_helper.sample_valid_actions(obs)


class DictGroupedPaths:
    def __init__(self, obs: Dict) -> None:
        globalstate = get_globalstate(obs)
        self.groups = {
            (plane_type, count): GroupedPaths(plane_type, obs, graph, subgroup) 
            for plane_type, graph in globalstate["route_map"].items()
            for count, subgroup in self._get_subgroup_dict(graph)
        }
    
    def _get_subgroup_dict(self, graph: nx.graph) -> ItemsView[int, Set[int]]:
        return {count: subgroup for count, subgroup in enumerate(self._get_subgroups(graph))}.items()
    
    def _get_subgroups(self, graph: nx.graph) -> List[Set[int]]:
        return [c for c in nx.weakly_connected_components(graph) if len(c) > 1]

class GroupedPaths:
    def __init__(self, plane_type: int, obs: Dict, graph: nx.graph, subgroup: List[int]) -> None:
        self.subgroup = subgroup
        self.subgraph = nx.subgraph(graph, subgroup)
        self.aircrafts = {
            aircraft 
            for aircraft, data in obs.items() 
            if data["plane_type"] == plane_type and data["current_airport"] in subgroup
        }

class CargoEstimate:
    def __init__(self, id: int, estimated_time: int, estimated_destination: int) -> None:
        self.id = id
        self.estimated_time = estimated_time
        self.estimated_destination = estimated_destination
        self.assigned: Optional[str] = None

    def assign_to(self, agent: str) -> None:
        self.assigned = agent
    
    def unassign(self) -> None:
        self.assigned = None

class Agent:
    def __init__(self, agent: str) -> None:
        self.name = agent
        self.assigned_to: Optional[int] = None 
        self.path: Optional[List[int]] = None

class PathInfo:
    def __init__(self, path: List[int], time: float) -> None:
        self.path = path
        self.time = time

class PathMatrix:
    # Paths stored based on cost
    def __init__(self, graph: nx.graph, globalstate: Dict) -> None:
        self.processing_time = get_processing_time(globalstate)
        self.graph: nx.graph = graph
        self._matrix: Dict[Tuple[int, int], PathInfo] = {}

    def _compute_shortest_path(self, orig: int, dest: int, from_to: Tuple[int, int]):
        def compute_path_time(path) -> int:
            return nx.path_weight(self.graph, path, weight="time") + (len(path*self.processing_time))

        path = nx.shortest_path(self.graph, orig, dest, weight="cost")
        self._matrix[from_to] = PathInfo(path, compute_path_time(path))
        for sub_ind in range(1, len(path)-1):
            sub_from_to = (path[sub_ind], dest)
            sub_path = path[sub_ind:len(path)]
            self._matrix[sub_from_to] = PathInfo(sub_path, nx.path_weight(self.graph, path, weight="time"))

    def get_path(self, orig: int, dest: int):
        from_to = (orig, dest)
        if from_to not in self._matrix:
            self._compute_shortest_path(orig, dest, from_to)       
        return self._matrix[from_to]