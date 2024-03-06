from typing import Dict, ItemsView, List, Set, Tuple

import networkx as nx

from airlift.envs.route_map import RouteMap
from solution.common import get_globalstate


class PathsOffline:
    def __init__(self) -> None:
        self.paths: Dict[Tuple[int, int], int] = dict() # (From, To): Till Timestep

    def _get_paths_online(self, cur_time: int) -> None:
        paths_online = {path for path, time in self.paths.items() if cur_time>=time}
        for path in paths_online:
            self.paths.pop(path)

    def _get_path_offline(self, cur_time: int, timesteps: int, path: Tuple[int, int]) -> None:
        if path not in self.paths:
            self.paths[path] = cur_time + timesteps

    def update(self, cur_time: int, infos: Dict[str, Dict[str, List[str]]]) -> None:
        self._get_paths_online(cur_time)
        for agent_info in infos.values():
            for warning in agent_info["warnings"]:
                warning_split = str.split(warning)
                if warning_split[:2] != ["ROUTE", "FROM:"]:
                    continue
                timesteps = int(warning_split[-2])
                path = tuple(sorted([int(warning_split[2]), int(warning_split[4])]))
                self._get_path_offline(cur_time, timesteps, path)


def get_processing_time(globalstate: Dict) -> Dict:
    if len(globalstate["scenario_info"]) != 1:
        print("ERROR!!!!! Number of scenarios in scenario_info != 1")
    return globalstate["scenario_info"][0].processing_time

class DictGroupedPaths:
    def __init__(self, obs: Dict) -> None:
        globalstate = get_globalstate(obs)
        self.groups = {
            (plane_type, count): GroupedPaths(plane_type, obs, graph, subgroup) 
            for plane_type, graph in globalstate["route_map"].items()
            for count, subgroup in self._get_subgroup_dict(graph)
        }

    def get_subpath_group(self, path: List[int]) -> Tuple[List[int], Tuple[int, int]]:
        subpath = path[:2]
        rempath = path[2:]
        subpath_group: Tuple[int, int] = (-1,-1)
        for group, grouped_path in self.groups.items():
            if grouped_path.subgraph.has_edge(subpath[0], subpath[1]):
                subpath_group = group
                break

        for node in rempath:
            if self.groups[subpath_group].subgraph.has_edge(subpath[-1], node):
                subpath.append(node)

        return subpath, subpath_group
    
    def _get_subgroup_dict(self, graph: nx.graph) -> ItemsView[int, Set[int]]:
        return {count: subgroup for count, subgroup in enumerate(self._get_subgroups(graph))}.items()
    
    def _get_subgroups(self, graph: nx.graph) -> List[Set[int]]:
        return [c for c in nx.weakly_connected_components(graph) if len(c) > 1]
    
    def get_agent_group(self, agent: str) -> Tuple[int, int]:
        for group, grouped_path in self.groups.items():
            if agent in grouped_path.agents:
                return group

        return -1,-1

class GroupedPaths:
    def __init__(self, plane_type: int, obs: Dict, graph: nx.graph, subgroup: List[int]) -> None:
        self.subgroup = subgroup
        self.subgraph = nx.subgraph(graph, subgroup)
        self.agents = {
            agent 
            for agent, data in obs.items() 
            if data["plane_type"] == plane_type and data["current_airport"] in subgroup
        }

class PathInfo:
    def __init__(self, path: List[int], time: int, cost: float) -> None:
        self.path = path
        self.time = time
        self.cost = cost

class PathMatrix:
    # Paths stored based on cost
    def __init__(self, graph: nx.graph, globalstate: Dict) -> None:
        self.processing_time = get_processing_time(globalstate)
        self.graph: nx.graph = graph
        self._matrix: Dict[Tuple[int, int], PathInfo] = {}

    def _compute_shortest_path(self, orig: int, dest: int, from_to: Tuple[int, int]) -> None:
        def compute_path_time(path) -> int:
            return nx.path_weight(self.graph, path, weight="time") + (len(path*self.processing_time))

        path = nx.shortest_path(self.graph, orig, dest, weight="cost")
        self._matrix[from_to] = PathInfo(path, compute_path_time(path), nx.path_weight(self.graph, path, weight="cost"))
        for sub_ind in range(1, len(path)-1):
            sub_from_to = (path[sub_ind], dest)
            sub_path = path[sub_ind:len(path)]
            self._matrix[sub_from_to] = PathInfo(
                sub_path, compute_path_time(path), nx.path_weight(self.graph, path, weight="cost")
            )

    def get_path(self, orig: int, dest: int) -> PathInfo:
        from_to = (orig, dest)
        if from_to not in self._matrix:
            self._compute_shortest_path(orig, dest, from_to)       
        return self._matrix[from_to]