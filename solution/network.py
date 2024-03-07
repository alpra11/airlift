from typing import Any, Dict, ItemsView, List, Optional, Set, Tuple

import networkx as nx

from airlift.envs import ObservationHelper


class Network:
    def __init__(self, state: Dict[str, Any]) -> None:
        graph = ObservationHelper.get_multidigraph(state)
        self.graph = Graph(graph, state)
        self.groups = Groups(state)
        self.agents = Agents(state)
        self.free_agents = self._get_agents_by_group()

    def _get_agents_by_group(self) -> Dict[Tuple[int, int], Set[str]]:
        return {count: group.agents.copy() for count, group in self.groups.groups.items()}
    
    def get_subpath_group(self, orig: int, dest: int) -> Tuple[List[int], Tuple[int, int]]:
        path = self.graph.shortest_path(orig, dest)
        return self.groups.get_initial_subpath_group(path)
    
    def assign_agent(self, agent: str, c_id: int, group: Tuple[int, int], subpath: List[int]) -> None:
        self.agents.assign(agent, c_id, subpath)
        self.free_agents[group].remove(agent)
    
    def free(self, agent: str) -> None:
        self.agents.free(agent)
        group = self.groups.get_agent_group(agent)
        self.free_agents[group].add(agent)

def get_processing_time(state: Dict) -> int:
    # TODO: Unnecessary check, remove before submitting
    if len(state["scenario_info"]) != 1:
        print("ERROR!!!!! Number of scenarios in scenario_info != 1")
    return state["scenario_info"][0].processing_time

class Graph:
    def __init__(self, graph: nx.graph, state: Dict[str, Any]) -> None:
        self.processing_time = get_processing_time(state)
        self.graph = graph

    def shortest_path(self, orig: int, dest: int) -> List[int]:
        return nx.shortest_path(self.graph, orig, dest, weight="cost")
    
    def shortest_path_time(self, orig: int, dest: int) -> int:
        path = self.shortest_path(orig, dest)
        return nx.path_weight(self.graph, path, weight="time") + (len(path*self.processing_time))
    
    def shortest_path_cost(self, orig: int, dest: int) -> int:
        path = self.shortest_path(orig, dest)
        return nx.path_weight(self.graph, path, weight="cost")


    @staticmethod
    def get_shortest_path_for_graph(graph: nx.graph, orig: int, dest: int) -> List[int]:
        return nx.shortest_path(graph, orig, dest, weight="cost")
    
class GroupedPaths:
    def __init__(self, plane_type: int, state: Dict[str, Any], graph: nx.graph, airports: List[int]) -> None:
        self.airports = airports
        self.subgraph = nx.subgraph(graph, airports)
        self.agents = self._get_agents(plane_type, state)

    def _get_agents(self, plane_type: int, state: Dict[str, Any]) -> Set[int]:
        return {
            agent
            for agent, data in state["agents"].items()
            if data["plane_type"] == plane_type and data["current_airport"] in self.airports
        }

class Groups:
    def __init__(self, state: Dict[str, Any]) -> None:
        self.groups = self._get_groups_dict(state)

    def _get_groups_dict(self, state: Dict[str, Any]) -> Dict[Tuple[int, int], GroupedPaths]:
        def _get_grouped_airports(graph: nx.graph) -> List[Set[int]]:
            return [aps for aps in nx.weakly_connected_components(graph) if len(aps) > 1]
        
        def _get_groups(graph: nx.graph) -> ItemsView[int, Set[int]]:
            return {c: aps for c, aps in enumerate(_get_grouped_airports(graph))}.items()

        return {
            (plane_type, count): GroupedPaths(plane_type, state, graph, airports)
            for plane_type, graph in state["route_map"].items()
            for count, airports in _get_groups(graph)
        }

    def get_initial_subpath_group(self, path: List[int]) -> Tuple[List[int], Tuple[int, int]]:
        subpath = path[:2]
        rempath = path[2:]
        sub_path_group = (-1,-1)
        for group, grouped_paths in self.groups.items():
            if grouped_paths.subgraph.has_edge(subpath[0], subpath[1]):
                sub_path_group = group
                subgraph = grouped_paths.subgraph
                break

        for node in rempath:
            if not subgraph.has_edge(subpath[-1], node):
                break
            subpath.append(node)

        return subpath, sub_path_group
    
    def get_agent_group(self, agent: str) -> Tuple[int, int]:
        for group, grouped_paths in self.groups.items():
            if agent in grouped_paths.agents:
                return group
        return -1,-1

class AgentInfo:
    def __init__(self, agent: str) -> None:
        self.name = agent
        self.cargo_id: Optional[int] = None 
        self.path: Optional[List[int]] = None

class Agents:
    def __init__(self, state: Dict[str, Any]) -> None:
        self.agents = {a: AgentInfo(a) for a in state["agents"]}

    def assign(self, agent: str, cargo_id: int, path: List[int]) -> None:
        self.agents[agent].cargo_id = cargo_id
        self.agents[agent].path = path

    def update_path(self, agent:str, path: List[int]) -> None:
        self.agents[agent].path = path

    def free(self, agent: str) -> None:
        self.agents[agent].cargo_id = None
        self.agents[agent].path = None

    def is_free(self, agent: str) -> bool:
        return self.agents[agent].cargo_id is None
    
    def get_cargo(self, agent: str) -> Optional[int]:
        return self.agents[agent].cargo_id
    
    def get_path(self, agent: str) -> Optional[List[int]]:
        return self.agents[agent].path