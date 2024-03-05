from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper
from airlift.envs.airlift_env import ObservationHelper
from airlift.envs.agents import PlaneState
from airlift.envs.airport import NOAIRPORT_ID

import networkx as nx

from typing import Tuple, Dict, List, Set, Optional, ItemsView

DEFAULT_ACTION = {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': NOAIRPORT_ID}

def action_pickup_cargo(cargo_id: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [cargo_id], 'cargo_to_unload': [], 'destination': NOAIRPORT_ID}

def action_drop_cargo(cargo_id: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [cargo_id], 'destination': NOAIRPORT_ID}

def action_set_destination(destination: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': destination}

def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]

def get_processing_time(globalstate: Dict) -> Dict:
    if len(globalstate["scenario_info"]) != 1:
        print("ERROR!!!!! Number of scenarios in scenario_info != 1")
    return globalstate["scenario_info"][0].processing_time

class LocInfo:
    def __init__(self, time: int, loc: int) -> None:
        self.time = time
        self.loc = loc
        self.agent: Optional[str] = None

    def is_assigned(self) -> bool:
        return self.agent is not None
    
    def assign_agent(self, agent: str) -> None:
        self.agent = agent
    
    def unassign(self) -> None:
        self.agent = None

class CargoEstimate:
    def __init__(self, id: int, dest: int, time: int, loc: int) -> None:
        self.id = id
        self.dest = dest
        self.cur_loc= LocInfo(time, loc)
        self.next_loc: Optional[LocInfo] = None

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

        self.agents: Dict[str, Agent] = {a: Agent(a) for a in obs}
        self.assignable_agents = self.get_assignable_agents()
        self.cargo_estimate: Dict[int, CargoEstimate] = {
            c.id: CargoEstimate(c.id, c.destination, c.earliest_pickup_time, c.location) 
            for c in globalstate["active_cargo"]
        }

        self.valid_actions: Dict[str, Dict] = dict()

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def increment_timestep(self) -> None:
        self.timestep += 1

    def update_estimate_dict(self, obs: Dict) -> Dict:
        globalstate = get_globalstate(obs)
        for c in globalstate["event_new_cargo"]:
            self.cargo_estimate[c.id] = CargoEstimate(c.id, c.destination, c.earliest_pickup_time, c.location)
    
    def get_sorted_cargo(self, globalstate: Dict) -> List:
        return sorted(
            globalstate["active_cargo"], 
            key=lambda x: 
            (x.soft_deadline - self.path_matrix.get_path(x.location, x.destination).time)
            if x.location != 0 else 100000
            #     x.location if x.location!=0 else self.cargo_estimate[x.id].next_loc.loc, 
            #     x.destination
            # ).time
        )
    
    def get_assignable_agents(self) -> Dict[Tuple[int, int], Set[str]]:
        return {
            group_no: {a for a in group.agents if self.agents[a].is_free()} 
            for group_no, group in self.grouped_paths.groups.items()
        }
    
    def make_agent_unassignable(self, agent: str, group: Tuple[int, int]) -> None:
        self.assignable_agents[group].remove(agent)
    
    def is_group_assignable(self, group: Tuple[int, int]) -> bool:
        return len(self.assignable_agents[group]) > 0

    def get_subpath_group(self, c_loc: int, c_dest: int) -> Tuple[List[int], Tuple[int, int]]:
        path = self.path_matrix.get_path(c_loc, c_dest)
        return self.grouped_paths.get_subpath_group(path.path)
    
    def assign_agent(self, c_id: int, c_dest: int, loc_info: LocInfo, obs: Dict) -> Optional[str]:
        if loc_info.is_assigned():
            return None
        c_loc = loc_info.loc
        subpath, group = self.get_subpath_group(c_loc, c_dest)
        if not self.is_group_assignable(group):
            return None

        agent = min(
            [
                (agent, self.path_matrix.get_path(obs[agent]["current_airport"], c_loc))   
                for agent in self.assignable_agents[group]
            ],
            key=lambda x: x[1].cost
        )[0]
        loc_info.assign_agent(agent)
        self.agents[agent].assign_cargo(c_id, subpath)
        self.make_agent_unassignable(agent, group)
        return agent

    def assign_free_agents(self, obs: Dict) -> None:
        globalstate = get_globalstate(obs)
        for cargo in self.get_sorted_cargo(globalstate):
            c_id = cargo.id
            c_dest = cargo.destination
            if cargo.location == 0:
                pass
            #     self.assign_agent(c_id, c_dest, self.cargo_estimate[c_id].next_loc, obs)
            else:
                self.assign_agent(c_id, c_dest, self.cargo_estimate[c_id].cur_loc, obs)

    def compute_waiting_actions(self, agent: str, agent_state: Dict) -> None:
        cargo_id = self.agents[agent].cargo_id
        cur_airport = agent_state["current_airport"]
        if cargo_id in agent_state["cargo_at_current_airport"]:
            self.valid_actions[agent] = action_pickup_cargo(cargo_id)
        elif (
            self.agents[agent].path[-1] == cur_airport and 
            cargo_id in agent_state["cargo_onboard"]
        ):
            self.valid_actions[agent] = action_drop_cargo(cargo_id)
            self.cargo_estimate[cargo_id].cur_loc.loc = cur_airport
            if cur_airport != self.cargo_estimate[cargo_id].dest:
                self.cargo_estimate[cargo_id].cur_loc.unassign()
        else:
            self.valid_actions[agent] = DEFAULT_ACTION
    
    def compute_takeoff_actions(self, agent: str, agent_state: Dict, obs: Dict) -> None:
        globalstate = get_globalstate(obs)
        graph = ObservationHelper.get_multidigraph(globalstate)
        cargo_id = self.agents[agent].cargo_id
        path = self.agents[agent].path
        cur_airport = agent_state["current_airport"]
        if cargo_id in agent_state["cargo_onboard"]:
            if path[0] != cur_airport:
                new_path = nx.shortest_path(graph, cur_airport, path[0], weight="cost")
                self.valid_actions[agent] = action_set_destination(new_path[1])
            else:
                if not graph.has_edge(path[0], path[1]):
                    new_path = nx.shortest_path(graph, path[0], path[-1], weight="cost")
                    self.valid_actions[agent] = action_set_destination(new_path[1])
                else:
                    self.agents[agent].path = path[1:]
                    self.valid_actions[agent] = action_set_destination(path[1])
        else:
            if len(self.agents[agent].path) == 1 and self.agents[agent].path[-1] == cur_airport:
                self.agents[agent].unassign()
                group = self.grouped_paths.get_agent_group(agent)
                self.assignable_agents[group].add(agent)
                self.valid_actions[agent] = DEFAULT_ACTION
            elif cur_airport == self.agents[agent].path[0]:
                if cargo_id in agent_state["cargo_at_current_airport"]:
                    self.valid_actions[agent] = action_pickup_cargo(cargo_id)
                else:
                    self.valid_actions[agent] = DEFAULT_ACTION
            else:
                new_path = nx.shortest_path(graph, cur_airport, path[0], weight="cost")
                self.valid_actions[agent] = action_set_destination(new_path[1])


    def execute_assigned_agents(self, obs: Dict) -> None:
        for agent in self.agents:
            agent_state = obs[agent]
            if self.agents[agent].is_free():
                self.valid_actions[agent] = DEFAULT_ACTION
                continue
            if agent_state["state"] == PlaneState.WAITING:
                self.compute_waiting_actions(agent, agent_state)            
            elif agent_state["state"] == PlaneState.READY_FOR_TAKEOFF:
                self.compute_takeoff_actions(agent, agent_state, obs)
            else:
                self.valid_actions[agent] = DEFAULT_ACTION
            
    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        self.update_estimate_dict(obs)

        self.assign_free_agents(obs)
        self.execute_assigned_agents(obs)

        self.increment_timestep()
        return self.valid_actions

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

class Agent:
    def __init__(self, agent: str) -> None:
        self.name = agent
        self.cargo_id: Optional[int] = None 
        self.path: Optional[List[int]] = None

    def assign_cargo(self, cargo_id: int, path: List[int]) -> None:
        self.cargo_id = cargo_id
        self.path = path

    def unassign(self) -> None:
        self.cargo_id = None
        self.path = None

    def is_free(self) -> bool:
        return self.cargo_id is None

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