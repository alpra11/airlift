from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from airlift.envs import ActionHelper, ObservationHelper
from airlift.envs.agents import PlaneState
from airlift.solutions import Solution
from solution.action import ValidActions
from solution.agent import Agent
from solution.cargo import CargoEstimate, LocInfo
from solution.common import get_globalstate
from solution.network import DictGroupedPaths, PathMatrix, PathsOffline


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

        self.paths_offline = PathsOffline()

        graph = ObservationHelper.get_multidigraph(globalstate)
        self.path_matrix = PathMatrix(graph, globalstate)
        self.grouped_paths = DictGroupedPaths(obs)

        self.agents: Dict[str, Agent] = {a: Agent(a) for a in obs}
        self.assignable_agents = self.get_assignable_agents()
        self.cargo_estimate: Dict[int, CargoEstimate] = {
            c.id: CargoEstimate(c.id, c.destination, c.earliest_pickup_time, c.location) 
            for c in globalstate["active_cargo"]
        }

        self.valid_actions = ValidActions()

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
            self.valid_actions.process_cargo(agent, cargo_to_load=[cargo_id])
        elif (
            self.agents[agent].path[-1] == cur_airport and 
            cargo_id in agent_state["cargo_onboard"]
        ):
            self.valid_actions.process_cargo(agent, cargo_to_unload=[cargo_id])
            self.cargo_estimate[cargo_id].cur_loc.loc = cur_airport
            if cur_airport != self.cargo_estimate[cargo_id].dest:
                self.cargo_estimate[cargo_id].cur_loc.unassign()
        else:
            # NO_ACTION
            pass
    
    def compute_takeoff_actions(self, agent: str, agent_state: Dict, obs: Dict) -> None:
        globalstate = get_globalstate(obs)
        graph = ObservationHelper.get_multidigraph(globalstate)
        cargo_id = self.agents[agent].cargo_id
        path = self.agents[agent].path
        cur_airport = agent_state["current_airport"]
        if cargo_id in agent_state["cargo_onboard"]:
            if path[0] != cur_airport:
                new_path = nx.shortest_path(graph, cur_airport, path[0], weight="cost")
                self.valid_actions.take_off(agent, new_path[1])
            else:
                if not graph.has_edge(path[0], path[1]):
                    new_path = nx.shortest_path(graph, path[0], path[-1], weight="cost")
                    self.valid_actions.take_off(agent, new_path[1])
                else:
                    self.agents[agent].path = path[1:]
                    self.valid_actions.take_off(agent, path[1])
        else:
            if len(self.agents[agent].path) == 1 and self.agents[agent].path[-1] == cur_airport:
                self.agents[agent].unassign()
                group = self.grouped_paths.get_agent_group(agent)
                self.assignable_agents[group].add(agent)
            elif cur_airport == self.agents[agent].path[0]:
                if cargo_id in agent_state["cargo_at_current_airport"]:
                    self.valid_actions.process_cargo(agent, cargo_to_load=[cargo_id])
                else:
                    # NO_ACTION
                    pass
            else:
                new_path = nx.shortest_path(graph, cur_airport, path[0], weight="cost")
                self.valid_actions.take_off(agent, new_path[1])


    def execute_assigned_agents(self, obs: Dict) -> None:
        for agent in self.agents:
            agent_state = obs[agent]
            if self.agents[agent].is_free():
                continue
            if agent_state["state"] == PlaneState.WAITING:
                self.compute_waiting_actions(agent, agent_state)            
            elif agent_state["state"] == PlaneState.READY_FOR_TAKEOFF:
                self.compute_takeoff_actions(agent, agent_state, obs)
            else:
                # NO_ACTION
                pass
            
    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        self.valid_actions.reset_actions(obs.keys())
        self.update_estimate_dict(obs)
        if infos:
            self.paths_offline.update(self.timestep, infos)

        self.assign_free_agents(obs)
        self.execute_assigned_agents(obs)

        self.increment_timestep()
        return self.valid_actions.get_actions()    