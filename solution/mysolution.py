from typing import Any, Dict, List, Optional

import networkx as nx

from airlift.envs import ActionHelper, ObservationHelper
from airlift.envs.agents import PlaneState
from airlift.envs.cargo import Cargo
from airlift.solutions import Solution
from solution.actions import Actions
from solution.cargo_plan import CargoPlan
from solution.helper import get_globalstate, process_infos
from solution.network import Network


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
        
        self.timestep = -1
        state = get_globalstate(obs)

        self.network = Network(state)
        self.cargo = CargoPlan(state)
        self.actions = Actions(state)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def increment_timestep(self) -> None:
        self.timestep += 1

    def update_data(self, state: Dict[str, Any]) -> None:
        self.increment_timestep()
        self.actions.reset()
        self.cargo.update(state)

    def get_sorted_cargo(self, state: Dict[str, Any]) -> List:
        # TODO: Add logic to use the next loc if the current location is already crossed
        return sorted(
            state["active_cargo"], 
            key=lambda x: 
            x.soft_deadline - self.network.graph.shortest_path_time(self.cargo.get_location(x.id), x.destination)
        )
    
    def assign_agent(self, cargo: Cargo, state: Dict[str, Any]) -> Optional[str]:
        if self.cargo.is_assigned(cargo.id):
            return None
        
        subpath, group = self.network.get_subpath_group(cargo.location, cargo.destination)

        if len(self.network.free_agents[group]) == 0:
            return None

        agent_cost_pairs = (
            (
                agent, 
                self.network.graph.shortest_path_cost(state["agents"][agent]["current_airport"], cargo.location)
            ) 
            for agent in self.network.free_agents[group]
        )
        agent = min(agent_cost_pairs, key=lambda x:x[1])[0]
        
        self.cargo.assign_agent(agent, cargo.id)
        self.network.assign_agent(agent, cargo.id, group, subpath)
        
        return agent
    
    def assign_free_agents(self, state: Dict[str, Any]) -> None:
        for cargo in self.get_sorted_cargo(state):
            if cargo.location != 0:
                self.assign_agent(cargo, state)

    def compute_waiting_actions(self, agent: str, agent_state: Dict[str, Any]) -> None:
        c_id = self.network.agents.get_cargo(agent)
        cur_airport = agent_state["current_airport"]

        if c_id in agent_state["cargo_at_current_airport"]:
            self.actions.process_cargo(agent, cargo_to_load=[c_id])
        elif cur_airport == self.network.agents.get_path(agent)[-1]:
            self.actions.process_cargo(agent, cargo_to_unload=[c_id])
            self.cargo.unassign(c_id, cur_airport)

    def compute_takeoff_actions(self, agent: str, agent_state: Dict[str, Any], state: Dict[str, Any]) -> None:
        graph = ObservationHelper.get_multidigraph(state)
        agents = self.network.agents
        c_id = agents.get_cargo(agent)
        ass_path = agents.get_path(agent)
        cur_airport = agent_state["current_airport"]

        if c_id in agent_state["cargo_onboard"]:
            if not graph.has_edge(ass_path[0], ass_path[1]):
                new_path = nx.shortest_path(graph, cur_airport, ass_path[-1], weight="cost")
                self.actions.take_off(agent, new_path[1])
                agents.update_path(agent, new_path[1:])
            else:
                self.actions.take_off(agent, ass_path[1])
                agents.update_path(agent, ass_path[1:])
        else:
            if len(ass_path) == 1:
                # TODO: Remove check when submitting
                if ass_path[0] == cur_airport:
                    self.network.free(agent)
                else:
                    print(f"ERROR: Agent {agent} has path {ass_path} while being at airport {cur_airport}!!")
            elif cur_airport == ass_path[0]:
                # TODO: Remove check when submitting
                if c_id in agent_state["cargo_at_current_airport"]:
                    self.actions.process_cargo(agent, cargo_to_load=[c_id])
            else:
                new_path = nx.shortest_path(graph, cur_airport, ass_path[0], weight="cost")
                self.actions.take_off(agent, new_path[1])
    
    def assign_actions(self, state: Dict[str, Any]) -> None:
        for agent, agent_state in state["agents"].items():
            if self.network.agents.is_free(agent):
                continue
            if agent_state["state"] in [PlaneState.PROCESSING, PlaneState.MOVING]:
                continue
            if agent_state["state"] == PlaneState.WAITING:
                self.compute_waiting_actions(agent, agent_state)            
            elif agent_state["state"] == PlaneState.READY_FOR_TAKEOFF:
                self.compute_takeoff_actions(agent, agent_state, state)

    def policies(self, obs, dones, infos):
        # Use the acion helper to generate an action
        state = get_globalstate(obs)
        process_infos(infos)

        self.update_data(state)

        self.assign_free_agents(state)
        self.assign_actions(state)

        print(self.actions.get_actions()["a_0"])

        return self.actions.get_actions()    