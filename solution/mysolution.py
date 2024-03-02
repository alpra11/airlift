from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airport import NOAIRPORT_ID
from airlift.envs.agents import PlaneState
from airlift.envs.airlift_env import ObservationHelper

import networkx as nx

class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()
        self.loaded_cargo = set()       

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones, infos):
        cargo_ids_assigned = set()
        # Use the action helper to generate an action
        actions = {}

        agents = [a for a in obs.items()]
        sorted(agents, key=lambda a: a[1]["state"])
        for (a, agent) in agents:
            plane_state = agent["state"]
            current_airport = agent["current_airport"]
            max_weight = agent["max_weight"]
            cur_weight = agent["current_weight"]
            available_destinations = agent["available_routes"]
            global_state = agent["globalstate"]
            plane_type = agent["plane_type"]
            active_cargo = global_state["active_cargo"]
            cargo_at_current_airport = agent["cargo_at_current_airport"]
            cargo_onboard = agent["cargo_onboard"]
            destination = agent["destination"]
        
            if plane_state == PlaneState.WAITING:
                cargo_to_unload = []
                cargo_to_load = []
                # unload
                for cargo in ObservationHelper.get_active_cargo_info(global_state, cargo_onboard) or []:
                    graph = ObservationHelper.get_multidigraph(global_state)
                    path = nx.shortest_path(graph, current_airport, cargo.destination, weight="cost")
                    if cargo.destination == current_airport or path[1] not in available_destinations:
                        cargo_to_unload.append(cargo.id)
                        if cargo.id in self.loaded_cargo:
                            self.loaded_cargo.remove(cargo.id)
                        cur_weight -= cargo.weight
                        #print(f"Unload {cargo.id} from {a}")

                # load
                for cargo in ObservationHelper.get_active_cargo_info(global_state, cargo_at_current_airport) or []:
                    if cargo.weight <= max_weight-cur_weight and cargo.id not in cargo_ids_assigned and cargo.id not in self.loaded_cargo:
                        graph = ObservationHelper.get_multidigraph(global_state)
                        path = nx.shortest_path(graph, current_airport, cargo.destination, weight="cost")
                        if path[1] in available_destinations:
                            cargo_to_load.append(cargo.id)
                            self.loaded_cargo.add(cargo.id)
                            cur_weight += cargo.weight
                            cargo_ids_assigned.add(cargo.id)
                            #print(f"Load {cargo.id} on {a} destination {cargo.destination}")
                
                actions[a] = {"priority": None,
                            "cargo_to_load": cargo_to_load,
                            "cargo_to_unload": cargo_to_unload,
                            "destination": NOAIRPORT_ID}

            elif plane_state == PlaneState.READY_FOR_TAKEOFF:

                cargo_to_ship = ObservationHelper.get_active_cargo_info(global_state, cargo_onboard) or []
                sorted(cargo_to_ship, key=lambda x: x.hard_deadline)

                # destination from cargo on board
                destination = NOAIRPORT_ID
                for cargo in cargo_to_ship:
                    if cargo.destination in available_destinations:
                        destination = cargo.destination
                        #print(f"Destination for {a} to {destination} for {cargo.id}")
                        break
                    else:
                        graph = ObservationHelper.get_multidigraph(global_state)
                        path = nx.shortest_path(graph, current_airport, cargo.destination, weight="cost")
                        if path[1] in available_destinations:
                            destination = path[1]
                            #print(f"Destination for {a} to {destination} for {cargo.id} final dest {cargo.destination}, path is {path}")
                            break
                        else:
                            print(f"Plane {a} can't deliver {cargo.id} to {cargo.destination}, path is {path}, avail dests {available_destinations}")

                # destination from cargo on destinations
                if destination == NOAIRPORT_ID:
                    sorted(active_cargo, key=lambda x: x.earliest_pickup_time)
                    for cargo in active_cargo:
                        if cargo.id not in cargo_ids_assigned and cargo.location > 0:
                            if cargo.location in available_destinations or cargo.location == current_airport:
                                destination = cargo.location
                                cargo_ids_assigned.add(cargo.id)
                                break
                            else:
                                graph = ObservationHelper.get_multidigraph(global_state)
                                path = nx.shortest_path(graph, current_airport, cargo.location, weight="cost")
                                
                                if path[1] in available_destinations:
                                    destination = path[1]
                                    cargo_ids_assigned.add(cargo.id)
                                    break
                
                # random destination
                if destination == NOAIRPORT_ID and len(available_destinations) > 0:
                    destination = self._action_helper._choice(available_destinations)

                actions[a] = {"priority": None,
                            "cargo_to_load": [],
                            "cargo_to_unload": [],
                            "destination": destination}
            else:
                actions[a] = ActionHelper.noop_action()
        return actions

