from airlift.solutions import Solution
from airlift.envs import ActionHelper, PlaneState, ObservationHelper
import networkx as nx
import matplotlib.pyplot as plt


class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()
        self.current_steps = 0
        self.loaded_cargos_by_cid = dict()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)
        self.current_steps = 0
        self.loaded_cargos_by_cid = dict()

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones, infos):
        state = list(obs.values())[0]['globalstate']
        # if self.current_steps % 50 == 0:
        #     print(f"step {self.current_steps}")
        #     print("Active cargos:")
        #     for cargo in state['active_cargo']:
        #         if cargo.is_available:
        #             print(f"  {cargo}")
        actions = dict()
        multi_route_map = ObservationHelper.get_multidigraph(state)
        cids_being_loaded = set()
        for agent, agent_state in state['agents'].items():
            agent_route_map = state['route_map'][agent_state['plane_type']]
            actions[agent] = ActionHelper.noop_action()
            if agent_state['state'] == PlaneState.READY_FOR_TAKEOFF:
                available_weight = agent_state['max_weight'] - agent_state['current_weight']
                active_pickup_locs = [(cargo.location, cargo.earliest_pickup_time) 
                                      for cargo in state['active_cargo']
                                      if cargo.weight <= available_weight]
                active_dropoff_locs = list()
                for c_id in agent_state['cargo_onboard']:
                    cargo = self.loaded_cargos_by_cid[c_id]
                    active_dropoff_locs.append((cargo.destination, 0))
                active_locs = set(active_pickup_locs + active_dropoff_locs)
                # Find the distance to each location
                best_time = 9e99
                best_next_dest = None
                for loc, avail_time in active_locs:
                    if loc not in multi_route_map.nodes:
                        continue
                    this_path = nx.shortest_path(multi_route_map,
                                            agent_state['current_airport'],
                                            loc,
                                            weight='time')
                    this_time = nx.shortest_path_length(multi_route_map,
                                                agent_state['current_airport'],
                                                loc,
                                                weight='time')
                    if avail_time > self.current_steps:
                        continue
                    if len(this_path) > 1:
                        if this_path[1] not in agent_state['available_routes']:
                            continue
                    if this_time < best_time:
                        best_time = this_time
                        if len(this_path) == 1:
                            best_next_dest = None
                        else:
                            best_next_dest = this_path[1]
                if best_next_dest is None:
                    actions[agent] = ActionHelper.noop_action()
                else:
                    # print(f"{agent} flying from {agent_state['current_airport']} to {best_next_dest}!")
                    actions[agent] = ActionHelper.takeoff_action(best_next_dest)
            elif agent_state['state'] == PlaneState.WAITING:
                cargo_to_unload = []
                cargo_to_load = []
                # check if there is cargo to unload
                available_weight = agent_state['max_weight'] - agent_state['current_weight']
                for c_id in agent_state['cargo_onboard']:
                    cargo = self.loaded_cargos_by_cid[c_id]
                    if cargo.destination == agent_state['current_airport']:
                        # Unload this cargo and account for the weight in aassinging loads
                        available_weight -= cargo.weight
                        cargo_to_unload.append(c_id)
                        # print(f"{agent} DELIVERING {c_id} @ {agent_state['current_airport']}!")
                    else:
                        # check if the next stop to the destination is not for this plane
                        path_to_dest = nx.shortest_path(multi_route_map,
                                                        agent_state['current_airport'],
                                                        cargo.destination,
                                                        weight='time')
                        if path_to_dest[1] not in agent_state['available_routes']:
                            # print(f"{agent} Unloading {c_id} @ {agent_state['current_airport']}!")
                            # Unload it here.
                            available_weight -= cargo.weight
                            cargo_to_unload.append(c_id)
                            
                for c_id in agent_state['cargo_at_current_airport']:
                    if c_id in cids_being_loaded:
                        continue
                    cargo = ObservationHelper.get_active_cargo_info(state, c_id)
                    if cargo is None:
                        # It isn't active????
                        continue
                    if not cargo.is_available:
                        continue
                    path_to_dest = nx.shortest_path(multi_route_map,
                                                    agent_state['current_airport'],
                                                    cargo.destination,
                                                    weight='time')
                    if path_to_dest[1] not in agent_state['available_routes']:
                        # don't load it!
                        continue
                    # greedily load the cargo
                    if cargo.weight <= available_weight:
                        # print(f"{agent} loading {c_id} @ {agent_state['current_airport']}!")
                        cids_being_loaded.add(c_id)
                        cargo_to_load.append(c_id)
                        self.loaded_cargos_by_cid[c_id] = cargo
                        available_weight -= cargo.weight
                actions[agent] = ActionHelper.process_action(cargo_to_load, cargo_to_unload)
        self.current_steps += 1
        return actions

