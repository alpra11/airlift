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

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        global_state = next(iter(obs.values()))["globalstate"]
        self.mgraph = ObservationHelper.get_multidigraph(global_state)
        self.path_matrix = PathMatrix(self.mgraph)

        self.current_time = 0

        self.agent_reroute_blocked_until = {a: 0 for a in obs.keys()}

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones, infos):
        cargo_ids_assigned = set()
        loaded_cargo = set()
        # Use the action helper to generate an action
        actions = {}

        global_state = next(iter(obs.values()))["globalstate"]
        current_mgraph = ObservationHelper.get_multidigraph(global_state)
        active_cargo = global_state["active_cargo"]
        active_cargo_by_pickup = sorted(
            active_cargo, key=lambda x: x.earliest_pickup_time
        )
        processing_time = global_state["scenario_info"][0].processing_time

        airport_capacity = nx.get_node_attributes(self.mgraph, "working_capacity")
        airport_planes = {a: 0 for a in airport_capacity.keys()}
        # Gather info on congestion

        agents = sorted(obs.items(), key=lambda a: a[1]["current_weight"], reverse=True)
        for a, agent in agents:
            plane_state = agent["state"]
            current_airport = agent["current_airport"]
            if plane_state in (PlaneState.WAITING, PlaneState.PROCESSING):
                if current_airport not in airport_planes:
                    airport_planes[current_airport] = 0
                airport_planes[current_airport] += 1

        airport_queue = {
            a: max(airport_planes[a] - airport_capacity[a], 0) * processing_time
            for a in airport_planes.keys()
        }

        for a, agent in agents:
            plane_state = agent["state"]
            current_airport = agent["current_airport"]
            max_weight = agent["max_weight"]
            cur_weight = agent["current_weight"]
            available_destinations = agent["available_routes"]
            cargo_at_current_airport = agent["cargo_at_current_airport"]
            cargo_onboard = agent["cargo_onboard"]
            destination = agent["destination"]
            plane_type = agent["plane_type"]

            if plane_state == PlaneState.WAITING:
                cargo_to_unload = []
                cargo_to_load = []
                # unload
                for c_id in cargo_onboard:
                    cargo_info = ObservationHelper.get_active_cargo_info(
                        global_state, c_id
                    )
                    if cargo_info is None:
                        # Dump the missing cargo
                        cargo_to_unload.append(c_id)
                        continue

                    path = self.path_matrix.get_path(
                        current_airport, cargo_info.destination
                    )
                    if (
                        cargo_info.destination == current_airport
                        or path[1] not in available_destinations
                    ):
                        cargo_to_unload.append(c_id)
                        cur_weight -= cargo_info.weight

                # load
                for cargo in (
                    ObservationHelper.get_active_cargo_info(
                        global_state, cargo_at_current_airport
                    )
                    or []
                ):
                    if (
                        cargo.weight <= max_weight - cur_weight
                        and cargo.id not in cargo_ids_assigned
                        and cargo.id not in loaded_cargo
                    ):
                        path = self.path_matrix.get_path(
                            current_airport, cargo.destination
                        )
                        if path[1] in available_destinations:
                            cargo_to_load.append(cargo.id)
                            loaded_cargo.add(cargo.id)
                            cur_weight += cargo.weight
                            # print(f"Load {cargo.id} plane {a} at {current_airport} for destination {cargo.destination}")

                actions[a] = {
                    "priority": None,
                    "cargo_to_load": cargo_to_load,
                    "cargo_to_unload": cargo_to_unload,
                    "destination": NOAIRPORT_ID,
                }

            elif plane_state == PlaneState.READY_FOR_TAKEOFF:

                cargo_to_ship = sorted(
                    ObservationHelper.get_active_cargo_info(global_state, cargo_onboard)
                    or [],
                    key=lambda x: x.soft_deadline,
                )

                # destination from cargo on board
                destination = NOAIRPORT_ID
                for cargo in cargo_to_ship:
                    path = self.path_matrix.get_path(current_airport, cargo.destination)
                    mals = nx.path_weight(current_mgraph, path, weight="mal")
                    queues = sum(airport_queue[airport] for airport in path)
                    path_time = (
                        self.current_time
                        + len(path)
                        * processing_time
                        * 2  # 2 to assume transfers in each airport, jank
                        + nx.path_weight(self.mgraph, path, weight="time")
                        + mals
                        + queues
                    )
                    if self.current_time > self.agent_reroute_blocked_until[a]:
                        if queues > 0:
                            # print(
                            #     f"Plane {a} attempting reroute to {cargo.destination} avoid congestion"
                            # )
                            filtered_mgraph = nx.subgraph_view(
                                current_mgraph,
                                filter_edge=lambda u, v, k: airport_queue[u] == 0
                                and airport_queue[v] == 0
                                and (k == plane_type if u == current_airport else True),
                            )
                            try:
                                new_path = nx.shortest_path(
                                    filtered_mgraph,
                                    current_airport,
                                    cargo.destination,
                                    weight="time",
                                )
                                new_mals = nx.path_weight(
                                    current_mgraph, new_path, weight="mal"
                                )
                                new_path_time = (
                                    self.current_time
                                    + len(new_path)
                                    * processing_time
                                    * 2  # 2 to assume transfers in each airport, jank
                                    + nx.path_weight(
                                        self.mgraph, new_path, weight="time"
                                    )
                                    + new_mals
                                )
                                if new_path_time < path_time:
                                    print(f"Rerouted via {new_path}")
                                    path = new_path
                                    mals = new_mals
                                    path_time = new_path_time
                                else:
                                    # Block rerouting for queues steps +1
                                    self.agent_reroute_blocked_until[a] = (
                                        self.current_time + queues + 1
                                    )
                            except nx.exception.NetworkXNoPath:
                                # Let it fall through to queueing
                                # Block rerouting for queues steps +1
                                self.agent_reroute_blocked_until[a] = (
                                    self.current_time + queues + 1
                                )
                        if mals > 0:
                            # print(
                            #     f"Plane {a} attempting reroute to {cargo.destination} avoid mal"
                            # )
                            filtered_mgraph = nx.subgraph_view(
                                current_mgraph,
                                filter_edge=lambda u, v, k: current_mgraph[u][v][k][
                                    "mal"
                                ]
                                and (k == plane_type if u == current_airport else True)
                                == 0,
                            )
                            try:
                                new_path = nx.shortest_path(
                                    filtered_mgraph,
                                    current_airport,
                                    cargo.destination,
                                    weight="time",
                                )
                                new_queues = sum(
                                    airport_queue[airport] for airport in path
                                )
                                new_path_time = (
                                    self.current_time
                                    + len(new_path)
                                    * processing_time
                                    * 2  # 2 to assume transfers in each airport, jank
                                    + nx.path_weight(
                                        self.mgraph, new_path, weight="time"
                                    )
                                    + new_queues
                                )
                                if new_path_time < path_time:
                                    print(f"Rerouted via {new_path}")
                                    path = new_path
                                    queues = new_queues
                                    path_time = new_path_time
                                else:
                                    # Block rerouting for mal steps + 1
                                    self.agent_reroute_blocked_until[a] = (
                                        self.current_time + mals + 1
                                    )

                            except nx.exception.NetworkXNoPath:
                                # Let it fall through to waiting and not rerouting
                                # Block rerouting for mal steps + 1
                                self.agent_reroute_blocked_until[a] = (
                                    self.current_time + mals + 1
                                )
                    if (
                        path_time
                        < (cargo.soft_deadline * 0 + cargo.hard_deadline * 1) / 1
                        and max_weight - cur_weight > active_cargo_by_pickup[0].weight
                    ):
                        break
                    else:
                        if path[1] in available_destinations:
                            destination = path[1]
                            actions[a] = {
                                "priority": None,
                                "cargo_to_load": [],
                                "cargo_to_unload": [],
                                "destination": destination,
                            }
                            # print(f"Destination for {a} to {destination} for {cargo.id} final dest {cargo.destination}, path is {path}")
                            break
                        else:
                            print(
                                f"Plane {a} can't deliver {cargo.id} to {cargo.destination}, path is {path}, avail dests {available_destinations}"
                            )

                # destination from cargo on destinations
                if destination == NOAIRPORT_ID:
                    for cargo in active_cargo_by_pickup:
                        if cargo.id not in cargo_ids_assigned and cargo.location > 0:
                            if cargo.location in available_destinations:
                                destination = cargo.location
                                cargo_ids_assigned.add(cargo.id)
                                actions[a] = {
                                    "priority": None,
                                    "cargo_to_load": [],
                                    "cargo_to_unload": [],
                                    "destination": destination,
                                }
                                break
                            elif cargo.location == current_airport:
                                if (
                                    cargo.is_available
                                    and cargo.weight <= max_weight - cur_weight
                                    and cargo.id not in cargo_ids_assigned
                                    and cargo.id not in loaded_cargo
                                ):
                                    path = self.path_matrix.get_path(
                                        current_airport, cargo.destination
                                    )
                                    if path[1] in available_destinations:
                                        loaded_cargo.add(cargo.id)
                                        cur_weight += cargo.weight
                                        actions[a] = {
                                            "priority": None,
                                            "cargo_to_load": [cargo.id],
                                            "cargo_to_unload": [],
                                            "destination": NOAIRPORT_ID,
                                        }
                                else:
                                    actions[a] = {
                                        "priority": None,
                                        "cargo_to_load": [],
                                        "cargo_to_unload": [],
                                        "destination": NOAIRPORT_ID,
                                    }
                            else:
                                path = self.path_matrix.get_path(
                                    current_airport, cargo.location
                                )
                                if path[1] in available_destinations:
                                    destination = path[1]
                                    cargo_ids_assigned.add(cargo.id)
                                    actions[a] = {
                                        "priority": None,
                                        "cargo_to_load": [],
                                        "cargo_to_unload": [],
                                        "destination": destination,
                                    }
                                    break
                if a not in actions:
                    actions[a] = {
                        "priority": None,
                        "cargo_to_load": [],
                        "cargo_to_unload": [],
                        "destination": NOAIRPORT_ID,
                    }
            else:
                actions[a] = ActionHelper.noop_action()

        self.current_time += 1
        return actions


class PathMatrix:
    def __init__(self, graph) -> None:
        self.graph = graph
        self._matrix = {}

    def get_path(self, orig, dest):
        from_to = (orig, dest)
        if from_to in self._matrix:
            return self._matrix[from_to]
        else:
            path = nx.shortest_path(self.graph, orig, dest, weight="cost")
            self._matrix[from_to] = path
            return path
