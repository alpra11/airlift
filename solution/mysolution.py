from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airport import NOAIRPORT_ID
from airlift.envs.agents import PlaneState
from airlift.envs.airlift_env import ObservationHelper

from typing import Tuple, Dict, List

import networkx as nx

from solution.strategic import Model


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

        self.current_time = 0

        self.planning = Model().create_planning(obs)

        global_state = next(iter(obs.values()))["globalstate"]

        self.path_matrices = dict()        
        for plane_type, route_map in global_state["route_map"].items()
            self.path_matrices[plane_type] = PathMatrix(route_map)

    def policies(self, obs, dones, infos):
        # Use the action helper to generate an action
        actions = {}

        global_state = next(iter(obs.values()))["globalstate"]
        if len(global_state["event_new_cargo"]) > 0:
            self.planning = Model().create_planning(obs)
            # TODO: Do this in a better way

        for a, agent in obs.items():
            plane = self.planning.planes[a]
            plane_state = agent["state"]
            plane_type = agent["plane_type"]
            current_airport = agent["current_airport"]
            max_weight = agent["max_weight"]
            cur_weight = agent["current_weight"]
            available_destinations = agent["available_routes"]
            cargo_at_current_airport = agent["cargo_at_current_airport"]
            cargo_onboard = agent["cargo_onboard"]
            destination = agent["destination"]

            if plane_state in (PlaneState.WAITING, PlaneState.READY_FOR_TAKEOFF):
                cargo_to_unload = []
                cargo_to_load = []
                # unload
                for cargo_id in cargo_onboard:
                    # Dump missed
                    cargo = ObservationHelper.get_active_cargo_info(
                        global_state, cargo_id
                    )
                    if cargo is None:
                        # Missed cargo
                        print(
                            f"Dumping missed cargo {cargo_id} from {a} at {current_airport}"
                        )
                        cargo_to_unload.append(cargo_id)
                        for plane in self.planning.planes.values():
                            plane.actions = [
                                ce for ce in plane.actions if ce.cargo_id != cargo_id
                            ]  # TODO This is slow
                        continue

                    # Unload cargo where the next CargoEdge is not an available destination
                    # TODO: this will happen if there is mal, not ideal
                    found_ce = False
                    for ce in plane.actions:
                        if ce.origin == current_airport and ce.cargo_id == cargo_id:
                            # If there is a cargoedge from this airport for this cargo, don't unload
                            found_ce = True
                            break
                    if not found_ce:
                        cargo_to_unload.append(cargo_id)
                        cur_weight -= cargo.weight

                # load
                for cargo in (
                    ObservationHelper.get_active_cargo_info(
                        global_state, cargo_at_current_airport
                    )
                    or []
                ):
                    for ce in plane.actions:
                        if (
                            ce.origin == current_airport
                            and ce.cargo_id == cargo.id
                            and cargo.weight <= max_weight - cur_weight
                        ):
                            # If there is a cargoedge from this airport for this cargo, load
                            cargo_to_load.append(cargo.id)
                            cur_weight += cargo.weight
                            # print(f"Loading {cargo.id} on {a} for {ce}")

                actions[a] = {
                    "priority": None,
                    "cargo_to_load": cargo_to_load,
                    "cargo_to_unload": cargo_to_unload,
                    "destination": NOAIRPORT_ID,
                }

            if (
                plane_state == PlaneState.READY_FOR_TAKEOFF
                and len(cargo_to_load) + len(cargo_to_unload) == 0
            ):

                ce_onboard = [
                    ce
                    for ce in plane.actions
                    if ce.cargo_id in cargo_onboard and ce.origin == current_airport
                ]

                # destination from cargoedge on board
                destination = NOAIRPORT_ID
                for ce in ce_onboard:
                    if self.current_time >= ce.lp:
                        # If it is time to dispatch any CE on board, then dispatch and break out
                        destination = ce.destination
                        actions[a] = {
                            "priority": None,
                            "cargo_to_load": [],
                            "cargo_to_unload": [],
                            "destination": ce.destination,
                        }
                        # print(f"Sending {a} to {destination} for {ce}")
                        for ce in ce_onboard:
                            if ce.destination == destination:
                                # Remove them from acitons as they are being executed
                                plane.actions.remove(ce)
                            else:
                                print(
                                    f"WARNING: plane being dispatched to {destination} with {ce} onboard"
                                )
                        break

                # destination to first cargoedge origin assigned
                if len(ce_onboard) == 0:
                    # If plane is empty find the first cargoedge assigned to the plane and go that way
                    for ce in plane.actions:
                        if ce.origin == current_airport:
                            # Stay here until the cargo is loaded
                            destination = NOAIRPORT_ID
                            actions[a] = {
                                "priority": None,
                                "cargo_to_load": [],
                                "cargo_to_unload": [],
                                "destination": NOAIRPORT_ID,
                            }
                            # print(f"{a} waiting at {current_airport} for {ce}")
                            break

                        path = self.path_matrices[plane_type].get_path(
                            current_airport, ce.origin
                        )
                        if path[1] in available_destinations:
                            # Head to it
                            destination = path[1]
                            actions[a] = {
                                "priority": None,
                                "cargo_to_load": [],
                                "cargo_to_unload": [],
                                "destination": destination,
                            }
                            # print(f"Sending {a} on {path} to pickup {ce}")
                            break
                        else:
                            print(f"WARNING: {a} cannot go on {path} to pickup {ce}")
            if a not in actions:
                actions[a] = ActionHelper.noop_action()
        self.current_time += 1
        return actions


class PathMatrix:
    def __init__(self, graph: nx.graph) -> None:
        self.graph: nx.graph = graph
        self._matrix: Dict[Tuple[int, int], List[int]] = {}

    def _compute_shortest_path(self, orig: int, dest: int, from_to: Tuple[int, int]):
        path = nx.shortest_path(self.graph, orig, dest, weight="cost")
        self._matrix[from_to] = path
        for sub_ind in range(1, len(path) - 1):
            sub_from_to = (path[sub_ind], dest)
            self._matrix[sub_from_to] = path[sub_ind : len(path)]

    def get_path(self, orig: int, dest: int):
        from_to = (orig, dest)
        if from_to not in self._matrix:
            self._compute_shortest_path(orig, dest, from_to)
        return self._matrix[from_to]
