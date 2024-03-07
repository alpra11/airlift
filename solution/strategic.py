import math
from typing import Dict, Iterable, List, Optional
from airlift.envs.airlift_env import ObservationHelper
from solution.common import (
    CargoEdge,
    CargoEdges,
    PathCache,
    Plane,
    PlaneTypeMap,
    Planning,
)


class Model:
    def __init__(self) -> None:
        pass

    def create_planning(self, obs) -> Planning:
        global_state = next(iter(obs.values()))["globalstate"]
        self.processing_time = global_state["scenario_info"][0].processing_time
        graph = ObservationHelper.get_multidigraph(global_state)
        self.paths = PathCache(graph)
        self.plane_type_map = PlaneTypeMap(global_state["route_map"])
        self.cargo_edges = self._create_cargo_edges(obs)
        self.planes = self._create_assignments(obs)
        # print_cargo_edges(self.cargo_edges)
        # print_planes(self.planes.values())
        return Planning(self.cargo_edges, self.planes)

    def update_planning(self, obs) -> Optional[Planning]:
        # TODO: we should probably put the planning in the model so it's more straightforward
        # Update the palnning for new cargo
        global_state = next(iter(obs.values()))["globalstate"]
        new_cargos = global_state["event_new_cargo"]
        if len(new_cargos) > 0:
            self.cargo_edges = self._add_cargo_edges_from_cargos(
                self.cargo_edges, new_cargos
            )
            self.planes = self._create_assignments(obs)
            return Planning(self.cargo_edges, self.planes)
        # TODO: Do something to return a modified planning if malfunctions happen...

    def _create_cargo_edges(self, obs) -> CargoEdges:
        global_state = next(iter(obs.values()))["globalstate"]
        cargo_edges = self._add_cargo_edges_from_cargos(
            CargoEdges(), global_state["active_cargo"]
        )
        return cargo_edges

    def _add_cargo_edges_from_cargos(
        self, cargo_edges: CargoEdges, cargos
    ) -> CargoEdges:
        for cargo in cargos:
            shortest_path = self.paths.get_path(cargo.location, cargo.destination)

            earliest_pickup = cargo.earliest_pickup_time

            # travel forward
            for orig, dest in zip(shortest_path[1:], shortest_path[:-1]):
                travel_time = self.paths.get_travel_time(orig, dest)
                earliest_pickup += (
                    self.processing_time + travel_time + self.processing_time
                )

            latest_pickup = cargo.soft_deadline
            sequence = len(shortest_path) - 1
            # travel backward
            for orig, dest in zip(shortest_path[-2::-1], shortest_path[::-1]):
                travel_time = self.paths.get_travel_time(orig, dest)
                earliest_pickup -= (
                    self.processing_time + travel_time + self.processing_time
                )
                latest_pickup -= (
                    self.processing_time + travel_time + self.processing_time
                )
                cargo_edges.add(
                    CargoEdge(
                        cargo.id,
                        orig,
                        dest,
                        travel_time + self.processing_time,
                        sequence,
                        earliest_pickup,
                        latest_pickup,
                        cargo.weight,
                        self.plane_type_map.get_allowable_plane_types(orig, dest),
                    )
                )
                sequence -= 1
        return cargo_edges

    def _create_assignments(self, obs) -> Dict[str, Plane]:
        planes: List[Plane] = []
        for a_id, agent in obs.items():
            planes.append(
                Plane(
                    a_id,
                    agent["current_airport"],
                    agent["current_airport"],
                    agent["plane_type"],
                    agent["max_weight"],
                )
            )

        for ce in sorted(
            self.cargo_edges.cargo_edges,
            key=lambda ce: (math.floor(ce.ep / 200), ce.sequence),
        ):
            sorted_planes = sorted(
                [p for p in planes if p.type in ce.allowed_plane_types],
                key=lambda p: p.matches(ce, self.paths),
            )
            found = False
            for plane in sorted_planes:
                if plane.can_service(ce, self.paths, self.plane_type_map):
                    tw_changes = plane.add_cargo_edge(ce, self.paths)
                    if tw_changes[0] > 0 or tw_changes[1] > 0:
                        for c in self.cargo_edges.cargo_edges:
                            if c.cargo_id == ce.cargo_id and c.sequence > ce.sequence:
                                c.ep += tw_changes[0]
                                c.lp -= tw_changes[1]
                    found = True
                    break
            if not found:
                print(f"No plane found for ce {ce}")

        return {plane.id: plane for plane in planes}


def print_cargo_edges(cargo_edges: CargoEdges) -> None:
    for ce in cargo_edges.cargo_edges:
        print(
            f"{ce.cargo_id};{ce.origin};{ce.destination};{ce.duration};{ce.sequence};{ce.ep};{ce.lp};{ce.weight};{ce.allowed_plane_types}"
        )


def print_planes(planes: Iterable[Plane]) -> None:
    cnt = 0
    for plane in planes:
        for actions in plane.actions:
            print("-")
            for action in actions:
                cnt += 1
                print(
                    f"{plane.id};{action.cargo_id};{action.origin};{action.destination};{action.ep};{action.lp}"
                )
    print(f"Planned {cnt} cargo edges")
