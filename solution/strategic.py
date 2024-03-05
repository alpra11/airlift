import math
import time
from airlift.envs.airlift_env import ObservationHelper
from solution.common import (
    Assignments,
    CargoEdge,
    CargoEdges,
    PathCache,
    Plane,
    Planning,
    TravelTimes,
)


class Model:
    def __init__(self) -> None:
        pass

    def create_planning(self, obs) -> Planning:
        global_state = next(iter(obs.values()))["globalstate"]
        graph = ObservationHelper.get_multidigraph(global_state)
        self.paths = PathCache(graph)
        self.travel_times = TravelTimes(global_state["route_map"])
        self.cargo_edges = self._create_cargo_edges(obs)
        self.assignments = self._create_assignments(obs)
        return Planning(self.cargo_edges, self.assignments)

    def _create_cargo_edges(self, obs) -> CargoEdges:
        start = time.time()
        cargo_edges = CargoEdges()

        global_state = next(iter(obs.values()))["globalstate"]
        processing_time = global_state["scenario_info"][0].processing_time

        for cargo in global_state["active_cargo"]:
            shortest_path = self.paths.get_path(cargo.location, cargo.destination)

            earliest_pickup = cargo.earliest_pickup_time

            # travel forward
            for orig, dest in zip(shortest_path[1:], shortest_path[:-1]):
                travel_time = self.travel_times.get_travel_time(orig, dest)
                earliest_pickup += processing_time + travel_time

            latest_pickup = cargo.hard_deadline
            sequence = len(shortest_path) - 1
            # travel backward
            for orig, dest in zip(shortest_path[-2::-1], shortest_path[::-1]):
                travel_time = self.travel_times.get_travel_time(orig, dest)
                earliest_pickup -= processing_time + travel_time
                latest_pickup -= processing_time + travel_time
                cargo_edges.add(
                    CargoEdge(
                        cargo.id,
                        orig,
                        dest,
                        travel_time + processing_time,
                        sequence,
                        earliest_pickup,
                        latest_pickup,
                        cargo.weight,
                        self.travel_times.get_allowable_plane_types(orig, dest),
                    )
                )
                sequence -= 1
        secs = time.time() - start
        print(f"Calculated {len(cargo_edges.cargo_edges)} edges in {secs} seconds")
        return cargo_edges

    def _create_assignments(self, obs) -> Assignments:
        assignments = Assignments()
        planes = []
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
            key=lambda ce: (math.floor(ce.ep / 30), ce.sequence),
        ):
            sorted_planes = sorted(
                [p for p in planes if p.type in ce.allowed_plane_types],
                key=lambda p: p.matches(ce),
            )
            found = False
            for plane in sorted_planes:
                if plane.can_service(ce, self.travel_times):
                    plane.add_cargo_edge(ce)
                    found = True
                    break
            if not found:
                print(f"No plane found for ce {ce}")

        cnt = 0
        for plane in planes:
            for action in plane.actions:
                cnt += 1
                print(
                    f"{plane.id};{action.cargo_id};{action.origin};{action.destination};{action.ep};{action.lp}"
                )
        print(f"Planned {cnt} cargo edges")

        return assignments
