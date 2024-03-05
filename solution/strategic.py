import time
from typing import Set
import networkx as nx
from airlift.envs.airlift_env import ObservationHelper

from solution.common import CargoEdge, CargoEdges, Planning


class Model:
    def __init__(self) -> None:
        pass

    def create_planning(self, obs) -> Planning:
        global_state = next(iter(obs.values()))["globalstate"]
        graph = ObservationHelper.get_multidigraph(global_state)
        self.paths = PathCache(graph)
        self.travel_times = TravelTimes(global_state["route_map"])
        self.cargo_edges = self.create_cargo_edges(obs)

    def create_cargo_edges(self, obs) -> CargoEdges:
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
                        earliest_pickup,
                        latest_pickup,
                        cargo.weight,
                        self.travel_times.get_allowable_plane_types(orig, dest),
                    )
                )
        secs = time.time() - start
        print(f"Calculated {len(cargo_edges.cargo_edges)} in {secs} seconds")
        return cargo_edges


class TravelTimes:
    # cache for travel time between 2 nodes
    def __init__(self, route_map) -> None:
        self.route_map = route_map
        self._cache = {}

    def get_allowable_plane_types(self, orig, dest) -> Set[int]:
        plane_types = set()
        for pt, graph in self.route_map.items():
            if graph.has_edge(orig, dest):
                plane_types.add(pt)
        return plane_types

    def get_travel_time(self, orig, dest) -> int:
        od = (orig, dest)
        if od in self._cache:
            return self._cache[od]
        else:
            tt = 0
            for graph in self.route_map.values():
                try:
                    tt = max(tt, nx.path_weight(graph, (orig, dest), "time"))
                except:
                    pass
            self._cache[od] = tt
            return tt


class PathCache:
    # shortest path cache
    def __init__(self, graph) -> None:
        self.graph = graph
        self._cache = {}

    def get_path(self, orig, dest):
        from_to = (orig, dest)
        if from_to in self._cache:
            return self._cache[from_to]
        else:
            path = nx.shortest_path(self.graph, orig, dest, weight="cost")
            self._cache[from_to] = path
            return path
