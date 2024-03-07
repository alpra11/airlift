from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

BIG_TIME = 100_000

TW_OVERLAP_MARGIN = 15


class PlaneTypeMap:
    # cache for travel time between 2 nodes
    def __init__(self, route_map) -> None:
        self.route_map = route_map

    def get_allowable_plane_types(self, orig: int, dest: int) -> Set[int]:
        plane_types = set()
        for pt, graph in self.route_map.items():
            if graph.has_edge(orig, dest):
                plane_types.add(pt)
        return plane_types

    def reachable(self, plane_type: int, orig: int, dest: int) -> bool:
        graph = self.route_map[plane_type]
        return nx.has_path(graph, orig, dest)


class PathCache:
    # shortest path cache
    def __init__(self, graph) -> None:
        self.graph = graph
        self._path_cache: Dict[Tuple[int, int], List[int]] = {}
        self._time_cache: Dict[Tuple[int, int], int] = {}

    def get_path(self, orig, dest):
        from_to = (orig, dest)
        if from_to in self._path_cache:
            return self._path_cache[from_to]
        else:
            path = nx.shortest_path(self.graph, orig, dest, weight="cost")
            self._path_cache[from_to] = path
            return path

    def get_travel_time(self, orig, dest):
        from_to = (orig, dest)
        if from_to in self._time_cache:
            return self._time_cache[from_to]
        else:
            path = self.get_path(orig, dest)
            time = nx.path_weight(self.graph, path, "time")
            self._time_cache[from_to] = time
            return time


@dataclass
class CargoEdge:
    cargo_id: int
    origin: int
    destination: int
    duration: int
    sequence: int
    ep: int  # earliest pickup
    lp: int  # latest pickup
    weight: int
    allowed_plane_types: Set[int]


class CargoEdges:
    def __init__(self) -> None:
        self.cargo_edges: List[List[CargoEdge]] = []

    def add(self, cargo_edges: List[CargoEdge]):
        self.cargo_edges.append(cargo_edges)


@dataclass
class Plane:
    id: str
    location: int
    next_destination: int
    type: int
    max_weight: int
    ep: int = field(default_factory=lambda: 0)
    lp: int = field(default_factory=lambda: BIG_TIME)
    cur_weight: int = field(default_factory=lambda: 0)
    actions: List[List[CargoEdge]] = field(default_factory=lambda: [])
    cargo_ids: Set[int] = field(default_factory=lambda: set())

    def has_actions(self) -> bool:
        return len(self.actions) > 0

    def get_next_deadline(self) -> Optional[int]:
        if self.has_actions():
            return min(ce.lp for ce in self.actions[0])

    def matches(
        self, ce: CargoEdge, path_cache: PathCache
    ) -> Tuple[int, int, int, int, int, int]:
        # lower is better match
        cargo = 0 if ce.cargo_id in self.cargo_ids else 1
        same_edge_and_tw_overlap = 1
        if (
            self.location == ce.origin and self.next_destination == ce.destination
        ) and tw_overlap(self.ep, self.lp, ce.ep, ce.lp):
            same_edge_and_tw_overlap = 0
        destination = 0 if self.next_destination == ce.origin else 1
        tt = path_cache.get_travel_time(self.location, ce.origin)
        timediff = self.ep + tt - ce.ep
        nr_actions = len(self.actions)

        return (cargo, same_edge_and_tw_overlap, destination, timediff, nr_actions)

    def can_service(
        self, ce: CargoEdge, path_cache: PathCache, plane_type_map: PlaneTypeMap
    ) -> bool:
        # can plane fly the edge
        if self.type not in ce.allowed_plane_types:
            return False

        # plane can not reach origin
        if not plane_type_map.reachable(self.type, self.location, ce.origin):
            return False

        # add cargo at location
        if not self.has_actions():
            return True

        if (
            self.location == ce.origin
            and self.next_destination == ce.destination
            and tw_overlap(self.ep, self.lp, ce.ep, ce.lp)
            and self.cur_weight + ce.weight <= self.max_weight
        ):
            return True
        # fly to cargo
        elif (
            self.next_destination == ce.origin
            and self.ep + self.actions[-1][-1].duration < ce.lp
        ):
            return True
        # fly to cargo
        elif (
            self.ep
            + self.actions[-1][-1].duration
            + path_cache.get_travel_time(self.next_destination, ce.origin)
            < ce.lp
        ):
            return True

        return False

    def add_cargo_edge(self, ce: CargoEdge, path_cache: PathCache) -> Tuple[int, int]:
        ep_change = 0
        lp_change = 0
        # add cargo at location
        if (
            self.location == ce.origin
            and tw_overlap(self.ep, self.lp, ce.ep, ce.lp)
            and self.cur_weight + ce.weight <= self.max_weight
        ):
            self.cur_weight += ce.weight
            self.cargo_ids.add(ce.cargo_id)
            ep_change = max(0, self.ep - ce.ep)
            lp_change = max(0, ce.lp - self.lp)
            self.ep = max(self.ep, ce.ep)
            self.lp = min(self.lp, ce.lp)

            if self.has_actions():
                self.actions[-1].append(ce)
            else:
                self.actions.append([ce])

        # fly to cargo
        else:
            self.ep += self.actions[-1][-1].duration if self.has_actions() else 0
            if self.next_destination != ce.origin:
                self.ep += path_cache.get_travel_time(self.next_destination, ce.origin)
            # unload everything and load current cargo
            self.cur_weight = ce.weight
            self.location = ce.origin
            self.cargo_ids = {ce.cargo_id}

            ep_change = max(0, self.ep - ce.ep)
            self.ep = max(self.ep, ce.ep)
            self.lp = ce.lp
            self.actions.append([ce])

        self.next_destination = ce.destination

        ce.ep = max(self.ep, ce.ep)
        ce.lp = min(self.lp, ce.lp)
        return (ep_change, lp_change)


class Planning:
    def __init__(self, cargo_edges: CargoEdges, planes: Dict[str, Plane]) -> None:
        self.cargo_edges = cargo_edges
        self.planes = planes


def tw_overlap(ep_1: int, lp_1: int, ep_2: int, lp_2: int) -> bool:
    return ep_1 <= lp_2 - TW_OVERLAP_MARGIN and ep_2 <= lp_1 - TW_OVERLAP_MARGIN
