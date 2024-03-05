from typing import Dict, List, NamedTuple, Set


class CargoEdge(NamedTuple):
    cargo_id: int
    origin: int
    destination: int
    ep: int  # earliest pickup
    lp: int  # latest pickup
    weight: int
    allowed_plane_types: Set[int]


class CargoEdges:
    cargo_edges: List[CargoEdge] = []

    def add(self, cargo_edge: CargoEdge):
        self.cargo_edges.append(cargo_edge)


class Assignments:
    assignments: Dict[int, CargoEdges] = {}


class Planning:
    def __init__(self, cargo_edges: CargoEdges, assignments: Assignments) -> None:
        self.cargo_edges = cargo_edges
        self.assignments = assignments
