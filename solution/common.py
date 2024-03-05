from typing import Dict, List, NamedTuple, Set, Tuple

class CargoEdge(NamedTuple):
    cargo_id: int
    edge: Tuple[int, int]
    ep: int # earliest pickup
    lp: int # latest pickup
    weight: int
    allowed_plane_types: Set[int]

class CargoEdges:
    cargo_edges: List[CargoEdge]

class Assignments:
    assignments: Dict[int, CargoEdges]

class CommonData:
    def __init__(self, cargo_edges: CargoEdges, assignments: Assignments) -> None:
        self.cargo_edges = cargo_edges
        self.assignments = assignments