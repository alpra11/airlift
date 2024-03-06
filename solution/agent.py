from typing import List, Optional


class Agent:
    def __init__(self, agent: str) -> None:
        self.name = agent
        self.cargo_id: Optional[int] = None 
        self.path: Optional[List[int]] = None

    def assign_cargo(self, cargo_id: int, path: List[int]) -> None:
        self.cargo_id = cargo_id
        self.path = path

    def unassign(self) -> None:
        self.cargo_id = None
        self.path = None

    def is_free(self) -> bool:
        return self.cargo_id is None