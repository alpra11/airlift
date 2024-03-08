from typing import Any, Dict, Optional

from airlift.envs.cargo import Cargo


class LocInfo:
    def __init__(self, time: int, loc: int) -> None:
        self.time = time
        self.loc = loc
        self.agent: Optional[str] = None

class CargoEstimate:
    def __init__(self, cargo: Cargo) -> None:
        self.id = cargo.id
        self.dest = cargo.destination
        self.cur_loc= LocInfo(cargo.earliest_pickup_time, cargo.location)
        self.is_waiting = False

    def is_assigned(self) -> bool:
        return self.cur_loc.agent is not None and not self.is_waiting
    
    def assign_agent(self, agent: str) -> None:
        self.cur_loc.agent = agent
        self.is_waiting = False
    
    def unassign(self, new_loc: int) -> None:
        self.cur_loc.loc = new_loc
        self.cur_loc.agent = None

class CargoPlan:
    def __init__(self, state: Dict[str, Any]) -> None:
        self.cargo = {c.id: CargoEstimate(c) for c in state["active_cargo"]}

    def is_assigned(self, c_id: int) -> bool:
        return self.cargo[c_id].is_assigned()
    
    def assign_agent(self, agent: str, c_id: int) -> None:
        self.cargo[c_id].assign_agent(agent)

    def unassign(self, c_id: int, cur_airport: int) -> None:
        self.cargo[c_id].unassign(cur_airport)
        if cur_airport == self.cargo[c_id].dest:
            self.assign_agent(-1, c_id)

    def update(self, state: Dict[str, Any]) -> None:
        for cargo in state["event_new_cargo"]:
            self.cargo[cargo.id] = CargoEstimate(cargo)

    def is_waiting(self, c_id: int) -> bool:
        return self.cargo[c_id].is_waiting
    
    def set_waiting(self, c_id: int) -> False:
        self.cargo[c_id].is_waiting = True

    def remove_waiting(self, c_id: int) -> False:
        self.cargo[c_id].is_waiting = False

    def get_location(self, c_id: int) -> int:
        return self.cargo[c_id].cur_loc.loc