from typing import Optional

class LocInfo:
    def __init__(self, time: int, loc: int) -> None:
        self.time = time
        self.loc = loc
        self.agent: Optional[str] = None

    def is_assigned(self) -> bool:
        return self.agent is not None
    
    def assign_agent(self, agent: str) -> None:
        self.agent = agent
    
    def unassign(self) -> None:
        self.agent = None

class CargoEstimate:
    def __init__(self, id: int, dest: int, time: int, loc: int) -> None:
        self.id = id
        self.dest = dest
        self.cur_loc= LocInfo(time, loc)
        self.next_loc: Optional[LocInfo] = None