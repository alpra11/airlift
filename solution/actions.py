from typing import Any, Dict, List, Optional

from airlift.envs.airport import NOAIRPORT_ID

NO_ACTION = {"priority": None, "cargo_to_load": [], "cargo_to_unload": [], "destination": NOAIRPORT_ID}

class Actions:
    def __init__(self, state: Dict[str, Any]) -> None:
        self._actions = {a: NO_ACTION.copy() for a in state["agents"]} 

    def reset(self) -> None:
        self._actions = {a: NO_ACTION.copy() for a in self._actions}

    def process_cargo(self, agent: str, cargo_to_load: List[int]=[], cargo_to_unload:List[int]=[], priority: Optional[int]=None) -> None:
        # Assuming single cargo per plane - update to load multiple cargos
        if priority:
            self._actions[agent]["priority"] = priority
        if cargo_to_load:
            self._actions[agent]["cargo_to_load"] = cargo_to_load
        if cargo_to_unload:
            self._actions[agent]["cargo_to_unload"] = cargo_to_unload

    def take_off(self, agent:str, destination: int) -> None:
        self._actions[agent]["destination"] = destination
    
    def get_actions(self) -> Dict[str, Dict[str, Any]]:
        return self._actions