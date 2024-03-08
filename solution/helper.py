from typing import Dict, List, Optional


def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]


# TODO: DELETE ONCE CODE IS READY, NO NEED TO PROCESS INFOS IN SUBMISSION
ACCEPTABLE_WARNINGS_11 = ["ROUTE FROM:"]
def process_infos(infos: Optional[Dict[str, Dict[str, List[str]]]], timestep: int) -> None:
    if not infos:
        return set()
    
    for agent, agent_info in infos.items():
        if list(agent_info.keys()) != ["warnings"]:
            print("ERROR: Structure of infos not as expected")
            print(agent_info)
        for warning in agent_info["warnings"]:
            # print(timestep, warning)
            # input()
            pass