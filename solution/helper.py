from typing import Dict, List, Optional


def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]


# TODO: DELETE ONCE CODE IS READY, NO NEED TO PROCESS INFOS IN SUBMISSION
ACCEPTABLE_WARNINGS_11 = ["ROUTE FROM:"]
def process_infos(infos: Optional[Dict[str, Dict[str, List[str]]]]) -> None:
    if not infos:
        return set()
    
    for agent, agent_info in infos.items():
        if list(agent_info.keys()) != ["warnings"]:
            print("ERROR: Structure of infos not as expected")
            print(agent_info)
        for warning in agent_info["warnings"]:
            if warning[:11] in ACCEPTABLE_WARNINGS_11:
                continue
            # TODO: Understand when this happens
            elif warning in ["Airport does not have capacity to process!", "The agent is not next in queue!"]:
                print(agent, warning)
            elif warning == "The destination airport is not reachable from here!":
                # TODO: This should not happen!!
                print(agent, warning)
            else:
                print("ERROR: Unexpected warning")
                print(agent, warning)