from typing import Dict, List, Optional, Set


def process_infos(infos: Optional[Dict[str, Dict[str, List[str]]]]) -> Set[str]:
    if not infos:
        return set()
    
    offline_warnings = [] 

    for agent_info in infos.values():
        # TODO: Delete once code is ready
        if list(agent_info.keys()) != ["warnings"]:
            print("ERROR: Structure of infos not as expected")
            print(agent_info)
        for warning in agent_info["warnings"]:
            if warning[:11] == "ROUTE FROM:":
                offline_warnings.append(warning)
            else:
                # TODO: Delete once code is ready
                print("ERROR: Unexpected warning")
                print(warning)

    return set(offline_warnings)