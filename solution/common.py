from typing import Dict

def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]