from airlift.envs.airport import NOAIRPORT_ID

from typing import Dict

DEFAULT_ACTION = {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': NOAIRPORT_ID}

def action_pickup_cargo(cargo_id: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [cargo_id], 'cargo_to_unload': [], 'destination': NOAIRPORT_ID}

def action_drop_cargo(cargo_id: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [cargo_id], 'destination': NOAIRPORT_ID}

def action_set_destination(destination: int) -> Dict:
    return {'priority': 1, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': destination}

def get_globalstate(obs: Dict) -> Dict:
    return next(iter(obs.values()))["globalstate"]

def get_processing_time(globalstate: Dict) -> Dict:
    if len(globalstate["scenario_info"]) != 1:
        print("ERROR!!!!! Number of scenarios in scenario_info != 1")
    return globalstate["scenario_info"][0].processing_time