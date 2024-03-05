import networkx as nx
from airlift.envs.airlift_env import ObservationHelper

class Model:
    def __init__(self) -> None:
        pass

    def initialize(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        global_state = next(iter(obs.values()))["globalstate"]
        graph = ObservationHelper.get_multidigraph(global_state)
        self.paths = PathCache(graph)
        self.travel_times = {plane_type: TravelTimes(graph) for plane_type, graph in global_state["route_map"]}

class TravelTimes:
    # cache for travel time between 2 nodes
    def __init__(self, graph) -> None:
        self.graph = graph
        self._cache = {}
    
    def get_travel_time(self, orig, dest):
        od = (orig,dest)
        if od in self._cache:
            return self._cache[od]
        else:
            # TODO : return None if segment not in graph
            tt = nx.path_weight(self.graph, (orig, dest), "time")
            self._cache[od] = tt
            return tt

class PathCache:
    # shortest path cache
    def __init__(self, graph) -> None:
        self.graph = graph
        self._cache = {}

    def get_path(self, orig, dest):
        from_to = (orig, dest)
        if from_to in self._cache:
            return self._cache[from_to]
        else:
            path = nx.shortest_path(self.graph, orig, dest, weight="cost")
            self._cache[from_to] = path
            return path