from top_view_renderer import EntityMap
from environment import get_entity_layer_path


class Expert(object):
    def _update_distance_to_goal(self):
        pass

    def _build_free_space_estimate(self):
        pass

    def __init__(self, entity_layer):
        self._entity_layer_name = entity_layer
        self._entity_map = EntityMap(get_entity_layer_path(entity_layer)[-1])
        self._goal = None

    def get_optimal_action(self, info):
        raise NotImplementedError("optimal action is not yet implemented")

    @property
    def entity_layer_name(self):
        return self._entity_layer_name
