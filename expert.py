from top_view_renderer import EntityMap
from environment import get_entity_layer_path


class Expert(object):
    def __init__(self, entity_layer):
        self._entity_layer_name = entity_layer

    @property
    def entity_layer_name(self):
        return self._entity_layer_name

    def get_optimal_action(self, info):
        raise NotImplementedError("optimal action is not yet implemented")
