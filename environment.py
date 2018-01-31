import os
import inspect
import deepmind_lab as dl
import deepmind_lab_gym as dlg
import multiprocdmlab as mpdmlab

DEEPMIND_RUNFILES_PATH = os.path.dirname(inspect.getfile(dl))
DEEPMIND_SOURCE_PATH = os.path.abspath(DEEPMIND_RUNFILES_PATH + '/..' * 5)
dl.set_runfiles_path(DEEPMIND_RUNFILES_PATH)


def get_entity_layer_path(entity_layer_name):
    global DEEPMIND_RUNFILES_PATH, DEEPMIND_SOURCE_PATH
    mode, size, num = entity_layer_name.split('-')
    path_format = '{}/assets/entityLayers/{}/{}/entityLayers/{}.entityLayer'
    path = path_format.format(DEEPMIND_SOURCE_PATH, size, mode, num)

    return path


def get_game_environment(mapname='training-09x09-0127', mode='training', multiproc=False):
    mapstrings = ','.join(open(get_entity_layer_path(m)).read() for m in mapname.split(','))

    params = {
        'level_script': 'random_mazes',
        'config': dict(width=320, height=320, fps=30
                       , rows=9
                       , cols=9
                       , mode=mode
                       , num_maps=1
                       , withvariations=True
                       , random_spawn_random_goal='True'
                       , chosen_map=mapname
                       , mapnames=mapname
                       , mapstrings=mapstrings
                       , apple_prob=0.9
                       , episode_length_seconds=5),
        'action_mapper': dlg.ActionMapperDiscrete,
        'additional_observation_types': ['GOAL.LOC', 'SPAWN.LOC', 'POSE', 'GOAL.FOUND']
    }

    if multiproc:
        params['deepmind_lab_class'] = dlg.DeepmindLab
        params['mpdmlab_workers'] = 1
        env = mpdmlab.MultiProcDeepmindLab(**params)
    else:
        env = dlg.DeepmindLab(**params)

    return env
