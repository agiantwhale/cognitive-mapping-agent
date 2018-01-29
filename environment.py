import os
import inspect
import deepmind_lab as dl
import deepmind_lab_gym as dlg
import multiprocdmlab as mpdmlab

DEEPMIND_RUNFILES_PATH = os.path.dirname(inspect.getfile(dl))
DEEPMIND_SOURCE_PATH = os.path.abspath(DEEPMIND_RUNFILES_PATH + "/.." * 5)
dl.set_runfiles_path(DEEPMIND_RUNFILES_PATH)


def get_entity_layer_path(entity_layer_name):
    global DEEPMIND_RUNFILES_PATH, DEEPMIND_SOURCE_PATH
    mode, size, num = entity_layer_name.split("-")
    path_format = "{}/assets/entityLayers/{}/{}/entityLayers/{}.entityLayer"
    path = path_format.format(DEEPMIND_SOURCE_PATH, size, mode, num)

    return mode, size, num, path


def main():
    mapname = "training-09x09-0127"

    mode, _, _, path = get_entity_layer_path(mapname)

    env = mpdmlab.MultiProcDeepmindLab(
        dlg.DeepmindLab
        , "random_mazes"
        , dict(width=320, height=320, fps=30
               , rows=9
               , cols=9
               , mode=mode
               , num_maps=1
               , withvariations=True
               , random_spawn_random_goal="True"
               , chosen_map=mapname
               , mapnames=mapname
               , mapstrings=open(path).read()
               , apple_prob=0.9
               , episode_length_seconds=5)
        , dlg.ActionMapperDiscrete
        , additional_observation_types=["GOAL.LOC", "SPAWN.LOC", "POSE", "GOAL.FOUND"]
        , mpdmlab_workers=1
    )
    env.reset()

    terminal = False
    while not terminal:
        _, _, terminal, _ = env.step(0)


if __name__ == '__main__':
    main()
