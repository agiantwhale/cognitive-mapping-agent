import os
import inspect
import deepmind_lab as dl
import deepmind_lab_gym as dlg
import multiprocdmlab as mpdmlab


# from top_view_renderer import EntityMap


def main():
    building_name = "training-09x09-0127"
    mode, size, num = building_name.split("-")

    deepmind_runfiles_path = os.path.dirname(inspect.getfile(dl))
    deepmind_source_path = os.path.abspath(deepmind_runfiles_path + "/.." * 5)
    mapstrings_path = "{}/assets/entityLayers/{}/{}/entityLayers/{}.entityLayer".format(deepmind_source_path,
                                                                                        size,
                                                                                        mode,
                                                                                        num)

    # entmap = EntityMap(mapstrings_path)

    dl.set_runfiles_path(deepmind_runfiles_path)

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
               , entitydir=deepmind_runfiles_path
               , chosen_map=building_name
               , mapnames=building_name
               , mapstrings=open(mapstrings_path).read()
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
