import numpy as np
import environment
import expert
import argparse
import cv2
import copy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('maps')
parser.add_argument('--random_goal', default=True, type=bool)
parser.add_argument('--random_spawn', default=True, type=bool)
parser.add_argument('--num_games', default=10 ** 8, type=int)
FLAGS = parser.parse_args()


def build_trajectory_summary(info_history, exp):
    image = np.ones((28 + exp._width * 100, 28 + exp._height * 100, 3), dtype=np.uint8) * 255

    def _node_to_game_coordinate(node):
        row, col = node
        return 14 + int((col - 0.5) * 100), 14 + int((row - 0.5) * 100)

    def _pose_to_game_coordinate(pose):
        x, y = pose[:2]
        return 14 + int(x), 14 + image.shape[1] - int(y)

    cv2.putText(image, exp._env_name, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    for row, col in exp._walls:
        loc = np.array([col, row])
        points = [loc, loc + np.array([0, 1]),
                  loc + np.array([1, 1]), loc + np.array([1, 0])]
        points = np.array([pts * 100 + np.array([14, 14]) for pts in points])
        cv2.fillConvexPoly(image, points, (224, 172, 52))

    for info in info_history:
        cv2.circle(image, _node_to_game_coordinate(info['GOAL.LOC']), 10, (82, 82, 255), -1)
        cv2.circle(image, _node_to_game_coordinate(info['SPAWN.LOC']), 10, (211, 111, 112), -1)
        cv2.circle(image, _pose_to_game_coordinate(info['POSE']), 4, (63, 121, 255), -1)

    cv2.imshow('trajectory', image)
    cv2.waitKey(-1)


def main():
    env = environment.get_game_environment(mapname=FLAGS.maps,
                                           random_goal=FLAGS.random_goal,
                                           random_spawn=FLAGS.random_spawn)
    exp = expert.Expert()

    games = 0
    while games < FLAGS.num_games:
        reward_history = []
        info_history = []

        terminal = False
        while not terminal:
            obs, info = env.observations()

            # cv2.imshow('visual', obs)
            # cv2.waitKey(30)

            action = np.argmax(exp.get_optimal_action(info))
            _, reward, terminal, info = env.step(action)

            if info_history:
                egomotion = environment.calculate_egomotion(info_history[-1]['POSE'], info['POSE'])

                print 'Optimal action: {}'.format(action)
                print 'Egomotion: {}'.format(egomotion)

            reward_history.append(copy.deepcopy(reward))
            info_history.append(copy.deepcopy(info))

        env.reset()

        build_trajectory_summary(info_history, exp)


if __name__ == '__main__':
    main()
