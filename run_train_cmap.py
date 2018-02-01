import numpy as np
import tensorflow as tf
import environment
import expert
from collections import deque
from model import CMAP

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run.')
flags.DEFINE_integer('history_length', 32, 'History length that should be maintained by mapper for back-propagation.')
FLAGS = flags.FLAGS


def main(_):
    random_rate = 0.5
    env = environment.get_game_environment()
    exp = expert.Expert()
    net = CMAP()

    observation_history = deque([], FLAGS.history_length)
    egomotion_history = deque([], FLAGS.history_length)
    rewards_history = deque([], FLAGS.history_length)

    terminal = False
    while not terminal:
        _, info = env.observations()

        optimal_action = exp.get_optimal_action(info)

        action = np.argmax(exp.get_optimal_action(info))
        _, _, terminal, _ = env.step(action)


if __name__ == '__main__':
    tf.app.run()
