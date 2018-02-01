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

    obs, previous_info = env.observations()

    observation_history = deque([obs], FLAGS.history_length)
    egomotion_history = deque([[0., 0.]],
                              FLAGS.history_length)
    rewards_history = deque([0.], FLAGS.history_length)
    estimate_maps = [np.zeros((1, 64, 64, 3))] * 2

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        terminal = False
        while not terminal:
            feed_data = {
                'visual_input': np.array([observation_history]),
                'egomotion': np.array([egomotion_history]),
                'reward': np.array([rewards_history]),
                'estimate_map_list': estimate_maps
            }

            feed_dict = {}
            for k, v in net.input_tensors.iteritems():
                if not isinstance(v, list):
                    feed_dict[v] = feed_data[k].astype(v.dtype.as_numpy_dtype)
                else:
                    for t, d in zip(v, feed_data[k]):
                        feed_dict[t] = d.astype(t.dtype.as_numpy_dtype)

            results = sess.run([net.output_tensors['action']] + net.intermediate_tensors['estimate_map_list'],
                               feed_dict=feed_dict)

            predict_action = results[0]
            optimal_action = exp.get_optimal_action(previous_info)
            dagger_action = random_rate * optimal_action + (1 - random_rate) * predict_action

            action = np.argmax(dagger_action)
            obs, info, terminal, reward = env.step(action)

            if len(observation_history) == FLAGS.history_length:
                estimate_maps = [np.squeeze(estimate_map[:, 0, :, :, :], axis=1) for estimate_map in results[1:]]

            observation_history.append([obs])
            egomotion_history.append([environment.calculate_egomotion(previous_info['POSE'], info['POSE'])])
            rewards_history.append([reward])

            previous_info = info


if __name__ == '__main__':
    tf.app.run()
