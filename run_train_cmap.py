import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
import expert
from collections import deque
from model import CMAP

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run.')
flags.DEFINE_integer('history_length', 32, 'History length that should be maintained by mapper for back-propagation.')
flags.DEFINE_string('logdir', 'output', 'Log directory')
FLAGS = flags.FLAGS


def prepare_feed_dict(tensors, data):
    feed_dict = {}
    for k, v in tensors.iteritems():
        if k not in data:
            continue

        if not isinstance(v, list):
            feed_dict[v] = data[k].astype(v.dtype.as_numpy_dtype)
        else:
            for t, d in zip(v, data[k]):
                feed_dict[t] = d.astype(t.dtype.as_numpy_dtype)

    return feed_dict


def main(_):
    random_rate = 0.5
    env = environment.get_game_environment()
    exp = expert.Expert()
    net = CMAP()

    obs, info = env.observations()

    previous_pose = info['POSE']

    observation_history = deque([obs], FLAGS.history_length)
    egomotion_history = deque([[0., 0.]],
                              FLAGS.history_length)
    rewards_history = deque([0.], FLAGS.history_length)
    estimate_maps = [np.zeros((1, 64, 64, 3))] * 2

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(net.output_tensors['loss'])
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        terminal = False
        while not terminal:
            optimal_action = exp.get_optimal_action(env.observations()[1])
            feed_dict = prepare_feed_dict(net.input_tensors, {'visual_input': np.array([observation_history]),
                                                              'egomotion': np.array([egomotion_history]),
                                                              'reward': np.array([rewards_history]),
                                                              'estimate_map_list': estimate_maps,
                                                              'optimal_action': np.array([optimal_action])})

            results = sess.run([net.output_tensors['action'], net.output_tensors['loss']] + net.intermediate_tensors[
                'estimate_map_list'],
                               feed_dict=feed_dict)
            for _ in range(100):
                sess.run(train_op, feed_dict=feed_dict)
            # save_path = saver.save(sess, FLAGS.logdir)
            # print("Model saved in path: %s" % save_path)

            predict_action = np.squeeze(results[0])
            loss = results[1]
            print loss
            dagger_action = random_rate * optimal_action + (1 - random_rate) * predict_action

            action = np.argmax(dagger_action)
            obs, reward, terminal, info = env.step(action)

            if len(observation_history) == FLAGS.history_length:
                estimate_maps = [estimate_map[:, 0, :, :, :] for estimate_map in results[2:]]

            observation_history.append(obs)
            egomotion_history.append(environment.calculate_egomotion(previous_pose, info['POSE']))
            rewards_history.append(reward)

            previous_pose = info['POSE']


if __name__ == '__main__':
    tf.app.run()
