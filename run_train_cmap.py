import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
import expert
from model import CMAP
import os
import copy
import time
import cv2

flags = tf.app.flags
flags.DEFINE_string('maps', 'training-09x09-0127', 'Comma separated game environment list')
flags.DEFINE_string('logdir', './output/dummy', 'Log directory')
flags.DEFINE_boolean('debug', False, 'Save debugging information')
flags.DEFINE_integer('num_games', 1000, 'Number of games to play')
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run')
FLAGS = flags.FLAGS


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    env = train_step_kwargs['env']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']
    summary_writer = train_step_kwargs['summary_writer']

    def _build_observation_summary(info_history):
        def _node_to_game_coordinate(node):
            row, col = node
            return (col + 0.5) * 100, (env._height - row - 0.5) * 100

        summary_text = os.linesep.join('{}[{}]-{}'.format(key, idx, step)
                                       for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE')
                                       for step, info in enumerate(info_history)
                                       for idx, value in enumerate(info[key]))
        text_summary = tf.summary.text('history', tf.convert_to_tensor(summary_text))

        image = np.zeros((env._width * 100, env._height * 100, 3), dtype=np.uint8)
        image.fill(255)
        for info in info_history:
            cv2.circle(image, _node_to_game_coordinate(info['GOAL.LOC']), 3, (255, 0, 0), -1)
            cv2.circle(image, _node_to_game_coordinate(info['SPAWN.LOC']), 3, (0, 255, 0), -1)
            cv2.circle(image, info['POSE'][:2], 1, (0, 0, 255), -1)
        image_summary = tf.summary.image('trajectory', tf.convert_to_tensor(image))

        return [text_summary, image_summary]

    def _build_walltime_summary(begin, data, end):
        return tf.Summary(value=[tf.Summary.Value(tag='DAGGER_eval_walltime', simple_value=(data - begin)),
                                 tf.Summary.Value(tag='DAGGER_train_walltime', simple_value=(end - data)),
                                 tf.Summary.Value(tag='DAGGER_complete_walltime', simple_value=(end - begin))])

    train_step_start = time.time()

    random_rate = 0.9

    env.reset()
    obs, info = env.observations()

    optimal_action_history = [exp.get_optimal_action(info)]
    observation_history = [obs]
    egomotion_history = [[0., 0.]]
    rewards_history = [0.]
    estimate_maps_history = [[np.zeros((1, 64, 64, 3))] * 2]
    info_history = [info]

    # Dataset aggregation
    terminal = False
    while not terminal:
        _, previous_info = env.observations()
        previous_info = copy.deepcopy(previous_info)

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': np.array([1]),
                                                          'visual_input': np.array([[observation_history[-1]]]),
                                                          'egomotion': np.array([[egomotion_history[-1]]]),
                                                          'reward': np.array([[rewards_history[-1]]]),
                                                          'estimate_map_list': estimate_maps_history[-1],
                                                          'is_training': False})

        results = sess.run([net.output_tensors['action']] +
                           net.intermediate_tensors['estimate_map_list'], feed_dict=feed_dict)
        predict_action = np.squeeze(results[0])
        optimal_action = exp.get_optimal_action(previous_info)

        dagger_action = random_rate * optimal_action + (1 - random_rate) * predict_action

        action = np.argmax(dagger_action)
        obs, reward, terminal, info = env.step(action)

        optimal_action_history.append(optimal_action)
        observation_history.append(obs)
        egomotion_history.append(environment.calculate_egomotion(previous_info['POSE'], info['POSE']))
        rewards_history.append(reward)
        estimate_maps_history.append([tensor[:, 0, :, :, :] for tensor in results[1:]])
        info_history.append(info)

    train_step_eval = time.time()

    assert len(optimal_action_history) == len(observation_history) == len(egomotion_history) == len(rewards_history)

    # Training
    cumulative_loss = 0
    for i in xrange(0, len(optimal_action_history), FLAGS.batch_size):
        batch_end_index = min(len(optimal_action_history), i + FLAGS.batch_size)
        batch_size = batch_end_index - i

        concat_observation_history = [observation_history[:batch_end_index]] * batch_size
        concat_egomotion_history = [egomotion_history[:batch_end_index]] * batch_size
        concat_reward_history = [rewards_history[:batch_end_index]] * batch_size
        concat_optimal_action_history = optimal_action_history[i:batch_end_index]
        concat_estimate_map_list = [np.zeros((batch_size, 64, 64, 3))] * 2

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': np.arange(i, batch_end_index) + 1,
                                                          'visual_input': np.array(concat_observation_history),
                                                          'egomotion': np.array(concat_egomotion_history),
                                                          'reward': np.array(concat_reward_history),
                                                          'optimal_action': np.array(concat_optimal_action_history),
                                                          'estimate_map_list': concat_estimate_map_list,
                                                          'is_training': True})

        total_loss = sess.run(train_op, feed_dict=feed_dict)
        cumulative_loss += total_loss

    train_step_end = time.time()

    np_global_step = sess.run(global_step)

    observation_summaries = sess.run(_build_observation_summary(info_history))
    for summary in observation_summaries:
        summary_writer.add_summary(summary, global_step=np_global_step)

    if FLAGS.debug:
        summary_writer.add_summary(_build_walltime_summary(train_step_start, train_step_eval, train_step_end),
                                   global_step=np_global_step)

    return cumulative_loss, False


def prepare_feed_dict(tensors, data):
    feed_dict = {}
    for k, v in tensors.iteritems():
        if k not in data:
            continue

        if not isinstance(v, list):
            if isinstance(data[k], np.ndarray):
                feed_dict[v] = data[k].astype(v.dtype.as_numpy_dtype)
            else:
                feed_dict[v] = data[k]
        else:
            for t, d in zip(v, data[k]):
                feed_dict[t] = d.astype(t.dtype.as_numpy_dtype)

    return feed_dict


def main(_):
    env = environment.get_game_environment(FLAGS.maps)
    exp = expert.Expert()
    net = CMAP(debug=FLAGS.debug)

    optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(net.output_tensors['loss'], optimizer)
    tf.summary.scalar('losses/total_loss', net.output_tensors['loss'])

    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.logdir,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(env=env, exp=exp, net=net),
                        number_of_steps=FLAGS.num_games,
                        save_summaries_secs=300 if not FLAGS.debug else 30,
                        save_interval_secs=600 if not FLAGS.debug else 60)


if __name__ == '__main__':
    tf.app.run()
