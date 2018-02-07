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
flags.DEFINE_boolean('multiproc', False, 'Multiproc environment')
flags.DEFINE_boolean('random_goal', True, 'Allow random goal')
flags.DEFINE_boolean('random_spawn', True, 'Allow random spawn')
flags.DEFINE_integer('num_games', 10 ** 8, 'Number of games to play')
flags.DEFINE_integer('batch_size', 1, 'Number of environments to run')
flags.DEFINE_float('decay', 0.999, 'DAGGER decay')
FLAGS = flags.FLAGS


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    env = train_step_kwargs['env']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']
    summary_writer = train_step_kwargs['summary_writer']

    step_history = train_step_kwargs['step_history']
    step_history_op = train_step_kwargs['step_history_op']

    update_global_step_op = train_step_kwargs['update_global_step_op']

    def _build_trajectory_summary(rate, loss, rewards_history, info_history, exp):
        image = np.ones((28 + exp._width * 100, 28 + exp._height * 100, 3), dtype=np.uint8) * 255

        def _node_to_game_coordinate(node):
            row, col = node
            return 14 + int((col - 0.5) * 100), 14 + int((row - 0.5) * 100)

        def _pose_to_game_coordinate(pose):
            x, y = pose[:2]
            return 14 + int(x), 14 + image.shape[1] - int(y)

        for info in info_history:
            cv2.circle(image, _node_to_game_coordinate(info['GOAL.LOC']), 10, (255, 0, 0), -1)
            cv2.circle(image, _node_to_game_coordinate(info['SPAWN.LOC']), 10, (0, 255, 0), -1)
            cv2.circle(image, _pose_to_game_coordinate(info['POSE']), 4, (0, 0, 255), -1)
        encoded = cv2.imencode('.png', image)[1].tostring()

        return tf.Summary(value=[tf.Summary.Value(tag='losses/trajectory',
                                                  image=tf.Summary.Image(encoded_image_string=encoded,
                                                                         height=image.shape[0],
                                                                         width=image.shape[1])),
                                 tf.Summary.Value(tag='losses/supervision_rate', simple_value=rate),
                                 tf.Summary.Value(tag='losses/loss', simple_value=loss),
                                 tf.Summary.Value(tag='losses/reward', simple_value=sum(rewards_history))])

    def _build_walltime_summary(begin, data, end):
        return tf.Summary(value=[tf.Summary.Value(tag='debug/DAGGER_eval_walltime', simple_value=(data - begin)),
                                 tf.Summary.Value(tag='debug/DAGGER_train_walltime', simple_value=(end - data)),
                                 tf.Summary.Value(tag='debug/DAGGER_complete_walltime', simple_value=(end - begin))])

    train_step_start = time.time()

    np_global_step = sess.run(global_step)

    random_rate = FLAGS.decay ** (np_global_step / 1000)

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
    while not terminal and len(info_history):
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

        optimal_action_history.append(copy.deepcopy(optimal_action))
        observation_history.append(copy.deepcopy(obs))
        egomotion_history.append(environment.calculate_egomotion(previous_info['POSE'], info['POSE']))
        rewards_history.append(copy.deepcopy(reward))
        estimate_maps_history.append([tensor[:, 0, :, :, :] for tensor in results[1:]])
        info_history.append(copy.deepcopy(info))

        summary_writer.add_summary(results[1], global_step=np_global_step)

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

        loss = sess.run(train_op, feed_dict=feed_dict)
        cumulative_loss += loss

    train_step_end = time.time()

    summary_text = os.linesep.join('{}[{}]-{}'.format(key, idx, step)
                                   for step, info in enumerate(info_history)
                                   for key in ('GOAL.LOC', 'SPAWN.LOC', 'POSE')
                                   for idx, value in enumerate(info[key]))
    step_history_summary = sess.run(step_history_op, feed_dict={step_history: summary_text})
    summary_writer.add_summary(step_history_summary, global_step=global_step)

    summary_writer.add_summary(_build_trajectory_summary(random_rate, cumulative_loss,
                                                         rewards_history, info_history, exp),
                               global_step=np_global_step)
    summary_writer.add_summary(_build_walltime_summary(train_step_start, train_step_eval, train_step_end),
                               global_step=np_global_step)

    should_stop = sess.run(update_global_step_op) >= FLAGS.num_games

    return cumulative_loss, should_stop


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
    tf.reset_default_graph()

    env = environment.get_game_environment(FLAGS.maps,
                                           multiproc=FLAGS.multiproc,
                                           random_goal=FLAGS.random_goal,
                                           random_spawn=FLAGS.random_spawn)
    exp = expert.Expert()
    net = CMAP()

    step_history = tf.placeholder(tf.string, name='step_history')
    step_history_op = tf.summary.text('game/step_history', step_history, collections=['game'])

    global_step = slim.get_or_create_global_step()
    update_global_step_op = tf.assign_add(global_step, 1)

    optimizer = tf.train.AdamOptimizer(learning_rate=100)
    train_op = slim.learning.create_train_op(net.output_tensors['loss'], optimizer,
                                             global_step=global_step, summarize_gradients=FLAGS.debug)

    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.logdir,
                        global_step=global_step,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(env=env, exp=exp, net=net,
                                               update_global_step_op=update_global_step_op,
                                               step_history=step_history,
                                               step_history_op=step_history_op),
                        number_of_steps=FLAGS.num_games,
                        save_summaries_secs=300 if not FLAGS.debug else 30,
                        save_interval_secs=600 if not FLAGS.debug else 60)


if __name__ == '__main__':
    tf.app.run()
