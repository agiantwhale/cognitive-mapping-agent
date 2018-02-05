import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import environment
import expert
from model import CMAP
import copy

flags = tf.app.flags
flags.DEFINE_string('maps', 'training-09x09-0127', 'Comma separated game environment list')
flags.DEFINE_string('logdir', 'output', 'Log directory')
flags.DEFINE_boolean('debug', False, 'Save debugging information')
flags.DEFINE_integer('num_games', 1000, 'Number of games to play.')
flags.DEFINE_integer('batch_size', 32, 'Number of environments to run.')
FLAGS = flags.FLAGS


def DAGGER_train_step(sess, train_op, global_step, train_step_kwargs):
    env = train_step_kwargs['env']
    exp = train_step_kwargs['exp']
    net = train_step_kwargs['net']

    random_rate = 0.9

    env.reset()
    obs, info = env.observations()

    optimal_action_history = [exp.get_optimal_action(info)]
    observation_history = [obs]
    egomotion_history = [[0., 0.]]
    rewards_history = [0.]
    estimate_maps_history = [[np.zeros((1, 64, 64, 3))] * 2]

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

    assert len(optimal_action_history) == len(observation_history) == len(egomotion_history) == len(rewards_history)

    # Training
    # We can just download more GPU ram from the internet, right?
    cumulative_loss = 0
    for i in xrange(0, len(optimal_action_history), FLAGS.batch_size):
        batch_end_index = i + FLAGS.batch_size

        concat_observation_history = [observation_history[:batch_end_index]] * FLAGS.batch_size
        concat_egomotion_history = [egomotion_history[:batch_end_index]] * FLAGS.batch_size
        concat_reward_history = [rewards_history[:batch_end_index]] * FLAGS.batch_size
        concat_optimal_action_history = optimal_action_history[i:batch_end_index]

        feed_dict = prepare_feed_dict(net.input_tensors, {'sequence_length': np.arange(1, batch_end_index - 1),
                                                          'visual_input': np.array(concat_observation_history),
                                                          'egomotion': np.array(concat_egomotion_history),
                                                          'reward': np.array(concat_reward_history),
                                                          'optimal_action': np.array(concat_optimal_action_history),
                                                          'estimate_map_list': estimate_maps_history[0],
                                                          'is_training': True})

        total_loss, np_global_step = sess.run([train_op, global_step], feed_dict=feed_dict)
        cumulative_loss += total_loss

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
    net = CMAP()

    optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(net.output_tensors['loss'], optimizer)

    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.logdir,
                        train_step_fn=DAGGER_train_step,
                        train_step_kwargs=dict(env=env, exp=exp, net=net),
                        number_of_steps=FLAGS.num_games,
                        save_summaries_secs=300,
                        save_interval_secs=600)


if __name__ == '__main__':
    tf.app.run()
