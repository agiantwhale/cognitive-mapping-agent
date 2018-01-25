import tensorflow as tf
from tensorflow.contrib import slim


class Model(object):
    @staticmethod
    def _build_mapper(visual_input, egomotion, m={}, estimator=None):
        ESTIMATE_SIZE = (16, 16)

        def _estimate(image):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                padding='SAME',
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    net = slim.conv2d(image, 64, [5, 5])
                    net = slim.conv2d(net, 128, [5, 5])
                    net = slim.conv2d(net, 256, [5, 5])
                    net = slim.fully_connected(net, 100)
                    net = slim.conv2d_transpose(net, 128, [5, 5])
                    net = slim.conv2d_transpose(net, 64, [5, 5])
            m['temporal_belief'] = net
            return m['temporal_belief']

        def _apply_egomotion(belief, ego):
            translation, rotation = tf.unstack(ego)
            # Transform relative to agent -- in other direction
            transform = tf.reshape(tf.stack([tf.cos(rotation),
                                             tf.sin(rotation),
                                             tf.negative(translation),
                                             tf.negative(tf.sin(rotation)),
                                             tf.cos(rotation),
                                             tf.zeros(1, ),
                                             tf.zeros(1, )]), (8,))
            m['warped_previous_belief'] = tf.contrib.image.transform(belief, transform)
            return m['warped_previous_belief']

        def _warp(temp_belief, prev_belief):
            temp_estimate, temp_confidence = tf.unstack(temp_belief, axis=2)
            prev_estimate, prev_confidence = tf.unstack(prev_belief, axis=2)

            current_confidence = temp_confidence + prev_confidence
            current_estimate = tf.divide(tf.multiply(temp_estimate, temp_confidence) +
                                         tf.multiply(prev_estimate, prev_confidence),
                                         current_confidence)
            return tf.stack([current_estimate, current_confidence], axis=2)

        class MapUnrollCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return tf.nn.rnn_cell.LSTMStateTuple(ESTIMATE_SIZE, ESTIMATE_SIZE)

            @property
            def output_size(self):
                return ESTIMATE_SIZE

            def __call__(self, inputs, state, scope=None):
                image, ego = inputs

                outputs = _warp(_estimate(image) if estimator is None else estimator(image),
                                _apply_egomotion(state, ego))

                return outputs, outputs

        m['current_belief'] = tf.nn.dynamic_rnn(MapUnrollCell(), (visual_input, egomotion),
                                                initial_state=tf.zeros(ESTIMATE_SIZE))
        return m['current_belief']

    @staticmethod
    def _build_planner():
        pass

    def __init__(self, image_size=(320, 320), map_size=(16, 16), estimator=None):
        self._image_size = image_size
        self._map_size = map_size

        tensors = {}

        current_input = tf.placeholder(tf.int32, self._image_size)
        previous_estimate = tf.placeholder(tf.float32, self._map_size)
        egomotion = tf.placeholder(tf.float32, (2,))

        self._build_mapper(current_input, previous_estimate, egomotion, tensors, estimator=estimator)
