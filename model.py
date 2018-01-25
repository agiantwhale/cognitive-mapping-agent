import tensorflow as tf
from tensorflow.contrib import slim


class Model(object):
    def _build_mapper(self, visual_input, egomotion, m={}, estimator=None):
        estimate_shape = self._estimate_size

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

            cos_rot = tf.cos(rotation)
            sin_rot = tf.sin(rotation)
            zero = tf.zeros(1, )

            transform = tf.reshape(tf.stack([cos_rot, sin_rot, tf.negative(translation),
                                             tf.negative(sin_rot), cos_rot, zero,
                                             zero, zero]), (8,))
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
                return tf.nn.rnn_cell.LSTMStateTuple(estimate_shape, estimate_shape)

            @property
            def output_size(self):
                return estimate_shape

            def __call__(self, inputs, state, scope=None):
                image, ego = inputs

                outputs = _warp(_estimate(image) if estimator is None else estimator(image),
                                _apply_egomotion(state, ego))

                return outputs, outputs

        m['current_belief'] = tf.nn.dynamic_rnn(MapUnrollCell(), (visual_input, egomotion),
                                                initial_state=tf.zeros(estimate_shape))
        return m['current_belief']

    @staticmethod
    def _build_planner():
        pass

    def __init__(self, image_size=(320, 320), estimate_size=(16, 16), estimator=None):
        self._image_size = image_size
        self._estimate_size = estimate_size

        tensors = {}

        current_input = tf.placeholder(tf.float32, [None, None] + list(self._image_size) + [3])
        egomotion = tf.placeholder(tf.float32, (None, None, 2))

        self._build_mapper(current_input, egomotion, tensors, estimator=estimator)
