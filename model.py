import tensorflow as tf
from tensorflow.contrib import slim


class Model(object):
    def _build_mapper(self, visual_input, egomotion, m={}, estimator=None):
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape

        def _estimate(image):
            def _constrain_confidence(belief):
                estimate, confidence = tf.unstack(belief, axis=3)
                return tf.stack([estimate, tf.nn.sigmoid(confidence)], axis=3)

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], stride=1, padding='SAME'):
                    net = slim.conv2d(image, 64, [5, 5])
                    net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                    net = slim.conv2d(net, 128, [5, 5])
                    net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                    net = slim.conv2d(net, 256, [5, 5])
                    net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                    net = slim.fully_connected(net, 200)
                    net = slim.conv2d_transpose(net, 64, [24, 24], padding='VALID')
                    net = slim.conv2d_transpose(net, 32, [24, 24], padding='VALID')
                    net = slim.conv2d_transpose(net, 2, [14, 14], padding='VALID')
                    beliefs = [net] + [slim.conv2d_transpose(net, 2, [6, 6])
                                       for _ in range(estimate_scale - 1)]
            m['temporal_belief'] = [_constrain_confidence(belief) for belief in beliefs]
            return m['temporal_belief']

        def _apply_egomotion(belief, scale_index, ego):
            translation, rotation = tf.unstack(ego, axis=1)

            cos_rot = tf.cos(rotation)
            sin_rot = tf.sin(rotation)
            zero = tf.zeros_like(rotation)
            scale = tf.get_variable("scale_value_{}".format(scale_index),
                                    shape=(self._batch_size,),
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

            transform = tf.stack([cos_rot, sin_rot, tf.multiply(tf.negative(translation), scale),
                                  tf.negative(sin_rot), cos_rot, zero,
                                  zero, zero], axis=1)
            m['warped_previous_belief'] = tf.contrib.image.transform(belief, transform)
            return m['warped_previous_belief']

        def _warp(temp_belief, prev_belief):
            temp_estimate, temp_confidence = tf.unstack(temp_belief, axis=3)
            prev_estimate, prev_confidence = tf.unstack(prev_belief, axis=3)

            current_confidence = temp_confidence + prev_confidence
            current_estimate = tf.divide(tf.multiply(temp_estimate, temp_confidence) +
                                         tf.multiply(prev_estimate, prev_confidence),
                                         current_confidence)
            current_belief = tf.stack([current_estimate, current_confidence], axis=3)
            return current_belief

        class BiLinearSamplingCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return [tf.TensorShape(estimate_shape) for _ in range(estimate_scale)]

            @property
            def output_size(self):
                return [tf.TensorShape(estimate_shape) for _ in range(estimate_scale)]

            def __call__(self, inputs, state, scope=None):
                image, ego = inputs

                current_scaled_estimates = _estimate(image) if estimator is None else estimator(image)
                previous_scaled_estimates = [_apply_egomotion(belief, index, ego) for index, belief in enumerate(state)]
                outputs = [_warp(c, p) for c, p in zip(current_scaled_estimates, previous_scaled_estimates)]

                return outputs, outputs

        bilinear_cell = BiLinearSamplingCell()
        m['current_belief'], _ = tf.nn.dynamic_rnn(bilinear_cell, (visual_input, egomotion),
                                                   initial_state=bilinear_cell.zero_state(self._batch_size, tf.float32))
        return m['current_belief']

    @staticmethod
    def _build_planner():
        pass

    def __init__(self, batch_size=1, image_size=(320, 320), estimate_size=64, estimate_scale=2, estimator=None):
        self._batch_size = batch_size
        self._image_size = image_size
        self._estimate_shape = (estimate_size, estimate_size, 2)
        self._estimate_scale = estimate_scale

        tensors = {}

        current_input = tf.placeholder(tf.float32, [batch_size, None] + list(self._image_size) + [3])
        egomotion = tf.placeholder(tf.float32, (batch_size, None, 2))

        self._build_mapper(current_input, egomotion, tensors, estimator=estimator)


if __name__ == "__main__":
    Model(batch_size=32)
