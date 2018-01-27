import tensorflow as tf
from tensorflow.contrib import slim


class Model(object):
    def _build_mapper(self, visual_input, egomotion, reward, m={}, estimator=None):
        batch_size = self._batch_size
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

        def _apply_egomotion(tensor, scale_index, ego):
            translation, rotation = tf.unstack(ego, axis=1)

            cos_rot = tf.cos(rotation)
            sin_rot = tf.sin(rotation)
            zero = tf.zeros_like(rotation)
            scale = tf.constant(2 ** scale_index, dtype=tf.float32, shape=(batch_size,))

            transform = tf.stack([cos_rot, sin_rot, tf.multiply(tf.negative(translation), scale),
                                  tf.negative(sin_rot), cos_rot, zero,
                                  zero, zero], axis=1)
            m['warped_previous_belief'] = tf.contrib.image.transform(tensor, transform)
            return m['warped_previous_belief']

        def _delta_reward_map(reward):
            h, w, c = estimate_shape
            m_h, m_w = int((h - 1) / 2), int((w - 1) / 2)

            return tf.scatter_nd(tf.constant([[i, m_h, m_w] for i in range(batch_size)]),
                                 tf.squeeze(reward), tf.constant([batch_size, h, w]))

        def _warp(temp_belief, prev_belief):
            temp_estimate, temp_confidence, temp_rewards = tf.unstack(temp_belief, axis=3)
            prev_estimate, prev_confidence, prev_rewards = tf.unstack(prev_belief, axis=3)

            current_confidence = temp_confidence + prev_confidence
            current_estimate = tf.divide(tf.multiply(temp_estimate, temp_confidence) +
                                         tf.multiply(prev_estimate, prev_confidence),
                                         current_confidence)
            current_rewards = temp_rewards + prev_rewards
            current_belief = tf.stack([current_estimate, current_confidence, current_rewards], axis=3)
            return current_belief

        class BiLinearSamplingCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            @property
            def output_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            def __call__(self, inputs, state, scope=None):
                image, ego, re = inputs

                delta_reward_map = tf.expand_dims(_delta_reward_map(re), axis=3)

                current_scaled_estimates = _estimate(image) if estimator is None else estimator(image)
                current_scaled_estimates = [tf.concat([estimate, delta_reward_map], axis=3)
                                            for estimate in current_scaled_estimates]
                previous_scaled_estimates = [_apply_egomotion(belief, scale_index, ego)
                                             for scale_index, belief in enumerate(state)]
                outputs = [_warp(c, p) for c, p in zip(current_scaled_estimates, previous_scaled_estimates)]

                return outputs, outputs

        bilinear_cell = BiLinearSamplingCell()
        m['current_belief'], _ = tf.nn.dynamic_rnn(bilinear_cell,
                                                   (visual_input, egomotion, tf.expand_dims(reward, axis=2)),
                                                   initial_state=bilinear_cell.zero_state(batch_size, tf.float32))
        return m['current_belief']

    def _build_planner(self, scaled_beliefs, m={}):
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape

        def _fuse_belief(belief):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                stride=1, kernel_size=[3, 3], padding='SAME'):
                with tf.variable_scope("fuser"):
                    net = slim.repeat(belief, 3, slim.conv2d, 6)
                    net = slim.conv2d(net, 1)
                    return net

        class ValueIterationCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            @property
            def output_size(self):
                return [tf.TensorShape(estimate_shape)] * estimate_scale

            def __call__(self, inputs, state, scope=None):
                return state, state

        vin_cell = ValueIterationCell()
        m['value_map'], _ = tf.nn.dynamic_rnn(vin_cell, stacked_scaled_beliefs,
                                              initial_state=vin_cell.zero_state(self._batch_size, tf.float32))
        return m['value_map']

    def __init__(self, batch_size=1, image_size=(320, 320), estimate_size=64, estimate_scale=2, estimator=None):
        self._batch_size = batch_size
        self._image_size = image_size
        self._estimate_shape = (estimate_size, estimate_size, 3)
        self._estimate_scale = estimate_scale

        tensors = {}

        current_input = tf.placeholder(tf.float32, [batch_size, None] + list(self._image_size) + [3])
        egomotion = tf.placeholder(tf.float32, (batch_size, None, 2))
        reward = tf.placeholder(tf.float32, (batch_size, None))

        self._build_mapper(current_input, egomotion, reward, tensors, estimator=estimator)


if __name__ == "__main__":
    Model(batch_size=32)
