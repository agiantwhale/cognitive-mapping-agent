import tensorflow as tf
from tensorflow.contrib import slim


class CMAP(object):
    def _upscale_image(self, image):
        estimate_size = self._estimate_size
        crop_size = int(estimate_size / 4)
        image = image[:, crop_size:-crop_size, crop_size:-crop_size, :]
        image = tf.image.resize_bilinear(image, tf.constant([estimate_size, estimate_size]),
                                         align_corners=True)
        return image

    def _build_mapper(self, m={}, estimator=None):
        is_training = self._is_training
        sequence_length = self._sequence_length
        visual_input = self._visual_input
        egomotion = self._egomotion
        reward = self._reward
        estimate_map = self._estimate_map_list
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape

        def _estimate(image):
            def _constrain_confidence(belief):
                estimate, confidence = tf.unstack(belief, axis=3)
                return tf.stack([estimate, tf.nn.sigmoid(confidence)], axis=3)

            beliefs = []

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.0003)):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], stride=1, padding='SAME'):
                    with tf.variable_scope("free_space_estimator", reuse=tf.AUTO_REUSE):
                        net = slim.batch_norm(image, is_training=is_training)
                        net = slim.conv2d(net, 64, [5, 5])
                        net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                        net = slim.conv2d(net, 128, [5, 5])
                        net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                        net = slim.conv2d(net, 256, [5, 5])
                        net = slim.max_pool2d(net, stride=4, kernel_size=[4, 4])
                        net = slim.fully_connected(net, 200)
                        net = slim.conv2d_transpose(net, 64, [24, 24], padding='VALID')
                        net = slim.conv2d_transpose(net, 32, [24, 24], padding='VALID')
                        net = slim.conv2d_transpose(net, 2, [14, 14], padding='VALID')

                        beliefs.append(net)
                        for i in xrange(estimate_scale - 1):
                            beliefs.append(self._upscale_image(slim.conv2d_transpose(net, 2, [6, 6])))

            return [_constrain_confidence(belief) for belief in beliefs]

        def _apply_egomotion(tensor, scale_index, ego):
            translation, rotation = tf.unstack(ego, axis=1)

            cos_rot = tf.cos(rotation)
            sin_rot = tf.sin(rotation)
            zero = tf.zeros_like(rotation)
            scale = tf.constant(2 ** scale_index, dtype=tf.float32)

            transform = tf.stack([cos_rot, sin_rot, tf.multiply(tf.negative(translation), scale),
                                  tf.negative(sin_rot), cos_rot, zero,
                                  zero, zero], axis=1)
            return tf.contrib.image.transform(tensor, transform)

        def _delta_reward_map(reward):
            h, w, c = estimate_shape
            m_h, m_w = int((h - 1) / 2), int((w - 1) / 2)

            return tf.pad(tf.expand_dims(reward, axis=2),
                          tf.constant([[0, 0], [m_h - 1, w - m_h], [m_w - 1, w - m_w]]))

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
                return self.state_size

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
        interm_beliefs, final_belief = tf.nn.dynamic_rnn(bilinear_cell,
                                                         (visual_input, egomotion, tf.expand_dims(reward, axis=2)),
                                                         sequence_length=sequence_length,
                                                         initial_state=estimate_map)
        m['estimate_map_list'] = interm_beliefs
        return final_belief

    def _build_planner(self, scaled_beliefs, m={}):
        batch_size = tf.shape(scaled_beliefs[0])[0]
        image_scaler = self._upscale_image
        estimate_size = self._estimate_size
        value_map_size = (estimate_size, estimate_size, 1)
        num_actions = self._num_actions
        num_iterations = self._num_iterations

        def _fuse_belief(belief):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                stride=1, padding='SAME'):
                with tf.variable_scope("fuser", reuse=tf.AUTO_REUSE):
                    net = slim.repeat(belief, 3, slim.conv2d, 6, [1, 1])
                    net = slim.conv2d(net, 1, [1, 1])
                    return net

        class HierarchicalVINCell(tf.nn.rnn_cell.RNNCell):
            @property
            def state_size(self):
                return tf.TensorShape(value_map_size)

            @property
            def output_size(self):
                return self.state_size

            def __call__(self, inputs, state, scope=None):
                # Upscale previous value map
                state = image_scaler(state)

                with tf.variable_scope("VIN_prior", reuse=tf.AUTO_REUSE):
                    rewards_map = _fuse_belief(tf.concat([inputs, state], axis=3))
                    actions_map = slim.conv2d(rewards_map, num_actions, [3, 3])
                    values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                with tf.variable_scope("VIN", reuse=tf.AUTO_REUSE):
                    for i in xrange(num_iterations - 1):
                        rv = tf.concat([rewards_map, values_map], axis=3)
                        actions_map = slim.conv2d(rv, num_actions, [3, 3])
                        values_map = tf.reduce_max(actions_map, axis=3, keep_dims=True)

                return values_map, values_map

        vin_cell = HierarchicalVINCell()
        interm_values_map, final_values_map = tf.nn.dynamic_rnn(vin_cell, tf.stack(scaled_beliefs, axis=1),
                                                                initial_state=vin_cell.zero_state(batch_size,
                                                                                                  tf.float32))
        m['values_maps'] = interm_values_map

        values_features = slim.flatten(final_values_map)
        actions_logit = slim.fully_connected(values_features, num_actions)

        return actions_logit

    def __init__(self, image_size=(320, 320), estimate_size=64, estimate_scale=2,
                 estimator=None, num_actions=4, num_iterations=3):
        self._image_size = image_size
        self._estimate_size = estimate_size
        self._estimate_shape = (estimate_size, estimate_size, 3)
        self._estimate_scale = estimate_scale
        self._num_actions = num_actions
        self._num_iterations = num_iterations
        self._is_training = tf.placeholder(tf.bool)

        self._sequence_length = tf.placeholder(tf.int32, [None], name='visual_input')
        self._visual_input = tf.placeholder(tf.float32, [None, None] + list(self._image_size) + [3],
                                            name='visual_input')
        self._egomotion = tf.placeholder(tf.float32, (None, None, 2), name='egomotion')
        self._reward = tf.placeholder(tf.float32, (None, None), name='reward')
        self._estimate_map_list = [tf.placeholder(tf.float32, (None, estimate_size, estimate_size, 3),
                                                  name='estimate_map_{}'.format(i))
                                   for i in xrange(estimate_scale)]
        self._optimal_action = tf.placeholder(tf.float32, (None, num_actions), name='optimal_action')

        tensors = {}
        scaled_beliefs = self._build_mapper(tensors, estimator=estimator)
        unscaled_action = self._build_planner(scaled_beliefs, tensors)

        self._action = tf.nn.softmax(unscaled_action)
        self._loss = tf.losses.softmax_cross_entropy(self._optimal_action, unscaled_action)
        self._loss += tf.losses.get_regularization_loss()

        self._intermediate_tensors = tensors

    @property
    def input_tensors(self):
        return {
            'is_training': self._is_training,
            'sequence_length': self._sequence_length,
            'visual_input': self._visual_input,
            'egomotion': self._egomotion,
            'reward': self._reward,
            'estimate_map_list': self._estimate_map_list,
            'optimal_action': self._optimal_action
        }

    @property
    def intermediate_tensors(self):
        return self._intermediate_tensors

    @property
    def output_tensors(self):
        return {
            'action': self._action,
            'loss': self._loss
        }


if __name__ == "__main__":
    CMAP()
