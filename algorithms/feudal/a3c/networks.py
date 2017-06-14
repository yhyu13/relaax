from __future__ import print_function
import tensorflow as tf
import numpy as np

from relaax.algorithm_lib.lstm import CustomBasicLSTMCell
from config import cfg


class _A3CNetwork(object):
    def __init__(self, thread_index):
        channels = cfg.state_size[-1]
        if cfg.history_len > 1:
            channels = cfg.history_len

        W_conv1 = _conv_weight_variable([8, 8, channels, 16])   # stride=4
        b_conv1 = _conv_bias_variable([16], 8, 8, channels)     # 3<>12-ch or 3D-3x4

        W_conv2 = _conv_weight_variable([4, 4, 16, 32])  # stride=2
        b_conv2 = _conv_bias_variable([32], 4, 4, 16)

        W_fc1 = _fc_weight_variable([2592, 256])
        b_fc1 = _fc_bias_variable([256], 2592)

        # lstm
        self.lstm = CustomBasicLSTMCell(256)

        # weight for policy output layer
        W_fc2 = _fc_weight_variable([256, cfg.action_size])
        b_fc2 = _fc_bias_variable([cfg.action_size], 256)

        # weight for value output layer
        W_fc3 = _fc_weight_variable([256, 1])
        b_fc3 = _fc_bias_variable([1], 256)

        # state (input)
        shape = [None] + cfg.state_size
        if cfg.history_len > 1:
            shape = [None] + cfg.state_size + [cfg.history_len]
        self.s = tf.placeholder("float", shape, name='input_state')  # (?, 84, 84, 3)

        h_conv1 = tf.nn.relu(_conv2d(self.s, W_conv1, 4) + b_conv1)
        # h_conv1 (?, 20, 20, 16)
        h_conv2 = tf.nn.relu(_conv2d(h_conv1, W_conv2, 2) + b_conv2)
        # h_conv2 (?, 9, 9, 32)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        # h_conv2_flat(?, 2592)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
        # h_fc1 (?, 256)
        h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 256])
        # h_fc_reshaped (1, ?, 256)

        # placeholders for LSTM unrolling time step size & initial_lstm_state
        self.step_size = tf.placeholder(tf.float32, [1])
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size])

        scope = "net_" + str(thread_index)
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          h_fc1_reshaped,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope=scope)
        # lstm_outputs (1, ?, 256)
        self.weights = [
            W_conv1, b_conv1,
            W_conv2, b_conv2,
            W_fc1, b_fc1,
            self.lstm.matrix, self.lstm.bias,
            W_fc2, b_fc2,
            W_fc3, b_fc3
        ]

        lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
        # lstm_outputs (?, 256)

        # policy (output)
        self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, W_fc2) + b_fc2)
        # self.pi(?, action_size)

        # value (output)
        v_ = tf.matmul(lstm_outputs, W_fc3) + b_fc3
        # v_(?, 1)
        self.v = tf.reshape(v_, [-1])
        # self.v (?,)

        self.lstm_state_out = np.zeros([1, self.lstm.state_size])


class A3CGlobalNetwork(_A3CNetwork):
    def __init__(self, thread_index=-1):
        super(A3CGlobalNetwork, self).__init__(thread_index)
        self.learning_rate_input = tf.placeholder(tf.float32, [], name="lr")

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input,
            decay=cfg.RMSP_ALPHA,
            momentum=0.0,
            epsilon=cfg.RMSP_EPSILON
        )


class A3CLocalNetwork(_A3CNetwork):
    def __init__(self, thread_index):
        super(A3CLocalNetwork, self).__init__(thread_index)
        # taken action (input for policy)
        self.a = tf.placeholder("float", [None, cfg.action_size])

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None])

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(self.pi, 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(
            tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * cfg.entropy_beta)

        # R (input for value)
        self.r = tf.placeholder("float", [None])

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.r - self.v))

        # gradient of policy and value are summed up
        self.total_loss = policy_loss + value_loss

    def sync_from(self, src_network):
        sync_ops = []
        for (src_var, dst_var) in zip(src_network.weights, self.weights):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state: self.lstm_state_out,
                                                                 self.step_size: [1]})
        # pi_out.shape(1, action_size), v_out.shape(1, 1)-> reshaped to (1,)
        return pi_out[0], v_out[0]

    def run_policy(self, sess, s_t):
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state: self.lstm_state_out,
                                                          self.step_size: [1]})
        # pi_out.shape(1, action_size)
        return pi_out[0]

    def run_value(self, sess, s_t):
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state: self.lstm_state_out,
                                       self.step_size: [1]})
        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        # v_out.shape(1, 1)-> reshaped to (1,)
        return v_out[0]


def _conv_weight_variable(shape):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv_bias_variable(shape, w, h, input_channels):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


# weight initialization
def _fc_weight_variable(shape):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _fc_bias_variable(shape, input_channels):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="VALID")
