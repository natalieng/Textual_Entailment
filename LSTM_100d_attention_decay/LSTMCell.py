#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state_tuple, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:

        z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)
        r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)
        o_t = tanh(x_t W_o + r_t * h_{t-1} U_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define U_r, W_r, b_r, U_z, W_z, b_z and U_o, W_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            xavier = tf.contrib.layers.xavier_initializer()
            constant = tf.constant_initializer(0)
            U_i = tf.get_variable("U_i", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_i = tf.get_variable("W_i", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_i = tf.get_variable("b_i", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_f = tf.get_variable("U_f", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_f = tf.get_variable("W_f", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_f = tf.get_variable("b_f", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_o = tf.get_variable("U_o", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_o = tf.get_variable("W_o", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_o = tf.get_variable("b_o", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_c = tf.get_variable("U_c", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_c = tf.get_variable("W_c", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_c = tf.get_variable("b_c", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(state_tuple.h, U_i) + b_i)
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(state_tuple.h, U_f) + b_f)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(state_tuple.h, U_o) + b_o)
            ct = tf.tanh(tf.matmul(inputs, W_c) + tf.matmul(state_tuple.h, U_c) + b_c)
            new_memory = f * state_tuple.c + i * ct
            new_state = o * tf.tanh(new_memory)
            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return output, tf.contrib.rnn.LSTMStateTuple(new_memory, new_state)

class LSTMCell2(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state_tuple, scope=None):
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            xavier = tf.contrib.layers.xavier_initializer()
            constant = tf.constant_initializer(0)
            U_i = tf.get_variable("U_i2", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_i = tf.get_variable("W_i2", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_i = tf.get_variable("b_i2", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_f = tf.get_variable("U_f2", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_f = tf.get_variable("W_f2", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_f = tf.get_variable("b_f2", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_o = tf.get_variable("U_o2", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_o = tf.get_variable("W_o2", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_o = tf.get_variable("b_o2", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            U_c = tf.get_variable("U_c2", shape=[self.state_size, self.state_size], dtype=tf.float32, initializer=xavier)
            W_c = tf.get_variable("W_c2", shape=[self.input_size, self.state_size], dtype=tf.float32, initializer=xavier)
            b_c = tf.get_variable("b_c2", shape=[self.state_size], dtype=tf.float32, initializer=constant)
            i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(state_tuple.h, U_i) + b_i)
            f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(state_tuple.h, U_f) + b_f)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(state_tuple.h, U_o) + b_o)
            ct = tf.tanh(tf.matmul(inputs, W_c) + tf.matmul(state_tuple.h, U_c) + b_c)
            new_memory = f * state_tuple.c + i * ct
            new_state = o * tf.tanh(new_memory)
            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return output, tf.contrib.rnn.LSTMStateTuple(new_memory, new_state)
