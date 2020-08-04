import numpy as np
import tensorflow as tf


class Lstm(object):
    def __init__(self, args):
        self.seq_len = args['seq_len']
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.l2_reg_lambda = args['l2_reg_lambda']
        self.keep_prob = args['keep_prob']

        self.initializer = tf.random_normal_initializer(stddev=0.02)

        self._add_ops()

        self.device = '/cpu:0'
        if args['gpus'] > 0:
            self.device = '/gpu:0'
        with tf.device(self.device):
            self._lstm_layer()
            self._loss_layer()
            self._add_train_op()

    def _add_ops(self):
        self.source = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.input_size], name='source')
        self.source_len = tf.placeholder(tf.int32, shape=[None], name='source_len')
        self.target = tf.placeholder(tf.float32, shape=[None, self.seq_len], name='target')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _lstm_layer(self):
        with tf.variable_scope('lstm_layer', reuse=tf.AUTO_REUSE):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, initializer=self.initializer,
                                              state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, initializer=self.initializer,
                                              state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.source,
                sequence_length=self.source_len,
                swap_memory=True,
                dtype=tf.float32)

            lstm_output = tf.concat(outputs, axis=-1)  # [batch_size, seq_len, hidden_size*2]

        with tf.name_scope('dropout'):
            self.lstm_output = tf.nn.dropout(lstm_output, self.dropout_keep_prob)

    def _loss_layer(self):
        with tf.variable_scope('loss_layer'):
            output_1 = tf.layers.dense(self.lstm_output, self.hidden_size, activation=tf.nn.relu, use_bias=True)
            output_2 = tf.layers.dense(output_1, 1, activation=tf.nn.relu, use_bias=True)
            self.output = tf.reshape(output_2, (-1, self.seq_len))   # [batch_size, seq_len]

            score = tf.log(tf.abs(self.output) + 1) - tf.log(tf.abs(self.target) + 1)
            self.loss = tf.reduce_mean(score * score)

    def _add_train_op(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def train_step(self, sess, batch_x, batch_y):
        feed_dict = {
            self.source: batch_x,
            self.source_len: np.ones(len(batch_x)) * self.seq_len,
            self.target: batch_y,
            self.dropout_keep_prob: self.keep_prob
        }
        _, step, loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict)
        return step, loss

    def eval_step(self, sess, batch_x):
        feed_dict = {
            self.source: batch_x,
            self.source_len: np.ones(len(batch_x)) * self.seq_len,
            self.dropout_keep_prob: 1.0
        }
        output = sess.run(self.output, feed_dict)
        return output
