import os
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.neighbors import NearestNeighbors


class Lstm_ranking(object):
    def __init__(self, embedding, args):

        self.embedding = embedding
        self.embedding_size = embedding.shape[-1]
        self.hidden_size = args.get('hidden_size', 100)
        self.seq_len = args['seq_len']
        self.l2_reg_lambda = args['l2_reg_lambda']
        self.max_grad_norm = args.get('max_grad_norm', 1.0)
        self.keep_prob = args['keep_prob']
        self.margin = args.get('margin', 0.5)

        self._add_ops()
        self._embedding_layer()

        self.device = '/gpu:0' if args['device'] >= 0 else '/cpu:0'
        with tf.device(self.device):
            self._build_nn()
            self._add_train_op()

    def _add_ops(self):
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_x_o = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x_o")
        self.input_x_c = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x_c")
        self.input_x_n = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x_n")

        self.x_o_len = tf.count_nonzero(self.input_x_o, axis=-1, dtype=tf.int32)
        self.x_c_len = tf.count_nonzero(self.input_x_c, axis=-1, dtype=tf.int32)
        self.x_n_len = tf.count_nonzero(self.input_x_n, axis=-1, dtype=tf.int32)

    def _embedding_layer(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_weight = tf.Variable(self.embedding, dtype=tf.float32, name='W', trainable=True)
            self.embedded_inputs_o = tf.nn.embedding_lookup(self.embedding_weight, self.input_x_o)
            self.embedded_inputs_c = tf.nn.embedding_lookup(self.embedding_weight, self.input_x_c)
            self.embedded_inputs_n = tf.nn.embedding_lookup(self.embedding_weight, self.input_x_n)

    def _lstm_layer(self, inputs, inputs_len, scope='lstm_nn', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)   # forward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)   # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                inputs,
                sequence_length=inputs_len,
                dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)  # [batch_size, seq_len, hidden_size*2]

            return self.attention(outputs, inputs_len)

    def attention(self, inputs, inputs_len):
        '''
        Args:
            inputs: [batch_size, seq_len, hidden_size]
            inputs_len: [batch_size]
        Return:
            outputs: [batch_size, hidden_size]
        '''
        _, maxlen, hidden_size = inputs.get_shape().as_list()

        inputs_expand = tf.expand_dims(inputs, axis=2)  # [batch_size, seq_len, 1, hidden_size]
        w_h = tf.get_variable('w_h', [1, 1, hidden_size, hidden_size])
        v = tf.get_variable('v', [hidden_size])

        h = tf.nn.conv2d(inputs_expand, w_h, [1, 1, 1, 1], 'SAME')
        e = tf.reduce_sum(v * tf.tanh(h), [2, 3])  # [batch_size, seq_len]

        mask = tf.sequence_mask(inputs_len, maxlen=maxlen, dtype=tf.float32)
        att = e * mask + (1 - mask) * (-1e6)
        att = tf.nn.softmax(att, axis=-1)

        o = tf.matmul(tf.transpose(inputs, [0, 2, 1]), tf.expand_dims(att, 2))

        return tf.reshape(o, [-1, hidden_size])

    def highway(self, inputs, w, b, scope='highway', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs = tf.layers.dropout(inputs, rate=1.0 - self.dropout_keep_prob)

            door = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, w, b))

            outputs = tf.nn.xw_plus_b(inputs, w, b)
            outputs = inputs * door + outputs * (1 - door)

            return tf.concat([inputs, outputs], axis=-1)

    def _build_nn(self):
        self.feature_o = self._lstm_layer(self.embedded_inputs_o, self.x_o_len, reuse=None)
        self.feature_c = self._lstm_layer(self.embedded_inputs_c, self.x_c_len, reuse=True)
        self.feature_n = self._lstm_layer(self.embedded_inputs_n, self.x_n_len, reuse=True)

        hidden_size = self.feature_o.get_shape()[-1]
        with tf.variable_scope('map'):
            w = tf.get_variable('w', shape=[hidden_size, hidden_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('b', shape=[hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

            self.feature_o = self.highway(self.feature_o, w, b, reuse=None)
            self.feature_c = self.highway(self.feature_c, w, b, reuse=True)
            self.feature_n = self.highway(self.feature_n, w, b, reuse=True)

            self.dis_o_c = self._dcos(self.feature_o, self.feature_c)
            self.dis_o_h = self._dcos(self.feature_o, self.feature_n)
            margin_loss = tf.nn.relu(self.margin - (self.dis_o_c - self.dis_o_h))
            positive_loss = tf.nn.relu(0.6 - self.dis_o_c)
            negative_loss = tf.nn.relu(self.dis_o_h - 0.4)
            self.d_loss = tf.reduce_sum(margin_loss + positive_loss + negative_loss)

            self.loss = self.d_loss + 0.2 * l2_loss
            # self.loss = self.d_loss

    def _dcos(self, a, b):
        # [batch_size, hidden_size]
        t_ab = tf.reduce_sum(a * b, axis=-1)
        t_a = tf.sqrt(tf.reduce_sum(a * a, axis=-1))
        t_b = tf.sqrt(tf.reduce_sum(b * b, axis=-1))

        return tf.div(t_ab, (t_a * t_b + 1e-8))

    def _add_train_op(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def train_step(self, sess, batch_o, batch_c, batch_n):
        feed_dict = {
            self.input_x_o: batch_o,
            self.input_x_c: batch_c,
            self.input_x_n: batch_n,
            self.dropout_keep_prob: self.keep_prob
        }

        output = {
            'step': self.global_step,
            'loss': self.loss,
            'dis_oc': self.dis_o_c,
            'dis_oh': self.dis_o_h,
            'dis_loss': self.d_loss
        }
        _, output = sess.run([self.train_op, output], feed_dict)
        return output['step'], output['loss']

    def dev_step(self, sess, data_o, data_c):
        feature_o = sess.run(self.feature_o, feed_dict={self.input_x_o: data_o, self.dropout_keep_prob: 1.0})
        feature_c = sess.run(self.feature_o, feed_dict={self.input_x_o: data_c, self.dropout_keep_prob: 1.0})

        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine').fit(feature_o)
        score, indices = nbrs.kneighbors(feature_c)

        c = 0
        c_3 = 0
        for i, idx in enumerate(indices):
            if idx[0] == i:
                c += 1
            if i in idx:
                c_3 += 1
        t = 1.0 * c / len(data_o)
        t_3 = 1.0 * c_3 / len(data_o)
        return t, t_3, indices, score

