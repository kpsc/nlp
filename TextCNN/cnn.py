import tensorflow as tf

class TextCNN(object):
    def __init__(self, embedding, args):

        self.embedding = embedding
        self.embedding_size = embedding.shape[-1]
        self.seq_len = args['seq_len']
        self.num_classes = args['num_classes']
        self.filter_sizes = args['filter_sizes']
        self.num_filters = args['num_filters']
        self.l2_reg_lambda = args['l2_reg_lambda']
        self.keep_prob = args['keep_prob']

        self._add_ops()
        self._embedding_layer()
        self._cnn_layer(self.embedded_inputs)
        self._loss_layer(self.h_drop)

        self._add_train_op()

    def _add_ops(self):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _embedding_layer(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_weight = tf.Variable(self.embedding, dtype=tf.float32, name='W', trainable=True)
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_weight, self.input_x)

    def _cnn_layer(self, inputs):
        '''
        inputs: [batch_size, seq_len, hidden_size]
        outputs: [batch_size, hidden_size_]
        '''
        inputs = tf.expand_dims(inputs, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                _filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                conv = tf.nn.conv2d(inputs,
                                    _filter,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def _loss_layer(self, inputs):
        hidden_size = inputs.get_shape()[-1]

        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            weight = tf.get_variable('weight_ouput',
                                     shape=[hidden_size, self.num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(weight)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(inputs, weight, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _add_train_op(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def train_step(self, sess, batch_x, batch_y):
        feed_dict = {
            self.input_x: batch_x,
            self.input_y: batch_y,
            self.dropout_keep_prob: self.keep_prob
        }
        _, step, loss, accuracy = sess.run([self.train_op, self.global_step, self.loss, self.accuracy], feed_dict)
        return step, loss, accuracy

    def dev_step(self, sess, batch_x, batch_y):
        feed_dict = {
            self.input_x: batch_x,
            self.input_y: batch_y,
            self.dropout_keep_prob: 1.0
        }
        step, loss, accuracy = sess.run([self.global_step, self.loss, self.accuracy], feed_dict)
        return step, loss, accuracy

    def infer(self, sess, query):
        feed_dict = {
            self.input_x: query,
            self.dropout_keep_prob: 1.0
        }
        prob = sess.run(self.scores, feed_dict)
        return prob


