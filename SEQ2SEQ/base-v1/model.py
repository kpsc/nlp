import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import (
    LSTMCell,
    GRUCell,
    MultiRNNCell,
    LSTMStateTuple,
    DropoutWrapper,
    ResidualWrapper)
from tensorflow.contrib.seq2seq import (
    BahdanauAttention,
    AttentionWrapper,
    TrainingHelper,
    BasicDecoder,
    BeamSearchDecoder)
from data import DataUnit


class Seq2Seq(object):
    def __init__(self, args, mode='train'):
        self.config = args
        self.embedding_size = self.config.get('embedding_size', 100)
        self.hidden_size = self.config.get('hidden_size', 100)
        self.num_layers = self.config.get('num_layers', 2)
        self.cell_type = self.config.get('cell_type', 'gru')

        self.max_decode_step = self.config.get('max_decode_step', 30)
        self.src_len = self.config.get('src_len', 30)
        self.tgt_len = self.config.get('tgt_len', 30)
        self.src_vcb_size = self.config.get('src_vcb_size', 10000)
        self.tgt_vcb_size = self.config.get('tgt_vcb_size', 10000)

        self.batch_size = self.config.get('batch_size', 64)
        self.beam_size = self.config.get('beam_size', 4)
        self.keep_prob_ = self.config.get('keep_prob', 0.9)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.learning_rate = self.config.get('learning_rate', 1e-3)

        self.bidirection = self.config.get('bidirection', True)
        self.share_embedding = self.config.get('share_embedding', True)
        self.mode = mode

        self.build_model()

    def build_model(self):
        self.add_ops()
        self.embedding()
        encoder_outputs, encoder_states = self.encoder()
        self.decoder(encoder_outputs, encoder_states)
        if self.mode == 'train':
            self.add_train_op()
        self.saver = tf.train.Saver()

    def add_train_op(self):
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars)

        with tf.device('/cpu:0'):
            grads, global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        with tf.device('/cpu:0'):
            self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                      global_step=self.global_step,
                                                      name='train_step')

    def add_ops(self):
        self.source = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='source')
        # self.source_len = tf.count_nonzero(self.source, axis=-1, dtype=tf.int32, name='source_eln')
        self.source_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='source_len')

        if self.mode == 'train':
            self.target = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='target')
            # self.target_len = tf.count_nonzero(self.target, axis=-1, dtype=tf.int32, name='target_len')
            self.target_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='target_len')

            self.decoder_start_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * DataUnit.START_INDEX
            self.decoder_input_train = tf.concat([self.decoder_start_token, self.target], axis=1)

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def embedding(self):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            self.encoder_embeddings = tf.Variable(tf.truncated_normal([self.src_vcb_size, self.embedding_size]),
                                                  dtype=tf.float32,
                                                  name='encoder_embedding_weight',
                                                  trainable=True)
            if self.share_embedding:
                self.decoder_embeddings = self.encoder_embeddings
            else:
                self.decoder_embeddings = tf.Variable(tf.truncated_normal([self.tgt_vcb_size, self.embedding_size]),
                                                      dtype=tf.float32,
                                                      name='decoder_embedding_weight',
                                                      trainable=True)

    def one_cell(self, hidden_size, cell_type):
        c = GRUCell if cell_type == 'gru' else LSTMCell
        cell = c(hidden_size)
        cell = DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=self.keep_prob)
        cell = ResidualWrapper(cell)

        return cell 

    def add_encoder_cell(self, hidden_size, cell_type, num_layers):
        cells = [self.one_cell(hidden_size, cell_type) for _ in range(num_layers)]

        return MultiRNNCell(cells)

    def encoder(self):
        with tf.variable_scope('encoder'):
            encoder_cell = self.add_encoder_cell(self.hidden_size, self.cell_type, self.num_layers)
            source_embedding = tf.nn.embedding_lookup(self.encoder_embeddings, self.source)
            source_embedding = tf.layers.dense(source_embedding, self.hidden_size, use_bias=False, name='source_projection')

            initial_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)
            if not self.bidirection:
                # encoder_outputs: [batch_size, seq_len, hidden_size]
                # encoder_state: N-tuple(LSTMStateTuple for each cell)
                encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=source_embedding,
                    sequence_length=self.source_len,
                    dtype=tf.float32,
                    initial_state=initial_state,
                    swap_memory=True
                )
            else:
                encoder_cell_bw = self.add_encoder_cell(self.hidden_size, self.cell_type, self.num_layers)
                encoder_outputs_, encoder_states_ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_cell,
                    cell_bw=encoder_cell_bw,
                    inputs=source_embedding,
                    sequence_length=self.source_len,
                    initial_state_fw=initial_state,
                    initial_state_bw=initial_state,
                    dtype=tf.float32,
                    swap_memory=True
                )
                encoder_outputs = tf.concat(encoder_outputs_, axis=-1)
                encoder_states = []
                for i in range(self.num_layers):
                    c_fw, h_fw = encoder_states_[0][i]
                    c_bw, h_bw = encoder_states_[1][i]
                    c_s = tf.concat([c_fw, c_bw], axis=-1)
                    h_s = tf.concat([h_fw, h_bw], axis=-1)
                    encoder_states.append(LSTMStateTuple(c_s, h_s))
                encoder_states = tuple(encoder_states)
            return encoder_outputs, encoder_states

    def add_decoder_cell(self, encoder_outputs, encoder_states, hidden_size, cell_type, num_layers):
        encoder_seq_len = self.source_len
        if self.mode == 'decode':
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
            encoder_states = tf.contrib.seq2seq.tile_batch(encoder_states, multiplier=self.beam_size)
            encoder_seq_len = tf.contrib.seq2seq.tile_batch(encoder_seq_len, multiplier=self.beam_size)

        hidden_size_ = hidden_size*2 if self.bidirection else hidden_size
        cell = MultiRNNCell([self.one_cell(hidden_size_, cell_type) for _ in range(num_layers)])
        self.attention = BahdanauAttention(self.hidden_size, encoder_outputs, encoder_seq_len)

        def cell_input_fn(inputs, attention):
            att_proj = tf.layers.Dense(hidden_size_, dtype=tf.float32, use_bias=False, name='att_proj')

            return att_proj(tf.concat([inputs, attention], axis=-1))

        decoder_cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention,
            attention_layer_size=hidden_size,
            cell_input_fn=cell_input_fn,
            name='attentionwrapper'
        )

        d_size = self.beam_size*self.batch_size if self.mode == 'decode' else self.batch_size
        decoder_initial_state = decoder_cell.zero_state(batch_size=d_size, dtype=tf.float32
                                                        ).clone(cell_state=encoder_states)

        return decoder_cell, decoder_initial_state

    def decoder(self, encoder_outputs, encoder_states):
        decoder_cell, decoder_init_state = self.add_decoder_cell(encoder_outputs, encoder_states, self.hidden_size,
                                                                 self.cell_type, self.num_layers)
        output_proj = tf.layers.Dense(self.tgt_vcb_size, dtype=tf.float32, use_bias=False,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      name='output_proj')
        if self.mode == 'train':
            target_embedding = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_input_train)
            training_helper = TrainingHelper(target_embedding, self.target_len, name='training_helper')
            training_decoder = BasicDecoder(decoder_cell, training_helper, decoder_init_state, output_proj)
            max_dec_len = tf.reduce_max(self.target_len)
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, maximum_iterations=max_dec_len)
            self.d_masks = tf.sequence_mask(self.target_len, max_dec_len, dtype=tf.float32, name='d_masks')
            self.prob = output.rnn_output
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.prob,
                targets=self.target,
                weights=self.d_masks,
                average_across_timesteps=True,
                average_across_batch=True
            )
        else:
            start_token = [DataUnit.START_INDEX] * self.batch_size
            end_token = DataUnit.END_INDEX
            inference_decoder = BeamSearchDecoder(
                cell=decoder_cell,
                embedding=lambda x: tf.nn.embedding_lookup(self.decoder_embeddings, x),
                start_tokens=start_token,
                end_token=end_token,
                initial_state=decoder_init_state,
                beam_width=self.beam_size,
                output_layer=output_proj
            )
            output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=self.max_decode_step)
            output_pred_ = output.predicted_ids
            self.decoder_output = tf.transpose(output_pred_, perm=[0, 2, 1])

    def train_step(self, sess, source, source_len, target, target_len, optim=True):
        feed_dict = {
            self.source: source,
            self.source_len: source_len,
            self.target: target,
            self.target_len: target_len,
            self.keep_prob: self.keep_prob_
        }

        output = {
            'loss': self.loss,
            'prob': self.prob
        }
        if optim:
            _, output = sess.run([self.train_op, output], feed_dict)
        else:
            output = sess.run({'prob': self.prob}, feed_dict)
        output['predict'] = np.argmax(output['prob'], axis=-1)

        return output

    def predict(self, sess, source, source_len):
        feed_dict = {
            self.source: source,
            self.source_len: source_len,
            self.keep_prob: 1.0
        }
        result = sess.run(self.decoder_output, feed_dict)
        return result[0]

    def get_response(self, sess, sententse, du):
        idx = du.transform_sentence(sententse.strip())
        sent = np.asarray(idx).reshape(1, -1)
        sent_len = np.asarray(len(idx)).reshape((1,))
        pred = self.predict(sess, np.array(sent), np.array(sent_len))
        result = du.transform_indexs(pred[0])

        return result

    def save(self, sess, path='./model/model.ckpt'):

        self.saver.save(sess, path)

    def load(self, sess, path='./model/model.ckpt'):

        self.saver.restore(sess, path)
