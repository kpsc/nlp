import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from encoder import Encoder
from decoder import Decoder
from utils import sequence_loss


class Seq2Seq(object):
    def __init__(self, args, mode='train'):
        self.config = args
        self.embedding_size = self.config.get('embedding_size')
        self.hidden_size = self.config.get('hidden_size')
        self.num_layers = self.config.get('num_layers')
        self.cell_type = self.config.get('cell_type')

        self.src_len = self.config.get('src_len', 30)
        self.tgt_len = self.config.get('tgt_len', 30)
        self.vcb_size = self.config.get('vcb_size')
        self.max_decode_step = self.config.get('max_decode_step')

        self.batch_size = self.config.get('batch_size')
        self.beam_size = self.config.get('beam_size')
        self.keep_prob_ = self.config.get('keep_prob', 0.9)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.learning_rate = self.config.get('learning_rate', 1e-3)

        self.mode = mode
        if self.mode != 'train':
            self.batch_size = self.batch_size * self.beam_size

        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        self.build_model()

    def build_model(self):
        self.add_ops()
        self.embedding()
        self.add_seq2seq()
        if self.mode == 'train':
            self.add_train_op()
        self.saver = tf.train.Saver()

    def add_train_op(self):
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars)
        with tf.device('/cpu:0'):
            grads, global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def add_ops(self):
        self.source = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='source')
        self.source_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='source_len')

        self.target = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='target')
        self.target_len = tf.placeholder(tf.int32, shape=[self.batch_size], name='target_len')

        if self.config.get('pointer_gen', True):
            self.source_extend = tf.placeholder(tf.int32, [self.batch_size, None], name='extend_vocab')
            self.oov_size = tf.placeholder(tf.int32, [], name='max_oovs')

        if self.mode == 'train':
            self.decoder_input_train = tf.placeholder(tf.int32, shape=[self.batch_size, self.tgt_len],
                                                      name='decoder_input')
        else:
            self.decoder_input_train = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def embedding(self):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            self.embedding_weight = tf.Variable(tf.truncated_normal([self.vcb_size, self.embedding_size]),
                                                dtype=tf.float32,
                                                name='encoder_embedding_weight',
                                                trainable=True)

            source_embedding = tf.nn.embedding_lookup(self.embedding_weight, self.source)
            self.source_embedding = tf.layers.dense(source_embedding, self.hidden_size, use_bias=False,
                                                    name='source_projection')

            self.decoder_embedding = [tf.nn.embedding_lookup(self.embedding_weight, x)
                                      for x in tf.unstack(self.decoder_input_train, axis=1)]

    def cal_final_dist(self, vocab_dists, att_dists, p_gens):
        '''
        Args:
            vocab_dists: List length max_dec_steps of [batch_size, vocab_size]
            att_dists: List length max_dec_steps of [batch_size, att_len]
        Returns:
        '''
        with tf.variable_scope('final_distribution'):
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
            att_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, att_dists)]

            extend_vcb_size = self.vcb_size + self.oov_size
            extra_vcb_zeros = tf.zeros((self.batch_size, self.oov_size))
            vocab_dists_extend = [tf.concat([dist, extra_vcb_zeros], axis=-1) for dist in vocab_dists]

            batch_nums = tf.range(0, self.batch_size)  # (batch_size, )
            batch_nums = tf.expand_dims(batch_nums, 1)  # (batch_size, 1)
            att_len = tf.shape(self.source_extend)[1]
            batch_nums = tf.tile(batch_nums, [1, att_len])  # [batch_size, att_len)
            indices = tf.stack([batch_nums, self.source_extend], axis=2)  # [batch_size, att_len, 2]
            shape = [self.batch_size, extend_vcb_size]
            att_dists_proj = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in att_dists]

            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extend, att_dists_proj)]

            return final_dists

    def add_seq2seq(self):
        self.encoder_outputs, self.encoder_states = self.encoder.encoder(self.source_embedding,
                                                                         self.source_len,
                                                                         self.keep_prob)
        self.dec_inp_state = self.encoder_states[-1]
        outputs, self.dec_out_state, self.att_dists, self.p_gens = self.decoder.decoder(
            self.decoder_embedding,
            self.dec_inp_state,
            self.encoder_outputs,
            self.source_len)

        with tf.variable_scope('output'):
            w = tf.get_variable('w', [self.hidden_size, self.vcb_size], dtype=tf.float32, initializer=self.init)
            b = tf.get_variable('b', [self.vcb_size], dtype=tf.float32, initializer=self.init)
            vocab_dists = []
            for i, o in enumerate(outputs):
                p = tf.nn.xw_plus_b(o, w, b)
                vocab_dists.append(tf.nn.softmax(p))

            final_dists = self.cal_final_dist(vocab_dists, self.att_dists, self.p_gens)
            self.prob = tf.stack(final_dists, axis=1)

            if self.mode == 'train':
                mask = tf.sequence_mask(self.target_len, maxlen=self.tgt_len, dtype=tf.float32)
                self.loss = sequence_loss(final_dists, self.target, mask)
            else:
                topk_probs, topk_ids = tf.nn.top_k(self.prob, self.batch_size * 2)
                self.topk_idx = tf.squeeze(topk_ids, axis=1)
                self.topk_log_prob = tf.log(tf.squeeze(topk_probs, axis=1))

    def encoder_run(self, sess, source, source_len):
        feed_dict = {
            self.source: source,
            self.source_len: source_len,
            self.keep_prob: 1.0
        }

        output = {
            'encoder_outputs': self.encoder_outputs,
            'dec_inp_state': self.dec_inp_state
        }
        output = sess.run(output, feed_dict)
        dec_inp_state = output['dec_inp_state']
        dec_inp_state = LSTMStateTuple(dec_inp_state.c[0], dec_inp_state.h[0])

        return output['encoder_outputs'], dec_inp_state

    def decode_onestep(self, sess, last_tokens, dec_pre_state, encoder_outputs, source_len, source_extend, oov_size):
        '''
        Args:
            last_tokens: tokens to be fed as input into the decoder for this timestep
            encoder_outputs: [beam_size, seq_len, hidden_size]
            dec_pre_state: List of bead_size LSTMStateTuples from the previous timestep
        return:
        '''
        beam_size = len(dec_pre_state)
        c = [np.expand_dims(state.c, axis=0) for state in dec_pre_state]
        h = [np.expand_dims(state.h, axis=0) for state in dec_pre_state]
        new_c = np.concatenate(c, axis=0)
        new_h = np.concatenate(h, axis=0)
        dec_pre_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        feed_dict = {
            self.decoder_input_train: np.transpose(np.array([last_tokens])),
            self.dec_inp_state: dec_pre_state,
            self.encoder_outputs: encoder_outputs,
            self.source_len: source_len,
            self.source_extend: source_extend,
            self.oov_size: oov_size,
            self.keep_prob: 1.0
        }
        output = {
            'idx': self.topk_idx,
            'probs': self.topk_log_prob,
            'states': self.dec_out_state
        }
        output = sess.run(output, feed_dict=feed_dict)
        dec_states = [LSTMStateTuple(output['states'].c[i, :], output['states'].h[i, :]) for i in range(beam_size)]

        return output['idx'], output['probs'], dec_states

    def train_step(self, sess, source, source_len, source_extend, oov_size, target, target_len, decoder_input,
                   optim=True):
        feed_dict = {
            self.source: source,
            self.source_len: source_len,
            self.source_extend: source_extend,
            self.oov_size: oov_size,
            self.target: target,
            self.target_len: target_len,
            self.decoder_input_train: decoder_input,
            self.keep_prob: self.keep_prob_
        }

        output = {
            'loss': self.loss,
            'prob': self.prob,
        }
        if optim:
            _, output = sess.run([self.train_op, output], feed_dict)
        else:
            output = sess.run(output, feed_dict)
        output['predict'] = np.argmax(output['prob'], axis=-1)

        return output

    def save(self, sess, path='./model/model.ckpt'):

        self.saver.save(sess, path)

    def load(self, sess, path='./model/model.ckpt'):

        self.saver.restore(sess, path)
