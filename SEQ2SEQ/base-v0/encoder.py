import tensorflow as tf
from tensorflow.contrib.rnn import (
    LSTMCell,
    GRUCell,
    MultiRNNCell,
    LSTMStateTuple,
    DropoutWrapper,
    ResidualWrapper)


class Encoder(object):
    def __init__(self, params={}):
        self.config = params
        self.hidden_size = self.config.get('hidden_size', 100)
        self.num_layers = self.config.get('num_layers', 1)
        self.cell_type = self.config.get('cell_type', 'lstm')

    def one_cell(self, hidden_size, cell_type, keep_prob=0.9):
        c = GRUCell if cell_type == 'gru' else LSTMCell
        cell = c(hidden_size)
        cell = DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=keep_prob)
        cell = ResidualWrapper(cell)

        return cell

    def add_encoder_cell(self, hidden_size, cell_type, num_layers, keep_prob=0.9):
        cells = [self.one_cell(hidden_size, cell_type, keep_prob) for _ in range(num_layers)]

        return MultiRNNCell(cells)

    def encoder(self, inputs, seq_len, keep_prob=0.9):
        batch_size = tf.shape(inputs)[0]
        with tf.variable_scope('encoder'):
            encoder_cell_fw = self.add_encoder_cell(self.hidden_size, self.cell_type, self.num_layers, keep_prob)
            encoder_cell_bw = self.add_encoder_cell(self.hidden_size, self.cell_type, self.num_layers, keep_prob)

            initial_state = encoder_cell_fw.zero_state(batch_size, dtype=tf.float32)
            encoder_outputs_, encoder_states_ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw,
                inputs=inputs,
                sequence_length=seq_len,
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
                # c_s = tf.concat([c_fw, c_bw], axis=-1)
                # h_s = tf.concat([h_fw, h_bw], axis=-1)
                c_s = tf.add(c_fw, c_bw)
                h_s = tf.add(h_fw, h_bw)
                encoder_states.append(LSTMStateTuple(c_s, h_s))
            encoder_states = tuple(encoder_states)

            return encoder_outputs, encoder_states
