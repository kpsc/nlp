import tensorflow as tf
from utils import linear


class Decoder(object):
    def __init__(self, params):
        self.config = params
        self.hidden_size = self.config.get('hidden_size', 100)
        self.num_layers = self.config.get('num_layers', 2)
        self.cell_type = self.config.get('cell_type', 'lstm')

        self.batch_size = self.config.get('batch_size', 64)
        self.beam_size = self.config.get('beam_size', 4)

        self.init = tf.truncated_normal_initializer(stddev=0.01)
        self.cell = tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True, initializer=self.init)

    def attention(self, decoder_state, encoder_outputs, encoder_len, reuse=None):
        hidden_size = encoder_outputs.get_shape().as_list()[-1]
        max_len = tf.shape(encoder_outputs)[1]
        with tf.variable_scope('Attention', reuse=reuse):
            encoder_outputs_ = tf.expand_dims(encoder_outputs, axis=2)     # [batch_size, seq_len, 1, hidden_size]
            w = tf.get_variable('w', [1, 1, hidden_size, hidden_size])
            encoder_features = tf.nn.conv2d(encoder_outputs_, w, [1, 1, 1, 1], 'SAME')

            decoder_feature = linear(decoder_state, hidden_size, True)  # [batch_size, hidden_size]
            decoder_feature = tf.expand_dims(tf.expand_dims(decoder_feature, 1), 1)

            v = tf.get_variable('v', [hidden_size])
            e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_feature), [2, 3])   # [batch_size, hidden_size]
            mask = tf.sequence_mask(encoder_len, maxlen=max_len, dtype=tf.float32)
            att = e * mask + (1 - mask) * (-1e6)
            att = tf.nn.softmax(att, axis=-1)   # [batch_size, seq_len]

            context_vector = tf.matmul(tf.transpose(encoder_outputs, [0, 2, 1]), tf.expand_dims(att, 2))
            context_vector = tf.reshape(context_vector, [-1, hidden_size])

            return context_vector, att

    def decoder(self, dec_inputs, pre_state, encoder_outputs, encoder_len):
        '''
        inputs:
            dec_inputs: A list of 2D Tensors [batch_size, embedding_size]
            encoder_outputs: [batch_size, seq_len, hidden_size]
            pre_state: [batch_size, hidden_size]
        returns:
            outputs: A list of the same length as decoder_inputs of 2D Tensor
            state: the final state of the decoder
        '''
        batch_size, _, hidden_size = encoder_outputs.get_shape().as_list()

        with tf.variable_scope('decoder'):
            # context_vector = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
            # if decode:
            #     context_vector, _ = self.attention(pre_state, encoder_outputs, encoder_len)

            state = pre_state
            outputs = []
            att_dists = []
            p_gens = []
            for i, dec_input in enumerate(dec_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # when we have output(y_i), then we can calculate the next time attention
                # if we are decode model, we decode only one step for each run
                context_vector, att = self.attention(state, encoder_outputs, encoder_len)
                att_dists.append(att)

                dec_size = dec_input.get_shape()[-1]
                x = linear([dec_input] + [context_vector], dec_size, True)   # merge input and context

                output, state = self.cell(x, state)

                with tf.variable_scope('decoder_output'):
                    output = linear([output] + [context_vector], self.cell.output_size, True)
                outputs.append(output)

                with tf.variable_scope('p_gen'):
                    # p_gen = sigma(w1*c + w2*st + w3*d_input + b)
                    # d_input: x or dec_input
                    p_gen = linear([context_vector, state.c, state.h, x], 1, True)
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

        return outputs, state, att_dists, p_gens


# Encoder: hj
# Decoder: si, s(i-1)

'''
time step 0:
    step1: [s(i-1), h] -> context
    step2: [y(i-1), context] -> y_input
    step3: [y_input, s(i-1)] --(lstm)-->>> yi, si
    
time step 1:
    step1: [si, h] -> context
    step2: [yi, context] -> y_input
    ...
'''

