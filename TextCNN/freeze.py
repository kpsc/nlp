#! /usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np

import data_helpers
vocab = data_helpers.Vocab('./data/word2vec-use')

def freeze_graph(path='./model/model_30.ckpt', output='./model/model.pb'):
    saver = tf.train.import_meta_graph(path+'.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, path)
        output_graph_def = graph_util.convert_variables_to_constants(
                           sess=sess,
                           input_graph_def=input_graph_def,   # = sess.graph_def,
                           output_node_names=['output/scores_rnn'])

        with tf.gfile.GFile(output, 'wb') as fgraph:
            fgraph.write(output_graph_def.SerializeToString())


def freeze_test(path='./model/model.pb'):
    with tf.gfile.GFile(path, 'rb') as fgraph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fgraph.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        input_x = graph.get_tensor_by_name('input_x:0')
        keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        pred = graph.get_tensor_by_name('output/scores:0')

        sess = tf.Session(graph=graph)

        # scores = sess.run(pred, feed_dict={input_x: x})
        with open('./data/intent.test', encoding='utf-8') as ftest:
            for line in ftest:
                arr = line.strip().split(',')
                sent = ','.join(arr[1:]).strip()
                feed_dict = {
                    input_x: np.array([vocab.encode(sent)]),
                    keep_prob: 1.0
                }
                score = sess.run(pred, feed_dict=feed_dict)
                score = score[0]

                idx = np.argmax(score)
                print(vocab.intent_type[idx], score[idx], line)


output_nodes = ['outputs']

def load_pb(path='model.pb'):
    with tf.gfile.GFile(path, 'rb') as fgraph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fgraph.read())

        return graph_def


def combined_graph():
    with tf.Graph().as_default() as g_combine:
        with tf.Session(graph=g_combine) as sess:
            graph_a = load_pb('./logs/pb/model.pb')
            graph_b = load_pb('./logs/pb/model_rnn.pb')

            tf.import_graph_def(graph_a, name='')
            tf.import_graph_def(graph_b, name='')

            g_combine_def = graph_util.convert_variables_to_constants(
                           sess=sess,
                           input_graph_def=sess.graph_def,
                           output_node_names=['output/scores_rnn', 'output/scores'])
            tf.train.write_graph(g_combine_def, './logs/pb/', 'model_combine.pb', as_text=False)


if __name__ == '__main__':
    freeze_graph(path='./logs/model_rnn/model_1', output='./logs/pb/model_rnn.pb')
    # freeze_test('./model/model.pb')
    # combined_graph()
