import tensorflow as tf
import numpy as np
import jieba


class Vocab(object):
    def __init__(self, vcb, seq_len=20):
        self.word2index = {}
        self.index2word = {}
        with open(vcb, encoding='utf-8') as fvcb:
            for line in fvcb:
                arr = line.strip().split()
                self.word2index[arr[0]] = arr[1]
                self.index2word[arr[1]] = arr[0]

        self.seq_len = seq_len
        self.vocab_size = len(self.index2word)
        self.intent_type = ['对象', '审阅', '页面布局', '文件', '表格', '样式']

    def encode(self, sent):
        if isinstance(sent, str):
            sent = sent.split()
        id = [self.word2index.get(s, 1) for s in sent]
        if len(id) < self.seq_len:
            id.extend([0]*(self.seq_len - len(id)))
        return id[:self.seq_len]


def model(path='./logs/pb/model.pb'):
    with tf.gfile.GFile(path, 'rb') as fgraph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fgraph.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        input_x = graph.get_tensor_by_name('input_x:0')
        keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        score = graph.get_tensor_by_name('output/scores:0')

        sess = tf.Session(graph=graph)

        return sess, input_x, keep_prob, score

def predict(sent, sess, input_x, keep_prob, score, topk=1):
    if isinstance(sent, str):
        sent = ' '.join(list(jieba.cut(sent.replace(' ', ''), cut_all=False)))
    feed_dict = {
        input_x: np.array([vocab.encode(sent)]),
        keep_prob: 1.0
    }
    prob = sess.run(score, feed_dict=feed_dict)
    prob = np.exp(prob[0]) / sum(np.exp(prob[0]))

    if topk == 1:
        idx = np.argmax(prob)
        return [(vocab.intent_type[idx], prob[idx])]
    else:
        idx = np.argsort(prob)[::-1][:topk]
        return [(vocab.intent_type[id], prob[id]) for id in idx]


if __name__ == '__main__':
    vocab = Vocab('./data/vocab')
    sess, input_x, keep_prob, score = model('./logs/pb/model.pb')

    with open('./data/intent.test', encoding='utf-8') as ftest:
        for line in ftest:
            arr = line.strip().split(',')
            sent = ','.join(arr[1:]).strip()

            result = predict(sent, sess, input_x, keep_prob, score, topk=1)
            print(result, line)
