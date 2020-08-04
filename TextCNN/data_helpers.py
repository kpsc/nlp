import numpy as np
from tqdm import tqdm

class Vocab(object):
    def __init__(self, embeddings, seq_len=20):
        self.embeddings = embeddings
        self.seq_len = seq_len
        self.index2word, self.word2index, embedding = self.load_embedding()
        self.embedding = np.array(embedding)
        self.embedding_size = 100
        self.vocab_size = len(self.index2word)
        self.intent_type = ['对象', '审阅', '页面布局', '文件', '表格', '样式']
        self.intents = self.intent()

    def load_embedding(self):
        index2word = {0: '<pad>', 1: '<unk>'}
        word2index = {'<pad>': 0, '<unk>': 1}
        embedding = [np.random.rand(100), np.random.rand(100)]
        with open(self.embeddings, 'r', encoding='utf-8') as femb:
            index = len(index2word.keys())
            for line in tqdm(femb):
                arr = line.strip().split()
                try:
                    assert len(arr) == 101
                    vector = np.array(list(map(float, arr[1:])))
                    index2word[index] = arr[0]
                    word2index[arr[0]] = index
                    embedding.append(vector)
                    index += 1
                except:
                    continue
        return index2word, word2index, embedding

    def encode(self, sent):
        if isinstance(sent, str):
            sent = sent.split()
        id = [self.word2index.get(s, 1) for s in sent]
        if len(id) < self.seq_len:
            id.extend([0]*(self.seq_len - len(id)))
        return id[:self.seq_len]

    def intent(self):
        intents = {}
        for i, s in enumerate(self.intent_type):
            t = [0, 0, 0, 0, 0, 0]
            t[i] = 1
            intents[s] = t
        return intents

def dataloader(vocab, path='./data/train'):
    sentences = []
    intents = []
    with open(path, encoding='utf-8') as fdata:
        for line in fdata:
            arr = line.strip().split(',')
            sent = ','.join(arr[1:]).strip()
            sentences.append(vocab.encode(sent))
            intent = arr[0].split('_')[-1]
            intents.append(vocab.intents[intent.strip()])
    return sentences, intents

def data_process(FLAGS):
    vocab = Vocab(FLAGS['embedding_file'])
    trainx, trainy = dataloader(vocab, FLAGS['train_data'])
    testx, testy = dataloader(vocab, FLAGS['test_data'])
    return np.array(trainx), np.array(trainy), np.array(testx), np.array(testy), vocab



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def test(sess, cnn, testdata, embedding_file, res='result-test'):
    vocab = Vocab(embedding_file)

    with open(testdata, encoding='utf-8') as fdata, \
        open(res, 'w', encoding='utf-8') as fres:
        for line in fdata:
            arr = line.strip().split(',')
            sent = ','.join(arr[1:]).strip()
            intent = arr[0].split('_')[-1]

            feed_dict = {
                cnn.input_x: np.array([vocab.encode(sent)]),
                cnn.input_y: np.array([vocab.intents[intent.strip()]]),
                cnn.dropout_keep_prob: 1.0
            }

            predictions = sess.run(cnn.predictions, feed_dict)
            pre_intent = vocab.intent_type[predictions[0]]
            fres.write(pre_intent + '\t' + line)


if __name__ == '__main__':
    vcb = Vocab('./data/word2vec-use')
    index2word = vcb.index2word
    word2index = vcb.word2index

    with open('./data/vocab', 'w', encoding='utf-8') as fvcb:
        for i in range(len(index2word.keys())):
            fvcb.write(index2word[i] + '\t' + str(i) + '\n')
