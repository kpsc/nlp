import numpy as np
from tqdm import tqdm
import json
import jieba
from gensim.models import word2vec
import matplotlib.pyplot as plt


class Vocab(object):
    def __init__(self, embeddings, seq_len=50, dim=100):
        self.embeddings = embeddings
        self.seq_len = seq_len
        self.embedding_size = dim
        self.index2word, self.word2index, embedding = self.load_embedding()
        self.embedding = np.array(embedding)
        self.vocab_size = len(self.index2word)

    def load_embedding(self):
        index2word = {0: '<pad>', 1: '<unk>'}
        word2index = {'<pad>': 0, '<unk>': 1}
        embedding = [np.random.rand(self.embedding_size), np.random.rand(self.embedding_size)]
        with open(self.embeddings, 'r', encoding='utf-8') as femb:
            index = len(index2word.keys())
            for line in tqdm(femb):
                arr = line.strip().split()
                try:
                    assert len(arr) == self.embedding_size + 1
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


class Dataloader():
    def __init__(self, embedding_file, seq_len, dim=100):
        self.vocab = Vocab(embedding_file, seq_len, dim)

    def dataload(self, path):
        sentences_o = []
        sentences_c = []
        sentences_n = []
        with open(path, encoding='utf-8') as fdata:
            for line in fdata:
                arr = line.strip().split('\t')
                sentences_o.append(self.vocab.encode(arr[0]))
                sentences_c.append(self.vocab.encode(arr[1]))
                sentences_n.append(self.vocab.encode(arr[-1]))
        return np.array(sentences_o), np.array(sentences_c), np.array(sentences_n)

    def batch_iter(self, data, batch_size, shuffle=True):
        data = np.array(data)
        data_len = len(data)
        num_batches = int((len(data) - 1) / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_len))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_len)
            yield shuffle_data[start_index:end_index]


def write_res(testdata, index, score, result='./logs/result-lstm'):
    sentences_o = []
    sentences_c = []
    with open(testdata, encoding='utf-8') as fdata:
        for line in fdata:
            arr = line.strip().split('\t')
            sentences_o.append(arr[0])
            sentences_c.append(arr[1])

    with open(result, 'w', encoding='utf-8') as fres, \
         open(result + '-mistake', 'w', encoding='utf-8') as fm:
        for i, (idx, s) in enumerate(zip(index, score)):
            p = 1 if idx[0] == i else 0

            fres.write(sentences_c[i] + '\n')
            fres.write('ori: ' + sentences_o[i] + '\n')
            fres.write(str(p) + ' pre: ' + str(1.0 - s[0])[:5] + '\t' + sentences_o[idx[0]] + '\n\n')

            if idx[0] != i:
                fm.write(sentences_c[i] + '\n')
                fm.write('ori: ' + sentences_o[i] + '\n')
                fm.write('pre: ' + str(1.0 - s[0])[:5] + '\t' + sentences_o[idx[0]] + '\n\n')


def w2v_train(path='./data/answer_dict.json', result='./data/w2v-all'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    corpus = []
    for k in tqdm(list(data.keys())):
        for q in data[k]:
            t = list(jieba.cut(q.replace(' ', '')))
            corpus.append(t)
    model = word2vec.Word2Vec(corpus, workers=16, size=100, min_count=2,
                              window=5, sg=1, sample=1e-3)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(result, binary=False)


def get_acc(path):
    acc = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if 'top1-acc' in line:
                ac = float(line.strip().split()[-3][:-1])
                acc.append(ac)
                print(ac)

    return acc


def plot(data, title='acc'):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # w2v_train()

    acc = get_acc('./logs/log')
    plot(acc)
