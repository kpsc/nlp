import random
import numpy as np


class Vocab(object):
    def __init__(self, path):
        self.path = path
        self.word2idx, self.idx2word = self.load_vocab()

        self.size = len(self.word2idx)
        self.pad = self.word2idx.get('<PAD>')
        self.unk = self.word2idx.get('<UNK>')
        self.start = self.word2idx.get('<SOS>')
        self.end = self.word2idx.get('<EOS>')

    def load_vocab(self):
        word2idx = {}
        idx2word = {}
        with open(self.path, encoding='utf-8') as fvcb:
            for line in fvcb:
                w, idx = line.strip().split()
                word2idx[w] = int(idx)
                idx2word[int(idx)] = w

        return word2idx, idx2word

    def encode(self, sent):
        if isinstance(sent, list):
            sent = ''.join(sent)
        sent = sent.replace(' ', '')
        idx = [self.word2idx.get(c, 1) for c in sent]
        # if len(idx) < seq_len:
        #     idx.extend([0] * (seq_len - len(idx)))

        return idx

    def decode(self, idx):
        # res = [self.idx2word.get(i, '<UNK>') for i in idx]

        res = []
        for index in idx:
            if index <= 3:
                continue
            res.append(self.idx2word.get(index, '<UNK>'))
        return ''.join(res)


class DataLoader(object):
    def __init__(self, path, vocab, src_len, tgt_len):
        self.path = path
        self.vcb = Vocab(vocab)
        self.src_len = src_len
        self.tgt_len = tgt_len

        self.vocab_size = self.vcb.size
        self.pad = self.vcb.word2idx.get('<PAD>')
        self.unk = self.vcb.word2idx.get('<UNK>')
        self.start = self.vcb.word2idx.get('<SOS>')
        self.end = self.vcb.word2idx.get('<EOS>')

        self.data = self.load_data()

    def __len__(self):

        return len(self.data)

    def transform_sentence(self, sentence):

        return self.vcb.encode(sentence)

    def transform_indexs(self, indexs):

        return self.vcb.decode(indexs)

    def load_data(self, path=None):
        if path is None:
            path = self.path

        data = []
        with open(path, encoding='utf-8') as fdata:
            for line in fdata:
                assert len(line.strip().split()) == 3

                src_tgt = line.strip().split()
                src = self.transform_sentence(src_tgt[0])[:self.src_len]
                tgt = self.transform_sentence(src_tgt[-1])[:self.tgt_len - 1]

                dec = [self.start] + tgt
                tgt = tgt + [self.end]

                s_len = len(src)
                t_len = len(tgt)

                if s_len < self.src_len:
                    src = src + [self.pad] * (self.src_len - s_len)
                if t_len < self.tgt_len:
                    tgt = tgt + [self.pad] * (self.tgt_len - t_len)
                    dec = dec + [self.pad] * (self.tgt_len - t_len)

                data.append([src, s_len, tgt, t_len, dec])

        return data

    def next_batch(self, batch_size):
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for (src, s_len, tgt, t_len, dec) in data_batch:
            batch.append((src, s_len, tgt, t_len, dec))
        batch = zip(*batch)
        batch = [np.array(x) for x in batch]

        return batch

    def eval_batch(self, batch_size):
        batch = self.data[:batch_size]
        batch = zip(*batch)
        batch = [np.array(x) for x in batch]

        return batch

    def test_data(self, path):
        data = self.load_data(path)
        batch = []
        for (src, s_len, tgt, t_len, dec) in data:
            batch.append([np.array([src]),
                          np.array([s_len]),
                          np.array([tgt]),
                          np.array([t_len]),
                          np.array([dec])])

        return batch
