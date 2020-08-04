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

    def encode_(self, w):
        return self.word2idx.get(w, self.unk)

    def decode_(self, idx):
        return self.idx2word.get(idx, '<UNK>')

    def encode(self, w):

        return self.word2idx.get(w, self.unk)

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

        self.oov_i2w = []
        self.oov_w2i = {}

        self.data = self.load_data()

    def __len__(self):

        return len(self.data)

    def reset_oov(self):
        self.oov_i2w = []
        self.oov_w2i = {}

    def transform_sentence(self, sent):
        idx, idx_ext = [], []
        for w in sent:
            idx.append(self.vcb.encode(w))
            if w in self.vcb.word2idx:
                idx_ext.append(self.vcb.encode(w))
            else:
                if w not in self.oov_i2w:
                    self.oov_i2w.append(w)
                idx_ext_ = self.vocab_size + self.oov_i2w.index(w)
                idx_ext.append(idx_ext_)
                self.oov_w2i[w] = idx_ext_

        return idx, idx_ext

    def transform_indexs(self, indexs):
        res = []
        for index in indexs:
            if index <= 3:
                continue
            elif index >= self.vocab_size:
                res.append(self.oov_i2w[index - self.vocab_size])
            else:
                res.append(self.vcb.idx2word.get(index))

        return ''.join(res)

    def encode(self, line):
        src_tgt = line.strip().split()
        src, src_ext = self.transform_sentence(src_tgt[0])
        dec, tgt_ext = self.transform_sentence(src_tgt[-1])

        dec = [self.start] + dec
        tgt_ext = tgt_ext + [self.end]

        s_len = len(src)
        t_len = len(dec)

        if s_len < self.src_len:
            src = src + [self.pad] * (self.src_len - s_len)
            src_ext = src_ext + [self.pad] * (self.src_len - s_len)
        else:
            src, src_ext = src[:self.src_len], src_ext[:self.src_len]

        if t_len < self.tgt_len:
            dec = dec + [self.pad] * (self.tgt_len - t_len)
            tgt_ext = tgt_ext + [self.pad] * (self.tgt_len - t_len)
        else:
            dec, tgt_ext = dec[:self.tgt_len], tgt_ext[:self.tgt_len]

        return src, s_len, src_ext, tgt_ext, t_len, dec

    def load_data(self, path=None):
        if path is None:
            path = self.path

        with open(path, encoding='utf-8') as fdata:
            data = fdata.read().splitlines()

        return data

    def next_batch(self, batch_size, data_batch=None):
        self.reset_oov()
        if data_batch is None:
            data_batch = random.sample(self.data, batch_size)

        batch = []
        for line in data_batch:
            assert len(line.strip().split()) == 3

            src, s_len, src_ext, tgt, t_len, dec = self.encode(line)

            batch.append([src, s_len, src_ext, tgt, t_len, dec])
        batch = zip(*batch)
        batch = [np.array(x) for x in batch]
        oov_size = len(self.oov_i2w)

        return batch, oov_size

    def eval_batch(self, batch_size):
        batch = self.data[:batch_size]
        batch, oov_size = self.next_batch(batch_size, batch)
        src, s_len, src_ext, tgt, t_len, dec = batch
        return src, s_len, src_ext, tgt, t_len, dec, oov_size

    def test_data(self, path):
        data = self.load_data(path)
        for line in data:
            self.reset_oov()
            src, s_len, src_ext, tgt, t_len, dec = self.encode(line)
            yield np.array([src]), np.array([s_len]), np.array([src_ext]), np.array([tgt]), len(self.oov_i2w)
