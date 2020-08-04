# encoding: utf-8

"""
    数据处理单元
    处理原始语料数据
    生成批训练数据
"""


import re
import os
import pickle
import json
import collections
import itertools
import random
import numpy as np
from config import data_config


class Vocab(object):
    def __init__(self, path):
        self.path = path
        self.word2idx, self.idx2word = self.load_vocab()

        self.size = len(self.word2idx)

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


class DataUnit(object):

    # 特殊标签
    PAD = '<PAD>'
    UNK = '<UNK>'
    START = '<SOS>'
    END = '<EOS>'

    # 特殊标签的索引
    START_INDEX = 0
    END_INDEX = 1
    UNK_INDEX = 2
    PAD_INDEX = 3

    def __init__(self, path, processed_path,
                 min_q_len, max_q_len,
                 min_a_len, max_a_len,
                 word2index_path):
        """
            初始化函数，参数意义可查看CONFIG.py文件中的注释
        :param
        """
        self.path = path
        self.processed_path = processed_path
        self.word2index_path = word2index_path
        self.min_q_len = min_q_len
        self.max_q_len = max_q_len
        self.min_a_len = min_a_len
        self.max_a_len = max_a_len
        self.vocab_size = 0
        self.index2word = {}
        self.word2index = {}
        self.data = self.load_data()
        self._fit_data_()

    def next_batch(self, batch_size):
        """
        生成一批训练数据
        :param batch_size: 每一批数据的样本数
        :return: 经过了填充处理的QA对
        """
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for qa in data_batch:
            encoded_q = self.transform_sentence(qa[0])[:self.max_q_len]
            encoded_a = self.transform_sentence(qa[1])[:self.max_a_len]
            q_len = len(encoded_q)

            # 填充句子
            encoded_q = encoded_q + \
                [self.func_word2index(self.PAD)] * (self.max_q_len - q_len)
            encoded_a = encoded_a + [self.func_word2index(self.END)]
            encoded_a = encoded_a[:self.max_a_len]
            a_len = len(encoded_a)
            encoded_a = encoded_a + \
                [self.func_word2index(self.PAD)] * (self.max_a_len - a_len)

            batch.append((encoded_q, q_len, encoded_a, a_len))
        batch = zip(*batch)
        batch = [np.asarray(x) for x in batch]
        return batch

    def transform_sentence(self, sentence):
        """
        将句子转化为索引
        :param sentence:
        :return:
        """
        res = []
        for word in sentence:
            res.append(self.func_word2index(word))
        return res

    def transform_indexs(self, indexs):
        """
        将索引转化为句子,同时去除填充的标签
        :param indexs:索引序列
        :return:
        """
        res = []
        for index in indexs:
            if (index == self.START_INDEX or index == self.PAD_INDEX
                    or index == self.END_INDEX or index == self.UNK_INDEX):
                continue
            res.append(self.func_index2word(index))
        return ''.join(res)

    def _fit_data_(self):
        """
        得到处理后语料库的所有词，并将其编码为索引值
        :return:
        """
        if not os.path.exists(self.word2index_path):
            vocabularies = [x[0] + x[1] for x in self.data]
            self._fit_word_(itertools.chain(*vocabularies))
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)
        else:
            with open(self.word2index_path, 'rb') as fr:
                self.word2index = pickle.load(fr)
            self.index2word = dict([(v, k)
                                    for k, v in self.word2index.items()])
        self.vocab_size = len(self.word2index)

    def load_data(self):
        """
        获取处理后的语料库
        :return:
        """
        if not os.path.exists(self.processed_path):
            data = self._extract_txt()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        else:
            with open(self.processed_path, 'rb') as fr:
                data = pickle.load(fr)
        # 根据CONFIG文件中配置的最大值和最小值问答对长度来进行数据过滤
        data = [
            x for x in data if self.min_q_len <= len(
                x[0]) <= self.max_a_len and self.min_a_len <= len(
                x[1]) <= self.max_a_len]
        return data

    def func_word2index(self, word):
        """
        将词转化为索引
        :param word:
        :return:
        """
        return self.word2index.get(word, self.word2index[self.UNK])

    def func_index2word(self, index):
        """
        将索引转化为词
        :param index:
        :return:
        """
        return self.index2word.get(index, self.UNK)

    def _fit_word_(self, vocabularies):
        """
        将词表中所有的词转化为索引，过滤掉出现次数少于4次的词
        :param vocabularies:词表
        :return:
        """
        vocab_counter = collections.Counter(vocabularies)
        index2word = ([self.START] + [self.END] + [self.UNK] + [self.PAD] +
                      [x[0] for x in vocab_counter if vocab_counter.get(x[0]) > 4])
        self.word2index = dict([(w, i) for i, w in enumerate(index2word)])
        self.index2word = dict([(i, w) for i, w in enumerate(index2word)])

    def _regular_(self, sen):
        """
        句子规范化，主要是对原始语料的句子进行一些标点符号的统一
        :param sen:
        :return:
        """
        sen = sen.replace('/', '')
        sen = re.sub(r'…{1,100}', '…', sen)
        sen = re.sub(r'\.{3,100}', '…', sen)
        sen = re.sub(r'···{2,100}', '…', sen)
        sen = re.sub(r',{1,100}', '，', sen)
        sen = re.sub(r'\.{1,100}', '。', sen)
        sen = re.sub(r'。{1,100}', '。', sen)
        sen = re.sub(r'\?{1,100}', '？', sen)
        sen = re.sub(r'？{1,100}', '？', sen)
        sen = re.sub(r'!{1,100}', '！', sen)
        sen = re.sub(r'！{1,100}', '！', sen)
        sen = re.sub(r'~{1,100}', '～', sen)
        sen = re.sub(r'～{1,100}', '～', sen)
        sen = re.sub(r'[“”]{1,100}', '"', sen)
        sen = re.sub(r'[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
        sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)

        return sen

    def _good_line_(self, line):
        """
        判断一句话是否是好的语料,即判断
        :param line:
        :return:
        """
        if len(line) == 0:
            return False
        ch_count = 0
        for c in line:
            # 中文字符范围
            if '\u4e00' <= c <= '\u9fff':
                ch_count += 1
        if ch_count / float(len(line)) >= 0.5 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 and len(
                re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3 and line.find("鸡") == -1:
            return True
        return False

    def _extract_data(self):
        res = []
        q = None
        with open(self.path, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line.startswith('M '):
                    if q is None:
                        q = self._regular_(line[2:-1])
                    else:
                        a = self._regular_(line[2:-1])
                        if self._good_line_(q) and self._good_line_(a):
                            res.append((q, a))
                        q = None
        return res

    def _extract_txt(self):
        res = []
        q = None
        with open(self.path, 'r', encoding='utf-8') as fr:
            for line in fr:
                qa = line.split('\t')
                if len(qa) < 2:
                    continue
                q = self._regular_(qa[0])
                a = self._regular_(qa[1])
                if self._good_line_(q) and self._good_line_(a):
                    res.append((q, a))
        return res

    def __len__(self):
        """
        返回处理后的语料库中问答对的数量
        :return:
        """
        return len(self.data)

    def normalize_txt(self):
        # normalize original txt data
        if not os.path.exists(self.processed_path):
            data = self._extract_txt()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        # normalize vocabulary
        if not os.path.exists(self.word2index_path):
            vocabularies = [x[0] + x[1] for x in data]
            self._fit_word_(itertools.chain(*vocabularies))
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)


if __name__ == '__main__':
    data_unit = DataUnit(
        path=data_config['path'], processed_path=data_config['processed_path'],
        min_q_len=data_config['min_q_len'], max_q_len=data_config['max_q_len'],
        min_a_len=data_config['min_a_len'], max_a_len=data_config['max_a_len'],
        word2index_path=data_config['word2index_path'])
    data_unit.normalize_txt()
