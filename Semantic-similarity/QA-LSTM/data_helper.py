import os
import random
import json
import jieba
from tqdm import tqdm
from itertools import combinations

neg_num = 10


def data_loader(path):
    data_q = []
    data_c = []
    with open(path, 'r', encoding='utf-8') as fdata:
        for line in tqdm(fdata):
            arr = json.loads(line, encoding='utf-8')
            data_q.append(arr['query'])
            data_c.append(arr['candidates'][0])
    return data_q, data_c


def rewrite_train(data_q, data_c, train_path, neg_num=2):
    with open(train_path, 'w', encoding='utf-8') as ftrain:
        for i, dq in enumerate(data_q):
            s = dq.strip().replace('\t', ' ') + '\t' + data_c[i].strip().replace('\t', ' ')
            t = list(range(len(data_q)))
            t.remove(i)
            neg_idx = random.sample(t, neg_num)
            for j in neg_idx:
                ftrain.write(s + '\t' + data_c[j].strip().replace('\t', ' ') + '\n')


def rewrite_test(data_q, data_c, test_path):
    with open(test_path, 'w', encoding='utf-8') as ftest:
        for i, dq in enumerate(data_q):
            s = dq.strip().replace('\t', ' ') + '\t' + data_c[i].strip().replace('\t', ' ') + '\t--'
            ftest.write(s + '\n')


def split_data(path='./data/answer_dict.json', result='./data/answer_dict_split.json'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    data_new = {}
    for k in tqdm(list(data.keys())):
        query = []
        for q in data[k]:
            query.append(' '.join(list(jieba.cut(q.replace(' ', '')))))
        data_new[k] = query

    with open(result, 'w', encoding='utf-8') as fres:
        json.dump(data_new, fres, ensure_ascii=False)


def generate_train(path='./data/bm25_sim/bm25_sim.json', result='./data/bm25_sim/train_data', neg_num=5):
    data = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            arr = json.loads(line.strip(), encoding='utf-8')
            data[arr['query']] = arr['candidates']

    with open(result, 'w', encoding='utf-8') as fres:
        keys = list(data.keys())
        for i, k in enumerate(tqdm(keys)):
            querys = [k] + data[k][:1]  # 很奇怪，这里使用多点数据的时候效果反而不好
            # if len(querys) < 2:
            #     continue

            c_n_m = list(combinations(list(range(len(querys))), 2))
            n = min(5, len(c_n_m))
            index = random.sample(c_n_m, n)
            for idx in index:
                # idx = random.sample(c_n_m, 1)[0]
                q = [querys[idx[0]], querys[idx[1]]]
                s = q[0].strip().replace('\t', ' ') + '\t' + q[1].strip().replace('\t', ' ')

                t = list(range(len(keys)))
                t.remove(i)
                neg = random.sample(t, neg_num)
                for j in neg:
                    n = random.sample(data[keys[j]], 1)[0]
                    fres.write(s + '\t' + n.strip().replace('\t', ' ') + '\n')
            fres.flush()


if __name__ == '__main__':
    data_a, data_b = data_loader('./data/sim_data/test_new.json')

    # rewrite_train(data_a, data_b, './data/train_data', 4)
    rewrite_test(data_a, data_b, './data/sim_data/test_data')

    # split_data()
    # generate_train()
