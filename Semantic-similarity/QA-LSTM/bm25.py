import time
import json
import numpy as np
from tqdm import tqdm
from gensim.summarization import bm25


# 基于 bm25 算法从无监督数据中构建相似语料
# 后续可以用 tfidf 及语言模型等对这些数据再过滤一遍
def bm25_sim(corpus, result, topk=5):
    data = []
    with open(corpus, encoding='utf-8') as fcor:
        for line in tqdm(fcor):
            line = json.loads(line.strip())
            data.append(line['query'])

    with open(result, 'w', encoding='utf-8') as fres:
        stime = time.time()
        step = 1
        while len(data) > 5000:
            query = data.pop(0)
            model = bm25.BM25(data)
            scores = model.get_scores(query)
            scores = sorted(list(enumerate(scores)), key=lambda k: k[1], reverse=True)

            index = [scores[0][0]]
            index_remove = [scores[0][0]]
            temp = ''.join(query)
            for i in range(1, 100):
                idx, score = scores[i]
                if abs(score - scores[i-1][1]) > 1e-4 and temp != ''.join(data[idx]):
                    index.append(idx)
                index_remove.append(idx)

                if len(index) >= topk or score < 15.0:
                    break
            ids = len(index_remove)
            for i in range(ids, 100):
                if scores[i][1] >= scores[ids-1][1] - 1.0:
                    index_remove.append(scores[i][0])

            if scores[0][1] > 20.0 and len(index) >= 3:
                output = {}
                output['query'] = ' '.join(query)
                output['candidates'] = [' '.join(data[i]) for i in index]
                fres.write(json.dumps(output, ensure_ascii=False) + '\n')

            index_remove = sorted(index_remove, reverse=True)
            for idx in index_remove:
                data.pop(idx)

            if step % 100 == 0:
                fres.flush()
                print('step: %d, spend-time: %.4f' % (step, time.time() - stime))
                stime = time.time()
            step += 1


if __name__ == '__main__':
    bm25_sim('./data/all_qa_word_segment.json', './data/bm25_sim.json', topk=10)
