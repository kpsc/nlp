import numpy as np
from jieba import analyse
from tqdm import tqdm
from gensim.models import word2vec
from gensim.summarization import bm25
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = analyse.extract_tags
textrank = analyse.textrank


def word2vec_train(corpus, vectorname='word2vec'):
    if isinstance(corpus, str):
        corpus = [corpus]
    sentences = []

    for data in corpus:
        with open(data, 'r', encoding='utf-8') as fori:
            for line in tqdm(fori):
                if any('\u4e00' <= c <= '\u9fff' for c in line):
                    sentences.append(line.strip().split())
            print('load %s finished!' % data)

    model = word2vec.Word2Vec(sentences, workers=16, size=100, min_count=1,
                              window=5, sg=1, sample=1e-3)

    model.init_sims(replace=True)
    model.wv.save_word2vec_format(vectorname, binary=False)

    
def weight_key(sent, mode='tfidf'):
    '''
    sent: '** ** * ***'
    '''
    weightmode = tfidf if mode == 'tfidf' else textrank
    weight = dict(weightmode(sentori, withWeight=True))
    if not weight:
        return np.array([1.0 for c in sent]) / len(sent)
    w = np.array([weight.get(c, -1e6) for c in sent])
    w_exp = np.exp(w)
    w_s = w_exp / np.sum(w_exp)
    return w_s.reshape(1, -1)


def bm25_sim(corpus, sent, topk=5):
    '''
    corpus:
        ['*****', '******', '******', ...]
    sent: ['**', '*', '***', '**', ...]
    '''
    model = bm25.BM25(corpus)
    scores = model.get_scores(sent)
    scores = sorted(list(enumerate(scores)), key=lambda k: k[1], reverse=True)[:topk]
    
    index = [idx[0] for idx in scores]
    

def tfidf_vector(data, stop_words, max_features=10000):
    '''
    data:
        ['*****', '******', '******', ...]
    stop_words:
        u'， 的 。 · ？ 了！ _ - ：'
    '''
    model = TfidfVectorizer(stop_words=stop_words.split(), max_features=max_features).fit(data)
    vocab = model.vocabulary_
    
    sparse_vector = model.transform(data)
    dense_vector = sparse_vector.todense()
    