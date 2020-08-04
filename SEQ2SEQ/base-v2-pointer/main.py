import os
import tensorflow as tf
from model import Seq2Seq
from data import DataLoader
from tqdm import tqdm
import numpy as np
from config import data_config, model_config
from beam_search import beam_search


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
params = model_config


def train():
    du = DataLoader(**data_config)
    params['src_vcb_size'] = du.vocab_size
    params['tgt_vcb_size'] = du.vocab_size
    params['vcb_size'] = du.vocab_size
    params['batch_size'] = 64
    batch_size = params['batch_size']

    steps = int(len(du) / batch_size) + 1

    tf.reset_default_graph()

    if not os.path.exists('./logs/model/'):
        os.makedirs('./logs/model/')

    with tf.Session() as sess:
        model = Seq2Seq(params, mode='train')
        sess.run(tf.global_variables_initializer())
        if continue_train:
            model.load(sess, tf.train.latest_checkpoint('./logs/model/'))

        src, s_len, src_ext, tgt, t_len, dec, osize = du.eval_batch(batch_size)
        for epoch in range(1, 3):
            costs = []
            bar = tqdm(range(steps), total=steps, desc='epoch {}, loss=0.000000'.format(epoch))
            for _ in bar:
                batch, oov_size = du.next_batch(batch_size)
                source, source_len, source_ext, target, target_len, decoder_input = batch
                output = model.train_step(sess, source, source_len, source_ext, oov_size, target, target_len, decoder_input)
                costs.append(output['loss'])
                bar.set_description('epoch {} loss={:.6f}'.format(epoch, np.mean(costs)))
            model.save(sess, os.path.join('./logs/model/', 'model_'+str(epoch)+'.ckpt'))

            output = model.train_step(sess, src, s_len, src_ext, osize, tgt, t_len, dec, False)
            print('source: ', du.transform_indexs(src_ext[0]))
            print('target: ', du.transform_indexs(tgt[0]))
            print('predict: ', du.transform_indexs(np.argmax(output['prob'], axis=-1)[0]))
            print('')


def test():
    du = DataLoader(**data_config)
    params['src_vcb_size'] = du.vocab_size
    params['tgt_vcb_size'] = du.vocab_size
    params['vcb_size'] = du.vocab_size
    params['batch_size'] = 1
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2Seq(params, mode='decode')
        sess.run(tf.global_variables_initializer())
        # model.load(sess, './logs/model/model_1.ckpt')
        model.load(sess, tf.train.latest_checkpoint('./logs/model/'))

        for source, source_len, source_ext, target, osize in du.test_data('../data/dialog.test'):
            result = beam_search(sess, model, du.vcb, source, source_len, source_ext, osize)
            print('source: ', du.transform_indexs(source_ext[0]))
            print('target: ', du.transform_indexs(target[0]))
            print('predict: ', du.transform_indexs(result))
            print('')


if __name__ == '__main__':
    for i in range(20):
        continue_train = False if i == 0 else True
        train()
        test()

    test()
