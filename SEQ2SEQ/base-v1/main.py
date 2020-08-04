import os
import tensorflow as tf
from model import Seq2Seq
from data import DataLoader
from tqdm import tqdm
import numpy as np
from config import data_config, model_config, n_epoch

continue_train = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

params = model_config


def train():
    du = DataLoader(**data_config)
    params['src_vcb_size'] = du.vocab_size
    params['tgt_vcb_size'] = du.vocab_size
    params['batch_size'] = 256
    batch_size = params['batch_size']

    steps = int(len(du) / batch_size) + 1

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        model = Seq2Seq(params, mode='train')
        sess.run(tf.global_variables_initializer())
        if continue_train:
            model.load(sess, tf.train.latest_checkpoint('./logs/model/'))

        src, s_len, tgt, t_len, dec = du.next_batch(batch_size)
        for epoch in range(1, 3):
            costs = []
            bar = tqdm(range(steps), total=steps, desc='epoch {}, loss=0.000000'.format(epoch))
            for _ in bar:
                source, source_len, target, target_len, _ = du.next_batch(batch_size)
                max_len = np.max(target_len)
                target = target[:, 0:max_len]
                output = model.train_step(sess, source, source_len, target, target_len)
                cost = output['loss']
                costs.append(cost)
                bar.set_description('epoch {} loss={:.6f}'.format(epoch, np.mean(costs)))
            model.save(sess, os.path.join('./logs/model/', 'model_'+str(epoch)+'.ckpt'))

            output = model.train_step(sess, src, s_len, tgt, t_len, False)
            print('source : ', du.transform_indexs(src[0]))
            print('target : ', du.transform_indexs(tgt[0]))
            print('predict: ', du.transform_indexs(output['predict'][0]))
            print('')


def test():
    du = DataLoader(**data_config)
    params['src_vcb_size'] = du.vocab_size
    params['tgt_vcb_size'] = du.vocab_size
    params['batch_size'] = 1
    tf.reset_default_graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    with tf.Session(config=config) as sess:
        model = Seq2Seq(params, mode='decode')
        sess.run(tf.global_variables_initializer())
        model.load(sess, tf.train.latest_checkpoint('./logs/model/'))
        # model.load(sess, './logs/model/model_16.ckpt')

        # sent = input('you: ')
        # while (sent):
        #     result = model.get_response(sess, sent, du)
        #     print('bot: ', result)
        #
        #     sent = input('you: ')

        sents = [('天王盖地虎', '宝塔镇妖河')]
        for sent in sents:
            result = model.get_response(sess, sent[0], du)

            print('source : ', sent[0])
            print('target : ', sent[1])
            print('predict: ', result)
            print('')


if __name__ == '__main__':
    for i in range(20):
        continue_train = False if i == 0 else True
        train()
        test()
