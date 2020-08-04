import os
import time
import tensorflow as tf
from data_helper import data_loader, rewrite_train, generate_train
from lstm_ranking import Lstm_ranking
from utils import Dataloader, write_res, plot

# word_embed.txt
# w2v-all

FLAGS = {
    'embedding_file': './data/w2v-all',
    'embedding_dim': 100,
    'hidden_size': 100,
    'seq_len': 25,
    'keep_prob': 0.9,
    'l2_reg_lambda': 0.1,
    'max_grad_norm': 1.0,
    'margin': 0.2,
    'batch_size': 64,
    'num_epochs': 40,
    'device': -1,

    'allow_soft_placement': True,
    'log_device_placement': False
}

FLAGS['embedding_file'] = './data/word_embed.txt'
FLAGS['embedding_dim'] = 200
FLAGS['hidden_size'] = 200

if FLAGS['device'] >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS['device'])

config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=tf.GPUOptions(allow_growth=True))


def train(train_path, test=None, ckpt='./checkpoint/'):
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)

    data = Dataloader(FLAGS['embedding_file'], FLAGS['seq_len'], FLAGS['embedding_dim'])
    if test:
        testo, testc, _ = data.dataload(test)

    # config = tf.ConfigProto(device_count={'gpu': 0})  # only cpu variable
    # tf.Session(config)
    with tf.Session(config=config) as sess:
        model = Lstm_ranking(data.vocab.embedding, FLAGS)
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())

        best_acc = 0.7
        acc_record = []
        for epoch in range(1, FLAGS['num_epochs']):
            # data_q, data_c = data_loader(train_path)
            # rewrite_train(data_q, data_c, './logs/train_data', 5)
            generate_train(train_path, './logs/train_data', 3)
            traino, trainc, trainn = data.dataload('./logs/train_data')
            # train
            stime = time.time()
            for batch in data.batch_iter(list(zip(traino, trainc, trainn)), FLAGS['batch_size']):
                batch_o, batch_c, batch_n = zip(*batch)
                step, loss = model.train_step(sess, batch_o, batch_c, batch_n)
                if step % 100 == 0:
                    print('train, epoch: %d, step: %d, loss: %.4f, spend-time: %.4fs' %
                          (epoch, step, loss, time.time() - stime))
                    stime = time.time()

            if test:
                acc, acc_3, index, score = model.dev_step(sess, testo, testc)
                acc_record.append(acc)
                print("Evaluation   epoch: %d, top1-acc: %.4f, top3-acc: %.4f" % (epoch, acc, acc_3))
                if acc > best_acc - 0.003:
                    saver.save(sess, os.path.join(ckpt, 'lstm-model-' + str(epoch) + '-' + str(acc*100)[:5] + '.ckpt'))
                    write_res(test, index, score, './logs/result-lstm/epoch-'+str(epoch))

                    best_acc = acc
            else:
                saver.save(sess, os.path.join(ckpt, 'lstm-model-' + str(epoch) + '.ckpt'))

        return acc_record


if __name__ == '__main__':
    if not os.path.exists('./logs/model-temp/'):
        os.makedirs('./logs/model-temp/')
    if not os.path.exists('./logs/result-lstm/'):
        os.makedirs('./logs/result-lstm/')

    test_data = './data/test_data'
    train_data = './data/bm25_sim/bm25_sim.json'
    acc = train(train_data, test_data, './logs/model-temp/')

    plot(acc, 'human')

    for c in acc:
        print(str(c)[:6])
