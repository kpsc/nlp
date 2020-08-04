import tensorflow as tf
import time
import data_helpers
from cnn import TextCNN
import matplotlib.pyplot as plt


FLAGS = {
    'train_data': './data/intent.train',
    'test_data': './data/intent.test',
    'num_classes': 6,

    'embedding_file': './data/word2vec-use',
    'embedding_dim': 100,
    'seq_len': 20,
    'filter_sizes': [2, 3, 4, 5],
    'num_filters': 128,
    'keep_prob': 0.9,
    'l2_reg_lambda': 0.1,
    'batch_size': 64,
    'num_epochs': 40,

    'allow_soft_placement': True,
    'log_device_placement': False
}


if __name__ == '__main__':
    trainx, trainy, testx, testy, vocab = data_helpers.data_process(FLAGS)

    conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(allow_growth=True))

    with tf.Graph().as_default() as graph, tf.Session(conf) as sess:
        model = TextCNN(vocab.embedding, FLAGS)
        saver = tf.train.Saver(max_to_keep=20)
        sess.run(tf.global_variables_initializer())

        stime = time.time()
        eval_acc = []
        for epoch in range(FLAGS['num_epochs']):
            # train
            for batch in data_helpers.batch_iter(list(zip(trainx, trainy)), FLAGS['batch_size'], 1):
                x_batch, y_batch = zip(*batch)
                step, loss, accuracy = model.train_step(sess, x_batch, y_batch)
                if (step + 1) % 200 == 0:
                    print("train    epoch: %d, spend-time: %.4f, step: %d, loss: %.4f, acc: %.4f" %
                          (epoch, time.time() - stime, step, loss, accuracy))
                    stime = time.time()

            # eval
            step, loss, accuracy = model.dev_step(sess, testx, testy)
            eval_acc.append(accuracy)
            print("Evaluation   epoch: %d, loss: %.4f, acc: %.4f" % (epoch, loss, accuracy))

            if accuracy > max(eval_acc) - 0.3:
                saver.save(sess, './logs/model/model_' + str(epoch) + '_' + str(accuracy*100)[:5] + '.ckpt')

        # saver.restore(sess, './logs/model/model_37_97.12.ckpt')
        # data_helpers.test(sess, model, FLAGS['test_data'], FLAGS['embedding_file'], './logs/res-test')

        plt.plot(eval_acc)
        plt.ylim([0.8, 0.98])
        plt.grid()
        plt.show()
