import tensorflow as tf
import pdb


def main():
    ## !!! change this to test the different behaviors !!!
    # optimizer = tf.train.GradientDescentOptimizer(1e-3)                 # This one is working
    optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.9, beta2=0.999999)  # This one is not working
    # optimizer = tf.train.AdagradOptimizer(1e-3)                         # This one is not working
    # optimizer = tf.train.AdadeltaOptimizer(1e-3)                        # This one is not working

    list_grads = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(2):
            with tf.device('/cpu:0'):
                with tf.name_scope('%d' % i) as scope:
                    W = tf.get_variable(name="filter", initializer=tf.random_uniform_initializer(dtype=tf.float32),
                                        shape=[5, 1])
                    X = tf.get_variable(name="data", initializer=tf.random_uniform_initializer(dtype=tf.float32),
                                        shape=[5, 1])
                    Y_ = tf.get_variable(name="out", initializer=tf.random_uniform_initializer(dtype=tf.float32),
                                         shape=[5, 1])
                    Y = W + X
                    loss = tf.reduce_mean(Y - Y_)
                    grad = optimizer.compute_gradients(loss)
                    list_grads.append(grad)

                    tf.get_variable_scope().reuse_variables()

    grads = list_grads[0] + list_grads[1]
    # pdb.set_trace()

    op_train = optimizer.apply_gradients(grads)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    sess = tf.Session()
    sess.run([init_global, init_local])

    _, sol = sess.run([op_train, loss])
    print(str(sol))


if __name__ == '__main__':
    main()
