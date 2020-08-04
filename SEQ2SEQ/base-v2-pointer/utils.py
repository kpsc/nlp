import tensorflow as tf


def reduce_states(fb_st, init):
    """a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = fb_st.c.get_shape()[-1] // 2
    with tf.variable_scope('reduce_final_st'):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=init)
        w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
                                     initializer=init)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                        initializer=init)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                        initializer=init)

        new_c = tf.nn.relu(tf.matmul(fb_st.c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul(fb_st.h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """
  Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term


def sequence_loss(logits, labels, mask):
    batch_size = tf.shape(mask)[0]

    loss_per_step = []
    batch_nums = tf.range(0, limit=batch_size)
    for step, dist in enumerate(logits):
        targets = labels[:, step]
        indices = tf.stack((batch_nums, targets), axis=1)
        p = tf.gather_nd(dist, indices)
        loss_ = -tf.log(p) * mask[:, step]  # (batch_size, )
        loss_per_step.append(loss_)

    loss = sum(loss_per_step) / tf.reduce_sum(mask, axis=1)
    loss = tf.reduce_mean(loss)

    return loss
