import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops

__default_gain__ = 1.0
__data_format__ = "NCHW"
__enable_bias__ = True

__init_weight_stddev__ = 1.0
__init_distribution_type__ = 'uniform'

__enable_wn__ = False

__enable_sn__ = False
__enable_snk__ = False

SPECTRAL_NORM_K_LIST = []
SPECTRAL_NORM_UV_LIST = []
SPECTRAL_NORM_SIGMA_LIST = []

SPECTRAL_NORM_K_INIT_OPS_LIST = []
SPECTRAL_NORM_UV_UPDATE_OPS_LIST = []


def set_enable_bias(boolvalue):
    global __enable_bias__
    __enable_bias__ = boolvalue


def set_enable_wn(boolvalue):
    global __enable_wn__
    __enable_wn__ = boolvalue


def set_enable_sn(value):
    global __enable_sn__
    __enable_sn__ = value


def set_enable_snk(value):
    global __enable_snk__
    __enable_snk__ = value


def set_default_gain(stddev):
    global __default_gain__
    __default_gain__ = stddev


def set_init_type(type_str):
    global __init_distribution_type__
    __init_distribution_type__ = type_str


def set_init_weight_stddev(stddev):
    global __init_weight_stddev__
    __init_weight_stddev__ = stddev


def set_data_format(data_format):
    global __data_format__
    __data_format__ = data_format


def identity(inputs):
    return inputs


def deconv2d(input, output_dim, ksize=3, stride=1, padding='SAME', name='deconv2d',
             enable_bias=None, enable_wn=None, enable_sn=None, data_format=None,
             initializer=None, init_weight_stddev=None, init_distribution_type=None, gain=None):

    enable_wn = __enable_wn__ if enable_wn is None else enable_wn
    enable_sn = __enable_sn__ if enable_sn is None else enable_sn
    enable_bias = __enable_bias__ if enable_bias is None else enable_bias
    data_format = __data_format__ if data_format is None else data_format

    gain = __default_gain__ if gain is None else gain
    init_weight_stddev = __init_weight_stddev__ if init_weight_stddev is None else init_weight_stddev
    init_distribution_type = __init_distribution_type__ if init_distribution_type is None else init_distribution_type

    def get_deconv_dim(spatial_size, stride_size, kernel_size, padding):
        spatial_size *= stride_size
        if padding == 'VALID':
            spatial_size += max(kernel_size - stride_size, 0)
        return spatial_size

    input_shape = input.get_shape().as_list()
    h_axis, w_axis, c_axis = [1, 2, 3] if data_format == "NHWC" else [2, 3, 1]
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]

    output_shape = list(input_shape)
    output_shape[h_axis] = get_deconv_dim(input_shape[h_axis], stride, ksize, padding)
    output_shape[w_axis] = get_deconv_dim(input_shape[w_axis], stride, ksize, padding)
    output_shape[c_axis] = output_dim

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if enable_wn:
            w = tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(w, [0, 1, 3], keepdims=True)) * stride * gain)
            w = g * tf.nn.l2_normalize(w, [0, 1, 3])
        else:
            if initializer is None:
                scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / init_weight_stddev * stride
                w = scale * tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev))
            else:
                w = tf.get_variable('w', [ksize, ksize, output_dim, input_shape[c_axis]], initializer=initializer)

        if enable_sn:
            w = spectral_normed_weight(w)[0]

        x = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding=padding, data_format=data_format)

        if enable_bias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b, data_format=data_format)

    return x


def conv2d(input, output_dim, ksize=3, stride=1, padding='SAME', name='conv2d',
           enable_bias=None, enable_wn=None, enable_sn=None, data_format=None,
           initializer=None, init_weight_stddev=None, init_distribution_type=None, gain=None):

    enable_wn = __enable_wn__ if enable_wn is None else enable_wn
    enable_sn = __enable_sn__ if enable_sn is None else enable_sn
    enable_bias = __enable_bias__ if enable_bias is None else enable_bias
    data_format = __data_format__ if data_format is None else data_format

    gain = __default_gain__ if gain is None else gain
    init_weight_stddev = __init_weight_stddev__ if init_weight_stddev is None else init_weight_stddev
    init_distribution_type = __init_distribution_type__ if init_distribution_type is None else init_distribution_type

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()
        h_axis, w_axis, c_axis = [1, 2, 3] if data_format == "NHWC" else [2, 3, 1]
        strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]

        if enable_wn:
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(w, [0, 1, 2], keepdims=True)) * gain)
            w = g * tf.nn.l2_normalize(w, [0, 1, 2])
        else:
            if initializer is None:
                scale = gain / np.sqrt(float(ksize * ksize * input_shape[c_axis])) / init_weight_stddev
                w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev)) * scale
            else:
                w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer)

        if enable_sn:
            w = spectral_normed_weight(w)[0]

        x = tf.nn.conv2d(input, w, strides=strides, padding=padding, data_format=data_format)

        if enable_bias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b, data_format=data_format)

    return x


def linear(input, output_dim, name='linear',
           enable_bias=None, enable_wn=None, enable_sn=None,
           initializer=None, init_weight_stddev=None, init_distribution_type=None, gain=None):

    enable_wn = __enable_wn__ if enable_wn is None else enable_wn
    enable_sn = __enable_sn__ if enable_sn is None else enable_sn
    enable_bias = __enable_bias__ if enable_bias is None else enable_bias

    gain = __default_gain__ if gain is None else gain
    init_weight_stddev = __init_weight_stddev__ if init_weight_stddev is None else init_weight_stddev
    init_distribution_type = __init_distribution_type__ if init_distribution_type is None else init_distribution_type

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 2

        if enable_wn:
            w = tf.get_variable('w', [input_shape[1], output_dim], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(tf.square(w), [0], keepdims=True)) * gain)
            w = g * tf.nn.l2_normalize(w, [0])
        else:
            if initializer is None:
                scale = gain / np.sqrt(float(input_shape[1])) / init_weight_stddev
                w = tf.get_variable('w', [input_shape[1], output_dim], initializer=simple_stddev_initializer(init_distribution_type, init_weight_stddev)) * scale
            else:
                w = tf.get_variable('w', [input_shape[1], output_dim], initializer=initializer)

        if enable_sn:
            w = spectral_normed_weight(w)[0]

        x = tf.matmul(input, w)

        if enable_bias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b)

    return x


def generalized_normalization(input, reduce_axis, parm_axis, enable_offset=True, enable_scale=True, epsilon=0.001, name='norm'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()
        params_shape = [1] * len(input_shape)
        for i in parm_axis:
            params_shape[i] = input_shape[i]

        offset, scale = None, None
        if enable_scale:
            scale = tf.get_variable('scale', shape=params_shape, initializer=tf.ones_initializer())
        if enable_offset:
            offset = tf.get_variable('offset', shape=params_shape, initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(input, reduce_axis, keep_dims=True)

        outputs = tf.nn.batch_normalization(input, mean, variance, offset, scale, epsilon)

    return outputs


def conditional_generalized_normalization(input, reduce_axis, parm_axis, labels, n_labels, enable_offset=True, enable_scale=True, epsilon=0.001, name='cond_bn'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()
        params_shape = [1] * len(input_shape)
        for i in parm_axis:
            params_shape[i] = input_shape[i]

        offset, scale = None, None
        if enable_scale:
            scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, params_shape], dtype='float32'))
            scale = tf.nn.embedding_lookup(scale_m, labels, name='embedding_scale')
        if enable_offset:
            offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, params_shape], dtype='float32'))
            offset = tf.nn.embedding_lookup(offset_m, labels, name='embedding_offset')

        mean, var = tf.nn.moments(input, reduce_axis, keep_dims=True)
        result = tf.nn.batch_normalization(input, mean, var, offset, scale, epsilon)

    return result


def avgpool(input, ksize, stride, name='avgpool', scale_up=False, data_format=None):

    data_format = __data_format__ if data_format is None else data_format

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        kernel = [1, ksize, ksize, 1] if data_format == "NHWC" else [1, 1, ksize, ksize]
        strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]

        input = tf.nn.avg_pool(input, ksize=kernel, strides=strides, padding='VALID', name=name, data_format=data_format)

        if scale_up:
            input *= ksize

    return input


def maxpool(input, ksize, stride, name='maxpool', data_format=None):

    data_format = __data_format__ if data_format is None else data_format

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        kernel = [1, ksize, ksize, 1] if data_format == "NHWC" else [1, 1, ksize, ksize]
        strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]

        input = tf.nn.max_pool(input, ksize=kernel, strides=strides, padding='VALID', name=name, data_format=data_format)

    return input


def image_nn_double_size(input, name='resize', data_format=None):

    data_format = __data_format__ if data_format is None else data_format

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        h_axis, w_axis, c_axis = [1, 2, 3] if data_format == "NHWC" else [2, 3, 1]
        input = tf.concat([input, input, input, input], axis=c_axis)
        input = tf.depth_to_space(input, 2, data_format=data_format)

    return input


def noise(input, stddev, by_add=False, by_multi=True, keep_prob=None, name='noise'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if by_add:
            input = input + tf.truncated_normal(tf.shape(input), 0, stddev, name=name)

        if by_multi:
            if keep_prob is not None:
                stddev = np.sqrt((1 - keep_prob) / keep_prob)  # get 'equivalent' stddev to dropout of keep_prob
            input = input * tf.truncated_normal(tf.shape(input), 1, stddev, name=name)

    return input


def lnoise(input, noise_std, drop_prob):

    if noise_std > 0:
        input = noise(input=input, stddev=noise_std, by_multi=True, by_add=False)

    if drop_prob > 0:
        input = dropout(input=input, drop_prob=drop_prob)

    return input


def dropout(input, drop_prob, name='dropout'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        keep_prob = 1.0 - drop_prob

        if keep_prob < 1.0:
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(tf.shape(input))
            binary_tensor = tf.floor(random_tensor)

            input = input * binary_tensor * np.sqrt(1.0 / keep_prob)
            # input = tf.nn.dropout(input, 1.0 - drop_prob, name=name) * (1.0 - drop_prob)

    return input


def normalized_orthogonal_initializer(flatten_axis, stddev=1.0):

    def _initializer(shape, dtype=None, partition_info=None):

        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")

        num_rows = 1
        for dim in [shape[i] for i in flatten_axis]:
            num_rows *= dim
        num_cols = shape[list(set(range(len(shape))) - set(flatten_axis))[0]]

        flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

        a = simple_stddev_random_value(flat_shape, type='uniform', stddev=1.0)
        q, r = linalg_ops.qr(a, full_matrices=False)

        q *= np.sqrt(flat_shape[0])

        if num_rows < num_cols:
            q = tf.matrix_transpose(q)

        return stddev * tf.reshape(q, shape)

    return _initializer


def simple_stddev_initializer(type, stddev):

    if type == 'normal':
        return tf.random_normal_initializer(stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform_initializer(minval=-stddev * np.sqrt(3.0), maxval=stddev * np.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal_initializer(stddev=stddev * np.sqrt(1.3))


def simple_stddev_random_value(shape, type, stddev):

    if type == 'normal':
        return tf.random_normal(shape, stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform(shape, minval=-stddev * np.sqrt(3.0), maxval=stddev * np.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal(shape, stddev=stddev * np.sqrt(1.3))


def spectral_normed_weight(W, num_iters=3, bUseCollection=True):

    def _l2normalize(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    u = tf.get_variable('u', [1, W_reshaped.shape.as_list()[1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    v = tf.get_variable('v', [1, W_reshaped.shape.as_list()[0]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32), u, v)
    )

    if bUseCollection:
        if u.name not in SPECTRAL_NORM_UV_LIST:
            SPECTRAL_NORM_UV_LIST.append(u.name)
            SPECTRAL_NORM_UV_UPDATE_OPS_LIST.append(u.assign(u_final))
        if v.name not in SPECTRAL_NORM_UV_LIST:
            SPECTRAL_NORM_UV_LIST.append(v.name)
            SPECTRAL_NORM_UV_UPDATE_OPS_LIST.append(v.assign(v_final))
        sigma = tf.matmul(tf.matmul(v, W_reshaped), tf.transpose(u))[0, 0]
    else:
        with tf.control_dependencies([u.assign(u_final), v.assign(v_final)]):
            sigma = tf.matmul(tf.matmul(v, W_reshaped), tf.transpose(u))[0, 0]

    if __enable_snk__:
        k = tf.get_variable('k', initializer=sigma, trainable=True)
        if k.name not in [var.name for var in SPECTRAL_NORM_K_LIST]:
            SPECTRAL_NORM_K_LIST.append(k)
            SPECTRAL_NORM_SIGMA_LIST.append(sigma)
            SPECTRAL_NORM_K_INIT_OPS_LIST.append(k.assign(sigma))
        W_bar = W_reshaped / sigma * k
    else:
        W_bar = W_reshaped / sigma

    W_bar = tf.reshape(W_bar, W_shape)

    return W_bar, sigma, u, v


def minibatch_feature(input, n_kernels=100, dim_per_kernel=5, name='minibatch'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()

        if len(input.get_shape()) > 2:
            input = tf.reshape(input, [input_shape[0], -1])

        batchsize = input_shape[0]

        x = linear(input, n_kernels * dim_per_kernel)
        x = tf.reshape(x, [-1, n_kernels, dim_per_kernel])

        mask = np.zeros([batchsize, batchsize])
        mask += np.eye(batchsize)
        mask = np.expand_dims(mask, 1)
        mask = 1. - mask
        rscale = 1.0 / np.sum(mask)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)), 2)
        masked = tf.exp(-abs_dif) * mask
        dist = tf.reduce_sum(masked, 2) * rscale

    return dist


def channel_concat(x, y):

    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    assert y_shapes[0] == x_shapes[0]

    y = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[1]]) * tf.ones([y_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])
    return tf.concat([x, y], 3)


def pwxb(input):

    input_shape = input.get_shape()

    if len(input_shape) == 4:
        if __data_format__ == 'NCHW':
            inputA = input[:, :input_shape[1] // 2, :, :]
            inputB = input[:, input_shape[1] // 2:, :, :]
        else:
            inputA = input[:, :, :, :input_shape[3] // 2]
            inputB = input[:, :, :, input_shape[3] // 2:]
    else:
        inputA = input[:, :input_shape[1] // 2]
        inputB = input[:, input_shape[1] // 2:]

    input = inputA * inputB

    # input = tf.nn.softplus(inputA) * inputB
    # xx = tf.nn.selu(tf.random_normal([1000000]))
    # yy = tf.nn.selu(tf.random_normal([1000000]))
    # print(tf.Session().run([tf.nn.moments(xx, 0), tf.nn.moments(tf.nn.softplus(xx) * yy, 0)]))

    return input


def scaled_activation(input, act, scale):

    xx = act(tf.random_normal([1000000]))
    mean, std = tf.Session().run(tf.nn.moments(xx, 0))
    input = (input - mean) / std * scale

    return input