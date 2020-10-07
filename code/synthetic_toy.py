import locale
import os
import sys
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

from common.ops import *
from common.optimizer import AdaShift
from common.data_loader import *
from common.logger import Logger

import numpy as np
import sklearn.datasets
import tensorflow as tf
from functools import partial

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("n", 2, "")
tf.app.flags.DEFINE_integer("iRun", 0, "")
tf.app.flags.DEFINE_string("sResultTagA", "case2", "your tag for each test case")
tf.app.flags.DEFINE_string("sResultTagB", "decay_ranged_re", "your tag for each test case")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")

tf.app.flags.DEFINE_string("discriminator", 'discriminator_mlp', "")

##################################################### Objectives ##############################################################################################

tf.app.flags.DEFINE_string("oGAN", 'lsgan', "")
tf.app.flags.DEFINE_boolean("bPsiMax", False, "")
tf.app.flags.DEFINE_float("alpha", 1.0, "")

tf.app.flags.DEFINE_boolean("bLipReg", False, "")
tf.app.flags.DEFINE_boolean("bMaxGrad", False, "")
tf.app.flags.DEFINE_string("oReg", 'al', 'cp, gp, lp, al')
tf.app.flags.DEFINE_float("fWeightLip", 1.0, "")
tf.app.flags.DEFINE_float("fBufferBatch", 0.0, "")
tf.app.flags.DEFINE_float("fLipTarget", 1.0, "")  # for gp, lp and al
tf.app.flags.DEFINE_float("fLrAL", 1e-3, "")

tf.app.flags.DEFINE_boolean("bUseSND", False, "")

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_integer("iMaxIter", 400000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 128, "")

tf.app.flags.DEFINE_string("oOpt", 'Adam', "")
tf.app.flags.DEFINE_float("fLrIni", 0.01, "")
tf.app.flags.DEFINE_boolean("bDecay", True, "")

tf.app.flags.DEFINE_float("fBeta1", 0.0, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_string("oAct", 'relu', "relu, elu, selu, swish, tanh, cos")

tf.app.flags.DEFINE_integer("iNumLayer", 16, "")
tf.app.flags.DEFINE_integer("iDimsPerLayer", 128, "")

tf.app.flags.DEFINE_string("oInitType", 'uniform', "truncated_normal, normal, uniform, orthogonal")
tf.app.flags.DEFINE_float("fInitWeightStddev", 0.10, "the initial weight stddev, if specified scale weight after init")

tf.app.flags.DEFINE_integer("GPU", -1, "")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/syn_toy/", "where to save the checkpoint and sample")

cfg(sys.argv)

############################################ MISC ################################################################################################

GPU_ID = allocate_gpu(cfg.GPU)

np.random.seed(1000)
tf.set_random_seed(1000)

set_enable_bias(True)
set_data_format('NCHW')
set_init_type(cfg.oInitType)
set_init_weight_stddev(cfg.fInitWeightStddev)

if cfg.bUseSND:
    cfg.bLipReg = False

sResultTag = cfg.sResultTagA
sResultTag += '_%s%.e' % (cfg.oOpt, cfg.fLrIni)
if cfg.oOpt != 'SGD':
    sResultTag += '_beta%.1f_%.3f' % (cfg.fBeta1, cfg.fBeta2)
sResultTag += ('_decay' if cfg.bDecay else '_none') + '%d' % cfg.iMaxIter
sResultTag += ('_mlp' if cfg.discriminator == 'discriminator_mlp' else '_densemlp') + '%dx%d' % (cfg.iDimsPerLayer, cfg.iNumLayer)
sResultTag += '_' + cfg.oAct
sResultTag += '_' + cfg.oGAN + ('_%.2f' % cfg.alpha if cfg.oGAN in ['lsgan', 'hinge'] else '')
sResultTag += '_SND' if cfg.bUseSND else ''
sResultTag += ('_lip_' + ('max' if cfg.bMaxGrad else '') + cfg.oReg + '%.e' % cfg.fWeightLip) if cfg.bLipReg else ''
sResultTag += ('_fBF%.2f' % cfg.fBufferBatch) if (cfg.fBufferBatch != 0 and cfg.bLipReg) else ''
sResultTag += ('_fAL%.e' % cfg.fLrAL) if ('al' in cfg.oReg and cfg.fLrAL != 0 and cfg.bLipReg) else ''
sResultTag += ('_' + cfg.sResultTagB) if len(cfg.sResultTagB) else ''
sResultTag += ('' if cfg.iRun == 0 else '_run%d' % cfg.iRun)

sTestName = 'toy' + ('_' + sResultTag if len(sResultTag) else '')
sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/sample/'
sCheckpointDir = sTestCaseDir + 'checkpoint/'

makedirs(sSampleDir)
makedirs(sCheckpointDir)
makedirs(sTestCaseDir + 'source/code/')
makedirs(sTestCaseDir + 'source/common/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.linebreak()
logger.log(sTestCaseDir)

commandline = ''
for arg in ['python3'] + sys.argv:
    commandline += arg + ' '
logger.log('\n' + commandline + '\n')

logger.log(str_flags(cfg.__flags))
logger.log('Using GPU%d\n' % GPU_ID)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

copydir(SOURCE_DIR + "code/", sTestCaseDir + 'source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + 'source/common/')
tf.logging.set_verbosity(tf.logging.ERROR)

############################################ Network ################################################################################################


is_training = tf.placeholder(bool, name='is_training')


def bn(x):
    return tf.layers.batch_normalization(x, axis=1, training=is_training)


def activation(x):
    if cfg.oAct == 'relu':
        return tf.nn.relu(x)
    elif cfg.oAct == 'selu':
        return tf.nn.selu(x)
    elif cfg.oAct == 'swish':
        return tf.nn.swish(x)
    elif cfg.oAct == 'elu':
        return tf.nn.elu(x)
    elif cfg.oAct == 'tanh':
        return tf.nn.tanh(x)
    elif cfg.oAct == 'cos':
        return tf.cos(x)
    else:
        raise Exception()


def discriminator_mlp(input):
    set_enable_sn(cfg.bUseSND)
    set_enable_sn(cfg.bUseSND)

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        h0 = input

        for i in range(cfg.iNumLayer):
            with tf.variable_scope('layer' + str(i)):
                h0 = linear(h0, cfg.iDimsPerLayer)
                h0 = activation(h0)

        h0 = linear(h0, 1, name='final_linear')

    return h0


def discriminator_mlp_dense(input):
    set_enable_sn(cfg.bUseSND)

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        h0 = tf.contrib.layers.flatten(input)

        for i in range(cfg.iNumLayer):
            with tf.variable_scope('layer' + str(i)):
                h1 = h0
                h1 = linear(h1, cfg.iDimsPerLayer)
                h1 = activation(h1)
                h0 = tf.concat(values=[h0, h1], axis=1)

        h0 = tf.contrib.layers.flatten(h0)
        h0 = linear(h0, 1)

    return h0


############################################ Function ################################################################################################


def transform(data, mean=(0, 0), size=1.0, rot=0.0, hflip=False, vflip=False):
    data *= size
    rotMatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    data = np.matmul(data, rotMatrix)
    if hflip: data[:, 0] *= -1
    if vflip: data[:, 1] *= -1
    data += mean
    return data


def sqaure_generator(num_sample, noise, transform):
    while True:
        x = np.random.rand(num_sample) - 0.5
        y = np.random.rand(num_sample) - 0.5
        data = np.asarray([x, y]).transpose().astype('float32') * 2.0
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def discrete_generator(num_sample, noise, transform, num_meta=2):
    meta_data = np.asarray(
        [[i, j] for i in range(-num_meta, num_meta + 1, 1) for j in range(-num_meta, num_meta + 1, 1)]) / float(
        num_meta)
    while True:
        idx = np.random.random_integers(0, len(meta_data) - 1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def boundary_generator(num_sample, noise, transform, num_meta=2):
    meta_data = []
    for i in range(-num_meta, num_meta + 1, 1):
        meta_data.append([i, -num_meta])
        meta_data.append([i, +num_meta])
    for i in range(-num_meta + 1, num_meta, 1):
        meta_data.append([-num_meta, i])
        meta_data.append([+num_meta, i])
    meta_data = np.asarray(meta_data) / float(num_meta)

    while True:
        idx = np.random.random_integers(0, len(meta_data) - 1, num_sample)
        data = meta_data[idx].astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def circle_generator(num_sample, noise, transform):
    while True:
        linspace = np.random.rand(num_sample)
        x = np.cos(linspace * 2 * np.pi)
        y = np.sin(linspace * 2 * np.pi)
        data = np.asarray([x, y]).transpose().astype('float32')
        data += noise * np.random.randn(num_sample, 2)
        data = transform(data)
        yield data


def scurve_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_s_curve(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 2.0
        data = transform(data)
        yield data


def swiss_generator(num_sample, noise, transform):
    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=num_sample,
            noise=noise
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 14.13717
        data = transform(data)
        yield data


def gaussian_generator(num_sample, noise, transform):
    while True:
        data = np.random.multivariate_normal([0.0, 0.0], noise * np.eye(2), num_sample)
        data = transform(data)
        yield data


def mix_generator(num_sample, generators, weights):
    while True:
        data = np.concatenate([generators[i].__next__() for i in range(len(generators))], 0)
        data_index = np.random.choice(len(weights), num_sample, replace=True, p=weights)
        data2 = np.concatenate(
            [data[num_sample * i:num_sample * i + np.sum(data_index == i)] for i in range(len(weights))], 0)
        np.random.shuffle(data2)
        yield data2


fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(-1.0, -0.0)))
real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.0, mean=(+1.0, +0.0)))

# fake_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5))
# real_gen = circle_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, 0), rot=np.pi / 2, hflip=True))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, 0), rot=np.pi / 2))

# fake_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.8, mean=(0, 0), hflip=False, rot=np.pi / 4))
# real_gen = scurve_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.5, mean=(0, 0)))

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0)))

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.5, 0.0)))], [0.9, 0.1])

# fake_gen = sqaure_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))
# real_gen = mix_generator(cfg.iBatchSize, [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(+1.0, 0.0))),
#                           gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.2, mean=(-1.0, 0.0)))], [0.9, 0.1])

# fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))
# real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(+0.0, 0)))

if 'case0' in sResultTag:

    fake_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0)))
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif 'case1' in sResultTag:

    fake_gen = mix_generator(cfg.iBatchSize,
                             [boundary_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.50, mean=(-1.0, -0.0))),
                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.25, mean=(-1.0, -0.0)))],
                             [0.20, 0.80])
    real_gen = sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))

elif 'case2' in sResultTag:

    fake_gen = mix_generator(cfg.iBatchSize,
                             [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))],
                             [0.80, 0.20])
    real_gen = mix_generator(cfg.iBatchSize,
                             [sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(-1.0, -0.0))),
                              sqaure_generator(cfg.iBatchSize, 0.0, partial(transform, size=0.5, mean=(+1.0, +0.0)))],
                             [0.20, 0.80])

# elif 'case3' in sResultTag:
#
#     fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-1.0, -0.0)))
#     real_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, +0.0)))

elif 'case3' in sResultTag:

    fake_gen = gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(-0.75, 0.0)))
    real_gen = mix_generator(cfg.iBatchSize,
                             [gaussian_generator(cfg.iBatchSize, 1.0, partial(transform, size=0.15, mean=(+1.0, 0.0))),
                              gaussian_generator(cfg.iBatchSize, 1.0,
                                                 partial(transform, size=0.15, mean=(-1.25, 0.0)))], [0.5, 0.5])

elif 'case4' in sResultTag:

    n = cfg.n
    np.random.seed(123456789)

    if n == 0:
        n = 2
        r0 = [[+0.8, +0.5], [-0.8, -0.5]]
        f0 = [[-0.8, +0.5], [+0.8, -0.5]]
    else:
        r0 = (np.random.rand(n, 2) - 0.5) * 2
        f0 = (np.random.rand(n, 2) - 0.5) * 2


    def get_mix_gen(centers):
        mix_gen = []
        for i in range(n):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0,
                                              partial(transform, size=0.0, mean=(centers[i][0], centers[i][1]))))
        return mix_gen


    fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0), [1 / n] * n)
    real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0), [1 / n] * n)

elif 'caseA' in sResultTag:

    fake_gen = swiss_generator(cfg.iBatchSize, 1.0, partial(transform, size=1.0, mean=(0, 0)))
    real_gen = swiss_generator(cfg.iBatchSize, 0.0, partial(transform, size=1.0, mean=(0, 0)))

elif 'caseB' in sResultTag:

    n = cfg.n
    std = 0

    if n < 0:
        n = -n
        std = 1e-2


    def get_mix_gen(centers, std):
        mix_gen = []
        for i in range(len(centers)):
            mix_gen.append(gaussian_generator(cfg.iBatchSize, 1.0,
                                              partial(transform, size=std, mean=(centers[i][0], centers[i][1]))))
        return mix_gen


    if n == 2:

        f0 = [[-0.5, +0.5], [+0.5, -0.5]]
        r0 = [[+0.5, +0.5], [-0.5, -0.5]]

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / n] * n)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / n] * n)

    elif n == 4:

        np.random.seed(123456789)
        f0 = (np.random.rand(2, 2) - 0.5) * 2
        r0 = (np.random.rand(4, 2) - 0.5) * 2

        fake_gen = mix_generator(cfg.iBatchSize, get_mix_gen(f0, std), [1 / 2] * 2)
        real_gen = mix_generator(cfg.iBatchSize, get_mix_gen(r0, 0), [1 / 4] * 4)


def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count


def log_netstate():
    _real_datas = real_gen.__next__()
    _fake_datas = fake_gen.__next__()

    logger.linebreak()
    _dis_vars, _disvar_lip_gradients, _disvar_gan_gradients, _disvar_tot_gradients = \
        sess.run([dis_vars, disvar_lip_gradients, disvar_gan_gradients, disvar_tot_gradients],
                 feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, fake_datas: _fake_datas,
                            is_training: True})

    for i in range(len(_dis_vars)):
        logger.log(
            'weight values: %8.5f %8.5f, lip gradient: %8.5f %8.5f, gan gradient: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (
                np.mean(_dis_vars[i]), np.std(_dis_vars[i]), np.mean(_disvar_lip_gradients[i]),
                np.std(_disvar_lip_gradients[i]), np.mean(_disvar_gan_gradients[i]),
                np.std(_disvar_gan_gradients[i]), np.mean(_disvar_tot_gradients[i]), np.std(_disvar_tot_gradients[i])) +
            dis_vars[i].name + ' shape: ' + str(dis_vars[i].shape))

    logger.linebreak()


def plot2(names, x_map_size, y_map_size, x_value_range, y_value_range, mode='fake', contour=False):

    def get_current_logits_map():

        logits_map = np.zeros([y_map_size, x_map_size, 1])
        gradients_map = np.zeros([y_map_size, x_map_size, 2])

        for i in range(y_map_size):  # the i-th row and j-th column
            locations = []
            for j in range(x_map_size):
                y = y_value_range[1] - (y_value_range[1] - y_value_range[0]) / y_map_size * (i + 0.5)
                x = x_value_range[0] + (x_value_range[1] - x_value_range[0]) / x_map_size * (j + 0.5)
                locations.append([x, y])
            locations = np.asarray(locations).reshape([x_map_size, 2])
            logits_map[i], gradients_map[i] = sess.run([real_logits, real_gradients], feed_dict={real_datas: locations})

        return logits_map.reshape(y_map_size, x_map_size), gradients_map

    def boundary_data(num_meta, mean, size):

        meta_data = []
        for i in range(-num_meta, num_meta + 1, 1):
            meta_data.append([i, -num_meta])
            meta_data.append([i, +num_meta])
        for i in range(-num_meta + 1, num_meta, 1):
            meta_data.append([-num_meta, i])
            meta_data.append([+num_meta, i])
        meta_data = np.asarray(meta_data) / float(num_meta)

        meta_data *= size
        meta_data += mean

        return meta_data

    def get_data_and_gradient(gen, num, pre_sample=None):
        data = []
        logit = []
        gradient = []
        if pre_sample is not None:
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: pre_sample})
            data.append(pre_sample)
            logit.append(_logit)
            gradient.append(_gradient)
        for i in range(num // cfg.iBatchSize + 1):
            _data = gen.__next__()
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: _data})
            data.append(_data)
            logit.append(_logit)
            gradient.append(_gradient)
        data = np.concatenate(data, axis=0)
        logit = np.concatenate(logit, axis=0)
        gradient = np.concatenate(gradient, axis=0)
        return data[:num], logit[:num], gradient[:num]

    pre_sample = None
    if 'case1' in sResultTag:
        pre_sample = boundary_data(5, [-1.0, 0.0], 0.25)
    elif 'case2' in sResultTag:
        pre_sample = np.concatenate([boundary_data(5, [-1.0, 0.0], 0.5), boundary_data(5, [1.0, 0.0], 0.5)], 0)

    _real_datas, _real_logits, _real_gradients = get_data_and_gradient(real_gen, 1024)
    _fake_datas, _fake_logits, _fake_gradients = get_data_and_gradient(fake_gen, 1024, pre_sample)

    rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(dpi=300)

    logits_map, gradients_map = get_current_logits_map()

    cmin = np.min(logits_map)
    cmax = np.max(logits_map)

    if contour:
        im = ax.contourf(logits_map, 50,
                         extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]])
    else:
        im = ax.imshow(logits_map, extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]],
                       vmin=cmin, vmax=cmax, cmap='viridis')

    plt.scatter(_real_datas[:, 0], _real_datas[:, 1], marker='+', s=50, label='real samples',
                color='navy')  # purple')#indigo')#navy')balck
    plt.scatter(_fake_datas[:, 0], _fake_datas[:, 1], marker='*', s=50, label='fake samples',
                color='ivory')  # ivory') #'silver')white

    plt.xlim(x_value_range[0], x_value_range[1])
    plt.ylim(y_value_range[0], y_value_range[1])

    if mode != 'none':

        if mode == 'fake':
            xx, yy, uu, vv = _fake_datas[:, 0], _fake_datas[:, 1], _fake_gradients[:, 0], _fake_gradients[:, 1]
        else:
            num_arrow = 20
            skip = (slice(y_map_size // num_arrow // 2, None, y_map_size // num_arrow),
                    slice(x_map_size // num_arrow // 2, None, x_map_size // num_arrow))
            y, x = np.mgrid[y_value_range[1]:y_value_range[0]:y_map_size * 1j,
                   x_value_range[0]:x_value_range[1]:x_map_size * 1j]
            xx, yy, uu, vv = x[skip], y[skip], gradients_map[skip][:, :, 0], gradients_map[skip][:, :, 1]

        ref_scale = np.max(np.linalg.norm(_fake_gradients, axis=1)) / 2 if np.max(
            np.linalg.norm(_fake_gradients, axis=1)) / np.mean(
            np.linalg.norm(_fake_gradients, axis=1)) < 2 else np.mean(np.linalg.norm(_fake_gradients, axis=1))

        if 'caseA' in sResultTag:
            ref_scale = np.mean(np.linalg.norm(_fake_gradients, axis=1)) / 5

        len = np.hypot(uu, vv)
        uu = uu / (len + 1e-8) * np.maximum(0.5 * ref_scale, np.minimum(len, 2 * ref_scale))
        vv = vv / (len + 1e-8) * np.maximum(0.5 * ref_scale, np.minimum(len, 2 * ref_scale))

        ax.quiver(xx, yy, uu, vv, color='red', angles='xy', width=0.03, minlength=0.8, minshaft=3, scale=ref_scale * 20)

    ax.set(aspect=1, title='')
    plt.legend(loc='upper left', prop={'size': 14})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both', format='%.1f')

    plt.tight_layout()

    for name in names:
        plt.savefig(sSampleDir + name)

    plt.close()


def plot(names, x_map_size, y_map_size, x_value_range, y_value_range, mode='fake', contour=False):

    def get_current_logits_map():

        logits_map = np.zeros([y_map_size, x_map_size, 1])
        gradients_map = np.zeros([y_map_size, x_map_size, 2])

        for i in range(y_map_size):  # the i-th row and j-th column
            locations = []
            for j in range(x_map_size):
                y = y_value_range[1] - (y_value_range[1] - y_value_range[0]) / y_map_size * (i + 0.5)
                x = x_value_range[0] + (x_value_range[1] - x_value_range[0]) / x_map_size * (j + 0.5)
                locations.append([x, y])
            locations = np.asarray(locations).reshape([x_map_size, 2])
            logits_map[i], gradients_map[i] = sess.run([real_logits, real_gradients], feed_dict={real_datas: locations})

        return logits_map.reshape(y_map_size, x_map_size), gradients_map

    def boundary_data(num_meta, mean, size):

        meta_data = []
        for i in range(-num_meta, num_meta + 1, 1):
            meta_data.append([i, -num_meta])
            meta_data.append([i, +num_meta])
        for i in range(-num_meta + 1, num_meta, 1):
            meta_data.append([-num_meta, i])
            meta_data.append([+num_meta, i])
        meta_data = np.asarray(meta_data) / float(num_meta)

        meta_data *= size
        meta_data += mean

        return meta_data

    def get_data_and_gradient(gen, num, pre_sample=None):
        data = []
        logit = []
        gradient = []
        if pre_sample is not None:
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: pre_sample})
            data.append(pre_sample)
            logit.append(_logit)
            gradient.append(_gradient)
        for i in range(num // cfg.iBatchSize + 1):
            _data = gen.__next__()
            _logit, _gradient = sess.run([real_logits, real_gradients], feed_dict={real_datas: _data})
            data.append(_data)
            logit.append(_logit)
            gradient.append(_gradient)
        data = np.concatenate(data, axis=0)
        logit = np.concatenate(logit, axis=0)
        gradient = np.concatenate(gradient, axis=0)
        return data[:num], logit[:num], gradient[:num]

    pre_sample = None
    if 'case1' in sResultTag:
        pre_sample = boundary_data(5, [-1.0, 0.0], 0.25)
    elif 'case2' in sResultTag:
        pre_sample = np.concatenate([boundary_data(5, [-1.0, 0.0], 0.5), boundary_data(5, [1.0, 0.0], 0.5)], 0)

    _real_datas, _real_logits, _real_gradients = get_data_and_gradient(real_gen, 1024)
    _fake_datas, _fake_logits, _fake_gradients = get_data_and_gradient(fake_gen, 1024, pre_sample)

    cmin = np.min(np.concatenate([_fake_logits, _real_logits], 0))
    cmax = np.max(np.concatenate([_fake_logits, _real_logits], 0))
    if cfg.oGAN == 'lsgan' and not cfg.bLipReg and cmax - cmin > 1e-1:
        cmin = -2.0
        cmax = +2.0

    rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(dpi=300)

    logits_map, gradients_map = get_current_logits_map()
    # pickle.dump([_real_datas, _real_logits, _real_gradients, _fake_datas, _fake_logits, _fake_gradients, logits_map, gradients_map], open(sSampleDir + names[0] + '.pck', 'wb'))

    if contour:
        im = ax.contourf(logits_map, 50,
                         extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]])
    else:
        im = ax.imshow(logits_map, extent=[x_value_range[0], x_value_range[1], y_value_range[0], y_value_range[1]],
                       vmin=cmin, vmax=cmax, cmap='viridis')

    plt.scatter(_real_datas[:, 0], _real_datas[:, 1], marker='+', s=1.5, label='real samples',
                color='navy')  # purple')#indigo')#navy')balck
    plt.scatter(_fake_datas[:, 0], _fake_datas[:, 1], marker='*', s=1.5, label='fake samples',
                color='ivory')  # ivory') #'silver')white

    plt.xticks(np.arange(x_value_range[0], x_value_range[1] + 1e-3, 0.5))
    plt.yticks(np.arange(y_value_range[0], y_value_range[1] + 1e-3, 0.5))

    if mode != 'none':

        if mode == 'fake':
            xx, yy, uu, vv = _fake_datas[:, 0], _fake_datas[:, 1], _fake_gradients[:, 0], _fake_gradients[:, 1]
        else:
            num_arrow = 20
            skip = (slice(y_map_size // num_arrow // 2, None, y_map_size // num_arrow),
                    slice(x_map_size // num_arrow // 2, None, x_map_size // num_arrow))
            y, x = np.mgrid[y_value_range[1]:y_value_range[0]:y_map_size * 1j,
                   x_value_range[0]:x_value_range[1]:x_map_size * 1j]
            xx, yy, uu, vv = x[skip], y[skip], gradients_map[skip][:, :, 0], gradients_map[skip][:, :, 1]

        ref_scale = np.max(np.linalg.norm(_fake_gradients, axis=1)) / 2 if np.max(
            np.linalg.norm(_fake_gradients, axis=1)) / np.mean(
            np.linalg.norm(_fake_gradients, axis=1)) < 2 else np.mean(np.linalg.norm(_fake_gradients, axis=1))

        if 'caseA' in sResultTag:
            ref_scale = np.mean(np.linalg.norm(_fake_gradients, axis=1)) / 5

        len = np.hypot(uu, vv)
        uu = uu / (len + 1e-8) * np.maximum(0.5 * ref_scale, np.minimum(len, 2 * ref_scale))
        vv = vv / (len + 1e-8) * np.maximum(0.5 * ref_scale, np.minimum(len, 2 * ref_scale))

        q = ax.quiver(xx, yy, uu, vv, color='red', angles='xy', width=0.001, minlength=0.8, minshaft=3,
                      scale=ref_scale * 30)  # violet, fuchsia
        plt.quiverkey(q, 0.65, 0.920, ref_scale, r'$\Vert\nabla_{\!x}f(x)\Vert$=%.2E' % (float(ref_scale)),
                      labelpos='E', coordinates='figure')

    ax.set(aspect=1, title='')
    plt.legend(loc='upper left', prop={'size': 10})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both', format='%.1f')

    plt.tight_layout()

    for name in names:
        plt.savefig(sSampleDir + name)

    plt.close()


def do_plot(iter):

    if 'case4' in sResultTag:
        plot2(['map_fake_%d.png' % iter], 900, 600, [-1.5, 1.5], [-1.5, 1.5], 'fake')
    else:
        plot(['map_fake_%d.png' % iter], 900, 600, [-2.0, 2.0], [-1.5, 1.5], 'fake')


########################################### Objective #################################################################################################


discriminator = globals()[cfg.discriminator]

real_datas = tf.placeholder(tf.float32, (None, 2), name='real_data')
fake_datas = tf.placeholder(tf.float32, (None, 2), name='fake_data')
iter_datas = tf.placeholder(tf.float32, (None, 2), name='iter_data')

fake_logits = discriminator(fake_datas)
real_logits = discriminator(real_datas)

if cfg.oGAN == 'lsgan':
    logger.log('using lsgan loss: %.2f' % cfg.alpha)
    dis_real_loss = tf.square(real_logits - cfg.alpha)
    dis_fake_loss = tf.square(fake_logits + cfg.alpha)
    gen_fake_loss = tf.square(fake_logits - cfg.alpha)

elif cfg.oGAN == 'x':
    logger.log('using wgan loss')
    dis_real_loss = -real_logits
    dis_fake_loss = fake_logits
    gen_fake_loss = -fake_logits

elif cfg.oGAN == 'sqrt':
    logger.log('using sqrt loss')
    dis_real_loss = tf.sqrt(tf.square(real_logits) + 1) - real_logits
    dis_fake_loss = tf.sqrt(tf.square(fake_logits) + 1) + fake_logits
    if cfg.bPsiMax:
        gen_fake_loss = tf.sqrt(tf.square(fake_logits) + 1) - fake_logits
    else:
        gen_fake_loss = -fake_logits

elif cfg.oGAN == 'log':
    logger.log('using log_sigmoid loss')
    dis_real_loss = -tf.log_sigmoid(real_logits)
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)
    if cfg.bPsiMax:
        gen_fake_loss = -tf.log_sigmoid(fake_logits)
    else:
        gen_fake_loss = -fake_logits

elif cfg.oGAN == 'exp':
    logger.log('using exp loss')
    dis_real_loss = tf.exp(-real_logits)
    dis_fake_loss = tf.exp(fake_logits)
    if cfg.bPsiMax:
        gen_fake_loss = tf.exp(-fake_logits)
    else:
        gen_fake_loss = -fake_logits

elif cfg.oGAN == 'hinge':
    logger.log('using hinge loss: %.2f' % cfg.alpha)
    dis_real_loss = -tf.minimum(tf.constant(0.), real_logits - tf.constant(cfg.alpha))
    dis_fake_loss = -tf.minimum(tf.constant(0.), -fake_logits - tf.constant(cfg.alpha))
    if cfg.bPsiMax:
        gen_fake_loss = -tf.minimum(tf.constant(0.), fake_logits - tf.constant(cfg.alpha))
    else:
        gen_fake_loss = -fake_logits

else:
    logger.log('using default gan loss')
    dis_real_loss = -tf.log_sigmoid(real_logits)
    dis_fake_loss = -tf.log_sigmoid(-fake_logits)
    gen_fake_loss = -tf.log_sigmoid(fake_logits)

dis_tot_loss = dis_gan_loss = tf.reduce_mean(dis_fake_loss) + tf.reduce_mean(dis_real_loss)
gen_tot_loss = gen_gan_loss = tf.reduce_mean(gen_fake_loss)

dis_lip_loss = tf.constant(0)

alpha = tf.random_uniform(shape=[tf.shape(real_datas)[0]] + [1] * (real_datas.get_shape().ndims - 1), minval=0.,
                          maxval=1.)
interpolates = real_datas * alpha + fake_datas * (1. - alpha)

if cfg.bMaxGrad:
    if cfg.fBufferBatch >= 0:
        interpolates = tf.concat([iter_datas, interpolates], 0)
    else:
        interpolates = tf.concat([iter_datas, interpolates[tf.shape(iter_datas)[0]:]], 0)

gradients = tf.gradients(discriminator(interpolates), interpolates)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=list(range(1, real_datas.get_shape().ndims))))

if cfg.bLipReg:

    slopes_t = tf.reduce_max(slopes) if cfg.bMaxGrad else slopes

    if cfg.oReg == 'cp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes_t))
    elif cfg.oReg == 'gp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(slopes_t - cfg.fLipTarget))
    elif cfg.oReg == 'lp':
        dis_lip_loss = cfg.fWeightLip * tf.reduce_mean(tf.square(tf.maximum(0.0, slopes_t - cfg.fLipTarget)))
    elif cfg.oReg == 'al':
        if not cfg.bMaxGrad:
            slopes_t = tf.reduce_mean(slopes)
        al_lambda = tf.get_variable('lambda', [], initializer=tf.zeros_initializer(), trainable=False)
        constraint = slopes_t - cfg.fLipTarget
        dis_lip_loss = cfg.fWeightLip * tf.square(constraint) + al_lambda * constraint
        if cfg.fLrAL != 0:
            al_lambda_update_op = tf.assign(al_lambda, al_lambda + cfg.fLrAL * constraint)
        else:
            al_lambda_update_op = tf.assign(al_lambda, al_lambda + 2 * cfg.fWeightLip * constraint)
    elif cfg.oReg == 'ali':
        if not cfg.bMaxGrad:
            slopes_t = tf.reduce_mean(slopes)
        al_lambda = tf.get_variable('lambda', [], initializer=tf.zeros_initializer(), trainable=False)
        constraint = slopes_t - cfg.fLipTarget
        al_s = tf.minimum(constraint + al_lambda / 2.0 / cfg.fWeightLip, 0.0)
        dis_lip_loss = cfg.fWeightLip * tf.square(constraint - al_s) + al_lambda * (constraint - al_s)
        if cfg.fLrAL != 0:
            al_lambda_update_op = tf.assign(al_lambda, tf.maximum(al_lambda + cfg.fLrAL * constraint, 0.0))
        else:
            al_lambda_update_op = tf.assign(al_lambda, tf.maximum(al_lambda + 2 * cfg.fWeightLip * constraint, 0.0))

    dis_tot_loss += dis_lip_loss

tot_vars = tf.trainable_variables()
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]

global_step = tf.Variable(0, trainable=False, name='global_step')
step_op = tf.assign_add(global_step, 1)

if cfg.bDecay:
    dis_lr = tf.constant(cfg.fLrIni) * tf.minimum(1., 2. * (1. - tf.cast(global_step, tf.float32) / cfg.iMaxIter))
else:
    dis_lr = tf.constant(cfg.fLrIni)

if 'AdaShift' == cfg.oOpt:
    logger.log('dis_optimizer: AdaShift')
    dis_optimizer = AdaShift(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
elif 'SGD' == cfg.oOpt:
    dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=dis_lr)
elif 'Adam' == cfg.oOpt:
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2,
                                           epsilon=cfg.fEpsilon)
else:
    assert False

dis_gradient_values = dis_optimizer.compute_gradients(dis_tot_loss, var_list=dis_vars)
dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
with tf.control_dependencies(dis_update_ops):
    dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values)

real_gradients = tf.gradients(real_logits, real_datas)[0]
fake_gradients = tf.gradients(fake_logits, fake_datas)[0]

varphi_gradients = tf.gradients(dis_real_loss, real_logits)[0]
phi_gradients = tf.gradients(dis_fake_loss, fake_logits)[0]

disvar_lip_gradients = tf.gradients(dis_lip_loss, dis_vars)
disvar_gan_gradients = tf.gradients(dis_gan_loss, dis_vars)
disvar_tot_gradients = tf.gradients(dis_tot_loss, dis_vars)

disvar_lip_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_lip_gradients]
disvar_gan_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_gan_gradients]
disvar_tot_gradients = [tf.constant(0.0) if grad is None else grad for grad in disvar_tot_gradients]

saver = tf.train.Saver()

################################## Traininig ##########################################################################################################

iter = 0
last_save_time = last_log_time = last_plot_time = last_imshave_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n\n')
            logger.flush()
            logger.log('\n\n')
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

_iter_datas = (gen_n_samples(real_gen, int(cfg.iBatchSize * cfg.fBufferBatch)) + gen_n_samples(fake_gen, int(
    cfg.iBatchSize * cfg.fBufferBatch))) / 2

log_netstate()
logger.log("Discriminator Total Parameter Count: {}\n\n".format(
    locale.format("%d", param_count(dis_gradient_values), grouping=True)))

start_iter = iter
start_time = time.time()

if cfg.bUseSND:
    sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

bSave = bPlot = False
do_plot(iter)

while iter < cfg.iMaxIter:

    iter += 1
    train_start_time = time.time()

    if cfg.bLipReg and cfg.oReg == 'al':
        _, _, _al_lambda, _al_constraint, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, al_lambda_update_op, al_lambda, constraint, dis_tot_loss, dis_gan_loss, dis_lip_loss,
             interpolates, phi_gradients, varphi_gradients, slopes, dis_lr, real_logits, fake_logits],
            feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, fake_datas: fake_gen.__next__(),
                       is_training: True})
    else:
        _, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients,
             slopes, dis_lr, real_logits, fake_logits],
            feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, fake_datas: fake_gen.__next__(),
                       is_training: True})
    if cfg.bUseSND:
        sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

    sess.run(step_op)
    logger.info('time_train', time.time() - train_start_time)

    log_start_time = time.time()

    logger.tick(iter)
    logger.info('klrD', _dis_lr * 1000)

    logger.info('logit_real', np.mean(_real_logits))
    logger.info('logit_fake', np.mean(_fake_logits))

    logger.info('loss_dis_gp', _dis_lip_loss)
    logger.info('loss_dis_gan', _dis_gan_loss)
    logger.info('loss_dis_tot', _dis_tot_loss)

    logger.info('d_phi', np.mean(_dphi))
    logger.info('d_varphi', np.mean(_dvarphi))

    logger.info('slopes_mean', np.mean(_slopes))
    logger.info('slopes_max', np.max(_slopes))

    if cfg.bMaxGrad:
        _iter_datas = _interpolates[np.argsort(-np.asarray(_slopes))[:len(_iter_datas)]]

    if cfg.bLipReg and cfg.oReg == 'al':
        logger.info('al_lambda', _al_lambda)
        logger.info('al_lambda_delta', _al_constraint)

    if np.any(np.isnan(_dis_tot_loss)):
        log_netstate()
        logger.flush()
        os.system('mv ' + sTestCaseDir + ' ' + sTestCaseDir[:-1] + '_NAN')
        exit(0)

    if time.time() - last_save_time > 60 * 30 or iter == cfg.iMaxIter or bSave:
        stime = time.time()
        log_netstate()
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()
        logger.log('Model saved')
        logger.log('Time: %.2f' % (time.time() - stime))
        bSave = False

    if time.time() - last_plot_time > 60 * 30 or iter == cfg.iMaxIter or bPlot:
        stime = time.time()
        logger.plot()
        logger.plot_together(['logit_real', 'logit_fake'],
                             [r'$\mathbb{E}_{x \sim P_r} f(x)$', r'$\mathbb{E}_{x \sim P_g} f(x)$'],
                             ['olive', 'skyblue'], 'logits.pdf')
        last_plot_time = time.time()
        logger.log('Plotted')
        logger.log('Time: %.2f' % (time.time() - stime))
        bPlot = False

    logger.info('time_log', time.time() - log_start_time)

    if time.time() - last_log_time > 60 * 1 or iter == 1:
        logger.info('time_used', (time.time() - start_time) / 3600)
        logger.info('time_remain', (time.time() - start_time) / (iter - start_iter) * (cfg.iMaxIter - iter) / 3600)
        logger.flush()
        last_log_time = time.time()

    if iter == 1 or iter == 10 or iter == 100 or iter == 1000 or iter == 4000 or iter == 7000 or iter % 10000 == 0:
        do_plot(iter)
