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
import tensorflow as tf

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("n", 10, "")
tf.app.flags.DEFINE_string("oDataSet", "cifar10", "cifar10, mnist, flowers,tiny")

tf.app.flags.DEFINE_string("sResultTagA", "case5", "your tag for each test case")
tf.app.flags.DEFINE_string("sResultTagB", "mlp1024", "your tag for each test case")
tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")

tf.app.flags.DEFINE_string("discriminator", 'discriminator_mlp', "")

##################################################### Objectives ##############################################################################################

tf.app.flags.DEFINE_string("oGAN", 'x', "")
tf.app.flags.DEFINE_boolean("bPsiMax", False, "")
tf.app.flags.DEFINE_float("alpha", 1.0, "")

tf.app.flags.DEFINE_boolean("bLipReg", True, "")
tf.app.flags.DEFINE_boolean("bMaxGrad", True, "")
tf.app.flags.DEFINE_string("oReg", 'al', 'cp, gp, lp, al')
tf.app.flags.DEFINE_float("fWeightLip", 1.0, "")
tf.app.flags.DEFINE_float("fBufferBatch", 0.0, "")
tf.app.flags.DEFINE_float("fLipTarget", 1.0, "")  # for gp, lp and al
tf.app.flags.DEFINE_float("fLrAL", 1e-3, "")

tf.app.flags.DEFINE_boolean("bUseSND", False, "")

################################################# Learning Process ###########################################################################################

tf.app.flags.DEFINE_integer("iRun", 0, "")
tf.app.flags.DEFINE_integer("iMaxIter", 300000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 128, "")

tf.app.flags.DEFINE_float("fLrIni", 2e-4, "")
tf.app.flags.DEFINE_string("oOpt", 'Adam', "")
tf.app.flags.DEFINE_string("oDecay", 'Exp', "Linear, Exp, None")

tf.app.flags.DEFINE_float("fBeta1", 0.0, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")

tf.app.flags.DEFINE_float("fLrDecay", 0.1, "")
tf.app.flags.DEFINE_integer("iLrStep", 100000, "")
tf.app.flags.DEFINE_boolean("bLrStair", False, "")

tf.app.flags.DEFINE_string("oAct", 'relu', "")

##################################################### Network Structure #######################################################################################

tf.app.flags.DEFINE_integer("iNumLayer", 16, "")
tf.app.flags.DEFINE_integer("iDimsPerLayer", 1024, "")

tf.app.flags.DEFINE_string("oInitType", 'uniform', "truncated_normal, normal, uniform, orthogonal")
tf.app.flags.DEFINE_float("fInitWeightStddev", 0.10, "the initial weight stddev, if specified scale weight after init")

tf.app.flags.DEFINE_integer("GPU", -1, "")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/syn_real/", "where to save the checkpoint and sample")

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
sResultTag += '_%s%s%.e_%d' %(cfg.oOpt, cfg.oDecay, cfg.fLrIni, cfg.iMaxIter)
sResultTag += '_beta%.1f_%.3f' % (cfg.fBeta1, cfg.fBeta2)
sResultTag += '_' + cfg.oGAN + ('_%.2f' % cfg.alpha if cfg.oGAN in ['lsgan', 'hinge'] else '')
sResultTag += '_SND' if cfg.bUseSND else ''
sResultTag += ('_lip_' + ('max' if cfg.bMaxGrad else '') + cfg.oReg + '%.e' % cfg.fWeightLip) if cfg.bLipReg else ''
sResultTag += ('_fBF%.2f' % cfg.fBufferBatch) if (cfg.fBufferBatch != 0 and cfg.bLipReg) else ''
sResultTag += ('_fAL%.e' % cfg.fLrAL) if ('al' in cfg.oReg and cfg.fLrAL != 0 and cfg.bLipReg) else ''
sResultTag += ('_' + cfg.sResultTagB) if len(cfg.sResultTagB) else ''
sResultTag += '_%s' % cfg.oAct
sResultTag += ('' if cfg.iRun == 0 else '_run%d' % cfg.iRun)

sTestName = cfg.oDataSet + ('_n%d' % cfg.n) + ('_' + sResultTag if len(sResultTag) else '')
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

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
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
    if cfg.oAct == 'softplus':
        return tf.nn.softplus(x)
    elif cfg.oAct == 'lrelu':
        return tf.nn.leaky_relu(x)
    else:
        return tf.nn.relu(x)


def discriminator_mlp(input):

    set_enable_sn(cfg.bUseSND)

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        h0 = tf.layers.flatten(input)

        for i in range(cfg.iNumLayer):
            with tf.variable_scope('layer' + str(i)):
                h0 = linear(h0, cfg.iDimsPerLayer)
                h0 = activation(h0)

        h0 = linear(h0, 1, name='final_linear')

    return h0


def discriminator_mlp_dense(input):

    set_enable_sn(cfg.bUseSND)

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        h0 = tf.layers.flatten(input)

        for i in range(cfg.iNumLayer):
            with tf.variable_scope('layer' + str(i)):
                h1 = h0
                h1 = linear(h1, cfg.iDimsPerLayer)
                h1 = activation(h1)
                h0 = tf.concat(values=[h0, h1], axis=1)

        h0 = linear(h0, 1)

    return h0


############################################ Function ################################################################################################


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        return load_cifar10()
    if dataset_name == 'mnist':
        return load_mnist()


n = cfg.n
cifarX, cifarY = load_cifar10()[:2]
cifarK = [1400, 1621, 808, 809, 3687, 6815, 2100, 4037, 9154, 8532]

mnistX, mnistY = load_mnist(useX32=True, useC3=True)[:2]
mnistK = [1916, 1936, 1923, 1994, 1992, 1996, 1642, 1946, 1790, 1949]

r0 = cifarX
f0 = mnistX

if 'case5' in sResultTag:

    if cfg.oDataSet == 'mnist':
        r0 = mnistX[mnistK[:n]] if n <= 10 else mnistX[:n]
    else:
        r0 = cifarX[cifarK[:n]] if n <= 10 else cifarX[:n]

    f0 = np.random.uniform(size=np.shape(r0), low=-1., high=1.)
    # f0 = ndimage.gaussian_filter(r0, sigma=(0.0, 5.0, 5.0, 5.0))
    # f0 = 0.9 * np.random.uniform(size=np.shape(r0), low=-1., high=1.) + 0.1 * f0

elif 'case7' in sResultTag:

    r0 = cifarX[cifarK[:n]] if n <= 10 else cifarX[:n]
    f0 = mnistX[mnistK[:n]] if n <= 10 else mnistX[:n]


real_gen = data_gen_random(r0, cfg.iBatchSize)
fake_gen = data_gen_random(f0, cfg.iBatchSize)


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
                 feed_dict={real_datas: _real_datas, iter_datas: _iter_datas, fake_datas: _fake_datas, is_training: True})

    for i in range(len(_dis_vars)):
        logger.log('weight values: %8.5f %8.5f, lip gradient: %8.5f %8.5f, gan gradient: %8.5f %8.5f, tot gradient: %8.5f %8.5f    ' % (
            np.mean(_dis_vars[i]), np.std(_dis_vars[i]), np.mean(_disvar_lip_gradients[i]), np.std(_disvar_lip_gradients[i]), np.mean(_disvar_gan_gradients[i]),
            np.std(_disvar_gan_gradients[i]), np.mean(_disvar_tot_gradients[i]), np.std(_disvar_tot_gradients[i])) + dis_vars[i].name + ' shape: ' + str(dis_vars[i].shape))

    logger.linebreak()


def path(r, f, g, n):

    images = []

    rr = []
    for i in range(len(f)):
        error = np.zeros(len(r))
        for j in range(len(r)):
            # s = np.mean(r[j] - f[i]) / np.mean(g[i])
            # error[j] = np.linalg.norm(r[j] - f[i] - s * g[i])
            g_dir = np.reshape(g[i] / np.linalg.norm(g[i]), [-1])
            rf_dir = np.reshape((r[j]-f[i]) / np.linalg.norm(r[j]-f[i]), [-1])
            error[j] = -g_dir.dot(rf_dir)
        ir = np.argmin(error)
        rr.append(r[ir])
    rr = np.asarray(rr)

    s = np.median((rr-f) / g, axis=(1, 2, 3), keepdims=True)
    # s = np.mean(rr - f, axis=(1, 2, 3), keepdims=True) / np.mean(g, axis=(1, 2, 3), keepdims=True)

    images.append(f)
    images.append(g / np.max(np.abs(g), axis=(1, 2, 3), keepdims=True))

    for i in range(n):
        nn = int(n // 3)
        ff = f + (i+1)/(n-nn) * g * s
        images.append(ff)
        # ff = ff / np.max(np.abs(ff), axis=(1, 2, 3), keepdims=True)
        # images.append(ff)

    images.append(rr)

    return np.stack(images, 1)


########################################### Objective #################################################################################################

discriminator = globals()[cfg.discriminator]

real_datas = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='real_data')
fake_datas = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='fake_data')
iter_datas = tf.placeholder(tf.float32, (None,) + np.shape(r0)[1:], name='iter_data')

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

elif cfg.oGAN == 'log_sigmoid':
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

alpha = tf.random_uniform(shape=[tf.shape(real_datas)[0]] + [1] * (real_datas.get_shape().ndims - 1), minval=0., maxval=1.)
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

if 'Linear' == cfg.oDecay:
    dis_lr = tf.constant(cfg.fLrIni) * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif 'Exp' == cfg.oDecay:
    dis_lr = tf.train.exponential_decay(cfg.fLrIni, global_step, cfg.iLrStep, cfg.fLrDecay, cfg.bLrStair)
else:
    dis_lr = tf.constant(cfg.fLrIni)

if 'AdaShift' == cfg.oOpt:
    logger.log('dis_optimizer: AdaShift')
    dis_optimizer = AdaShift(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)
else:
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=dis_lr, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

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

_iter_datas = (gen_n_samples(real_gen, int(cfg.iBatchSize * cfg.fBufferBatch)) + gen_n_samples(fake_gen, int(cfg.iBatchSize * cfg.fBufferBatch))) / 2

log_netstate()
logger.log("Discriminator Total Parameter Count: {}\n\n".format(locale.format("%d", param_count(dis_gradient_values), grouping=True)))

start_iter = iter
start_time = time.time()

if cfg.bUseSND:
    sess.run(SPECTRAL_NORM_UV_UPDATE_OPS_LIST)

bSave = bPlot = False

while iter < cfg.iMaxIter:

    iter += 1
    train_start_time = time.time()

    if cfg.bLipReg and cfg.oReg == 'al':
        _, _, _al_lambda, _al_constraint, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, al_lambda_update_op, al_lambda, constraint, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients, slopes, dis_lr, real_logits, fake_logits],
            feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, fake_datas: fake_gen.__next__(), is_training: True})
    else:
        _, _dis_tot_loss, _dis_gan_loss, _dis_lip_loss, _interpolates, _dphi, _dvarphi, _slopes, _dis_lr, _real_logits, _fake_logits = sess.run(
            [dis_optimize_ops, dis_tot_loss, dis_gan_loss, dis_lip_loss, interpolates, phi_gradients, varphi_gradients, slopes, dis_lr, real_logits, fake_logits],
            feed_dict={real_datas: real_gen.__next__(), iter_datas: _iter_datas, fake_datas: fake_gen.__next__(), is_training: True})
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
        logger.plot_together(['logit_real', 'logit_fake'], [r'$\mathbb{E}_{x \sim P_r} f(x)$', r'$\mathbb{E}_{x \sim P_g} f(x)$'], ['olive', 'skyblue'], 'logits.pdf')
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

    if iter % 1000 == 0:
        m = 9
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_datas: f0}), m)
        save_images(grad_path.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path9_%d.png' % iter)

        m = 14
        grad_path = path(r0, f0, sess.run(real_gradients, feed_dict={real_datas: f0}), m)
        save_images(grad_path.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]), [np.shape(grad_path)[0], np.shape(grad_path)[1]], sTestCaseDir + 'grad_path14_%d.png' % iter)
