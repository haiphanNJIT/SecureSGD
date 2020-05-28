# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import math
import random
import scipy.integrate as integrate
import scipy.stats
import mpmath as mp
from gaussian_moments import *
from tensorflow.python.platform import flags
from datetime import datetime
import time
from tensorflow.python.training import optimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import accountant
import utils
from more_attack import *
from cleverhans.attacks_tf import fgm, fgsm
import copy
import os
from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, MadryEtAl, MomentumIterativeMethod
from cleverhans.attacks_tf import fgm, fgsm
from cleverhans import utils_tf
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper
from cleverhans.utils import set_log_level
import logging
import robustnessGGaussian
from math import sqrt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
set_log_level(logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#FLAGS = flags.FLAGS
#flags.DEFINE_float('clip_bound',8,'the clip bound of the gradients')
#flags.DEFINE_float('clip_bound_2',1/1.5,'the clip bound for r_kM')

#flags.DEFINE_float('small_num',1e-5,'a small number')
#flags.DEFINE_float('large_num',1e5,'a large number')
#flags.DEFINE_integer('num_images',60000,'number of images N')

#flags.DEFINE_integer('batch_size',200,'batch_size L')
#flags.DEFINE_float('sample_rate',200/60000,'sample rate q = L / N')
#flags.DEFINE_integer('num_steps',10000,'number of steps T = E * N / L = E / q')
#flags.DEFINE_integer('num_epoch',10,'number of epoches E')

# flags.DEFINE_float('sigma',4.0,'sigma')
# flags.DEFINE_float('epsilon',1.24,'epsilon')
# flags.DEFINE_float('delta',1e-5,'delta')

#flags.DEFINE_float('lambd',1e3,'exponential distribution parameter')

# flags.DEFINE_integer('iterative_clip_step',10,'iterative_clip_step')

#flags.DEFINE_integer('clip',1,'whether to clip the gradient')
#flags.DEFINE_integer('noise',1,'whether to add noise')
#flags.DEFINE_integer('redistribute',1,'whether to redistribute the noise')

def parse_time(time_sec):
    time_string = "{} hours {} minutes".format(int(time_sec/3600), int((time_sec%3600)/60))
    return time_string

def idle():
    return

# compute sigma using strong composition theory given epsilon


def compute_sigma(epsilon, delta):
    return 1/epsilon * np.sqrt(np.log(2/math.pi/np.square(delta))+2*epsilon)

# compute sigma using moment accountant given epsilon


def comp_sigma(q, T, delta, epsilon):
    c_2 = 4 * 1.26 / (0.01 * np.sqrt(10000 * np.log(100000)))  # c_2 = 1.485
    return c_2 * q * np.sqrt(T * np.log(1 / delta)) / epsilon

# compute epsilon using abadi's code given sigma


def comp_eps(lmbda, q, sigma, T, delta):
    lmbds = range(1, lmbda+1)
    log_moments = []
    for lmbd in lmbds:
        log_moment = compute_log_moment(q, sigma, T, lmbd)
        log_moments.append((lmbd, log_moment))

    eps, delta = get_privacy_spent(log_moments, target_delta=delta)
    return eps


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def train(fgsm_eps, _dp_epsilon, _attack_norm_bound, log_filename, ratio):
    FLAGS = None
    

    #ratio = 16
    #target_eps = [0.125,0.25,0.5,1,2,4,8]
    #target_eps = [0.25 + 0.25*ratio]
    target_eps = [0.2 + 0.2*ratio]
    #print(target_eps[0])
    #fgsm_eps = 0.1
    dp_epsilon=_dp_epsilon
    image_size = 28
    _log_filename = log_filename + str(target_eps[0]) + '_fgsm_' + str(fgsm_eps) + '_dpeps_' + str(dp_epsilon) + '_attack_norm_bound_' + str(_attack_norm_bound) + '.txt'

    clip_bound = 0.001  # 'the clip bound of the gradients'
    clip_bound_2 = 1/1.5  # 'the clip bound for r_kM'

    small_num = 1e-5  # 'a small number'
    large_num = 1e5  # a large number'
    num_images = 50000  # 'number of images N'

    batch_size = 125  # 'batch_size L'
    sample_rate = batch_size/50000  # 'sample rate q = L / N'
    # 900 epochs
    num_steps = 1800000  # 'number of steps T = E * N / L = E / q'
    num_epoch = 24  # 'number of epoches E'

    sigma = 5  # 'sigma'
    delta = 1e-5  # 'delta'

    lambd = 1e3  # 'exponential distribution parameter'

    iterative_clip_step = 2  # 'iterative_clip_step'

    clip = 1  # 'whether to clip the gradient'
    noise = 0  # 'whether to add noise'
    redistribute = 0  # 'whether to redistribute the noise'

    D = 50000

    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 25])
    b_fc1 = bias_variable([25])
    W_fc2 = weight_variable([25, 10])
    b_fc2 = bias_variable([10])
    
    def inference(x, dp_mult):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu((conv2d(x_image, W_conv1) + b_conv1) + dp_mult)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, h_conv1

    def inference_prob(x):
        logits, _ = inference(x, 0)
        y_prob = tf.nn.softmax(logits)
        return y_prob

    shape     = W_conv1.get_shape().as_list()
    w_t       = tf.reshape(W_conv1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivityW = tf.reduce_max(sing_vals)
    dp_delta=0.05
    attack_norm_bound = _attack_norm_bound
    dp_mult = attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon
    noise = tf.placeholder(tf.float32, [None, 28, 28, 32]);
    
    #y_conv, h_conv1 = inference(x, dp_mult * noise)
    y_conv, h_conv1 = inference(x, attack_norm_bound * noise)
    softmax_y = tf.nn.softmax(y_conv)
    # Define loss and optimizer

    priv_accountant = accountant.GaussianMomentsAccountant(D)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending(
        [None, None], sigma, batch_size)

    # sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    #train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy);
    #train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    
    # noise redistribution #
    grad, = tf.gradients(cross_entropy, h_conv1)
    normalized_grad = tf.sign(grad)
    normalized_grad = tf.stop_gradient(normalized_grad)
    normalized_grad_r = tf.abs(tf.reduce_mean(normalized_grad, axis = (0)))
    #print(normalized_grad_r)
    sum_r = tf.reduce_sum(normalized_grad_r, axis = (0,1,2), keepdims=False)
    #print(sum_r)
    normalized_grad_r = 256*32*normalized_grad_r/sum_r
    print(normalized_grad_r)
    
    shape_grad     = normalized_grad_r.get_shape().as_list()
    grad_t       = tf.reshape(normalized_grad_r, [-1, shape_grad[-1]])
    g         = tf.transpose(grad_t)
    sing_g_vals = tf.svd(g, compute_uv=False)
    sensitivity_2 = tf.reduce_max(sing_g_vals)
    ########################
    
    opt = GradientDescentOptimizer(learning_rate=1e-1)

    # compute gradient
    gw_W1 = tf.gradients(cross_entropy, W_conv1)[0]  # gradient of W1
    gb1 = tf.gradients(cross_entropy, b_conv1)[0]  # gradient of b1

    gw_W2 = tf.gradients(cross_entropy, W_conv2)[0]  # gradient of W2
    gb2 = tf.gradients(cross_entropy, b_conv2)[0]  # gradient of b2

    gw_Wf1 = tf.gradients(cross_entropy, W_fc1)[0]  # gradient of W_fc1
    gbf1 = tf.gradients(cross_entropy, b_fc1)[0]  # gradient of b_fc1

    gw_Wf2 = tf.gradients(cross_entropy, W_fc2)[0]  # gradient of W_fc2
    gbf2 = tf.gradients(cross_entropy, b_fc2)[0]  # gradient of b_fc2

    # clip gradient
    gw_W1 = tf.clip_by_norm(gw_W1, clip_bound)
    gw_W2 = tf.clip_by_norm(gw_W2, clip_bound)
    gw_Wf1 = tf.clip_by_norm(gw_Wf1, clip_bound)
    gw_Wf2 = tf.clip_by_norm(gw_Wf2, clip_bound)

    # sigma = FLAGS.sigma # when comp_eps(lmbda,q,sigma,T,delta)==epsilon

    # sensitivity = 2 * FLAGS.clip_bound #adjacency matrix with one tuple different
    sensitivity = clip_bound  # adjacency matrix with one more tuple

    gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0,
                              stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0,
                            stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0,
                              stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0,
                            stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gw_Wf1 += tf.random_normal(shape=tf.shape(gw_Wf1), mean=0.0,
                               stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gbf1 += tf.random_normal(shape=tf.shape(gbf1), mean=0.0,
                             stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gw_Wf2 += tf.random_normal(shape=tf.shape(gw_Wf2), mean=0.0,
                               stddev=(sigma * sensitivity)**2, dtype=tf.float32)
    gbf2 += tf.random_normal(shape=tf.shape(gbf2), mean=0.0,
                             stddev=(sigma * sensitivity)**2, dtype=tf.float32)

    train_step = opt.apply_gradients([(gw_W1, W_conv1), (gb1, b_conv1), (gw_W2, W_conv2), (
        gb2, b_conv2), (gw_Wf1, W_fc1), (gbf1, b_fc1), (gw_Wf2, W_fc2), (gbf2, b_fc2)])

    # craft adversarial samples from x for testing
    #softmax_y_test = tf.nn.softmax(y_conv)

    #====================== attack =========================

    attack_switch = {'fgsm':True, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}

    # define cleverhans abstract models for using cleverhans attacks
    ch_model_logits = CallableModelWrapper(callable_fn=inference, output_layer='logits')
    ch_model_probs = CallableModelWrapper(callable_fn=inference_prob, output_layer='probs')

    # define each attack method's tensor
    attack_tensor_dict = {}
    # FastGradientMethod
    if attack_switch['fgsm']:
        print('creating attack tensor of FastGradientMethod')
        fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
        x_adv_test_fgsm = fgsm_obj.generate(x=x, eps=fgsm_eps, clip_min=0.0, clip_max=1.0) # testing now
        attack_tensor_dict['fgsm'] = x_adv_test_fgsm

    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    # default: eps_iter=0.05, nb_iter=10
    if attack_switch['ifgsm']:
        print('creating attack tensor of BasicIterativeMethod')
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
        attack_tensor_dict['ifgsm'] = x_adv_test_ifgsm

    # MomentumIterativeMethod
    # default: eps_iter=0.06, nb_iter=10
    if attack_switch['mim']:
        print('creating attack tensor of MomentumIterativeMethod')
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        x_adv_test_mim = mim_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, decay_factor=1.0, clip_min=0.0, clip_max=1.0)
        attack_tensor_dict['mim'] = x_adv_test_mim

    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # default: eps_iter=0.01, nb_iter=40
    if attack_switch['madry']:
        print('creating attack tensor of MadryEtAl')
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        x_adv_test_madry = madry_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=0.0, clip_max=1.0)
        attack_tensor_dict['madry'] = x_adv_test_madry

    #====================== attack =========================

    #Define the correct prediction and accuracy#
    correct_prediction_x = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy_x = tf.reduce_mean(tf.cast(correct_prediction_x, tf.float32))

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    s = math.log(sqrt(2.0/math.pi)*1e+5)
    sigmaEGM = sqrt(2.0)*1.0*(sqrt(s) + sqrt(s+dp_epsilon))/(2.0*dp_epsilon)
    print(sigmaEGM)
    __noiseE = np.random.normal(0.0, sigmaEGM**2, 28*28*32).astype(np.float32)
    __noiseE = np.reshape(__noiseE, [-1, 28, 28, 32]);

    start_time = time.time()
    logfile = open(_log_filename, 'w')
    last_eval_time = -1
    accum_time = 0
    accum_epoch = 0
    max_benign_acc = -1
    max_adv_acc_dict = {}
    test_size = len(mnist.test.images)
    print("Computing The Noise Redistribution Vector")
    for i in range(4000):
        batch = mnist.train.next_batch(batch_size)
        sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, noise: __noiseE*0})
    batch = mnist.train.next_batch(batch_size*10)
    grad_redis = sess.run([normalized_grad_r], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, noise: __noiseE*0})
    #print(grad_redis)
    _sensitivity_2 = sess.run([sensitivity_2], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, noise: __noiseE*0})
    #print(_sensitivity_2)
    
    _sensitivityW = sess.run(sensitivityW)
    #print(_sensitivityW)
    Delta_redis = _sensitivityW/sqrt(_sensitivity_2[0])
    #print(Delta_redis)
    sigmaHGM = sqrt(2.0)*Delta_redis*(sqrt(s) + sqrt(s+dp_epsilon))/(2.0*dp_epsilon)
    #print(sigmaHGM)
    __noiseH = np.random.normal(0.0, sigmaHGM**2, 28*28*32).astype(np.float32)
    __noiseH = np.reshape(__noiseH, [-1, 28, 28, 32])*grad_redis;

    sess.run(tf.global_variables_initializer())
    print("Training")
    for i in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, noise: (__noiseE + __noiseH)/2})
        sess.run([privacy_accum_op])
        spent_eps_deltas = priv_accountant.get_privacy_spent(
            sess, target_eps=target_eps)
        if i % 1000 == 0:
            print(i, spent_eps_deltas)
        _break = False
        for _eps, _delta in spent_eps_deltas:
            if _delta >= delta:
                _break = True
                break
        if _break == True:
            break
    print("Testing")
    benign_acc = accuracy_x.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, noise: (__noiseE + __noiseH)/2})
    ### PixelDP Robustness ###
    adv_acc_dict = {}
    robust_adv_acc_dict = {}
    robust_adv_utility_dict = {}
    for atk in attack_switch.keys():
        if atk not in adv_acc_dict:
            adv_acc_dict[atk] = -1
            robust_adv_acc_dict[atk] = -1
            robust_adv_utility_dict[atk] = -1
        
        if attack_switch[atk]:
            adv_images_dict = sess.run(attack_tensor_dict[atk], feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0})
            #grad_redis = sess.run([normalized_grad_r], feed_dict={x: adv_images_dict, y_: mnist.test.labels, keep_prob: 1.0, noise:__noise})
            ### Robustness ###
            predictions_form_argmax = np.zeros([test_size, 10])
            softmax_predictions = softmax_y.eval(feed_dict={x: adv_images_dict, keep_prob: 1.0, noise:(__noiseE + __noiseH)/2})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
            for n_draws in range(0, 2000):
                if n_draws % 1000 == 0:
                    print(n_draws)
                _noiseE = np.random.normal(0.0, sigmaEGM**2, 28*28*32).astype(np.float32)
                _noiseE = np.reshape(_noiseE, [-1, 28, 28, 32]);
                _noise = np.random.normal(0.0, sigmaHGM**2, 28*28*32).astype(np.float32)
                _noise = np.reshape(_noise, [-1, 28, 28, 32])*grad_redis;
                for j in range(test_size):
                    pred = argmax_predictions[j]
                    predictions_form_argmax[j, pred] += 1;
                softmax_predictions = softmax_y.eval(feed_dict={x: adv_images_dict, keep_prob: 1.0, noise: (__noiseE + __noiseH)/2 + (_noiseE + _noise)/4})
                argmax_predictions = np.argmax(softmax_predictions, axis=1)
            final_predictions = predictions_form_argmax;
            is_correct = []
            is_robust = []
            for j in range(test_size):
                is_correct.append(np.argmax(mnist.test.labels[j]) == np.argmax(final_predictions[j]))
                robustness_from_argmax = robustnessGGaussian.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=dp_epsilon, dp_delta=1e-5, dp_mechanism='gaussian') / dp_mult
                is_robust.append(robustness_from_argmax >= fgsm_eps)
            adv_acc_dict[atk] = np.sum(is_correct)*1.0/test_size
            robust_adv_acc_dict[atk] = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
            robust_adv_utility_dict[atk] = np.sum(is_robust)*1.0/test_size
            print(" {}: {:.4f} {:.4f} {:.4f} {:.4f}".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk]*robust_adv_utility_dict[atk]))
            ##############################
    log_str = "step: {}\t target_epsilon: {}\t dp_epsilon: {:.1f}\t attack_norm_bound: {:.1f}\t benign_acc: {:.4f}\t".format(i, target_eps, dp_epsilon, attack_norm_bound, benign_acc)
    for atk in attack_switch.keys():
        if attack_switch[atk]:
            log_str += " {}: {:.4f} {:.4f} {:.4f} {:.4f}".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk]*robust_adv_utility_dict[atk])
    print(log_str)
    logfile.write(log_str + '\n')
    ##############################
    duration = time.time() - start_time
    logfile.write(str(duration) + '\n')
    logfile.flush()
    logfile.close()
    ###

def main(_):
    log_filename = './results/Run1_DPSGD_HGM_eps_'
    ratio = 4;
    for fgsm_eps in [10]: #[5, 10, 20, 30, 40, 50, 60]:
        for dp_epsilon in [4.0]: #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            for attack_norm_bound in [0.2]: #[0.1, 0.2, 0.3]:
                train((fgsm_eps*1.0)/100.0, dp_epsilon, attack_norm_bound, log_filename, ratio)

if __name__ == '__main__':
    if tf.gfile.Exists('./tmp/mnist_logs'):
        tf.gfile.DeleteRecursively('./tmp/mnist_logs')
    tf.gfile.MakeDirs('./tmp/mnist_logs')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
