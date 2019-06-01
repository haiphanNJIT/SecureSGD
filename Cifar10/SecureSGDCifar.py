"""
An implementation of the Adaptive Laplace Mechanism (AdLM)
Author: Hai Phan, CCS, NJIT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
from math import sqrt;
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from gaussian_moments import *;
import accountant, utils
from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans import utils_tf
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper
from cleverhans.utils import set_log_level
from more_attack import *
from cleverhans.attacks_tf import fgm, fgsm
import random
import logging
import cifar10;
import cifar10_read
import robustnessGGaussian

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = '/tmp/DPSGD2/cifar10_train_DPSGD_HGM'
FLAGS = tf.app.flags.FLAGS;
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + path,
                           """Directory where to read model checkpoints.""")

#############################
##Hyper-parameter Setting####
#############################
hk = 256; #number of hidden units at the last layer
D = 50000;
infl = 1; #inflation rate in the privacy budget redistribution
R_lowerbound = 1e-5; #lower bound of the LRP
c = [0, 40, 50, 75] #norm bounds
image_size = 28;
padding = 4;
batch_size = 125
lr = 0.1
epochs = 5000; #number of epochs
MOVING_AVERAGE_DECAY = 0.9999

clip_bound = 0.01 # 'the clip bound of the gradients'
sigma = 1.5 # 'sigma'
delta = 1e-5 # 'delta'
sensitivity = clip_bound #adjacency matrix with one more tuple
target_eps = [4.0];
fgsm_eps = 0.2

## Robustness ##
dp_epsilon=4.0
dp_delta=0.05
attack_norm_bound = 0.2
#############################

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
        Returns:
        Variable Tensor
        """
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = cifar10._variable_on_cpu(name, shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in xrange(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features

def inference(images, params, dp_mult):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  
  ###
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  #xavier = tf.contrib.layers.xavier_initializer_conv2d()
  with tf.variable_scope('conv1') as scope:
    conv = tf.nn.conv2d(images, params[0], [1, 2, 2, 1], padding='SAME')
    #conv = tf.nn.dropout(conv, 0.9)
    pre_activation = tf.nn.bias_add(conv, params[1])
    conv1 = tf.nn.relu(pre_activation + dp_mult, name = scope.name)
    cifar10._activation_summary(conv1)
  
  norm1 = tf.contrib.layers.batch_norm(conv1, scale=True, is_training=True, updates_collections=None)

  # conv2
  with tf.variable_scope('conv2') as scope:
    conv = tf.nn.conv2d(norm1, params[2], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[3])
    conv2 = tf.nn.relu(pre_activation, name = scope.name)
    #conv2 = tf.nn.dropout(conv2, 0.9)
    cifar10._activation_summary(conv2)
  
  # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
  current = tf.concat((conv2, norm1), axis=3)
  # norm2
  norm2 = tf.contrib.layers.batch_norm(current, scale=True, is_training=True, updates_collections=None)

  # conv3
  with tf.variable_scope('conv3') as scope:
    conv = tf.nn.conv2d(norm2, params[4], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[5])
    conv3 = tf.nn.relu(pre_activation, name = scope.name)
    #conv3 = tf.nn.dropout(conv3, 0.9)
    cifar10._activation_summary(conv3)

  # norm3
  norm3 = tf.contrib.layers.batch_norm(conv3, scale=True, is_training=True, updates_collections=None)
  #pool3, row_pooling_sequence, col_pooling_sequence = tf.nn.fractional_max_pool(norm3, pooling_ratio=[1.0, 2.0, 2.0, 1.0])
  pool3 = avg_pool(norm3, 2)
    
  # local4
  with tf.variable_scope('local4') as scope:
    h_pool2_flat = tf.reshape(pool3, [-1, int(image_size/4)**2*256]);
    z2 = tf.add(tf.matmul(h_pool2_flat, params[6]), params[7], name=scope.name)
    #Applying normalization for the flat connected layer h_fc1#
    batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
    BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,params[11],params[10],1e-3)
    ###
    local4 = max_out(BN_norm, hk)
    cifar10._activation_summary(local4)
    
  """print(images.get_shape());
  print(norm1.get_shape());
  print(norm2.get_shape());
  print(pool3.get_shape());
  print(local4.get_shape());"""

  # linear layer(WX + b),
  # We don't apply softmax here because 
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits 
  # and performs the softmax internally for efficiency.
  softmax_linear = tf.add(tf.matmul(local4, params[8]), params[9], name=scope.name)
  cifar10._activation_summary(softmax_linear)
  return softmax_linear, conv1

def inference_test_input_probs(x, params, image_size):
    logits, _ = inference(x, params, 0)
    return tf.nn.softmax(logits)

def train(cifar10_data, logfile):
  """Train CIFAR-10 for a number of steps."""
  logfile.write("fgsm_eps \t %g, epsilon \t %d \n"%(fgsm_eps, target_eps[0]))
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Parameters Declarification
    #with tf.variable_scope('conv1') as scope:
    kernel1 = _variable_with_weight_decay('kernel1',
                                      shape=[3, 3, 3, 128],
                                      stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                      wd=0.0)
    biases1 = cifar10._variable_on_cpu('biases1', [128], tf.constant_initializer(0.0))
    #with tf.variable_scope('conv2') as scope:
    kernel2 = _variable_with_weight_decay('kernel2',
                                          shape=[5, 5, 128, 128],
                                          stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                          wd=0.0)
    biases2 = cifar10._variable_on_cpu('biases2', [128], tf.constant_initializer(0.1))
    #with tf.variable_scope('conv3') as scope:
    kernel3 = _variable_with_weight_decay('kernel3',
                                        shape=[5, 5, 256, 256],
                                        stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                        wd=0.0)
    biases3 = cifar10._variable_on_cpu('biases3', [256], tf.constant_initializer(0.1))
    #with tf.variable_scope('local4') as scope:
    kernel4 = cifar10._variable_with_weight_decay('kernel4', shape=[int(image_size/4)**2*256, hk], stddev=0.04, wd=0.004)
    biases4 = cifar10._variable_on_cpu('biases4', [hk], tf.constant_initializer(0.1))
    #with tf.variable_scope('local5') as scope:
    kernel5 = cifar10._variable_with_weight_decay('kernel5', [hk, 10],
                                        stddev=np.sqrt(2.0/(int(image_size/4)**2*256))/math.ceil(5 / 2), wd=0.0)
    biases5 = cifar10._variable_on_cpu('biases5', [10], tf.constant_initializer(0.1))
                                          
    scale2 = tf.Variable(tf.ones([hk]))
    beta2 = tf.Variable(tf.zeros([hk]))

    params = [kernel1, biases1, kernel2, biases2, kernel3, biases3, kernel4, biases4, kernel5, biases5, scale2, beta2]
    ########

    # Build a Graph that computes the logits predictions from the
    # inference model.
    shape     = kernel1.get_shape().as_list()
    w_t       = tf.reshape(kernel1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivityW = tf.reduce_max(sing_vals)
    dp_delta=0.05
    #dp_mult = attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon
    noise = tf.placeholder(tf.float32, [None, 28, 28, 32]);
    
    dp_mult = attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_delta)) / dp_epsilon
    noise = tf.placeholder(tf.float32, [None, 14, 14, 128]);
    x = tf.placeholder(tf.float32, [None,image_size,image_size,3]);
    #y_conv, h_conv1 = inference(x, params, dp_mult**2 * noise);
    y_conv, h_conv1 = inference(x, params, attack_norm_bound * noise);
    softmax_y_conv = tf.nn.softmax(y_conv)
    y_ = tf.placeholder(tf.float32, [None, 10]);
    
    #logits = inference(images)

    # Calculate loss. Apply Taylor Expansion for the output layer
    loss = cifar10.lossDPSGD(y_conv, y_)
    
    # noise redistribution #
    grad, = tf.gradients(loss, h_conv1)
    normalized_grad = tf.sign(grad)
    normalized_grad = tf.stop_gradient(normalized_grad)
    normalized_grad_r = tf.abs(tf.reduce_mean(normalized_grad, axis = (0)))**2
    sum_r = tf.reduce_sum(normalized_grad_r, axis = (0,1,2), keepdims=False)
    normalized_grad_r = 14*14*128*normalized_grad_r/sum_r
    print(normalized_grad_r)
    
    shape_grad     = normalized_grad_r.get_shape().as_list()
    grad_t       = tf.reshape(normalized_grad_r, [-1, shape_grad[-1]])
    g         = tf.transpose(grad_t)
    sing_g_vals = tf.svd(g, compute_uv=False)
    sensitivity_2 = tf.reduce_max(sing_g_vals)
    ########################
    
    opt = tf.train.GradientDescentOptimizer(lr)
    
    gw_K1 = tf.gradients(loss, kernel1)[0]
    gb1 = tf.gradients(loss, biases1)[0]
    
    gw_K2 = tf.gradients(loss, kernel2)[0]
    gb2 = tf.gradients(loss, biases2)[0]
    
    gw_K3 = tf.gradients(loss, kernel3)[0]
    gb3 = tf.gradients(loss, biases3)[0]
    
    gw_K4 = tf.gradients(loss, kernel4)[0]
    gb4 = tf.gradients(loss, biases4)[0]
    
    gw_K5 = tf.gradients(loss, kernel5)[0]
    gb5 = tf.gradients(loss, biases5)[0]
    
    #clip gradient
    gw_K1 = tf.clip_by_norm(gw_K1,clip_bound)
    gw_K2 = tf.clip_by_norm(gw_K2,clip_bound)
    gw_K3 = tf.clip_by_norm(gw_K3,clip_bound)
    gw_K4 = tf.clip_by_norm(gw_K4,clip_bound)
    gw_K5 = tf.clip_by_norm(gw_K5,clip_bound)
    
    #perturb
    gw_K1 += tf.random_normal(shape=tf.shape(gw_K1), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gw_K2 += tf.random_normal(shape=tf.shape(gw_K2), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gw_K3 += tf.random_normal(shape=tf.shape(gw_K3), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gw_K4 += tf.random_normal(shape=tf.shape(gw_K4), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gw_K5 += tf.random_normal(shape=tf.shape(gw_K5), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gb3 += tf.random_normal(shape=tf.shape(gb3), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gb4 += tf.random_normal(shape=tf.shape(gb4), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    gb5 += tf.random_normal(shape=tf.shape(gb5), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
    
    # apply gradients and keep tracking moving average of the parameters
    apply_gradient_op = opt.apply_gradients([(gw_K1,kernel1),(gb1,biases1),(gw_K2,kernel2),(gb2,biases2),(gw_K3,kernel3),(gb3,biases3),(gw_K4,kernel4),(gb4,biases4),(gw_K5,kernel5),(gb5,biases5)], global_step=global_step);
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = cifar10.trainDPSGD(loss, global_step, clip_bound, sigma, sensitivity)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    attack_switch = {'fgsm':True, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}
    
    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_input_probs, output_layer='probs', params=params, image_size=image_size)
    
    # define each attack method's tensor
    attack_tensor_dict = {}
    # FastGradientMethod
    if attack_switch['fgsm']:
        print('creating attack tensor of FastGradientMethod')
        fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_fgsm = fgsm_obj.generate(x=x, eps=fgsm_eps, clip_min=-1.0, clip_max=1.0, ord=2) # testing now
        x_adv_test_fgsm = fgsm_obj.generate(x=x, eps=fgsm_eps, clip_min=-1.0, clip_max=1.0) # testing now
        attack_tensor_dict['fgsm'] = x_adv_test_fgsm

    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    # default: eps_iter=0.05, nb_iter=10
    if attack_switch['ifgsm']:
        print('creating attack tensor of BasicIterativeMethod')
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/3, nb_iter=3, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['ifgsm'] = x_adv_test_ifgsm
    
    # MomentumIterativeMethod
    # default: eps_iter=0.06, nb_iter=10
    if attack_switch['mim']:
        print('creating attack tensor of MomentumIterativeMethod')
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_mim = mim_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, decay_factor=1.0, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_mim = mim_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/3, nb_iter=3, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['mim'] = x_adv_test_mim

    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # default: eps_iter=0.01, nb_iter=40
    if attack_switch['madry']:
        print('creating attack tensor of MadryEtAl')
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        #x_adv_test_madry = madry_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_madry = madry_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/3, nb_iter=3, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['madry'] = x_adv_test_madry
    #====================== attack =========================
    
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
    
    # Privacy accountant
    priv_accountant = accountant.GaussianMomentsAccountant(D)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], sigma, batch_size)
    
    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(os.getcwd() + path, sess.graph)
    
    # load the most recent models
    _global_step = 0
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path);
        saver.restore(sess, ckpt.model_checkpoint_path)
        _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found')
    
    T = int(int(math.ceil(D/batch_size))*epochs + 1) # number of steps
    step_for_epoch = int(math.ceil(D/batch_size)); #number of steps for one epoch

    s = math.log(sqrt(2.0/math.pi)*1e+5)
    sigmaEGM = sqrt(2.0)*1.0*(sqrt(s) + sqrt(s+dp_epsilon))/(2.0*dp_epsilon)
    #print(sigmaEGM)
    __noiseE = np.random.normal(0.0, sigmaEGM**2, 14*14*128).astype(np.float32)
    __noiseE = np.reshape(__noiseE, [-1, 14, 14, 128]);
    print("Compute The Noise Redistribution Vector")
    for step in xrange(_global_step, 100*step_for_epoch):
        batch = cifar10_data.train.next_batch(batch_size); #Get a random batch.
        _, loss_value = sess.run([train_op, loss], feed_dict = {x: batch[0], y_: batch[1], noise: __noiseE*0})
        if step % (5*step_for_epoch) == 0:
            print(loss_value)
    batch = cifar10_data.train.next_batch(40*batch_size);
    grad_redis = sess.run([normalized_grad_r], feed_dict = {x: batch[0], y_: batch[1], noise: __noiseE*0})
    _sensitivity_2 = sess.run([sensitivity_2], feed_dict={x: batch[0], y_: batch[1], noise: __noiseE*0})
    #print(_sensitivity_2)
    
    _sensitivityW = sess.run(sensitivityW)
    #print(_sensitivityW)
    Delta_redis = _sensitivityW/sqrt(_sensitivity_2[0])
    #print(Delta_redis)
    sigmaHGM = sqrt(2.0)*Delta_redis*(sqrt(s) + sqrt(s+dp_epsilon))/(2.0*dp_epsilon)
    #print(sigmaHGM)
    __noiseH = np.random.normal(0.0, sigmaHGM**2, 14*14*128).astype(np.float32)
    __noiseH = np.reshape(__noiseH, [-1, 14, 14, 128])*grad_redis;
    
    sess.run(init)
    print("Training")
    for step in xrange(_global_step, _global_step + T):
      start_time = time.time()
      batch = cifar10_data.train.next_batch(batch_size); #Get a random batch.
      #grad_redis = sess.run([normalized_grad_r], feed_dict = {x: batch[0], y_: batch[1], noise: (__noise + grad_redis)/2})
      _, loss_value = sess.run([train_op, loss], feed_dict = {x: batch[0], y_: batch[1], noise: (__noiseE + __noiseH)/2})
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      sess.run([privacy_accum_op])
      spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
      if step % (5*step_for_epoch) == 0:
        print(spent_eps_deltas)
      _break = False;
      for _eps, _delta in spent_eps_deltas:
        if _delta >= delta:
            _break = True;
            break;
      if _break == True:
        break;
    
    ## Robustness
    print("Testing")
    adv_acc_dict = {}
    robust_adv_acc_dict = {}
    robust_adv_utility_dict = {}
    test_bach_size = 5000
    for atk in attack_switch.keys():
        if atk not in adv_acc_dict:
            adv_acc_dict[atk] = -1
            robust_adv_acc_dict[atk] = -1
            robust_adv_utility_dict[atk] = -1
        if attack_switch[atk]:
            test_bach = cifar10_data.test.next_batch(test_bach_size)
            adv_images_dict = sess.run(attack_tensor_dict[atk], feed_dict ={x:test_bach[0]})
            ### PixelDP Robustness ###
            predictions_form_argmax = np.zeros([test_bach_size, 10])
            softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: (__noiseE + __noiseH)/2})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
            for n_draws in range(0, 1000):
                _noiseE = np.random.normal(0.0, sigmaEGM**2, 14*14*128).astype(np.float32)
                _noiseE = np.reshape(_noiseE, [-1, 14, 14, 128]);
                _noise = np.random.normal(0.0, sigmaHGM**2, 14*14*128).astype(np.float32)
                _noise = np.reshape(_noise, [-1, 14, 14, 128])*grad_redis;
                for j in range(test_bach_size):
                    pred = argmax_predictions[j]
                    predictions_form_argmax[j, pred] += 1;
                softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: (__noiseE + __noiseH)/2 + (_noiseE + _noise)/4})
                argmax_predictions = np.argmax(softmax_predictions, axis=1)
            final_predictions = predictions_form_argmax;
            is_correct = []
            is_robust = []
            for j in range(test_bach_size):
                is_correct.append(np.argmax(test_bach[1][j]) == np.argmax(final_predictions[j]))
                robustness_from_argmax = robustnessGGaussian.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=dp_epsilon, dp_delta=0.05, dp_mechanism='gaussian') / dp_mult
                is_robust.append(robustness_from_argmax >= fgsm_eps)
            adv_acc_dict[atk] = np.sum(is_correct)*1.0/test_bach_size
            robust_adv_acc_dict[atk] = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
            robust_adv_utility_dict[atk] = np.sum(is_robust)*1.0/test_bach_size
            ##############################
    log_str = "";
    for atk in attack_switch.keys():
        if attack_switch[atk]:
            # added robust prediction
            log_str += " {}: {:.4f} {:.4f} {:.4f} {:.4f}".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk]*robust_adv_utility_dict[atk])
    print(log_str)
    logfile.write(log_str + '\n')

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract();
  if tf.gfile.Exists('/tmp/cifar10_train'):
    tf.gfile.DeleteRecursively('/tmp/cifar10_train');
  tf.gfile.MakeDirs('/tmp/cifar10_train');
  cifar10_data = cifar10_read.read_data_sets("cifar-10-batches-bin/", one_hot = True);
  print('Done getting images')
  logfile = open('./tmp/results/IJCAICameraReady/DPSGD_HGM_' + str(target_eps[0]) + '_' + str(fgsm_eps) + '_PixelDP4.0_run2.txt','w')
  train(cifar10_data, logfile)
  logfile.flush()
  logfile.close();

if __name__ == '__main__':
  tf.app.run()
