#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "rgbNet: A deep network architecture for rgb-sharpening", ICCV,2017.
# author: Junfeng Yang
"""

import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.layers as ly
import os
import h5py
import scipy.io as sio
import time
# import tensorlayer as tl
# from tensorlayer.layers import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], - 1, (6, 6))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (6, 6))
    return rs


def _phase_shift(I, r):
    bsize, w, h, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]
    X = tf.reshape(I, (bsize, w, h, r, r))
    X = tf.split(X, w, 1)  # 在w通道上分成了w份， 将每一维分成了1
    # tf.squeeze删除axis上的1，然后在第三通道 即r通道上 将w个小x重新级联变成r * w
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # 最终变成 bsize, h, r * w, r
    X = tf.split(X, h, 1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)

    return tf.reshape(X, (bsize, w * r, h * r, 1))  # 最后变成这个shape


def PS(X, r):
    #print("Input X shape:",X.get_shape(),"scale:",r)

    Xc = tf.split(X, 31, 3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)   #each of x in Xc is r * r channel 分别每一个通道变为r*r

    #print("output X shape:",X.get_shape())
    return X


def vis_ms(data):
    _, b, g, _, r, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tf.split(data, 31,
                                                                                                           axis=3)
    vis = tf.concat([r, g, b], axis=3)
    return vis


# rgbNet structures
def rgbNet(ms, RGB, num_spectral=31, num_res=6, num_fm=64, reuse=False):
    weight_decay = 1e-5

    with tf.variable_scope('net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        gap_ms_c = tf.reduce_mean(ms, [1, 2], name='global_pool', keep_dims=True)
        with tf.compat.v1.variable_scope('CA'):
            CA = ly.conv2d(gap_ms_c, num_outputs=1, kernel_size=1, stride=1,
                           weights_regularizer=ly.l2_regularizer(weight_decay),
                           weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            CA = ly.conv2d(CA, num_outputs=num_spectral, kernel_size=1, stride=1,
                           weights_regularizer=ly.l2_regularizer(weight_decay),
                           weights_initializer=tf.random_normal_initializer(), activation_fn=tf.nn.sigmoid)

        gap_RGB_s = tf.reduce_mean(RGB, [3], name='global_pool', keep_dims=True)
        SA = ly.conv2d(gap_RGB_s, num_outputs=1, kernel_size=6, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.sigmoid)
        sa = SA
        rgb = ly.conv2d(RGB, 3, 6, 8, activation_fn=None,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
        # ms = ly.conv2d_transpose(ms, num_spectral, 8, 4, activation_fn=None,
        #                          weights_initializer=ly.variance_scaling_initializer(),
        #                          weights_regularizer=ly.l2_regularizer(weight_decay))
        rslice, gslice, bslice = tf.split(rgb, 3, axis=3)
        msp1, msp2 = tf.split(ms, [15, 16], axis=3)
        ms = tf.concat([rslice, msp1, gslice, msp2, bslice], axis=3)

        rs = ly.conv2d(ms, num_outputs=num_spectral * 2 * 2, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        rs = PS(rs, 2)
        rs = ly.conv2d(rs, num_outputs=num_spectral * 2 * 2, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        rs = PS(rs, 2)
        rs = ly.conv2d(rs, num_outputs=num_spectral * 2 * 2, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        rs = PS(rs, 2)
        ########PS#########
        # rs = ly.conv2d(ms, num_outputs=num_spectral * 8 * 8, kernel_size=3, stride=1,
        #                weights_regularizer=ly.l2_regularizer(weight_decay),
        #                weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        # rs = PS(rs, 8)

        Rslice, Gslice, Bslice = tf.split(RGB, 3, axis=3)
        Msp1, Msp2 = tf.split(rs, [15, 16], axis=3)
        rs = tf.concat([Rslice, Msp1, Gslice, Msp2, Bslice], axis=3)

        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)

        for i in range( num_res):
            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            rs1 = ly.conv2d(rs1, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
            rs = tf.add(rs, rs1)

        # 2D convolution
        rs = SA * rs
        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)

        rs = CA * rs

        return rs,CA,sa,SA

def train():

    tf.reset_default_graph()

    train_batch_size = 32  # training batch size
    test_batch_size = 32  # validation batch size
    image_size = 80 # patch size
    bands = 31
    iterations = 150010  # total number of iterations to use.
    train_data_name = './DATA/trainCSRH.mat'  # training data
    test_data_name = './DATA/validationCSRH.mat'  # validation data
    # train_data_name = 'G:\JF_Hu\DATA/train_harvard20_3136.mat'  # training data
    # test_data_name = 'G:\JF_Hu\DATA/validation_harvard20_784.mat'  # validation data
    restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD

    ############## loading data
    # train_data = sio.loadmat(train_data_name)
    # test_data = sio.loadmat(test_data_name)

    train_data = h5py.File(train_data_name)  # for large data ( v7.3 data)
    # train_data = train_data['feature_data'][:]

    test_data = h5py.File(test_data_name)
    # test_data = test_data['feature_data'][:]

    ############## placeholder for training
    gt = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, bands])
    lms = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, bands])
    ms_hp = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size // 8, image_size // 8, bands])
    rgb_hp = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, image_size, image_size, 3])

    ############# placeholder for testing
    test_gt = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, bands])
    test_lms = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, bands])
    test_ms_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size // 8, image_size // 8, bands])
    test_rgb_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, 3])

    ######## network architecture
    mrs,_,_,_ = rgbNet(ms_hp, rgb_hp)
    mrs = tf.add(mrs, lms)  # last in the architecture: add two terms together

    test_rs,_,_,_ = rgbNet(test_ms_hp, test_rgb_hp, reuse=True)
    test_rs = test_rs + test_lms  # same as: test_rs = tf.add(test_rs,test_lms)

    ######## loss function
    mse = tf.reduce_mean(tf.abs(mrs - gt))
    test_mse = tf.reduce_mean(tf.abs(test_rs - test_gt))

    ##### Loss summary
    mse_loss_sum = tf.summary.scalar("mse_loss", mse)

    test_mse_sum = tf.summary.scalar("test_loss", test_mse)

    lms_sum = tf.summary.image("lms", tf.clip_by_value(vis_ms(lms), 0, 1))
    mrs_sum = tf.summary.image("rs", tf.clip_by_value(vis_ms(mrs), 0, 1))

    label_sum = tf.summary.image("label", tf.clip_by_value(vis_ms(gt), 0, 1))

    all_sum = tf.summary.merge([mse_loss_sum, mrs_sum, label_sum, lms_sum])

    #########   optimal    Adam or SGD

    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')

    if method == 'Adam':
        g_optim = tf.train.AdamOptimizer(0.0001, beta1=0.9) \
            .minimize(mse, var_list=t_vars)

    else:
        global_steps = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.1, global_steps, decay_steps=50000, decay_rate=0.1)
        clip_value = 0.1 / lr
        optim = tf.train.MomentumOptimizer(lr, 0.9)
        gradient, var = zip(*optim.compute_gradients(mse, var_list=t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient, clip_value)
        g_optim = optim.apply_gradients(zip(gradient, var), global_step=global_steps)

    ##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Run the above

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=11)
    with tf.Session() as sess:
        sess.run(init)

        if restore:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_directory)
            saver.restore(sess, ckpt.model_checkpoint_path)

        #### read training data #####
        gt1 = train_data['gt'][...]  ## ground truth N*H*W*C
        rgb1 = train_data['rgb'][...]  #### rgb image N*H*W
        ms_lr1 = train_data['ms'][...]  ### low resolution MS image
        lms1 = train_data['lms'][...]  #### MS image interpolation to rgb scale

        gt1 = np.array(gt1, dtype=np.float32) / (2 ** 16 - 1)  ### normalization, WorldView L = 11
        rgb1 = np.array(rgb1, dtype=np.float32) / (2 ** 8 - 1)
        ms_lr1 = np.array(ms_lr1, dtype=np.float32) / (2 ** 16 - 1)
        lms1 = np.array(lms1, dtype=np.float32) / (2 ** 16 - 1)

        N = gt1.shape[0]

        #### read validation data #####
        gt2 = test_data['gt'][...]  ## ground truth N*H*W*C
        rgb2 = test_data['rgb'][...]  #### rgb image N*H*W
        ms_lr2 = test_data['ms'][...]  ### low resolution MS image
        lms2 = test_data['lms'][...]  #### MS image interpolation -to rgb scale

        gt2 = np.array(gt2, dtype=np.float32) / (2 ** 16 - 1)  ### normalization, WorldView L = 11
        rgb2 = np.array(rgb2, dtype=np.float32) / (2 ** 8 - 1)
        ms_lr2 = np.array(ms_lr2, dtype=np.float32) / (2 ** 16 - 1)
        lms2 = np.array(lms2, dtype=np.float32) / (2 ** 16 - 1)
        N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []
        t_time = []
        time_s = time.time()

        for i in range(iterations):
            ###################################################################
            #### training phase! ###########################

            bs = train_batch_size
            batch_index = np.random.randint(0, N, size=bs)

            train_gt = gt1[batch_index, :, :, :]
            rgb_batch = rgb1[batch_index, :, :, :]
            ms_lr_batch = ms_lr1[batch_index, :, :, :]
            train_lms = lms1[batch_index, :, :, :]

            train_rgb_hp = rgb_batch # exrgbd to N*H*W*1

            train_ms_hp = ms_lr_batch

            # train_gt, train_lms, train_rgb_hp, train_ms_hp = get_batch(train_data, bs = train_batch_size)

            _, mse_loss, merged = sess.run([g_optim, mse, all_sum], feed_dict={gt: train_gt, lms: train_lms,
                                                                               ms_hp: train_ms_hp,
                                                                               rgb_hp: train_rgb_hp})

            if i % 1000 == 0:
                mse_train.append(mse_loss)  # record the mse of trainning
                print("Iter: " + str(i) + " MSE: " + str(mse_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609


            if i % 10000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/model-' + str(i) + '.ckpt')
                print("Save Model")

            ###################################################################
            #### compute the mse of validation data ###########################
            bs_test = test_batch_size
            batch_index2 = np.random.randint(0, N2, size=bs_test)

            if i % 2000 == 0 and i != 0:
                test_gt_batch = gt2[batch_index2, :, :, :]
                test_rgb_batch = rgb2[batch_index2, :, :, :]
                test_ms_lr_batch = ms_lr2[batch_index2, :, :, :]
                test_lms_batch = lms2[batch_index2, :, :, :]
                test_rgb_hp_batch = test_rgb_batch  #
                test_ms_hp_batch = test_ms_lr_batch
                # train_gt, train_lms, train_rgb, train_ms = get_batch(train_data, bs = train_batch_size)
                # test_gt_batch, test_lms_batch, test_rgb_batch, test_ms_batch = get_batch(test_data, bs=test_batch_size)
                test_mse_loss, merged = sess.run([test_mse, test_mse_sum],
                                                 feed_dict={test_gt: test_gt_batch, test_lms: test_lms_batch,
                                                            test_ms_hp: test_ms_hp_batch,
                                                            test_rgb_hp: test_rgb_hp_batch})
                mse_valid.append(test_mse_loss)  # record the mse of trainning
                print("Iter: " + str(i) + " Valid MSE: " + str(test_mse_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609

def test():
    # warnings.filterwarnings("ignore")
    # test_data = 'G:\JF_Hu\DATA/test_cave_11(pRGB).mat'
    test_data = 'G:\JF_Hu/1-HSRnet/DATA/test_harvard_10_v3(pRGB)scale8.mat'
    # test_data = 'G:\JF_Hu/1-HSRnet\DATA/test10CSR.mat'
    tf.reset_default_graph()
    N = 10
    sz = 1000
    OUT = np.zeros((N, sz, sz, 31))
    allCA = np.zeros((N, 1, 1, 31))
    allSA = np.zeros((N, sz, sz, 1))
    allHP = np.zeros((N, sz, sz, 31))
    r_hp = tf.placeholder(shape=[1, sz, sz, 3], dtype=tf.float32)
    m_hp = tf.placeholder(shape=[1, sz // 8, sz // 8, 31], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[1, sz, sz, 31], dtype=tf.float32)


    [rs, CA, sa, SA] = rgbNet(m_hp, r_hp)  # output high-frequency parts

    mrs = tf.add(rs, lms_p)

    output = tf.clip_by_value(mrs, 0, 1)  # final output



    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto() # 对session进行参数配置
    config.allow_soft_placement = True # 如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.7#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True #按需分配显存，这个比较重要


    with tf.Session(config=config) as sess:
        sess.run(init)

        # loading  model
        ckpt = tf.train.latest_checkpoint(model_directory)
        saver.restore(sess, ckpt)
        print("load new model")
        data = h5py.File(test_data)
        # data = sio.loadmat(test_data)
        Inms = data['ms']  # MS image: 16x16x8
        Inlms = data['lms'][...]  # up-sampled LRMS image: 64x64x8
        Inrgb = data['rgb1'][...]  # rgb image: 64x64x3
        for i in range(N):


            ms = Inms[i, :, :, :] / (2 ** 16 - 1)
            # ms = np.array(ms, dtype=np.float32) / (pow(2, 16) - 1)
            ms = ms[np.newaxis,:,:,:]
            print(ms.shape)


            lms = Inlms[i, :, :, :] / (2 ** 16 - 1)
            # lms = np.array(lms, dtype=np.float32) / (pow(2, 16) - 1)
            lms = lms[np.newaxis,:,:,:]


            rgb = Inrgb[i, :, :, :] / (2 ** 8 - 1)
            # rgb = np.array(rgb, dtype=np.float32) / (pow(2, 8) - 1)
            rgb = rgb[np.newaxis,:,:,:]

            ms_hp = ms  # high-frequency parts of MS image
            rgb_hp = rgb  # high-frequency parts of rgb image

            time_s = time.time()
            [final_output, CA1, SA1, HP] = sess.run([output, CA, SA, rs],
                                                     feed_dict={r_hp: rgb_hp, m_hp: ms_hp, lms_p: lms})
            time_e = time.time()
            print(time_e - time_s)
            OUT[i, :, :, :] = final_output
            allCA[i, :, :, :] = CA1
            allSA[i, :, :, :] = SA1
            allHP[i, :, :, :] = HP
            print('4200test' + str(i) + ' done!')
        sio.savemat('output-HSRnet(scale8-2x2x2)(Harvard)-harvard.mat',
            {'output': OUT, 'SA': allSA, "CA": allCA, 'HP': allHP})

if __name__ == '__main__':
    global model_directory
    model_directory = './3_models/HSRnet(scale8-2x2x2)(Harvard)'
    train()
    test()







