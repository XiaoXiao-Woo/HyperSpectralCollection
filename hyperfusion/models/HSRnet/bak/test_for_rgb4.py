#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
# This is a re-implementation of training code of this paper:
# J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "rgbNet: A deep network architecture for rgb-sharpening", ICCV,2017. 
# author: Junfeng Yang

"""
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import scipy.io as sio
import cv2
import os
import h5py
from PIL import Image
import time
import warnings
import matplotlib.pyplot as plt

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

def vis_ms(data):
    _, b, g, _, r, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = tf.split(data, 31,
                                                                                                           axis=3)
    vis = tf.concat([r, g, b], axis=3)
    return vis


def rgbNet(ms, RGB, num_spectral=31, num_res=3, num_fm=64, reuse=False):
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
            CA = tf.nn.softmax(CA)

        gap_RGB_s = tf.reduce_mean(RGB, [3], name='global_pool', keep_dims=True)
        SA = ly.conv2d(gap_RGB_s, num_outputs=1, kernel_size=5, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.sigmoid)

        sa = ly.conv2d(SA, 1, 5, 8, activation_fn=tf.nn.sigmoid,
                       weights_initializer=ly.variance_scaling_initializer(),
                       weights_regularizer=ly.l2_regularizer(weight_decay))
        rgb = ly.conv2d(RGB, 3, 5, 8, activation_fn=None,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
        # ms = ly.conv2d_transpose(ms, num_spectral, 8, 4, activation_fn=None,
        #                          weights_initializer=ly.variance_scaling_initializer(),
        #                          weights_regularizer=ly.l2_regularizer(weight_decay))
        rslice, gslice, bslice = tf.split(rgb, 3, axis=3)
        msp1, msp2 = tf.split(ms, [15, 16], axis=3)
        ms = tf.concat([rslice, msp1, gslice, msp2, bslice], axis=3)
        ms = sa * ms


        rs = ly.conv2d(ms, num_outputs=num_fm, kernel_size=5, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size=5, stride=1,rate=2,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            rs1 = ly.conv2d(rs1, num_outputs=num_fm, kernel_size=5, stride=1,rate=2,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
            rs = tf.add(rs, rs1)

        rs = ly.conv2d_transpose(rs, num_spectral, 5, 8, activation_fn=None,
                                 weights_initializer=ly.variance_scaling_initializer(),
                                 weights_regularizer=ly.l2_regularizer(weight_decay))

        Rslice, Gslice, Bslice = tf.split(RGB, 3, axis=3)
        Msp1, Msp2 = tf.split(rs, [15, 16], axis=3)
        rs = tf.concat([Rslice, Msp1, Gslice, Msp2, Bslice], axis=3)
        rs = SA * rs
        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=5, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)

        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size=5, stride=1,rate=2,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.leaky_relu)
            rs1 = ly.conv2d(rs1, num_outputs=num_fm, kernel_size=5, stride=1,rate=2,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
            rs = tf.add(rs, rs1)

        # 2D convolution
        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=5, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
        rs = CA*rs
        # 3D convolution
        # temp = tf.transpose(rs,[0,3,1,2])
        # ms = tms[:,:,:,:,tf.newaxis]
        # InitLayer = InputLayer(rs,name='InitLayer')
        # Layer1 = Conv3dLayer(InitLayer,act=tf.nn.leaky_relu, shape=(6, 6, 6, 1,1), strides=(1, 1, 1, 1, 1), padding='SAME',
        #                    W_init=tf.contrib.layers.xavier_initializer(), name='3Dlayer')
        # rs = Layer1.outputs
        # [num,_depth,height,width,channel] = rs.shape
        # rs = tf.reshape(rs,[num,depth,height,width])
        # rs = tf.transpose(rs,[0,2,3,1])

        return rs,CA,sa,SA


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # test_data = '/home/office/桌面/Machine Learning/Jin-Fan Hu/DATA/test_cave_11.mat'
    test_data = 'G:\JF_Hu/1-HSRnet\DATA/test/test10CSR.mat'

    model_directory = 'G:\JF_Hu/1-HSRnet/3_models\models(3res_slice_rgb+RGB)_SACA(detail_from_nohp)(dilation)'

    tf.reset_default_graph()

    data = h5py.File(test_data)
    # data = sio.loadmat(test_data)
    ms = data['ms'][...]  # MS image: 16x16x8
    ms = np.array(ms, dtype=np.float32) / (pow(2, 16) - 1)
    # ms = ms[np.newaxis,:,:,:]
    print(ms.shape)

    lms = data['lms'][...]  # up-sampled LRMS image: 64x64x8
    lms = np.array(lms, dtype=np.float32) / (pow(2, 16) - 1)
    # lms = lms[np.newaxis,:,:,:]
    rgb = data['rgb'][...]  # rgb image: 64x64x3
    rgb = np.array(rgb, dtype=np.float32) / (pow(2, 8) - 1)
    # rgb = rgb[np.newaxis,:,:,:]

    spectral_bands = ms.shape[3]

    N = rgb.shape[0]
    # batch_index = range(bs)
    batch_index = range(N)
    ms_hp = ms # high-frequency parts of MS image
    ms_hp = ms_hp[batch_index, :, :, :]

    rgb_hp = rgb # high-frequency parts of rgb image
    rgb_hp = rgb_hp[batch_index, :, :, :]
    lms = lms[batch_index, :, :, :]

    h = rgb.shape[1]  # height
    w = rgb.shape[2]  # width
    # print(h)
    # placeholder for tensor
    gt_p = tf.placeholder(shape=[N, h, w, spectral_bands], dtype=tf.float32)
    r_hp = tf.placeholder(shape=[N, h, w, 3], dtype=tf.float32)
    m_hp = tf.placeholder(shape=[N, h // 8, w // 8, 31], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[N, h, w, 31], dtype=tf.float32)

    [rs,CA,sa,SA] = rgbNet(m_hp, r_hp)  # output high-frequency parts

    mrs = tf.add(rs, lms_p)

    output = tf.clip_by_value(mrs, 0, 1)  # final output

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # loading  model
        ckpt = tf.train.latest_checkpoint(model_directory)
        saver.restore(sess, ckpt)
        print("load new model")

        time_s = time.time()
        [final_output,CA1,sa1,SA1] = sess.run([output,CA,sa,SA], feed_dict={ r_hp: rgb_hp, m_hp: ms_hp, lms_p: lms})
        time_e = time.time()
        print(time_e-time_s)
        sio.savemat('output-models(3res_slice_rgb+RGB)_SACA(detail_from_nohp)(dilation).mat', {'output': final_output,'sa':sa1,'SA':SA1,"CA":CA1})