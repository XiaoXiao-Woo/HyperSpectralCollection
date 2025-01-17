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


def rgbNet(ms, RGB, num_spectral=31, num_res=6, num_fm=64, reuse=False):
    weight_decay = 1e-5
    with tf.variable_scope('net'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        rgb = ly.conv2d(RGB, 3, 6, 4, activation_fn=None,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
        # ms = ly.conv2d_transpose(ms, num_spectral, 8, 4, activation_fn=None,
        #                          weights_initializer=ly.variance_scaling_initializer(),
        #                          weights_regularizer=ly.l2_regularizer(weight_decay))

        ms = tf.concat([rgb, ms], axis=3)

        rs = ly.conv2d(ms, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.relu)
        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.relu)

        rs = ly.conv2d_transpose(rs, num_fm, 6, 4, activation_fn=None,
                                 weights_initializer=ly.variance_scaling_initializer(),
                                 weights_regularizer=ly.l2_regularizer(weight_decay))


        rs = tf.concat([RGB, rs], axis=3)

        # rs = tf.concat([RGB, rs], axis=3)
        rs = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.relu)

        for i in range(num_res):
            rs1 = ly.conv2d(rs, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=tf.nn.relu)
            rs1 = ly.conv2d(rs1, num_outputs=num_fm, kernel_size=3, stride=1,
                            weights_regularizer=ly.l2_regularizer(weight_decay),
                            weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
            rs = tf.add(rs, rs1)

        # 2D convolution
        rs = ly.conv2d(rs, num_outputs=num_spectral, kernel_size=3, stride=1,
                       weights_regularizer=ly.l2_regularizer(weight_decay),
                       weights_initializer=ly.variance_scaling_initializer(), activation_fn=None)
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

        return rs



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test_data = '/home/office/桌面/Machine Learning/Jin-Fan Hu/DATA/test_cave_11.mat'
    # test_data = '/home/office/桌面/Machine Learning/Jin-Fan Hu/DATA/test_harvard_new10.mat'

    model_directory = './Umodels/models(rgb+RGB-6res-64fm-l2)(simplec)-10w'

    tf.reset_default_graph()

    # data = h5py.File(test_data)
    data = sio.loadmat(test_data)
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
    ms_hp = get_edge(ms)  # high-frequency parts of MS image
    ms_hp = ms_hp[batch_index, :, :, :]

    rgb_hp = get_edge(rgb)  # high-frequency parts of rgb image
    rgb_hp = rgb_hp[batch_index, :, :, :]
    lms = lms[batch_index, :, :, :]

    h = rgb.shape[1]  # height
    w = rgb.shape[2]  # width
    # print(h)
    # placeholder for tensor
    gt_p = tf.placeholder(shape=[N, h, w, spectral_bands], dtype=tf.float32)
    r_hp = tf.placeholder(shape=[N, h, w, 3], dtype=tf.float32)
    m_hp = tf.placeholder(shape=[N, h // 4, w // 4, 31], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[N, h, w, 31], dtype=tf.float32)

    rs = rgbNet(m_hp, r_hp)  # output high-frequency parts

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
        final_output = sess.run(output, feed_dict={ r_hp: rgb_hp, m_hp: ms_hp, lms_p: lms})
        time_e = time.time()
        print(time_e-time_s)
        sio.savemat('output-cave_11(models(rgb+RGB-6res-64fm-l2)(simplec)-10w.mat', {'output': final_output})