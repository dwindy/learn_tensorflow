"""
Created on Sun Mar 10 20:18:02 2019
used to detect cat face.
based on human face detector
ref:https://yinguobing.com/facial-landmark-localization-by-deep-learning-network-model/
@author: bingxin
"""

import tensorflow as tf
import os
import numpy as np

def init_net(input_data):
    with tf.variable_scope('layer1'):
        kernel1 = tf.Variable(tf.random_normal([3,3,3,32], stddev=1), name="kernel1")
        conv1 = tf.nn.conv2d(input_data, kernel1, strides = [1,1,1,1], padding='VALID')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=(2, 2), padding='valid', name="pool1")
        print("conv1 shape ", conv1.shape)
        print("pool1 shape ", pool1.shape)
    with tf.variable_scope('layer2'):
        kernel2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=1), name="kernel2")
        conv2 = tf.nn.conv2d(pool1, kernel2, strides = [1,1,1,1], padding='VALID')
        kernel3 = tf.Variable(tf.random_normal([3,3,64,64], stddev=1), name="kernel3")
        conv3 = tf.nn.conv2d(conv2, kernel3, strides = [1,1,1,1], padding='VALID')
        pool2 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=(2, 2), padding='valid', name="pool2")
        print("conv2 shape ", conv2.shape)
        print("conv3 shape ", conv3.shape)
        print("pool2 shape ", pool2.shape)
    with tf.variable_scope('layer3'):
        kernel4 = tf.Variable(tf.random_normal([3,3,64,64], stddev=1), name="kernel4")
        conv4 = tf.nn.conv2d(pool2, kernel4, strides = [1,1,1,1], padding='VALID')
        kernel5 = tf.Variable(tf.random_normal([3,3,64,64], stddev=1), name="kernel5")
        conv5 = tf.nn.conv2d(conv4, kernel5, strides = [1,1,1,1], padding='VALID')
        pool3 = tf.layers.max_pooling2d(conv5, pool_size=[2, 2], strides=(1, 1), padding='valid', name="pool3")
        print("conv4 shape ", conv4.shape)
        print("conv5 shape ", conv5.shape)
        print("pool3 shape ", pool3.shape)
    with tf.variable_scope('layer4'):
        kernel6 = tf.Variable(tf.random_normal([3,3,64,128], stddev=1), name="kernel6")
        conv6 = tf.nn.conv2d(pool3, kernel6, strides = [1,1,1,1], padding='VALID')
        kernel7 = tf.Variable(tf.random_normal([3,3,128,128], stddev=1), name="kernel7")
        conv7 = tf.nn.conv2d(conv6, kernel7, strides = [1,1,1,1], padding='VALID')
        pool4 = tf.layers.max_pooling2d(conv7, pool_size=[2, 2], strides=(1, 1), padding='valid', name="pool4")
        print("conv6 shape ", conv6.shape)
        print("conv7 shape ", conv7.shape)
        print("pool4 shape ", pool4.shape)
    with tf.variable_scope('layer5'):
        kernel8 = tf.Variable(tf.random_normal([3,3,128,256], stddev=1), name="kernel8")
        conv8 = tf.nn.conv2d(pool4, kernel8, strides = [1,1,1,1], padding='VALID')
        flatten = tf.reshape(conv8, [-1,228096])
        #flatten = tf.layers.flatten(conv8, name="flatten")
        print("conv8 shape ", conv8.shape)
        print("flatten shape ", flatten.shape)
    with tf.variable_scope('layer6'):
        dense1 = tf.layers.dense(flatten, units=1024, activation = tf.nn.relu, use_bias=True)
        dense2 = tf.layers.dense(dense1, units=136, activation = None, use_bias=True, name="logits")
        print("dense1 shape ", dense1.shape)
        print("dense2 shape ", dense2.shape)
    return dense2


image = tf.placeholder(tf.float32, shape=[None, 192, 168, 3], name='input_image_tensor')
pred = init_net(image)