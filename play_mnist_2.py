import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Training data size:", mnist.train.num_examples)
x = tf.placeholder(tf.float32, [None, 784], name="X")
y_ = tf.placeholder(tf.float32, [None, 10], name="Y")


def init_net(input_data):
    regularizers = 0
    with tf.name_scope('conv1') as scope:
        x_image = tf.reshape(input_data, [-1, 28, 28, 1])
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 1, 6], dtype=tf.float32, stddev=1e-1), name='kernel1')
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[6], dtype=tf.float32), trainable=True, name='biases1')
        result = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(result, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 6, 12], dtype=tf.float32, stddev=1e-1, name='kernel2'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[12], dtype=tf.float32), trainable=True, name='biases2')
        result = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(result, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(kernel,
                                                    "regularizer_loss"))

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 12, 24], dtype=tf.float32, stddev=1e-1, name='kernel2'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[24], dtype=tf.float32), trainable=True, name='biases3')
        result = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(result, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(kernel,
                                                    "regularizer_loss"))
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool2.get_shape()[1:]))
        #shape = pool2 wid * hei * layers = 14 * 14 * 6 = 1176
        fc1w = tf.Variable(tf.truncated_normal([shape, 128], dtype=tf.float32, stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32), trainable=True, name = 'biases')
        pool4_flat = tf.reshape(pool2, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool4_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc1w, "regularizer_loss"))

    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([128,256], dtype=tf.float32, stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32), trainable = True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc2w, "regularizer_loss"))
    
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([256, 10], dtype=tf.float32,
                           stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                                trainable=True, name='biases')
        fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc3w, "regularizer_loss"))

    return fc3, regularizers, fc2

def compute_cost(Y_output, Y):
    with tf.name_scope("SC_loss"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=Y_output, labels=Y))

Y_pro, w_loss, features = init_net(x)

c_loss = compute_cost(Y_pro, y_)

loss = c_loss + 0.0001 * w_loss

accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(Y_pro, 1), tf.argmax(y_, 1)), tf.float32))

train_op = tf.train.AdamOptimizer(tf.train.exponential_decay(
    0.0002, tf.Variable(0), 100, 0.98)).minimize(loss)

saver = tf.train.Saver()

def train():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        _, cost, acc = sess.run([train_op, c_loss, accuracy], feed_dict={x:batch[0], y_:batch[1]})
        if i%100 == 0:
            print ("step %d, cost %g , training accuracy %g" % (i, cost, acc))
            saver.save(sess, 'save_model/new')
    writer = tf.summary.FileWriter("tensorboard/",sess.graph)
    writer.close()
def predict():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'save_model/new')
    print( "test accuracy %g" % accuracy.eval(feed_dict={
        x:mnist.test.images, y_:mnist.test.labels}))

train()
#predict()
