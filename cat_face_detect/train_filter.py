import os
import sys
import time
import random
import json
import cv2

import numpy as np
import tensorflow as tf

from os.path import join, exists
from glob import glob

train_dir = "test26_VFA/"
train_data_dir = "data-train/extend_train_VFA/new_criterion_2/"
test_data_dir = "data-test/VFA/new_criterrion/img-0"
saved_models ="test26_VFA/bestcheck/9833"
bestcheck_dir = join(train_dir, "bestcheck")
input_channels = 3
output_channels = 2
H = 192
W = 144
learning_rate = 0.0001
batch_size = 32
training_epoches = 100
is_train =False

def init_net(input_data):
    regularizers = 0
    with tf.name_scope('conv1')as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,input_channels,32], dtype=tf.float32, stddev=1e-1),name='kernel1')
        conv = tf.nn.conv2d(input_data, kernel, [1,1,1,1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='bias1')
        out = tf.nn.bias_add(conv, bias)
        conv1 = tf.nn.relu(out, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
    print "conv1 shape ", conv1.shape

    with tf.name_scope('pool1')as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    print "pool1 shape ", pool1.shape

    with tf.name_scope('conv2')as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,32,64], dtype=tf.float32, stddev=1e-1),name='kernel1')
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='bias1')
        out = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(out, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
    print "conv2 shape ", conv2.shape

    with tf.name_scope('pool2')as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    print "pool2 shape ", pool2.shape

    with tf.name_scope('conv3')as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,64,128], dtype=tf.float32, stddev=1e-1),name='kernel1')
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='bias1')
        out = tf.nn.bias_add(conv, bias)
        conv3 = tf.nn.relu(out, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
    print "conv3 shape ", conv3.shape

    with tf.name_scope('conv4')as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32, stddev=1e-1),name='kernel1')
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='bias1')
        out = tf.nn.bias_add(conv, bias)
        conv4 = tf.nn.relu(out, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
    print "conv4 shape ", conv4.shape

    with tf.name_scope('conv5')as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,128,64], dtype=tf.float32, stddev=1e-1),name='kernel1')
        conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='bias1')
        out = tf.nn.bias_add(conv, bias)
        conv5 = tf.nn.relu(out, name=scope)
        regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
    print "conv5 shape ", conv5.shape

    with tf.name_scope('pool3')as scope:
        pool3 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool3')
    print "pool3 shape", pool3.shape

    with tf.name_scope('flatten')as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        flatten = tf.reshape(pool3, [-1, shape])
        dropout = tf.nn.dropout(tf.nn.relu(flatten), 0.2)
    print "flatten shape", flatten.shape
    
    with tf.name_scope('fc1')as scope:
        shape = int(np.prod(dropout.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 256], dtype=tf.float32,stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
        fc1 = tf.nn.bias_add(tf.matmul(flatten, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc1w, "regularizer_loss"))  # 1/2 L2 distance
    print "fc1 shape", fc1.shape

    with tf.name_scope('fc2')as scope:
        shape = int(np.prod(fc1.get_shape()[1:]))
        fc2w = tf.Variable(tf.truncated_normal([shape, 128], dtype=tf.float32,stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
        fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2, name=scope)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc2w, "regularizer_loss"))  # 1/2 L2 distance
    print "fc2 shape", fc2.shape

    with tf.name_scope('fc3')as scope:
        shape = int(np.prod(fc2.get_shape()[1:]))
        fc3w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),trainable=True, name='biases')
        fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b, name=scope)
        #fc3 = tf.nn.relu(fc3, name=scope)
        #fc3 = tf.nn.sigmoid(fc3)
        regularizers += tf.reduce_sum(tf.nn.l2_loss(fc3w, "regularizer_loss"))  # 1/2 L2 distance
    print "fc3 shape", fc3.shape

    return fc3, regularizers

# def init_net2(input_data):
#     regularizers = 0
#     with tf.name_scope('conv1')as scope:
#         kernel = tf.Variable(tf.truncated_normal([3,3,input_channels,32], dtype=tf.float32, stddev=1e-1),name='kernel1')
#         conv = tf.nn.conv2d(input_data, kernel, [1,1,1,1], padding='SAME')
#         bias = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='bias1')
#         out = tf.nn.bias_add(conv, bias)
#         conv1 = tf.nn.relu(out, name=scope)
#         regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
#     print "conv1 shape ", conv1.shape

#     with tf.name_scope('pool1')as scope:
#         pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
#     print "pool1 shape ", pool1.shape

#     with tf.name_scope('conv2')as scope:
#         kernel = tf.Variable(tf.truncated_normal([3,3,32,32], dtype=tf.float32, stddev=1e-1),name='kernel1')
#         conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
#         bias = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='bias1')
#         out = tf.nn.bias_add(conv, bias)
#         conv2 = tf.nn.relu(out, name=scope)
#         regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
#     print "conv2 shape ", conv2.shape

#     with tf.name_scope('pool2')as scope:
#         pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
#     print "pool2 shape ", pool2.shape

#     with tf.name_scope('conv3')as scope:
#         kernel = tf.Variable(tf.truncated_normal([3,3,32,64], dtype=tf.float32, stddev=1e-1),name='kernel1')
#         conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
#         bias = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='bias1')
#         out = tf.nn.bias_add(conv, bias)
#         conv3 = tf.nn.relu(out, name=scope)
#         regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
#     print "conv3 shape ", conv3.shape

#     with tf.name_scope('pool3')as scope:
#         pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
#     print "pool3 shape ", pool3.shape  

#     with tf.name_scope('conv4')as scope:
#         kernel = tf.Variable(tf.truncated_normal([3,3,64,32], dtype=tf.float32, stddev=1e-1),name='kernel1')
#         conv = tf.nn.conv2d(conv3, kernel, [1,2,2,1], padding='SAME')
#         bias = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='bias1')
#         out = tf.nn.bias_add(conv, bias)
#         conv4 = tf.nn.relu(out, name=scope)
#         regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
#     print "conv4 shape ", conv4.shape

#     with tf.name_scope('conv5')as scope:
#         concat_Data = tf.concat([conv4, pool3], axis=3) # to check
#         kernel = tf.Variable(tf.truncated_normal([3,3,96,64], dtype=tf.float32, stddev=1e-1),name='kernel1')
#         conv = tf.nn.conv2d(concat_Data, kernel, [1,2,2,1], padding='SAME')
#         bias = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='bias1')
#         out = tf.nn.bias_add(conv, bias)
#         conv5 = tf.nn.relu(out, name=scope)
#         regularizers = tf.reduce_sum(tf.nn.l2_loss(kernel, "regularizer_loss"))
#     print "conv5 shape ", conv5.shape

#     with tf.name_scope('pool4')as scope:
#         pool4 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
#     print "pool4 shape ", pool4.shape  
   
#     with tf.name_scope('flatten')as scope:
#         shape = int(np.prod(pool4.get_shape()[1:]))
#         flatten = tf.reshape(pool4, [-1, shape])
#         dropout = tf.nn.dropout(tf.nn.relu(flatten), 0.2)
#     print "flatten shape", flatten.shape

#     with tf.name_scope('fc1')as scope:
#         shape = int(np.prod(dropout.get_shape()[1:]))
#         fc1w = tf.Variable(tf.truncated_normal([shape, 256], dtype=tf.float32,stddev=1e-1), name='weights')
#         fc1b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
#         fc1 = tf.nn.bias_add(tf.matmul(flatten, fc1w), fc1b)
#         fc1 = tf.nn.relu(fc1, name=scope)
#         regularizers += tf.reduce_sum(tf.nn.l2_loss(fc1w, "regularizer_loss"))  # 1/2 L2 distance
#     print "fc1 shape", fc1.shape

#     with tf.name_scope('fc2')as scope:
#         shape = int(np.prod(fc1.get_shape()[1:]))
#         fc2w = tf.Variable(tf.truncated_normal([shape, 128], dtype=tf.float32,stddev=1e-1), name='weights')
#         fc2b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
#         fc2 = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
#         fc2 = tf.nn.relu(fc2, name=scope)
#         regularizers += tf.reduce_sum(tf.nn.l2_loss(fc2w, "regularizer_loss"))  # 1/2 L2 distance
#     print "fc2 shape", fc2.shape

#     with tf.name_scope('fc3')as scope:
#         shape = int(np.prod(fc2.get_shape()[1:]))
#         fc3w = tf.Variable(tf.truncated_normal([shape, 2], dtype=tf.float32,stddev=1e-1), name='weights')
#         fc3b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),trainable=True, name='biases')
#         fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
#         regularizers += tf.reduce_sum(tf.nn.l2_loss(fc3w, "regularizer_loss"))  # 1/2 L2 distance
#     print "fc3 shape", fc3.shape

#     return fc3, regularizers

def compute_cost(Y_output, Y):  
    with tf.name_scope("SC_loss"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( #already contains logits, need input a linear result.
            logits=Y_output, labels=Y))

def read_names(data_dir, ratio):
    abnormalAddress = os.path.join(os.path.join(os.getcwd(),data_dir), 'data0')
    normalAddress = os.path.join(os.path.join(os.getcwd(),data_dir), 'data1')
    abnormalNames = os.listdir(abnormalAddress) 
    normalNames = os.listdir(normalAddress)
    abnormalNames.sort()
    normalNames.sort()
    normalNum = len(normalNames)
    abnormalNum = len(abnormalNames)

    #shuffle the names
    random.shuffle(normalNames)
    random.shuffle(abnormalNames)

    #extract valid dataset
    #read files and store in np
    normal_validnum = int(normalNum*ratio)
    abnormal_validnum = int(abnormalNum*ratio)
    normal_valid_names = normalNames[0:normal_validnum]
    normal_train_names = normalNames[normal_validnum:]
    abnormal_valid_names = abnormalNames[0:abnormal_validnum]
    abnormal_train_names = abnormalNames[abnormal_validnum:]

    for name in normal_valid_names:
        if name in normal_train_names:
            print name
            print "train and valid overlap"
    for name in abnormal_valid_names:
        if name in abnormal_train_names:
            print name
            print "train and valid overlap"

    normal_valid = []
    normal_label = []

    for eachName in normal_valid_names:
        img_path = os.path.join(normalAddress, eachName)
        img = cv2.imread(img_path)
        # if int(eachName[:-4])>1280:
        #     print "normal get wrong "
        #print img_path
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey (0)
        img = np.array(img)
        img = img/255.
        normal_valid.append(img)
        normal_label.append([0,1])

    abnormal_valid = []
    abnormal_label = []
    for eachName in abnormal_valid_names:
        img_path = os.path.join(abnormalAddress, eachName)
        # if int(eachName[:-4])<1280:
        #     print "abnormal get wrong "
        img = cv2.imread(img_path)
        img = np.array(img)
        img = img/255.
        abnormal_valid.append(img)
        abnormal_label.append([1, 0])

    normal_valid.extend(abnormal_valid)
    normal_label.extend(abnormal_label)
    valid_data =  np.array(normal_valid)
    valid_label =  np.array(normal_label)

    return valid_data, valid_label, normal_train_names, abnormal_train_names

def get_batch_data(data_dir, normalnames, abnormalnames, batch_size, batch_index):
    normalNum = len(normalnames)
    abnormalNum = len(abnormalnames)
    imgs_train_batch = []
    labels_train_batch = []
    i = 0
    #print batch_index,batch_size
    index_start = batch_index * (batch_size/2)
    #print index_start
    while i < batch_size / 2:
        #index = random.randint(0,normalNum-1)
        #print "normal : ", i + index_start, " : ", normalNum
        imgName = normalnames[i + index_start]
        imgAdd = os.path.join(os.path.join(data_dir,'data1'),imgName)
        img = cv2.imread(imgAdd)
        # print imgAdd
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey (0)
        img = np.array(img)
        img = img/255.
        imgs_train_batch.append(img)
        x = np.expand_dims(img,axis=0)
        labels_train_batch.append([0,1])
        i = i +1

    i = 0
    while i < batch_size / 2:
        index = random.randint(0,abnormalNum-1)
        imgName = abnormalnames[index]
        #print "abnormal : ", index, " : ", abnormalNum," name ",imgName
        imgAdd = os.path.join(os.path.join(data_dir,'data0'),imgName)
        if os.path.exists(imgAdd):
            img = cv2.imread(imgAdd)

        # print imgAdd
        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey (0)
        img = np.array(img)
        img = img/255.
        imgs_train_batch.append(img)
        labels_train_batch.append([1,0])
        i = i + 1
    
    #print labels_train_batch
    return np.array(imgs_train_batch), np.array(labels_train_batch)

def main(_):
    #define inputs, outputs, and net
    X = tf.placeholder(tf.float32, shape=(None, H, W, input_channels), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, output_channels), name="Y")
    y_pred, regularizers = init_net(X)

    #define cost function
    c_loss = compute_cost(y_pred, Y)
    loss = c_loss + 0.0001 * regularizers #add weights to prevent weight from changing huge

    #define accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(Y,1)), tf.float32))

    #define optimizer
    train_op = tf.train.AdamOptimizer(tf.train.exponential_decay(learning_rate, tf.Variable(0), 100, 0.2)).minimize(loss)
    #define initializer
    init_op = tf.global_variables_initializer()
    #defines a saver
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        start_time = time.time()
        each_time = start_time

        sess.run(init_op)

        if is_train:
            stopCount = 0
            earlysTopAcc = 0

            if not exists(bestcheck_dir):
                os.makedirs(bestcheck_dir)
            
            #extract valida dataset, labels and namelist of train data
            valid_data, valid_label, normal_train_names, abnormal_train_names = read_names(train_data_dir, 0.2)
            train_normalNum = len(normal_train_names)
            train_abnormalNum = len(abnormal_train_names)

            for epoch in range(training_epoches):
                epoch_cost = 0
                epoch_acc = 0
                batch_count = int((train_normalNum+ train_abnormalNum)/batch_size)
                #print "epoch :",epoch, "batch_count ",batch_count
                for b in range(batch_count):

                    imgs_train_batch, labels_train_batch = get_batch_data(train_data_dir, normal_train_names, abnormal_train_names, batch_size, b)
                    _, cost, acc = sess.run([train_op, c_loss, accuracy], feed_dict={X: imgs_train_batch, Y: labels_train_batch})
                    #print "batch index ", b, " cost ",cost, " acc ", acc
                    epoch_cost += cost
                    epoch_acc += acc

                epoch_acc_AVG = epoch_acc / batch_count
                epoch_cost_AVG = epoch_cost/batch_count
                #after all batch
                elapsed_time = time.time() - each_time
                each_time = time.time()
                
                #validating
                valid_loops = int(valid_data.shape[0] / batch_size)
                valid_cost = 0
                valid_acc = 0
                for v in range(valid_loops):
                    #print v, v*batch_size, ((v+1)*batch_size)
                    each_data = valid_data[v*batch_size:(v+1)*batch_size,]
                    each_label = valid_label[v*batch_size:(v+1)*batch_size,]
                    test_cost, test_acc = sess.run([c_loss, accuracy], feed_dict={X: each_data, Y: each_label})
                    #print "valid index ", v, " cost ",test_cost, " acc ", test_acc
                    valid_cost = valid_cost + test_cost
                    valid_acc = valid_acc + test_acc

                valid_acc = valid_acc / valid_loops
                valid_cost = valid_cost / valid_loops

                #save if this batch improved acc
                print 'earlystopacc %.6f '%(earlysTopAcc)
                if valid_acc < earlysTopAcc:
                    stopCount += 1
                    print 'Epoch: %2d|'%(epoch), ' Validation Accuracy %.6f'%(valid_acc),' not improved from %.6f ' %(earlysTopAcc)
                else:

                    stopCount = 0
                    save_path = saver.save(sess, join(bestcheck_dir, '%04d'%(valid_acc*10000)))
                    print "Epoch: %2d|"%(epoch), " AVG train Accuracy: %.06f|"%(epoch_acc_AVG), "Cost: %.4f|"%(epoch_cost_AVG),"Vliad-Accuracy: %.6f|"%(valid_acc) , "valid-Cost: %.4f|"%valid_cost, "early valid acc %.6f"%(earlysTopAcc)
                    earlysTopAcc = valid_acc
                if stopCount >=10:
                    break
        
        else:
            if saved_models==None:
                print "None trained model loaded!"
                exit(1)

            
            if saved_models!=None:
                print(' trained_models address:' , saved_models)
                saver.restore(sess, saved_models)
                print(' restore done %s'%saved_models)
            
            test_img0s = glob(join(test_data_dir, '*_0.jpg'))
            print(len(test_img0s))

            count_w = 0
            count_r = 0

            fw = []
            lw = []

            cnt = 0
            for test_img0 in test_img0s:
                img_name = test_img0.split('/')[-1].split('_')[0]

                imgs = []
                for j in range(9):
                    img_file = join(test_data_dir, img_name+'_%d.jpg'%(j))
                    img = cv2.imread(img_file)
                    img = cv2.resize(img, (48,64), interpolation=cv2.INTER_AREA)
                    imgs.append(img)    

                img_row1 = np.hstack([imgs[0], imgs[1], imgs[2]])
                img_row2 = np.hstack([imgs[3], imgs[4], imgs[5]])
                img_row3 = np.hstack([imgs[6], imgs[7], imgs[8]])

                tmp_img = np.vstack([img_row1, img_row2, img_row3])
                # tmp_img = np.vstack([img_row1, img_row2])
                tmp_img = np.expand_dims(tmp_img, axis=0)

                tmp_img = tmp_img/255.
                # test_y = sess.run([features], feed_dict={X: tmp_img})
                test_y = sess.run([y_pred], feed_dict={X: tmp_img})
                res = np.argmax(np.squeeze(test_y))
                if res!=1:
                    count_w += 1
                else:
                    count_r += 1

                cnt+=1
                print("normal %d error %d of %d | %.3f %.3f | result %d " %(count_r, count_w, cnt, test_y[0][0][0], test_y[0][0][1], res))
                # fw.append(np.array2string(np.squeeze(test_y), separator=' ', max_line_width=9999999).split('[')[-1].split(']')[0]+'\n')
                # lw.append('2\n')
            
            print count_w
            print count_r
            tf.train.write_graph(sess.graph_def, './', 'graph.pbtxt')
            # f = open('features.txt', 'a+')
            # f.writelines(fw)
            # f.close()
            # f = open('labels.txt', 'a+')
            # f.writelines(lw)
            # f.close()


        # #test
        # else:
        #     if trained_models==None:
        #         print "None trained model loaded!"
        #         exit(1)

        #     if trained_models!=None:
        #         print(' trained_models address:' , trained_models)
        #         saver.restore(sess, trained_models)
        #         print('restore done %s'%trained_models)
            
        #     abnormalAddress = os.path.join(os.path.join(os.getcwd(),test_dir), 'data0')
        #     normalAddress = os.path.join(os.path.join(os.getcwd(),test_dir), 'data1')
        #     abnormalNames = os.listdir(abnormalAddress) 
        #     normalNames = os.listdir(normalAddress)
        #     abnormalNames.sort()
        #     normalNames.sort()
        #     normalNum = len(normalNames)
        #     abnormalNum = len(abnormalNames)
        #     #shuffle the names
        #     random.shuffle(normalNames)
        #     random.shuffle(abnormalNames)

        #     #extract dataset
        #     normal_valid = []
        #     normal_label = []
        #     for eachName in normalNames:
        #         img_path = os.path.join(normalAddress, eachName)
        #         img = cv2.imread(img_path)
        #         # if int(eachName[:-4])>1280:
        #         #     print "normal get wrong "
        #         #print img_path
        #         # cv2.namedWindow("Image")
        #         # cv2.imshow("Image", img)
        #         # cv2.waitKey (0)
        #         img = np.array(img)
        #         img = img/255.
        #         normal_valid.append(img)
        #         normal_label.append([0,1])
        #     abnormal_valid = []
        #     abnormal_label = []
        #     for eachName in abnormalNames:
        #         img_path = os.path.join(abnormalAddress, eachName)
        #         # if int(eachName[:-4])<1280:
        #         #     print "abnormal get wrong "
        #         img = cv2.imread(img_path)
        #         img = np.array(img)
        #         img = img/255.
        #         abnormal_valid.append(img)
        #         abnormal_label.append([1, 0])

        #     count_w = 0
        #     count_r = 0
        #     for i in range(normalNum):
        #         input_img = np.expand_dims(normal_valid[i], axis=0)
                
        #         test_y = sess.run([y_pred], feed_dict={X:input_img})
        #         print test_y
        #         exit(0)
        #         res = np.argmax(np.squeeze(test_y))
        #         if res!=1:
        #             count_w += 1
        #         else:
        #             count_r += 1

        #         print("error %d noram %d | %.3f %.3f | result %d " %(count_w, count_r, test_y[0][0][0], test_y[0][0][1], res))
        #         exit(0)

        #     tf.train.write_graph(sess.graph_def, './', 'graph.pbtxt')

if __name__ == "__main__":
    tf.app.run()


