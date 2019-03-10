# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:18:02 2019

@author: bingxin
"""

import tensorflow as tf
import csv
import os
import numpy as np

def loadData():
    filename = 'train.csv'
    train_data = []
    train_label = []
    if os.path.exists(filename):
        f = open(filename)
        reader = csv.reader(f)
#        print(list(reader)) 
        head = next(reader)
        print(head)
        for row in reader:
#            print(reader.line_num, row)
            if 0==int(row[1]):
                train_label.append([0])
            if 1==int(row[1]):
                train_label.append([1])   
            
            thisPerson = [0]*7
            #Pclass
            thisPerson[0]=int(row[2])
            #Gender
            #print(row[4])
            if 'male'==row[4]:
                thisPerson[1]=(0)
            if 'female'==row[4]:
                thisPerson[1]=(1)
            #Age
            if row[5]:
                thisPerson[2]=float(row[5])
            else:
                thisPerson[2]=(0)
            #SibSp
            thisPerson[3]=int(row[6])
            #Parch
            thisPerson[4]=int(row[7])
            #Fare
            thisPerson[5]=float(row[9])
            #Embarked
            if 'S'==row[11]:
                thisPerson[6]=1
            if 'C'==row[11]:
                thisPerson[6]=2
            if 'Q'==row[11]:
                thisPerson[6]=3
            if ''==row[11]:
                thisPerson[6]=0
            #print(thisPerson)
            train_data.append(thisPerson)
        print("load train data ",len(train_data), " of shape ", len(train_data[0]))
        print("load train label ",len(train_label))#, " of shape ", len(train_label[0]))
    
    valid_data = []
    valid_label = []
    filename = 'test.csv'
    if os.path.exists(filename):
        f = open(filename)
        reader = csv.reader(f)
        head = next(reader)
        print(head)
        for row in reader:
            thisPerson = [0]*7
            #Pclass
            thisPerson[0]=int(row[1])
            #Gender
            #print(row[4])
            if 'male'==row[3]:
                thisPerson[1]=(0)
            if 'female'==row[3]:
                thisPerson[1]=(1)
            #Age
            if row[4]:
                thisPerson[2]=float(row[4])
            else:
                thisPerson[2]=(0)
            #SibSp
            thisPerson[3]=int(row[5])
            #Parch
            thisPerson[4]=int(row[6])
            #Fare
            if row[8]:
                thisPerson[5]=float(row[8])
            else:
                thisPerson[5]=0
            #Embarked
            if 'S'==row[10]:
                thisPerson[6]=1
            if 'C'==row[10]:
                thisPerson[6]=2
            if 'Q'==row[10]:
                thisPerson[6]=3
            if ''==row[10]:
                thisPerson[6]=0
            #print(thisPerson)
            valid_data.append(thisPerson)
        print("load valid data ",len(valid_data), " of shape ", len(valid_data[0]))
    filename = 'gender_submission.csv'
    if os.path.exists(filename):
        f = open(filename)
        reader = csv.reader(f)
        head = next(reader)
        print(head)
        for row in reader:
#            print(reader.line_num, row)
            if 0==int(row[1]):
                valid_label.append([0])
            if 1==int(row[1]):
                valid_label.append([1])   
        print("load valid label ",len(valid_label))#, " of shape ", len(valid_label[0].shape))
        
    return train_data, train_label, valid_data, valid_label

#load data
train_data, train_label, valid_data, valid_label=loadData()

#define batch size
batch_size = 8
#define variables of trainning
w1 = tf.Variable(tf.random_normal([7,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#define inputs
x = tf.placeholder(tf.float32,shape=(None,7),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
#define propogating functions
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#define loss function
#cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))

#define optimizing function
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

##generate a dataset ramdomly
#rdm = RandomState(1)
##define dataset size
#dataset_size = 128
##generate input data : 128 * 2
#X = rdm.rand(dataset_size,2)
##generate inputs data: define x1 +x2 < 1 as positive sample
#Y = [ [ int( x1 + x2 < 1) ] for (x1, x2) in X ]

dataSize = len(train_data)

#define session
with tf.Session() as sess:
    #initial all variables
    init = tf.initialize_all_variables()
    sess.run(init)
    #define training steps
    steps = 5000
    for i in range(steps):
        #select 128 samples in each epoch
        start = (i * batch_size)% dataSize
        end = min(start + batch_size, dataSize)
        
        sess.run(train_step, feed_dict={x:train_data[start:end],y_:train_label[start:end]})
        
        if i%1000 == 0:
            all_cost = sess.run(cross_entropy, feed_dict={x:train_data, y_:train_label})
            print("epoch %d cost %g" %(i, all_cost))
    
    print(w1.eval(session=sess))
    print(w2.eval(session=sess))