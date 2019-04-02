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
                train_label.append([1,0])
            if 1==int(row[1]):
                train_label.append([0,1])   
            
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
                valid_label.append([1,0])
            if 1==int(row[1]):
                valid_label.append([0,1])   
        print("load valid label ",len(valid_label))#, " of shape ", len(valid_label[0].shape))
        
    return train_data, train_label, valid_data, valid_label

def init_net(input_data):
    with tf.variable_scope('layer1'):
        weight = tf.Variable(tf.random_normal([7,3],stddev=1),name="w1")
        biase = tf.Variable(tf.random_normal([1,3])+ 0.1,name="b1") 
        L1 = tf.matmul(input_data,weight)+biase
        print("weight shape ",weight.shape)
        print("biase shape ",biase.shape)
    print("layer1 shape ",L1.shape)
    #L1 = tf.nn.relu(L1)
    with tf.variable_scope('layer2'):
        weight = tf.Variable(tf.random_normal([3,1],stddev=1),name="w2")
        biase = tf.Variable(tf.random_normal([1,2])+ 0.1,name="b2") 
        Y_pre = tf.matmul(L1,weight)+biase
        print("weight shape ",weight.shape)
        print("biase shape ",biase.shape)
    print("Y_pre shape ",Y_pre.shape)
    
    Y_pre = tf.nn.relu(Y_pre)
#    exit("stop")
    return Y_pre
def init_single_layer(input_data):
    with tf.variable_scope('layer1'):
        weight = tf.Variable(tf.random_normal([7,2],stddev=1),name="w1")
        biase = tf.Variable(tf.random_normal([1,2])+ 0.001,name="b1") 
        result = tf.matmul(input_data,weight)+biase
        pred = tf.nn.relu(result)  #softmax apply on one-hot
        print("weight shape ",weight.shape)
        print("biase shape ",biase.shape)
        print("pred shape ",pred.shape)
    return pred
#load data
train_data, train_label, valid_data, valid_label=loadData()

X = tf.placeholder(tf.float32, shape=[None,7], name="input")
Y = tf.placeholder(tf.float32, shape=[None,2], name="output")

Y_pro = init_net(X)
#Y_pro = init_single_layer(X)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y_pro - Y)))
#define accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_pro,1), tf.argmax(Y,1)), tf.float32))

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_data_feed = np.array(train_data)
    print(train_data_feed.shape)
    #print(train_data_feed[0:5])
    train_label_feed = np.array(train_label)
    print(train_label_feed.shape)
    #print(train_label_feed[0:5])
    print("training starts")
    epoches = 100000
    for e in range(epoches):
        sess.run(train_step, feed_dict={X:train_data_feed, Y:train_label_feed})
        if e%500==0:
            print("Epoch ", e)
            Y_pred, this_loss = sess.run([Y_pro, loss], feed_dict={X:train_data_feed, Y:train_label_feed})
            #print(Y_pred[0:5])
            print("loss ", this_loss)
        if e%1000==0:
            valid_data_feed = np.array(valid_data)
            valid_label_feed = np.array(valid_label)
            print("VALID : ")
            Y_pred_test, this_loss_test, acc_rate = sess.run([Y_pro, loss, accuracy], feed_dict={X:valid_data_feed, Y:valid_label_feed})
            #print(Y_pred_test)
            print("loss ", this_loss_test, "acc_rate ", acc_rate)

# todo output channel 1 or 2? that's why accuracy 100%
