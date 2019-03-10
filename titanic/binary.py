import tensorflow as tf
from numpy.random import RandomState

#define batch size
batch_size = 8
#define variables of trainning
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#define inputs
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
#define propogating functions
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
#define loss function
#cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))

#define optimizing function
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

#generate a dataset ramdomly
rdm = RandomState(1)
#define dataset size
dataset_size = 128
#generate input data : 128 * 2
X = rdm.rand(dataset_size,2)
#generate inputs data: define x1 +x2 < 1 as positive sample
Y = [ [ int( x1 + x2 < 1) ] for (x1, x2) in X ]

#define session
with tf.Session() as sess:
    #initial all variables
    init = tf.initialize_all_variables()
    sess.run(init)
    #define training steps
    steps = 5000
    for i in range(steps):
        #select 128 samples in each epoch
        start = (i * batch_size)% dataset_size
        end = min(start + batch_size, dataset_size)
        print(X[start:end].shape)
        print(Y[start:end].shape)
        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        
        if i%1000 == 0:
            all_cost = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("epoch %d cost %g" %(i, all_cost))
    
    print(w1.eval(session=sess))
    print(w2.eval(session=sess))
            