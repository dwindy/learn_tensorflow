import tensorflow as tf 


a = tf.Variable([1,0,0,1,1])
b = tf.Variable([0,0,0,1,1])
c = a + b
equal = tf.equal(a,b)
cast = tf.cast(equal, tf.float32)
mean = tf.reduce_mean(cast)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(equal))
print(sess.run(cast))
print(sess.run(mean))