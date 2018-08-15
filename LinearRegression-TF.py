import tensorflow as tf
import numpy as np

#dummy data
xi = [-2,-1,0,1,2]
yi = [0,0,1,1,3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#model parameters
w = tf.Variable(1.0, name = "weight")
b = tf.Variable(2.0, name = "bias")

#model
Yt = w*X + b 

loss = tf.square(Y - Yt)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

numberIterations = 50

initialize = tf.global_variables_initializer()

with tf.Session() as sess:
    #initialize the variables
    sess.run(initialize) 
    for i in range(numberIterations):
        sess.run(optimizer, {X:xi , Y:yi})
    print(sess.run(w))
    print(sess.run(b))




    
    

