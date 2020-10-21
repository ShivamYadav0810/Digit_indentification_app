# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:31:48 2020

@author: shivam
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#y=wx+b
#input to the graph, takes in any number of images(784 elements pixel array)
x_input=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x_input')
#weights to be multiplied by input
w=tf.Variable(initial_value=tf.zeroes(shape=[784,10],name='w'))
#Biaases to be added to wieghts * input
b=tf.Variable(initial_value=tf.zeroes(shape=[10]),name='b')
#actual mode prediction based on imput and current values of w and b
y_actual =tf.add(x=tf.matmul(a=x_input,b=w,name='matmul'),
                 y=b,name='y_actual')
#input to enter correct answer for comparison during training 
y_expected = tf.placeholder(dtype=tf.float32,shape=[None,10],name='y_expected')
#cross entropy loss function because output is a list of possibilities(% certainity of the correct answer
cross_entropy_loss=tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_expected,logits=y_actual),name='cross_entropy_loss')
#Classic gradient decent optimizer aims at minimizing the difference between expected and eactual values of loss
optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.5,name='optimizer')
train_step=optimizer.minimize(loss=cross_entropy_loss,name='train_step')
#Writing the graph
saver=tf.train.Saver()
session=tf.InteractiveSession();
session.run(tf.global_variables_initializer())
#Training the graph
tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='mnist_model.pdtxt',
                     as_text=False)
for _ in range(1000):
    batch=mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input:batch[0],y_expected:batch[1]})
correction_prediction =tf.equal(x=tf.argmax(y_actual,1,y=tf.float32))
saver.save(sess=session,
           save_path='mnist_model.ckpt')
accuracy=tf.reduce_mean(tf.cast(x=correction_prediction,dtype=tf.float32))
print(accuracy.eval(feed_dict={x_input:mnist_data.test.images,y_expected:mnist_data.test.labels}))
print(session.run(fetches=y_actual,feed_dict={x_input:[mnist_data.test.images[0]]}))
