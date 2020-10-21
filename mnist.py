# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:12:30 2020

@author: shivam
"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
mnist_data=input_data.read_data_sets('MNIST_data',one_hot=True)
train_image=mnist_data.train.image[0]
train_label=mnist_data.train.label[0]
print(train_image)
print(train_label)