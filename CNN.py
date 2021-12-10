from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
#from convolution import conv2d


import os
import tensorflow as tf
import numpy as np
import random
import math



class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 100

        self.input_size = 120*160*3
        self.num_classes = 16
        self.loss_list = np.empty(self.batch_size)
        self.optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-3)

        self.filter1 = tf.Variable(tf.random.normal([5, 5, 3, 16], stddev=.1))
        self.bi1 = tf.Variable(tf.random.normal([16]))

        self.filter2 = tf.Variable(tf.random.normal([5, 5, 16, 20], stddev=.1))
        self.bi2 = tf.Variable(tf.random.normal([20]))

        self.filter3 = tf.Variable(tf.random.normal([5, 5, 20, 20], stddev=.1))
        self.bi3 = tf.Variable(tf.random.normal([20]))


        self.W1 = tf.Variable(tf.random.truncated_normal([1600, 300], stddev=.1))
        self.b1 = tf.Variable(tf.random.normal([300], stddev=.1))

        self.W2 = tf.Variable(tf.random.truncated_normal([300, 30], stddev=.1))
        self.b2 = tf.Variable(tf.random.normal([30], stddev=.1)) #30

        self.W3 = tf.Variable(tf.random.truncated_normal([30, 100], stddev=.1))
        self.b3 = tf.Variable(tf.random.normal([100], stddev=.1))

        self.W4 = tf.Variable(tf.random.truncated_normal([100, 400], stddev=.1))
        self.b4 = tf.Variable(tf.random.normal([400], stddev=.1))

        self.W5 = tf.Variable(tf.random.truncated_normal([400, 16], stddev=.1))
        self.b5 = tf.Variable(tf.random.normal([16], stddev=.1))


    def call(self, inputs, is_testing=False):
        if is_testing:

            layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=[2,2], padding="SAME")
            layer1Output = tf.nn.bias_add(layer1Output, self.bi1)
            mean1, var1 = tf.nn.moments(layer1Output, axes=[0,1,2])
            layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean1, variance=var1, offset=None, scale=None, variance_epsilon= 1e-5)
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.max_pool(layer1Output, ksize=[3,3], strides=[2,2], padding="SAME")

            layer1Output = tf.nn.conv2d(layer1Output, self.filter2, strides=[2,2], padding="SAME")
            layer1Output = tf.nn.bias_add(layer1Output, self.bi2)
            mean2, var2 = tf.nn.moments(layer1Output, axes=[0,1,2])
            layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean2, variance=var2,offset=None, scale=None, variance_epsilon= 1e-5)
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.max_pool(layer1Output, ksize=[3,3], strides=[2,2], padding="SAME")

            layer1Output = tf.nn.conv2d(layer1Output, self.filter3, strides=[1,1], padding="SAME")
            layer1Output = tf.nn.bias_add(layer1Output, self.bi3)
            mean3, var3 = tf.nn.moments(layer1Output, axes=[0,1,2])
            layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean3, variance=var3, offset=None, scale=None, variance_epsilon= 1e-5)
            layer1Output = tf.nn.relu(layer1Output)

            layer1Output = tf.matmul(tf.reshape(layer1Output,[len(inputs), 1600]), self.W1) + self.b1
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.dropout(layer1Output, 0.1)

            layer1Output = tf.matmul(layer1Output, self.W2) + self.b2
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.dropout(layer1Output, 0.1)


            layer1Output = tf.matmul(layer1Output, self.W3) + self.b3
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.dropout(layer1Output, 0.1)

            layer1Output = tf.matmul(layer1Output, self.W4) + self.b4
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.dropout(layer1Output, 0.1)

            layer1Output = tf.matmul(layer1Output, self.W5) + self.b5
            layer1Output = tf.nn.relu(layer1Output)
            layer1Output = tf.nn.dropout(layer1Output, 0.1)

            layer1Output = tf.nn.softmax(layer1Output)

            return layer1Output

        layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=[2,2], padding="SAME")
        layer1Output = tf.nn.bias_add(layer1Output, self.bi1)
        mean1, var1 = tf.nn.moments(layer1Output, axes=[0,1,2])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean1, variance=var1, offset=None, scale=None, variance_epsilon= 1e-5)
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, ksize=[3,3], strides=[2,2], padding="SAME")

        layer1Output = tf.nn.conv2d(layer1Output, self.filter2, strides=[2,2], padding="SAME")
        layer1Output = tf.nn.bias_add(layer1Output, self.bi2)
        mean2, var2 = tf.nn.moments(layer1Output, axes=[0,1,2])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean2, variance=var2,offset=None, scale=None, variance_epsilon= 1e-5)
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, ksize=[3,3], strides=[2,2], padding="SAME")

        layer1Output = tf.nn.conv2d(layer1Output, self.filter3, strides=[1,1], padding="SAME")
        layer1Output = tf.nn.bias_add(layer1Output, self.bi3)
        mean3, var3 = tf.nn.moments(layer1Output, axes=[0,1,2])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean=mean3, variance=var3, offset=None, scale=None, variance_epsilon= 1e-5)
        layer1Output = tf.nn.relu(layer1Output)

        layer1Output = tf.matmul(tf.reshape(layer1Output,[len(inputs), 1600]), self.W1) + self.b1
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.dropout(layer1Output, 0.4)

        layer1Output = tf.matmul(layer1Output, self.W2) + self.b2
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.dropout(layer1Output, 0.1)

        layer1Output = tf.matmul(layer1Output, self.W3) + self.b3
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.dropout(layer1Output, 0.1)

        layer1Output = tf.matmul(layer1Output, self.W4) + self.b4
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.dropout(layer1Output, 0.1)

        layer1Output = tf.matmul(layer1Output, self.W5) + self.b5
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.dropout(layer1Output, 0.1)

        "NOTE: We do sigmoid instead of softmax: this is important for multiple labels"
        layer1Output = tf.nn.sigmoid(layer1Output)

        return layer1Output

    def loss(self, logits, labels):
        loss = tf.keras.losses.binary_crossentropy(labels, logits)
        loss = tf.reduce_mean(loss)
        
        return loss

    def accuracy(self, logits, labels):
        # pick any prediction above 50%
        one_zero = tf.where(logits > 0.5, 1, 0)
        correct_predictions = tf.where(labels == one_zero, 1, 0)

        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
