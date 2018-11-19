import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

def Discriminator(inputs,reuse=None):
	with tf.variable_scope('Discriminator',reuse=reuse):
		
		conv1 = slim.conv2d(inputs, 16, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv1 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu,reuse =reuse,scope='Discriminator')
		conv1 = tf.layers.batch_normalization(conv1)
		pool1=tf.space_to_depth(conv1,2)

		conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv2 = tf.layers.batch_normalization(conv2)
		pool2=tf.space_to_depth(conv2,2)

		conv3 = slim.conv2d(pool2, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv3 = tf.layers.batch_normalization(conv3)
		pool3=tf.space_to_depth(conv3,2)

		conv4 = slim.conv2d(pool3, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv4 = tf.layers.batch_normalization(conv4)
		pool4=tf.space_to_depth(conv4,2)
		

		conv5 = slim.conv2d(pool4, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv5 = slim.conv2d(conv5, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv5 = tf.layers.batch_normalization(conv5)
		pool5=tf.space_to_depth(conv5,2)

		conv6 = slim.conv2d(pool5, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv6 = slim.conv2d(conv6, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv6 = tf.layers.batch_normalization(conv6)
		pool6=tf.space_to_depth(conv6,2)

		conv7 = slim.conv2d(pool6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv7 = slim.conv2d(conv7, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu)
		conv7 = tf.layers.batch_normalization(conv7)
		pool7=tf.space_to_depth(conv7,2)

		DenseLayer=tf.reduce_mean(pool7,axis=[1,2])
		DenseLayer=tf.layers.dense(inputs=DenseLayer,units=100,activation=tf.nn.leaky_relu)
		DenseLayer = tf.layers.batch_normalization(DenseLayer)

		DenseLayer=tf.layers.dense(inputs=DenseLayer,units=1,activation=None)

		
		return DenseLayer,tf.nn.sigmoid(DenseLayer)


		



		
