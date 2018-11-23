import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

def Discriminator(Target,DarkInput,reuse=None,is_training=True):
	with tf.variable_scope('Discriminator',reuse=reuse):
		with slim.arg_scope([slim.conv2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.001)):

			def TargetConv(inputs=Target,reuse=reuse):
				"""
				The objective of this method is to simply take the Expected or 
				Gemerated target image and apply convolutions operations on it 
				adn bring it from a res of (None,16a,16b,3) to (None,a,b,1024)
				"""	

				conv1 = slim.conv2d(inputs, 16, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv1 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv1 = slim.batch_norm(conv1, is_training=is_training)
				
				conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv2 = slim.batch_norm(conv2, is_training=is_training)
				
				conv3 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv3 = slim.batch_norm(conv3, is_training=is_training)

				conv4 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv4 = slim.batch_norm(conv4, is_training=is_training)
				
				return conv4
			
			def DarkInputConv(inputs=DarkInput,reuse=reuse):
				"""
				The objective of this method is to simply take Dark image
				and apply convolutions operations on it 
				adn bring it from a res of (None,8a,8b,4) to (None,a,b,1024)
				"""	
				conv2 = slim.conv2d(inputs, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv2 = slim.batch_norm(conv2, is_training=is_training)
			
				conv3 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv3 = slim.batch_norm(conv3, is_training=is_training)
			
				conv4 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
				# conv4 = slim.batch_norm(conv4, is_training=is_training)
				return conv4
			
			def DCNx(DCNinputs,BlockSize=9,FilterCount=7,KernelSize=5,reuse=reuse):
				TensorList=[]
				x = slim.conv2d(DCNinputs, FilterCount, [KernelSize, KernelSize], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
				TensorList.append(x)
				for i in range(BlockSize):
					x = tf.concat(TensorList,axis=3)
					x = slim.conv2d(x, FilterCount, [KernelSize, KernelSize], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
					
					if i == BlockSize-1:
						return x
					else:
						TensorList.append(x)

			Target = TargetConv(inputs=Target,reuse=reuse)
			DarkInput = DarkInputConv(inputs=DarkInput,reuse=reuse)

			combined = tf.concat([Target,DarkInput],axis=3)

			dcn = DCNx(combined)

			conv5 = slim.conv2d(dcn, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
			conv5 = slim.conv2d(conv5, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
			# conv5 = slim.batch_norm(conv5, is_training=is_training)
			
			conv6 = slim.conv2d(conv5, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
			conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
			# conv6 = slim.batch_norm(conv6, is_training=is_training)
			
			conv7 = slim.conv2d(conv6, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=1)
			conv7 = slim.conv2d(conv7, 512, [3, 3], rate=1, activation_fn=tf.nn.leaky_relu, stride=2)
			# conv7 = slim.batch_norm(conv7, is_training=is_training)
			
			DenseLayer=tf.reduce_mean(conv7,axis=[1,2])
			DenseLayer=tf.layers.dense(inputs=DenseLayer,units=100,activation=tf.nn.leaky_relu)

			# DenseLayer = tf.layers.batch_normalization(DenseLayer)

			DenseLayer=tf.layers.dense(inputs=DenseLayer,units=1,activation=None)

			
			return DenseLayer,tf.nn.sigmoid(DenseLayer)


			



			
