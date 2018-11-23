from tensorflow.contrib import slim
import tensorflow as tf
import numpy as np



def network(inputs,reuse=None, is_training=True):
	with tf.variable_scope('Generator',reuse=reuse):
		def lrelu(x,alpha=0.2):
			return tf.maximum(x * alpha, x)
		
		def DCNx(DCNinputs=inputs,BlockSize=9,FilterCount=7,KernelSize=5,reuse=reuse):
			TensorList=[]
			x = slim.conv2d(inputs, FilterCount, [KernelSize, KernelSize], rate=1, activation_fn=tf.nn.leaky_relu,reuse=reuse, stride=1)
			TensorList.append(x)
			for i in range(BlockSize):
				x=tf.concat(TensorList,axis=3)
				x = slim.conv2d(inputs, FilterCount, [KernelSize, KernelSize], rate=1, activation_fn=tf.nn.leaky_relu,reuse=reuse, stride=1)
				
				if i == BlockSize-1:
					return x
				else:
					TensorList.append(x)
		
		with slim.arg_scope([slim.conv2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.001)):

			conv1 = slim.conv2d(inputs, 16, [3, 3], rate=1, activation_fn=lrelu,reuse=reuse, stride=1)
			conv1 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=lrelu,reuse =reuse, stride=2)
			conv1 =  slim.batch_norm(conv1, is_training=is_training)
			
			conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=2)
			conv2 =  slim.batch_norm(conv2, is_training=is_training)
			
			conv3 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=2)
			conv3 =  slim.batch_norm(conv3, is_training=is_training)
			
			conv4 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=2)
			conv4 =  slim.batch_norm(conv4, is_training=is_training)
			
			conv5 = slim.conv2d(conv4, 512, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			latent = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=2)



			up6 = tf.depth_to_space(latent,2)

			conv6 = slim.conv2d(up6, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv6 = slim.conv2d(conv6, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv6 = slim.batch_norm(conv6, is_training=is_training)

			up5=tf.concat([conv6,conv4],axis=3)
			up5 = tf.depth_to_space(up5,2)

			conv7 = slim.conv2d(up5, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv7 = slim.conv2d(conv7, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv7 = slim.batch_norm(conv7, is_training=is_training)

			up4=tf.concat([conv7,conv3],axis=3)
			up4 = tf.depth_to_space(up4,2)


			conv8 = slim.conv2d(up4, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv8 = slim.conv2d(conv8, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv8 = slim.batch_norm(conv8, is_training=is_training)

			up3 = tf.concat([conv8,conv2],axis=3)
			up3 =tf.depth_to_space(up3,2)

			conv9 = slim.conv2d(up3, 16,[3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv9 = slim.conv2d(conv9, 16, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv9 = slim.batch_norm(conv9, is_training=is_training)
			
			up2 = tf.concat([conv9,conv1],axis=3)
			up2 = tf.depth_to_space(up2,2)



			conv10 = slim.conv2d(up2, 12, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv10 = slim.conv2d(conv10, 12, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			up1 = tf.depth_to_space(conv10,2)

			conv11 = slim.conv2d(up1, 3, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse, stride=1)
			conv11 = slim.conv2d(conv11, 3, [3, 3], rate=1, activation_fn=tf.nn.tanh, reuse=reuse, stride=1)
		
			return conv11





