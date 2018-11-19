from tensorflow.contrib import slim
import tensorflow as tf
import numpy as np

encoder_dict={"1":{"m":64,"n":64},"2":{"m":64,"n":128},"3":{"m":128,"n":256},"4":{"m":256,"n":512}}
decoder_dict={"1":{"n":64,"m":64},"2":{"n":64,"m":128},"3":{"n":128,"m":256},"4":{"n":256,"m":512}}

def Generator(inputs,classes=3,reuse=None):
	"""
	This is the primary method of this module.
	this block calls the pre defied methods for 
	the sub blocks of the ennocder and the decoder blocks.

	we make use of the encoder dict and the decoder dict 
	for specifying the umber of outpyt filters to be generated  for a given input

	for more info about the enncoder annd decoder dict please refer to the
	comments above.
	"""
	with tf.variable_scope('Generator',reuse=reuse):


		def lrelu(x,alpha=0.2):
			return tf.maximum(x * alpha, x)


		def encoder_level_2b(inputs,encoder_number,kernelsize=3):
			"""
			This is a straight forward convolution block of the subblock 
			of encoder
			"""
			conv_1b=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernelsize, stride=1, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			conv_1b=tf.layers.batch_normalization(conv_1b)
			conv_2b=slim.conv2d(conv_1b, encoder_dict[encoder_number]["n"], kernelsize, stride=1, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			conv_2b=tf.layers.batch_normalization(conv_2b)
			return conv_2b

		def encoder_level_2a(inputs,encoder_number,kernelsize=3):
			"""
			This is the residual block that later concatennates with the 
			output of the convolution block encoder level 2b
			"""
			conv_1a=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernel_size=kernelsize, stride=2, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			conv_1a=tf.layers.batch_normalization(conv_1a)
			conv_2a=slim.conv2d(conv_1a, encoder_dict[encoder_number]["n"], kernel_size=kernelsize,stride=1, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			conv_2a=tf.layers.batch_normalization(conv_2a)
			return conv_2a

		def encoder_level_1(inputs,encoder_number):
			"""
			The encoder is an implementation of a simple 
			residual convolution block 
			"""
			conv1=encoder_level_2a(inputs=inputs,encoder_number=encoder_number)
			input_downsampled=slim.conv2d(inputs, encoder_dict[encoder_number]["n"], kernel_size=3, stride=2, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			input_downsampled=tf.layers.batch_normalization(input_downsampled)
			conv1_combined=tf.concat([input_downsampled,conv1],axis=3)
			
			conv2=encoder_level_2b(conv1_combined,encoder_number=encoder_number)
			conv2_combined=tf.concat([conv1_combined,conv2],axis=3)
			return conv2_combined

		def decoder_block(inputs,decoder_number,kernelsize=3):
			"""
			This is a straight forward block that simply convolves and feeds forward 
			into the compute graph.
			"""
			
			dconv_1=slim.conv2d(inputs, decoder_dict[decoder_number]["m"]//4, kernel_size=1, stride=1, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			dconv_1=tf.layers.batch_normalization(dconv_1)
			dconv_2=tf.depth_to_space(dconv_1,2)
			dconv_2=tf.layers.batch_normalization(dconv_2)
			dconv_3=slim.conv2d(dconv_2,decoder_dict[decoder_number]["m"]//4,kernel_size=1,stride=1,rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			dconv_3=tf.layers.batch_normalization(dconv_3)

			return dconv_3

		def pre_encoder(inputs):
			"""
			This is the most fundamental feature extraction bblock used before feeding 
			the input to the encoder sub blocks.
			"""
			pre_enc_1=slim.conv2d(inputs,encoder_dict["1"]["m"], kernel_size=7, stride=2, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			pre_enc_2=slim.conv2d(pre_enc_1,encoder_dict["1"]["m"], kernel_size=3, stride=2, rate=1,padding='SAME',activation_fn=lrelu,reuse=reuse)
			return pre_enc_2
		def post_decoder(inputs,classes):
			"""
			The objective of this method is to simply brinng the 
			CNN output to the required filter ssize=channels  and also 
			apply relevannt activations to bound the output space of the outputs 
			to an 8bit image.
			"""

			post_decoder_dict={"a":{"m":64,"n":32},"b":{"m":32,"n":32},"c":{"m":32,"n":classes}}


			post_decoder_1=slim.conv2d_transpose(inputs,post_decoder_dict["a"]["n"],kernel_size=3,stride=2,padding='SAME',activation_fn=lrelu)
			post_decoder_1= tf.layers.batch_normalization(post_decoder_1)

			post_decoder_2=slim.conv2d(post_decoder_1,post_decoder_dict["b"]["n"], kernel_size=3, stride=1, rate=1,padding='SAME',activation_fn=lrelu)
			
			post_decoder_3=slim.conv2d_transpose(post_decoder_2,post_decoder_dict["c"]["n"],kernel_size=3,stride=2,padding='SAME',activation_fn=tf.nn.relu)
			
			return post_decoder_3



		input_1=pre_encoder(inputs)
		encoder_1=encoder_level_1(input_1,"1")

		print(encoder_1.get_shape())
		encoder_2=encoder_level_1(encoder_1,"2")
		print(encoder_2.get_shape())

		encoder_3=encoder_level_1(encoder_2,"3")
		print(encoder_3.get_shape())
		encoder_4=encoder_level_1(encoder_3,"4")
		print(encoder_4.get_shape())

		decoder_4=decoder_block(encoder_4,"4")
		decode_encode_concat4=tf.concat([encoder_3,decoder_4],axis=3)

		decoder_3=decoder_block(decode_encode_concat4,"3")
		decode_encode_concat3=tf.concat([encoder_2,decoder_3],axis=3)

		decoder_2=decoder_block(decode_encode_concat3,"2")
		decode_encode_concat2=tf.concat([encoder_1,decoder_2],axis=3)

		decoder_1=decoder_block(decode_encode_concat2,"1")
		
		output=post_decoder(decoder_1,classes)

		return output
  

def network(inputs,reuse=None):
	with tf.variable_scope('Generator',reuse=reuse):
		def lrelu(x,alpha=0.2):
			return tf.maximum(x * alpha, x)

		conv1 = slim.conv2d(inputs, 16, [3, 3], rate=1, activation_fn=lrelu,reuse=reuse)
		conv1 = slim.conv2d(conv1, 16, [3, 3], rate=1, activation_fn=lrelu,reuse =reuse)
		conv1 = tf.layers.batch_normalization(conv1)
		pool1=tf.space_to_depth(conv1,2)

		conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv2 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv2 = tf.layers.batch_normalization(conv2)
		pool2=tf.space_to_depth(conv2,2)

		conv3 = slim.conv2d(pool2, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv3 = slim.conv2d(conv3, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv3 = tf.layers.batch_normalization(conv3)
		pool3=tf.space_to_depth(conv3,2)

		conv4 = slim.conv2d(pool3, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv4 = slim.conv2d(conv4, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv4 = tf.layers.batch_normalization(conv4)
		pool4=tf.space_to_depth(conv4,2)

		conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		latent = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)


		up6 = tf.depth_to_space(latent,2)

		conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv6 = tf.layers.batch_normalization(conv6)

		up5=tf.concat([conv6,pool3],axis=3)
		up5 = tf.depth_to_space(up5,2)

		conv7 = slim.conv2d(up5, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv7 = tf.layers.batch_normalization(conv7)

		up4=tf.concat([conv7,pool2],axis=3)
		up4 = tf.depth_to_space(up4,2)


		conv8 = slim.conv2d(up4, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv8 = tf.layers.batch_normalization(conv8)

		conv9=tf.depth_to_space(conv8,2)

		conv10 = slim.conv2d(conv9, 16, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv10 = slim.conv2d(conv10, 16, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv10 = tf.layers.batch_normalization(conv10)
		conv10=tf.depth_to_space(conv10,2)


		conv11 = slim.conv2d(conv10, 3, [3, 3], rate=1, activation_fn=lrelu, reuse=reuse)
		conv11 = slim.conv2d(conv11, 3, [3, 3], rate=1, activation_fn=tf.nn.tanh, reuse=reuse)
		
		

		return conv11





