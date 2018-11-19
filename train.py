import tensorflow as tf
import numpy as np
import rawpy
import discriminator 
import generator

class DarkGAN:
	"""docstring for ClassName"""
	def __init__(self):
		
		self.GeneratorInput=tf.placeholder(tf.float32, [None, 512, 512, 4])#placeholder
		
		self.DiscriminatorLabelsFake=tf.placeholder(tf.float32, [None, 1])#placeholder
		self.DiscriminatorLabelsReal=tf.placeholder(tf.float32, [None, 1])#placeholder
		self.GeneratorLabels=tf.placeholder(tf.float32, [None, 1])#placeholder

		self.RealImagePlaceholder=tf.placeholder(tf.float32, [None, 512, 512, 3])#

		
		self.GeneratedImage=generator.network(self.GeneratorInput)
		self.DiscriminatorOutReal,self.DiscriminatorLogitsReal=discriminator.Discriminator(self.RealImagePlaceholder)
		self.DiscriminatorOutFake,self.DiscriminatorLogitsFake=discriminator.Discriminator(self.GeneratedImage,reuse=True)
		
		self.DiscriminatorRealLoss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DiscriminatorOutReal, labels=self.DiscriminatorLabelsReal))
		self.DiscriminatorFakeLoss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DiscriminatorOutFake, labels=self.DiscriminatorLabelsFake))
		self.DiscriminatorLoss=tf.reduce_mean(self.DiscriminatorRealLoss+self.DiscriminatorFakeLoss)
		
		self.GeneratorLoss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.DiscriminatorLogitsFake, labels=self.GeneratorLabels))
		
		self.GeneratorMSE=tf.losses.mean_squared_error(self.GeneratedImage,self.RealImagePlaceholder)

		self.TrainableVars=tf.trainable_variables()
		self.d_vars=[var for var in self.TrainableVars if 'Discriminator' in var.name]
		self.g_vars=[var for var in self.TrainableVars if 'Generator' in var.name] 
		print(self.g_vars)
		
		self.DiscriminatorOptimizerReal=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorRealLoss,var_list=self.d_vars)
		self.DiscriminatorOptimizerFake=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorFakeLoss,var_list=self.d_vars)
		self.GeneratorOptimizer=tf.train.AdamOptimizer(0.01).minimize(self.GeneratorLoss,var_list=self.g_vars)
		self.DiscriminatorOptimizer=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorLoss,var_list=self.d_vars)
		self.Session = tf.Session()
		
		self.init_op= tf.initialize_all_variables()
		self.Session.run(self.init_op)
		self.BatchSize=1
		self.TrainSize=10
		#self.Trainlist
		self.HmEpochs=10
		"""
		declare the same for discriminator 
		too
		"""

	def pack_raw(raw):
		# pack Bayer image to 4 channels
		im = raw.raw_image_visible.astype(np.float32)
		im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

		im = np.expand_dims(im, axis=2)
		img_shape = im.shape
		H = img_shape[0]
		W = img_shape[1]

		out = np.concatenate((im[0:H:2, 0:W:2, :],
							  im[0:H:2, 1:W:2, :],
							  im[1:H:2, 1:W:2, :],
							  im[1:H:2, 0:W:2, :]), axis=2)
		return out


	def FetchImage(self):
		for i in range(self.TrainSize):
			"""
			image = rawpy.imread(self.TrainInputList[i])
			image = np.expand_dims(pack_raw(image), axis=0)# * self.Exposure[self.Trainlist[i]]
			
			GT = rawpy.imread(self.TrainGTList[i])
			GT = GT.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
			GT = np.expand_dims(np.float32(GT / 65535.0), axis=0)
			"""
			yield np.ones((1,512,512,4)),10*np.ones((1,512,512,3))
				
	def train(self):
		for epoch in range(self.HmEpochs):
			ImageBatch=self.FetchImage()
			for image,brightimage in ImageBatch:
				FakeLabels=np.expand_dims(np.ones(1),axis=1)
				RealLabels=np.expand_dims(np.zeros(1),axis=1)
				Discriminator_feed_dict={self.RealImagePlaceholder : brightimage ,self.GeneratorInput:image ,self.DiscriminatorLabelsReal:RealLabels,self.DiscriminatorLabelsFake:FakeLabels }
				_,DiscCost,log=self.Session.run([self.DiscriminatorOptimizer,self.DiscriminatorRealLoss,self.DiscriminatorLogitsReal],feed_dict=Discriminator_feed_dict)
				print(log)
				GeneratorFeedDict={self.GeneratorLabels:RealLabels, self.GeneratorInput:image}
				#_,GenCos=self.Session.run([self.GeneratorOptimizer,self.GeneratorLoss],feed_dict=GeneratorFeedDict)
				#print(GenCos)
				
				mseloss,im=self.Session.run([self.GeneratorMSE,self.GeneratedImage],feed_dict={self.RealImagePlaceholder:brightimage,self.GeneratorInput:image})
				print("MSE loss",mseloss)
				

dgan=DarkGAN()
dgan.train()



	

		
		