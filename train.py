import tensorflow as tf
import numpy as np
import rawpy
import discriminator 
import generator
import json

class DarkGAN:
	"""docstring for ClassName"""
	def __init__(self):
		
		"""
		This is placeholder for the input for the Generator it is a4 channnel input
		"""
		self.GeneratorInput=tf.placeholder(tf.float32, [None, 512, 512, 4])
		
		"""
		The DiscriminatorLabelsFake is always an array of zeros expanded
		of the same sized as batch size

		For the DiscriminatorLabelsReal are an array of ones expanded 
		of the same sized as batch size

		The GeneratorLabels shall take the same labels as that of DiscriminatorRealLabels

		The RealImagePlaceholder is for when we need to feed the target image for the discriminnator

		"""
		self.DiscriminatorLabelsFake=tf.placeholder(tf.float32, [None, 1])
		self.DiscriminatorLabelsReal=tf.placeholder(tf.float32, [None, 1])
		self.GeneratorLabels=tf.placeholder(tf.float32, [None, 1])

		self.RealImagePlaceholder=tf.placeholder(tf.float32, [None, 512, 512, 3])
		self.GeneratedImage=generator.network(self.GeneratorInput)

		"""
		The discriminator generates 2 outputs , the activation and the logits = sigmoid(activation) =P(Input=FakeImage)
		This is generated for both when fake image is fed(logit expected to be 1),and realimage is fed(logit expected to be 0)

		"""

		self.DiscriminatorOutReal,self.DiscriminatorLogitsReal=discriminator.Discriminator(self.RealImagePlaceholder)
		self.DiscriminatorOutFake,self.DiscriminatorLogitsFake=discriminator.Discriminator(self.GeneratedImage,reuse=True)

		"""
		As the target and the computation graph are different when the generated image is fed or the true target 
		is, we compute 2 sigmoid crossentropy losses and the sum of both is the Final loss for the discriminnator

		Discriminatorloss=SigLoss(P(Discriminator thinks True target is correct),logit=1)+SigLoss(P(Discriminator thinks Generated Image is correct),logit =0)
		the first term is the RealLoss and the second term is FakeLoss

		"""
		
		self.DiscriminatorRealLoss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DiscriminatorOutReal, labels=self.DiscriminatorLabelsReal))
		self.DiscriminatorFakeLoss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DiscriminatorOutFake, labels=self.DiscriminatorLabelsFake))
		self.DiscriminatorLoss=tf.reduce_mean(self.DiscriminatorRealLoss+self.DiscriminatorFakeLoss)
		
		"""
		GeneratorLoss=SigLoss(P(Discriminator thinks Generated Image is correct),logit =1)

		"""
		self.GeneratorLoss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DiscriminatorLogitsFake, labels=self.GeneratorLabels))
		

		"""
		This is for the user to observe how well with time is the model able to generate images simillar to the target
		"""
		self.GeneratorMSE=tf.losses.absolute_difference(self.GeneratedImage,self.RealImagePlaceholder)

		self.TrainableVars=tf.trainable_variables()

		"""
		d_vars are the weights of the discriminator as all of them is under the scope of "Discriminator"
		This is necessary for allowing the optimizer to minimize the loss wrt discriminator weights

		g_vars are the weights of the generator as all of them is under the scope of "Generator"
		This is necessary for allowing the optimizer to minimize the loss wrt generator weights

		"""
		self.d_vars=[var for var in self.TrainableVars if 'Discriminator' in var.name]
		self.g_vars=[var for var in self.TrainableVars if 'Generator' in var.name] 
		

		"""
		In Soumith Chintala's GAN hacks tutorial, he sugest to feed all Generated and Real  Targets be fed
		in exclusicve batches the following 2 optimizers allows us to do the same
		They optimize Discriminator weights only
		"""
		self.DiscriminatorOptimizerReal=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorRealLoss,var_list=self.d_vars)
		self.DiscriminatorOptimizerFake=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorFakeLoss,var_list=self.d_vars)
		
		"""
		The followinng Optimizers are for minizing the Generator and Discriminator Loss wrt their weights
		"""
		self.GeneratorOptimizer=tf.train.AdamOptimizer(0.01).minimize(self.GeneratorLoss,var_list=self.g_vars)
		self.DiscriminatorOptimizer=tf.train.AdamOptimizer(0.01).minimize(self.DiscriminatorLoss,var_list=self.d_vars)
		
		self.Session = tf.Session()
		
		self.init_op= tf.initialize_all_variables()
		self.Session.run(self.init_op)
		self.BatchSize=1
		self.TrainSize=10
		self.HmEpochs=10
		with open('DoDtrain_exposures.json') as f:
			self.TrainDict = json.load(f)
	

	def pack_raw(self,raw):
		# pack Bayer image to 4 channels
		im = raw.raw_image_visible.astype(np.float32)
		im = np.maximum(im - 512, 0) / (16383 - 512)
		  # subtract the black level
		
		im = np.expand_dims(im, axis=2)
		img_shape = im.shape
		H = img_shape[0]
		W = img_shape[1]

		out = np.concatenate((im[0:H:2, 0:W:2, :],
							  im[0:H:2, 1:W:2, :],
							  im[1:H:2, 1:W:2, :],
							  im[1:H:2, 0:W:2, :]), axis=2)
		return out


	def FetchImage(self,key):
			
			image = self.pack_raw(rawpy.imread(key))
			shape=image.shape
			H,W=shape[0],shape[1]
			h,w=np.random.randint(low=0,high=H-513),np.random.randint(low=0,high=W-513)

			image=np.expand_dims(image,axis=0)
			image=image[:,h:h+512,w:w+512,:]

			image = self.TrainDict[key]["Exposure"]*image# * self.Exposure[self.Trainlist[i]]
			
			GT = rawpy.imread(self.TrainDict[key]["Target"])
			GT = GT.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=16)
			GT = np.expand_dims(np.float32(GT / 65535.0), axis=0)[:,h:h+512,w:w+512,:]
			return image,GT

			
	 			
	def train(self):
		for epoch in range(self.HmEpochs):
			
			for key in self.TrainDict.keys():
				image,brightimage=self.FetchImage(key)

				FakeLabels=np.expand_dims(np.ones(1),axis=1)
				RealLabels=np.expand_dims(np.zeros(1),axis=1)
				
				Discriminator_feed_dict={self.RealImagePlaceholder : brightimage ,self.GeneratorInput:image ,self.DiscriminatorLabelsReal:RealLabels,self.DiscriminatorLabelsFake:FakeLabels }
				_,DiscCost,logreal,logfake=self.Session.run([self.DiscriminatorOptimizerReal,self.DiscriminatorRealLoss,self.DiscriminatorLogitsReal,self.DiscriminatorLogitsFake],feed_dict=Discriminator_feed_dict)
				_,DiscCost,logreal,logfake=self.Session.run([self.DiscriminatorOptimizerFake,self.DiscriminatorRealLoss,self.DiscriminatorLogitsReal,self.DiscriminatorLogitsFake],feed_dict=Discriminator_feed_dict)
				
				GeneratorFeedDict={self.GeneratorLabels:RealLabels, self.GeneratorInput:image}
				_,GenCos=self.Session.run([self.GeneratorOptimizer,self.GeneratorLoss],feed_dict=GeneratorFeedDict)
				
				mseloss,im=self.Session.run([self.GeneratorMSE,self.GeneratedImage],feed_dict={self.RealImagePlaceholder:brightimage,self.GeneratorInput:image})
				print("ABS loss",mseloss)
				

dgan=DarkGAN()
dgan.train()



	

		
		