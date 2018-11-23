import tensorflow as tf
import numpy as np
import rawpy
import discriminator 
import generator
import json
import matplotlib.pyplot as plt
from scipy import misc

class DarkGAN:
	"""docstring for ClassName"""
	def __init__(self):
		
		self.IsTraining = tf.placeholder(tf.bool)

		"""
		This is placeholder for the input for the Generator it is a4 channnel input
		"""
		self.GeneratorInput=tf.placeholder(tf.float32, [None, None, None, 4])
		
		"""
		The DiscriminatorLabelsFake is always an array of zeros expanded
		of the same sized as batch size

		For the DiscriminatorLabelsReal are an array of ones expanded 
		of the same sized as batch size

		The GeneratorLabels shall take the same labels as that of DiscriminatorRealLabels

		The TargetImagePlaceholder is for when we need to feed the target image for the discriminnator

		"""
		self.DiscriminatorLabelsFake=tf.placeholder(tf.float32, [None, 1])
		self.DiscriminatorLabelsReal=tf.placeholder(tf.float32, [None, 1])
		self.GeneratorLabels=tf.placeholder(tf.float32, [None, 1])

		self.TargetImagePlaceholder=tf.placeholder(tf.float32, [None, None, None, 3])
		self.GeneratedImage=generator.network(self.GeneratorInput,is_training=self.IsTraining)

		"""
		The discriminator generates 2 outputs , the activation and the logits = sigmoid(activation) =P(Input=FakeImage)
		This is generated for both when fake image is fed(logit expected to be 1),and realimage is fed(logit expected to be 0)

		"""

		self.DiscriminatorOutReal,self.DiscriminatorLogitsReal=discriminator.Discriminator(Target=self.TargetImagePlaceholder,DarkInput=self.GeneratorInput)
		self.DiscriminatorOutFake,self.DiscriminatorLogitsFake=discriminator.Discriminator(Target=self.GeneratedImage,DarkInput=self.GeneratorInput,reuse=True)

		"""
		As the target and the computation graph are different when the generated image is fed or the true target 
		is, we compute 2 sigmoid crossentropy losses and the sum of both is the Final loss for the discriminnator

		Discriminatorloss=SigLoss(P(Discriminator thinks True target is correct),logit=1)+SigLoss(P(Discriminator thinks Generated Image is correct),logit =0)
		the first term is the RealLoss and the second term is FakeLoss

		"""
		
		self.DiscriminatorRealLoss=tf.reduce_mean(self.cross_entropy(logits=self.DiscriminatorLogitsReal, labels=self.DiscriminatorLabelsReal))
		self.DiscriminatorFakeLoss=tf.reduce_mean(self.cross_entropy(logits=self.DiscriminatorLogitsFake, labels=self.DiscriminatorLabelsFake))
		self.DiscriminatorLoss=tf.reduce_mean(self.DiscriminatorRealLoss+self.DiscriminatorFakeLoss)
		
		"""
		GeneratorLoss=SigLoss(P(Discriminator thinks Generated Image is correct),logit =1)

		"""
		self.GeneratorLoss=tf.reduce_mean(self.cross_entropy(logits=self.DiscriminatorLogitsFake, labels=self.GeneratorLabels))
		

		"""
		This is for the user to observe how well with time is the model able to generate images simillar to the target
		"""
		self.GeneratorABS=tf.losses.absolute_difference(self.GeneratedImage,self.TargetImagePlaceholder)

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
		self.DiscriminatorOptimizerReal=tf.train.AdamOptimizer(0.00001).minimize(self.DiscriminatorRealLoss,var_list=self.d_vars)
		self.DiscriminatorOptimizerFake=tf.train.AdamOptimizer(0.00001).minimize(self.DiscriminatorFakeLoss,var_list=self.d_vars)
		
		"""
		The followinng Optimizers are for minizing the Generator and Discriminator Loss wrt their weights
		"""
		self.gen_loss_lambda1 = 1.0
		self.gen_loss_lambda2 = 10.0
		self.GeneratorOptimizer=tf.train.AdamOptimizer(0.00001).minimize(self.GeneratorLoss+\
			self.gen_loss_lambda1*self.GeneratorABS,var_list=self.g_vars)
		self.GeneratorGradients=tf.train.AdamOptimizer(0.00001).compute_gradients(self.GeneratorLoss,var_list=self.g_vars)
		self.DiscriminatorOptimizer=tf.train.AdamOptimizer(0.00001).minimize(self.DiscriminatorLoss,var_list=self.d_vars)
		
		self.Session = tf.Session()
		self.saver=tf.train.Saver()
		
		self.init_op= tf.initialize_all_variables()
		self.Session.run(self.init_op)
		self.BatchSize=1
		self.TrainSize=10
		self.HmEpochs=10
		self.PatchSize=64
		self.save_path="./models/"
		with open('DoDtrain_exposures.json') as f:
			self.TrainDict = json.load(f)
		with open('DoDval_exposures.json') as f:
			self.ValidDict = json.load(f)
		if tf.train.latest_checkpoint(self.save_path) is not None:
			self.checkpoint = tf.train.latest_checkpoint(self.save_path)
			self.saver.restore(self.Session,self.checkpoint)	

	def cross_entropy(self,logits,labels,K=0.99999):
		logits = logits*K +(1-K)/2
		loss = -tf.multiply(labels,tf.log(logits)) - tf.multiply((1-labels),tf.log(1-logits))
		return loss

	def pack_raw(self,raw):
		# pack Bayer image to 4 channels
		im = raw.raw_image_visible.astype(np.float32)
		im = np.maximum(im - 512, 0) / (16383 - 512)
		  # subtract the black level
		
		im = np.expand_dims(im, axis=2)
		img_shape = im.shape
		H = img_shape[0]
		W = img_shape[1]
		# print("Shape of image",img_shape)
		out = np.concatenate((im[0:H:2, 0:W:2, :],
							  im[0:H:2, 1:W:2, :],
							  im[1:H:2, 1:W:2, :],
							  im[1:H:2, 0:W:2, :]), axis=2)
		return out


	def FetchImage(self,key,patch=True):
			
			image = np.array(self.pack_raw(rawpy.imread(key))*float(self.TrainDict[key]["Exposure"]))
			shape=image.shape
			H,W=shape[0],shape[1]
			print("H",H,"  W",W)
			if(patch==True):
				h,w=np.random.randint(low=0,high=H-self.PatchSize-1,size=self.BatchSize),np.random.randint(low=0,high=W-self.PatchSize-1,size=self.BatchSize)
			else:
				h,w=np.random.randint(low=0,high=H-self.PatchSize-1),np.random.randint(low=0,high=W-self.PatchSize-1)

			if(patch==True):
				image = np.array([image[h[i]:h[i]+self.PatchSize,w[i]:w[i]+self.PatchSize,:] for i in range(self.BatchSize)])
			else:
				image = np.expand_dims(image,axis=0)
			print(np.shape(image))
			#image = np.multiply(exps,image,axis=0)# * self.Exposure[self.Trainlist[i]]
			
			GT = rawpy.imread(self.TrainDict[key]["Target"])
			GT = GT.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
			GT = np.float32(np.array(GT) / float(65535.0))
			if(patch==True):
				GT = np.array([GT[2*h[i]:2*h[i]+2*self.PatchSize,2*w[i]:2*w[i]+2*self.PatchSize,:] for i in range(self.BatchSize)])
			else:
				GT = np.expand_dims(GT,axis=0)
			print(np.shape(GT))
			return image,GT

			
	 			
	def train(self):
		counter=0
		gen_loss, disc_loss, abs_loss = [],[],[]
		keys = self.TrainDict.keys()
		batch_replay = True
		replay_key = keys[-1]
		fig1,ax1 = plt.subplots(1,3,sharex=True,squeeze=True)
		fig2,ax2 = plt.subplots(1,2,squeeze=True)
		while True:
			mean_gen_loss, mean_disc_loss, mean_abs_loss = 0.0,0.0,0.0
			counter+=1
			for key in keys:
				# Fetch batch_size no. of patches from each image
				image,brightimage=self.FetchImage(key,patch=False)

				FakeLabels=np.expand_dims(np.ones(self.BatchSize),axis=1)
				RealLabels=np.expand_dims(np.zeros(self.BatchSize),axis=1)
				
				Discriminator_feed_dict={self.TargetImagePlaceholder : brightimage ,self.DiscriminatorLabelsReal:RealLabels ,self.GeneratorInput:image ,self.DiscriminatorLabelsFake:FakeLabels ,self.IsTraining:True}


				_,DiscCost=self.Session.run([self.DiscriminatorOptimizer,self.DiscriminatorLoss],feed_dict=Discriminator_feed_dict)
				print("Discriminator cost",str(DiscCost))

				GeneratorFeedDict={self.GeneratorLabels:RealLabels, self.GeneratorInput:image, self.IsTraining:True}
				_,GenCost,grad=self.Session.run([self.GeneratorOptimizer,self.GeneratorLoss,self.GeneratorGradients],feed_dict=GeneratorFeedDict)
				print("Discriminator unfooling loss",GenCost)
				
				absloss,im=self.Session.run([self.GeneratorABS,self.GeneratedImage],feed_dict={self.TargetImagePlaceholder:brightimage,self.GeneratorInput:image})
				absloss = np.random.random()
				print("ABS loss",absloss)

				if(batch_replay==True):
					image,brightimage=self.FetchImage(replay_key,patch=False)
					Discriminator_feed_dict={self.TargetImagePlaceholder : brightimage ,self.DiscriminatorLabelsReal:RealLabels ,self.GeneratorInput:image ,self.DiscriminatorLabelsFake:FakeLabels ,self.IsTraining:True}
					print("Batch replayed")
					_,DiscCost=self.Session.run([self.DiscriminatorOptimizer,self.DiscriminatorLoss],feed_dict=Discriminator_feed_dict)
					print("Discriminator cost",str(DiscCost))
					replay_key = key

				mean_gen_loss += GenCost
				mean_disc_loss += DiscCost
				mean_abs_loss += absloss
				ax2[0].imshow((brightimage[0]*255).astype('uint8'))
				ax2[1].imshow((brightimage[0]*255).astype('uint8'))
				plt.draw()
				plt.pause(0.0001)

			N = float(len(keys))
			mean_gen_loss /= N
			mean_abs_loss /= N
			mean_disc_loss /= N
			gen_loss.append(mean_gen_loss)
			disc_loss.append(mean_disc_loss)
			abs_loss.append(mean_abs_loss)	
			ax1[0].set_xlabel('Epoch')
			ax1[0].set_title('Generator Loss')
			ax1[0].plot(np.arange(counter)+1,gen_loss)
			ax1[1].set_title('Discriminator Loss')
			ax1[1].plot(np.arange(counter)+1,disc_loss)
			ax1[2].set_title('L1 Loss')
			ax1[2].plot(np.arange(counter)+1,abs_loss)
			plt.tight_layout()
			plt.draw()
			fig1.savefig('losses.png')			
			plt.pause(0.0001)
			
			save_path = self.saver.save(self.Session,self.save_path+"model"+str(counter)+".ckpt")

			valid_keys = self.ValidDict.keys()
			for key in valid_keys:
				image,_=self.FetchImage(key,patch=False)
				im=self.Session.run([self.GeneratedImage],feed_dict={self.GeneratorInput:image,self.IsTraining:False})
				misc.imsave("./Sony/results/"+key[13:-4]+"result.png",im)
				

dgan=DarkGAN()
dgan.train()



	

		
		