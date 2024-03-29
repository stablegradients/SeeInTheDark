import tensorflow as tf
import numpy as np
import rawpy
import discriminator 
import generator
import json
import matplotlib.pyplot as plt
from scipy import misc
import math

class DarkGAN:
	"""docstring for ClassName"""
	def __init__(self):
		
		# self.IsTraining = tf.placeholder(tf.bool)

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
		self.GeneratedImage=generator.network(self.GeneratorInput)

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
		self.GeneratorOptimizer=tf.train.AdamOptimizer(0.00001).minimize(self.GeneratorLoss+self.gen_loss_lambda1*self.GeneratorABS,var_list=self.g_vars)
		self.GeneratorGradients=tf.train.AdamOptimizer(0.00001).compute_gradients(self.GeneratorLoss,var_list=self.g_vars)
		self.DiscriminatorOptimizer=tf.train.AdamOptimizer(0.00001).minimize(self.DiscriminatorLoss,var_list=self.d_vars)
		
		self.Session = tf.Session()
		self.saver=tf.train.Saver()
		
		self.init_op= tf.initialize_all_variables()
		self.Session.run(self.init_op)
		self.BatchReplayBool=True
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


	def FetchImageTraining(self,key,patch=True):
		"""
		In this method if the patch is true then we return multiple patches of the same image
		else we return just one patch of size self.PatchSize
		"""
		
		image = np.array(self.pack_raw(rawpy.imread(key))*float(self.TrainDict[key]["Exposure"]))
		
		shape=image.shape
		H,W=shape[0],shape[1]
		
		if(patch==True):
			h,w=np.random.randint(low=0,high=H-self.PatchSize-1,size=self.BatchSize),np.random.randint(low=0,high=W-self.PatchSize-1,size=self.BatchSize)
			image = np.array([image[h[i]:h[i]+self.PatchSize,w[i]:w[i]+self.PatchSize,:] for i in range(self.BatchSize)])

			GT = rawpy.imread(self.TrainDict[key]["Target"])
			GT = GT.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
			GT = np.float32(np.array(GT) / float(65535.0))
			GT = np.array([GT[2*h[i]:2*h[i]+2*self.PatchSize,2*w[i]:2*w[i]+2*self.PatchSize,:] for i in range(self.BatchSize)])
			return image,GT
		else:
			h,w=np.random.randint(low=0,high=H-self.PatchSize-1),np.random.randint(low=0,high=W-self.PatchSize-1)
			image = np.expand_dims(image,axis=0)
			GT = rawpy.imread(self.TrainDict[key]["Target"])
			GT = GT.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
			GT = np.float32(np.array(GT) / float(65535.0))

			GT = np.expand_dims(GT,axis=0)
			return image,GT



	def FetchImageExecute(self,key):
		"""
		We return the maximum image patch of size (2^k,2^j,3) where k,j are integers > 4
		This is done due to upsample dimension mismatch in Generator nnetwork
		"""
		
		image = np.array(self.pack_raw(rawpy.imread(key))*float(self.ValidDict[key]["Exposure"]))
		
		shape=image.shape
		H,W=int(2**int(math.log(shape[0],2))),int(2**int(math.log(shape[1],2)))
		
		image = np.expand_dims(image[0:H,0:W,:],axis=0)
		
		GT = rawpy.imread(self.ValidDict[key]["Target"])
		GT = GT.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
		GT = np.float32(np.array(GT) / float(65535.0))
		GT = np.expand_dims(GT[0:2*H,0:2*W,:],axis=0)
		return image,GT		

	def show_images(self,im,brightimage):
		im = np.clip(im[0]*255,0,255).astype('uint8')
		self.ax2[0].imshow(im)
		brightimage = np.clip(brightimage[0]*255,0,255).astype('uint8')
		self.ax2[1].imshow(brightimage)
		plt.draw()
		plt.pause(0.0001)
	
	def init_plots(self):
		self.gen_loss_list, self.disc_loss_list, self.abs_loss_list, self.valid_abs_loss_list = [],[],[],[]
		self.fig1,self.ax1 = plt.subplots(1,3,sharex=True,squeeze=True)
		self.fig2,self.ax2 = plt.subplots(1,2,squeeze=True)

	def plots(self, mean_gen_loss, mean_disc_loss, mean_abs_loss, mean_valid_abs_loss):
		self.gen_loss_list.append(mean_gen_loss)
		self.disc_loss_list.append(mean_disc_loss)
		self.abs_loss_list.append(mean_abs_loss)
		self.valid_abs_loss_list.append(mean_valid_abs_loss)
		counter = len(self.gen_loss_list)
		self.ax1[0].set_xlabel('Epoch')
		self.ax1[0].set_title('Generator Loss')
		self.ax1[0].plot(np.arange(counter)+1,self.gen_loss_list)
		self.ax1[1].set_title('Discriminator Loss')
		self.ax1[1].plot(np.arange(counter)+1,self.disc_loss_list)
		self.ax1[2].set_title('L1 Loss')
		self.ax1[2].set_color_cycle(['red', 'green'])
		self.ax1[2].plot(np.arange(counter)+1,self.abs_loss_list)
		self.ax1[2].plot(np.arange(counter)+1,self.valid_abs_loss_list)
		self.ax1[2].legend(['Training','Validation'])
		plt.tight_layout()
		plt.draw()
		self.fig1.savefig('losses.png')			
		plt.pause(0.0001)

	def train(self):
		counter=0
		keys = self.TrainDict.keys()
		batch_replay = True
		self.init_plots()
		while True:
			
			counter+=1
			samples=0
			mean_gen_loss, mean_disc_loss, mean_abs_loss=0,0,0
			for key in keys:
				# Fetch batch_size no. of patches from each image
				samples+=1
				print("Epoch "+str(counter)+", Sample "+str(samples))
				image,brightimage=self.FetchImageTraining(key,patch=True)

				FakeLabels=np.expand_dims(np.ones(self.BatchSize),axis=1)
				RealLabels=np.expand_dims(np.zeros(self.BatchSize),axis=1)
				

				"""
				The training op of Discriminator follows,the generated images and real target image for given
				short exposure images are fed to thhe computation graph
				"""
				Discriminator_feed_dict={self.TargetImagePlaceholder : brightimage ,self.DiscriminatorLabelsReal:RealLabels ,self.GeneratorInput:image ,self.DiscriminatorLabelsFake:FakeLabels}

				_,DiscCost=self.Session.run([self.DiscriminatorOptimizer,self.DiscriminatorLoss],feed_dict=Discriminator_feed_dict)
				print("Discriminator cost",str(DiscCost))

				"""
				The training op of Generator follows,the generated images and real target image for given
				short exposure images are fed to thhe computation graph
				"""

				GeneratorFeedDict={self.GeneratorLabels:RealLabels, self.GeneratorInput:image, self.TargetImagePlaceholder:brightimage}
				_,GenCost,grad=self.Session.run([self.GeneratorOptimizer,self.GeneratorLoss,self.GeneratorGradients],feed_dict=GeneratorFeedDict)
				print("Discriminator unfooling loss",GenCost)
				
				absloss,im=self.Session.run([self.GeneratorABS,self.GeneratedImage],feed_dict={self.TargetImagePlaceholder:brightimage,self.GeneratorInput:image})
				print("ABS loss",absloss)

				"""
				Batch replay plays the train op of discriminator againn and helps preventing GAN collapse: REF soumith gan hacks
				"""
				if(self.BatchReplayBool==True):
					self.BatchReplay(key)
					
				mean_gen_loss += GenCost
				mean_disc_loss += DiscCost
				mean_abs_loss += absloss
			
			valid_abs_loss=self.Validation()
			self.plots(mean_gen_loss/samples,mean_disc_loss/samples, mean_abs_loss/samples, valid_abs_loss)	
			save_path = self.saver.save(self.Session,self.save_path+"model"+str(counter)+".ckpt")

	def BatchReplay(self,key):
		image,brightimage=self.FetchImageTraining(key,patch=True)
		FakeLabels=np.expand_dims(np.ones(self.BatchSize),axis=1)
		RealLabels=np.expand_dims(np.zeros(self.BatchSize),axis=1)


		Discriminator_feed_dict={self.TargetImagePlaceholder : brightimage ,self.DiscriminatorLabelsReal:RealLabels ,self.GeneratorInput:image ,self.DiscriminatorLabelsFake:FakeLabels}
		
		_,DiscCost=self.Session.run([self.DiscriminatorOptimizer,self.DiscriminatorLoss],feed_dict=Discriminator_feed_dict)
		print("Discriminator cost",str(DiscCost))
		
		
	
	def Validation(self):
		valid_keys = self.ValidDict.keys()
		mean_valid_abs_loss = 0.0
		
		for key in valid_keys:
			print("Validating on "+key)

			
			image,brightimage=self.FetchImageExecute(key)
			im,valid_abs_loss=self.Session.run([self.GeneratedImage,self.GeneratorABS],feed_dict={self.GeneratorInput:image,self.TargetImagePlaceholder:brightimage})
			
			mean_valid_abs_loss += valid_abs_loss
			
			filename = key.replace("short","ResultPatch").replace(".ARW",".png")
			misc.imsave(filename,(np.clip(im[0]*255,0,255)).astype('uint8'))
			filename = key.replace("short","GTPatch").replace(".ARW",".png")
			misc.imsave(filename,(np.clip(brightimage[0]*255,0,255)).astype('uint8'))

		samples=float(len(keys))
		print("Validation done!!!!!!!")
		print("Validation Absolute Loss "+str(mean_valid_abs_loss/samples))
		return  mean_valid_abs_loss/samples
		
			

				
				

dgan=DarkGAN()
dgan.Validation()



	

		
		