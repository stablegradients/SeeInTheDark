import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
TensorDict={}
OutputFilters={"A":16,"B":32,"C":64,"D":128,"E":256}

def lrelu(x,alpha=0.2):
    return tf.maximum(x * alpha, x)


def DiscriminatorConvloutions(inputs,block_code="A"):
	inputs=slim.conv2d(inputs,OutputFilters[block_code]//2, kernel_size=3, stride=1, rate=1,padding='SAME',activation_fn=lrelu)
	inputs=slim.conv2d(inputs,OutputFilters[block_code], kernel_size=3, stride=2, rate=1,padding='SAME',activation_fn=lrelu)
	inputs=slim.conv2d(inputs,OutputFilters[block_code], kernel_size=3, stride=1, rate=1,padding='SAME',activation_fn=lrelu)
	TensorDict[block_code+"2"]=inputs
	inputs=slim.conv2d(inputs,OutputFilters[block_code], kernel_size=3, stride=2, rate=1,padding='SAME',activation_fn=lrelu)
	TensorDict[block_code+"4"]=inputs
	
	return 
def FetchInputs(block_code="A"):
	if block_code == "B":
		return TensorDict["A2"]
	if block_code == "C":
		return tf.concat([TensorDict["A4"],TensorDict["B2"]],axis=3)
	if block_code == "D":
		return tf.concat([TensorDict["B4"],TensorDict["C2"]],axis=3)
	if block_code == "E":
		return tf.concat([TensorDict["C4"],TensorDict["D2"]],axis=3)


def Discriminator(inputs,reuse=None):
	with tf.variable_scope('Discriminator',reuse=reuse):
		DiscriminatorConvloutions(inputs,"A")

		DiscriminatorConvloutions(FetchInputs("B"),"B")

		DiscriminatorConvloutions(FetchInputs("C"),"C")
		
		DiscriminatorConvloutions(FetchInputs("D"),"D")
		
		DiscriminatorConvloutions(FetchInputs("E"),"E")

		ConvOutput=slim.conv2d(TensorDict["E4"],512, kernel_size=3, stride=2, rate=1,padding='SAME',activation_fn=lrelu)

		DenseLayer=tf.reduce_mean(ConvOutput, axis=[1,2])

		DenseLayer=tf.layers.dense(DenseLayer,50,activation=tf.nn.relu)

		DenseLayer=tf.layers.dense(DenseLayer,2,activation=tf.nn.relu)

		DenseLayerSoft=tf.nn.softmax(DenseLayer)
		return DenseLayer,DenseLayerSoft


		



		
