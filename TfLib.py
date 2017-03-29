import numpy as np
import tensorflow as tf

############################################################################################################################################################

# linear is tensor, eg tf.matmul(Wx+b)
# name is a string, naming the activationFunc for each unit
def ActivationOp(linear, name):
	if name == 'linear':
		return linear
	elif name == 'sigmoid': # scipy.special.expit
		return tf.nn.sigmoid(linear, name='encoded')
	elif name == 'softmax': # NumberLib.softmax
		return tf.nn.softmax(linear, name='encoded')
	elif name == 'tanh': # np.tanh
		return tf.nn.tanh(linear, name='encoded')
	elif name == 'relu': # np.maximum(x, 0)
		return tf.nn.relu(linear, name='encoded')
	else:
		print("ActivationOp: bad name " + name)
		assert False
		return linear
		
############################################################################################################################################################
		
def PrintTrainableVariables(sessionArg, printFunc, printNodes:bool = False):
	printFunc("*" * 80)
	printFunc("Start_PrintTrainableVariables")
	currGraph = sessionArg.graph
	nodeIndex = 0
	for node in currGraph.get_collection('trainable_variables'):
		if printNodes:
			printFunc("nodeIndex {0}, node-name {1}, node-dtype {2}, node [{3}]".format(nodeIndex, node.name, node.dtype, node))
		else:
			printFunc("nodeIndex {0}, node-name {1}, node-dtype {2}".format(nodeIndex, node.name, node.dtype))
		nodeValue = node.value()
		printFunc("value: name {0}, dtype {1}, shape {2}".format(nodeValue.name, nodeValue.dtype, nodeValue.get_shape()))
		nodeIndex += 1 
	printFunc("End_PrintTrainableVariables")
	printFunc("*" * 80)
	
############################################################################################################################################################

