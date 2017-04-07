############################################################################################################################################################

# three use cases
# 	case A: use new features to improve MLP; save sdae final weights, MLP sets them as initial weights
# 	case B: directly use new features to improve a classifier eg logReg
# 	case C: generate images, should be close to originals

# Note on control parameters
"""
numLayers, numHiddensPerLayer (hiddenDims)
f, g (encodeFunc, decodeFunc; hiddenEncodeFuncName, hiddenDecodeFuncName)
lossFunc (lossName)
bool, tiedWeights (bTiedWeights)
maskNoiseFraction
"""

############################################################################################################################################################

import collections
import datetime
import numpy as np
import os
import pickle
import random
import scipy.stats
import sys
import tensorflow as tf


import DatetimeLib
import FileLib
import GlobalSettings
import LogLib
import NumberLib
import PlotLib
import TfLib

############################################################################################################################################################
# block, init
############################################################################################################################################################

harnessStartTime = datetime.datetime.utcnow()

# set up logging
rootResultsDir = GlobalSettings.GlobalSettings_GetRootResultsDir()
thisResultsDir = rootResultsDir + "Tf_Results/"
logResult = LogLib.StartUniqueLog(thisResultsDir)
print("logResult", str(logResult))

LocalPrintAndLogFunc = LogLib.MakeLogBothFunc(logResult['file'])

############################################################################################################################################################

LocalPrintAndLogFunc("SdaeTwo")
LocalPrintAndLogFunc("SdaeTwo harnessStartTime=" + str(harnessStartTime))
rseed = harnessStartTime.microsecond
random.seed(rseed)
np.random.seed(rseed)
LocalPrintAndLogFunc("after setting randomSeed=" + str(rseed))

############################################################################################################################################################
# block: load data
############################################################################################################################################################

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# these are np-ndarrays:
trainCovariates = mnist.train.images
trainLabels = mnist.train.labels
testCovariates = mnist.test.images
testLabels = mnist.test.labels
LocalPrintAndLogFunc("shape-trainCovariates=" + str(trainCovariates.shape))
assert len(trainCovariates.shape) == 2
LocalPrintAndLogFunc("shape-trainLabels=" + str(trainLabels.shape))
LocalPrintAndLogFunc("shape-testCovariates=" + str(testCovariates.shape))
LocalPrintAndLogFunc("shape-testLabels=" + str(testLabels.shape))

############################################################################################################################################################
# block: create overall parameters
############################################################################################################################################################

# live
inputDim = trainCovariates.shape[1] # 784
hiddenDims = (256, 128, 64) # 3-layer
#hiddenDims = (256, 128) # 2-layer
#hiddenDims = (256,) # 1-layer
assert len(hiddenDims) > 0 # SDAE needs at least one hidden layer

totalDimsList = list(); totalDimsList.append(inputDim); totalDimsList.extend(hiddenDims); totalDims = tuple(totalDimsList)
LocalPrintAndLogFunc("inputDim=" + str(inputDim))
LocalPrintAndLogFunc("hiddenDims=" + str(hiddenDims))
LocalPrintAndLogFunc("numHiddenLayers=" + str(len(hiddenDims)))
FileLib.WriteTextFileForParam(logResult['path'], "numHiddenLayers", str(len(hiddenDims)))
LocalPrintAndLogFunc("totalDims=" + str(totalDims))

# typical combinations
# combo1: hiddenEncodeFuncName=sigmoid, hiddenDecodeFuncName=affine[linear], loss=rmse [advised when inputData is real]
# combo2: hiddenEncodeFuncName=sigmoid, hiddenDecodeFuncName=sigmoid, loss=crossEntropy [advised when inputData is binary]
hiddenEncodeFuncName = "sigmoid"
LocalPrintAndLogFunc("hiddenEncodeFuncName=" + str(hiddenEncodeFuncName))

hiddenDecodeFuncName = "sigmoid"
LocalPrintAndLogFunc("hiddenDecodeFuncName=" + str(hiddenDecodeFuncName))

lossName = "rmse" # "cross-entropy"
LocalPrintAndLogFunc("lossName=" + str(lossName))

learnRate = 0.007
LocalPrintAndLogFunc("learnRate=" + str(learnRate))

maskNoiseFraction = 0.1 
#maskNoiseFraction = 0.25
LocalPrintAndLogFunc("maskNoiseFraction=" + str(maskNoiseFraction))

bTiedWeights = True
#bTiedWeights = False
LocalPrintAndLogFunc("bTiedWeights=" + str(bTiedWeights))

bTrainDataBatchesRandom = True
#bTrainDataBatchesRandom = False
LocalPrintAndLogFunc("bTrainDataBatchesRandom=" + str(bTrainDataBatchesRandom))



numTrainBatches = 100
LocalPrintAndLogFunc("numTrainBatches=" + str(numTrainBatches))

batchSize = int(trainCovariates.shape[0] / numTrainBatches) 
LocalPrintAndLogFunc("batchSize " + str(batchSize) + " out of " + str(trainCovariates.shape[0]))

printBatchFreq = int(numTrainBatches / 10)
LocalPrintAndLogFunc("printBatchFreq=" + str(printBatchFreq))

nTrainEpochs = 5 # 15 # Live
#nTrainEpochs = 1 # Devel
LocalPrintAndLogFunc("nTrainEpochs=" + str(nTrainEpochs))

printEpochFreq = 1
LocalPrintAndLogFunc("printEpochFreq=" + str(printEpochFreq))




initWbTruncNormal = True
#initWbTruncNormal = False
LocalPrintAndLogFunc("initWbTruncNormal=" + str(initWbTruncNormal))
print("*" * 80)

############################################################################################################################################################
# block: create network transform data (weights and biases)
############################################################################################################################################################

# the encodeWeights
# 	dataGenerator options: zeros, linearSeq eg tf.linspace, truncNormal, regularNormal
# 	tf.truncated_normal clips at 2*stdDev, ie +2,-2 for stdGaussian

if initWbTruncNormal:
	initialEncodeWeights1 = scipy.stats.truncnorm.rvs(-2, 2, size=inputDim * hiddenDims[0]).reshape(inputDim, hiddenDims[0])
	if len(hiddenDims) > 1: initialEncodeWeights2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0] * hiddenDims[1]).reshape(hiddenDims[0], hiddenDims[1])
	if len(hiddenDims) > 2: initialEncodeWeights3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1] * hiddenDims[2]).reshape(hiddenDims[1], hiddenDims[2])
else:
	npMat1 = np.arange(0, inputDim * hiddenDims[0]).reshape(inputDim, hiddenDims[0]);
	npMat1 = npMat1 / np.max(npMat1)
	initialEncodeWeights1 = npMat1
	if len(hiddenDims) > 1: 
		npMat2 = np.arange(0, hiddenDims[0] * hiddenDims[1]).reshape(hiddenDims[0], hiddenDims[1])
		npMat2 = npMat2 / np.max(npMat2)
		initialEncodeWeights2 = npMat2
	if len(hiddenDims) > 2: 
		npMat3 = np.arange(0, hiddenDims[1] * hiddenDims[2]).reshape(hiddenDims[1], hiddenDims[2])
		npMat3 = npMat3 / np.max(npMat3)
		initialEncodeWeights3 = npMat3

# decodeWeights
if bTiedWeights:
	# tied (shared) weights
	initialDecodeWeights1 = np.transpose(initialEncodeWeights1)
	if len(hiddenDims) > 1: initialDecodeWeights2 = np.transpose(initialEncodeWeights2)
	if len(hiddenDims) > 2: initialDecodeWeights3 = np.transpose(initialEncodeWeights3)
else:
	if initWbTruncNormal:
		initialDecodeWeights1 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0] * inputDim).reshape(hiddenDims[0], inputDim)
		if len(hiddenDims) > 1: initialDecodeWeights2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1] * hiddenDims[0]).reshape(hiddenDims[1], hiddenDims[0])
		if len(hiddenDims) > 2: initialDecodeWeights3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[2] * hiddenDims[1]).reshape(hiddenDims[2], hiddenDims[1])
	else: 
		npMat4 = np.arange(0, hiddenDims[0] * inputDim).reshape(hiddenDims[0], inputDim);
		npMat4 = npMat4 / np.max(npMat4)
		initialDecodeWeights1 = npMat4
		if len(hiddenDims) > 1: 
			npMat5 = np.arange(0, hiddenDims[1] * hiddenDims[0]).reshape(hiddenDims[1], hiddenDims[0])
			npMat5 = npMat5 / np.max(npMat5)
			initialDecodeWeights2 = npMat5
		if len(hiddenDims) > 2: 
			npMat6 = np.arange(0, hiddenDims[2] * hiddenDims[1]).reshape(hiddenDims[2], hiddenDims[1])
			npMat6 = npMat6 / np.max(npMat6)
			initialDecodeWeights3 = npMat6

# encodeBiases
if initWbTruncNormal:
	initialEncodeBiases1 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0])
	if len(hiddenDims) > 1: initialEncodeBiases2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1])
	if len(hiddenDims) > 2: initialEncodeBiases3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[2])
else:
	initialEncodeBiases1 = np.arange(0, hiddenDims[0]) / hiddenDims[0]
	if len(hiddenDims) > 1: initialEncodeBiases2 = np.arange(0, hiddenDims[1]) / hiddenDims[1]
	if len(hiddenDims) > 2: initialEncodeBiases3 = np.arange(0, hiddenDims[2]) / hiddenDims[2]

# decodeBiases, NOT same shape as encodeBiases
if initWbTruncNormal:
	initialDecodeBiases1 = scipy.stats.truncnorm.rvs(-2, 2, size=inputDim)
	if len(hiddenDims) > 1: initialDecodeBiases2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0])
	if len(hiddenDims) > 2: initialDecodeBiases3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1])
else:
	initialDecodeBiases1 = np.arange(0, inputDim) / inputDim
	if len(hiddenDims) > 1: initialDecodeBiases2 = np.arange(0, hiddenDims[0]) / hiddenDims[0]
	if len(hiddenDims) > 2: initialDecodeBiases3 = np.arange(0, hiddenDims[1]) / hiddenDims[1]

############################################################################################################################################################

threeLayerInitEncodeWeightsDict = { "h1": initialEncodeWeights1 }
threeLayerInitEncodeBiasesDict = { "b1": initialEncodeBiases1 }
threeLayerInitDecodeWeightsDict = { "h1": initialDecodeWeights1 }
threeLayerInitDecodeBiasesDict = { "b1": initialDecodeBiases1}
if len(hiddenDims) > 1: 
	threeLayerInitEncodeWeightsDict["h2"] = initialEncodeWeights2
	threeLayerInitEncodeBiasesDict["b2"] =initialEncodeBiases2
	threeLayerInitDecodeWeightsDict["h2"] = initialDecodeWeights2
	threeLayerInitDecodeBiasesDict["b2"] = initialDecodeBiases2
if len(hiddenDims) > 2: 
	threeLayerInitEncodeWeightsDict["h3"] = initialEncodeWeights3
	threeLayerInitEncodeBiasesDict["b3"] =initialEncodeBiases3
	threeLayerInitDecodeWeightsDict["h3"] = initialDecodeWeights3
	threeLayerInitDecodeBiasesDict["b3"] = initialDecodeBiases3

initEncodeWeightsDict = dict()
initEncodeBiasesDict = dict()
initDecodeWeightsDict = dict()
initDecodeBiasesDict = dict()

finalEncodeWeightsDict = dict()
finalEncodeBiasesDict = dict()
finalDecodeWeightsDict = dict()
finalDecodeBiasesDict = dict()

numLayers = len(hiddenDims)
for hLayerIndex in range(numLayers):
	hKeyStr = "h" + str(hLayerIndex+1)
	bKeyStr = "b" + str(hLayerIndex+1)
	initEncodeWeightsDict[hKeyStr] = threeLayerInitEncodeWeightsDict[hKeyStr]
	initEncodeBiasesDict[bKeyStr] = threeLayerInitEncodeBiasesDict[bKeyStr]
	initDecodeWeightsDict[hKeyStr] = threeLayerInitDecodeWeightsDict[hKeyStr]
	initDecodeBiasesDict[bKeyStr] = threeLayerInitDecodeBiasesDict[bKeyStr]
	finalEncodeWeightsDict[hKeyStr] = np.copy(initEncodeWeightsDict[hKeyStr])
	finalEncodeBiasesDict[bKeyStr] = np.copy(initEncodeBiasesDict[bKeyStr])
	finalDecodeWeightsDict[hKeyStr] = np.copy(initDecodeWeightsDict[hKeyStr])
	finalDecodeBiasesDict[bKeyStr] = np.copy(initDecodeBiasesDict[bKeyStr])
	
############################################################################################################################################################

# x is input state; weightDict, biasDict are encode
# hidden layers only
	
# numLayers: start at top, input, and go down
def FeedForwardHiddensPartial_SessionEval(sessionArg, x, numLayers:int, weightDict, biasDict, hiddenEncodeFuncNameStr:str):
	assert numLayers >= 0 and numLayers <= 3
	
	if numLayers == 0: return x
	
	# input to Hidden layer 1
	x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h1']), biasDict['b1']), hiddenEncodeFuncNameStr)
	
	if numLayers >= 2:
		# Hidden layer 1 to 2
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h2']), biasDict['b2']), hiddenEncodeFuncNameStr)
	
	if numLayers >= 3:	
		# Hidden layer 2 to 3
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h3']), biasDict['b3']), hiddenEncodeFuncNameStr)
	
	result = x.eval(session=sessionArg)
	return result
	
# x is state of final hidden layer	
# numLayers: start at numLayers, go up
def GenerateInputBackwardHiddensPartial_SessionEval(sessionArg, x, numLayers:int, decodeWeightDict, decodeBiasDict, hiddenDecodeFuncNameStr:str):
	assert numLayers >= 0 and numLayers <= 3
	
	if numLayers == 0: return x
	
	if numLayers >= 3:
		# Hidden layer 3 to 2
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, decodeWeightDict['h3']), decodeBiasDict['b3']), hiddenDecodeFuncNameStr)
	if numLayers >= 2:
		# Hidden layer 2 to 1
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, decodeWeightDict['h2']), decodeBiasDict['b2']), hiddenDecodeFuncNameStr)
	if numLayers >= 1:
		# Hidden layer 1 to Input
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, decodeWeightDict['h1']), decodeBiasDict['b1']), hiddenDecodeFuncNameStr)
	
	result = x.eval(session=sessionArg)
	return result
	
def PrintNetParamsSummary1(initialReportStr, printFunc, sessionArg, weightsDict, biasesDict, decodeWeightsDict, decodeBiasesDict, printValues:bool, printShape:bool):
	printFunc(initialReportStr)
	numLayers = len(weightsDict.keys())
	printFunc("numLayers=" + str(numLayers))
	for hLayerIndex in range(numLayers):
		printFunc("hLayerIndex=" + str(hLayerIndex))
		hKeyStr = "h" + str(hLayerIndex+1)
		bKeyStr = "b" + str(hLayerIndex+1)
		assert hKeyStr in weightsDict.keys() and hKeyStr in decodeWeightsDict.keys()
		assert bKeyStr in biasesDict.keys() and bKeyStr in decodeBiasesDict.keys()
		encodeWeightsThisLayer = weightsDict[hKeyStr]
		encodeBiasesThisLayer = biasesDict[bKeyStr]
		decodeWeightsThisLayer = decodeWeightsDict[hKeyStr]
		decodeBiasesThisLayer = decodeBiasesDict[bKeyStr]
	
		if printValues:
			if isinstance(encodeWeightsThisLayer, np.ndarray):
				printFunc("encodeWeightsThisLayer [0,0] " + str(encodeWeightsThisLayer[0,0]))
				printFunc("encodeBiasesThisLayer [0] " + str(encodeBiasesThisLayer[0]))
				printFunc("decodeWeightsThisLayer [0,0] " + str(decodeWeightsThisLayer[0,0]))
				printFunc("decodeBiasesThisLayer [0] " + str(decodeBiasesThisLayer[0]))
			else:
				printFunc("encodeWeightsThisLayer [0,0] " + str(sessionArg.run(encodeWeightsThisLayer)[0,0]))
				printFunc("encodeBiasesThisLayer [0] " + str(sessionArg.run(encodeBiasesThisLayer)[0]))
				printFunc("decodeWeightsThisLayer [0,0] " + str(sessionArg.run(decodeWeightsThisLayer)[0,0]))
				printFunc("decodeBiasesThisLayer [0] " + str(sessionArg.run(decodeBiasesThisLayer)[0]))
			printFunc("-" * 80)
		if printShape:
			if isinstance(encodeWeightsThisLayer, np.ndarray):
				printFunc("shape-encodeWeightsThisLayer " + str(encodeWeightsThisLayer.shape))
				printFunc("shape-encodeBiasesThisLayer " + str(encodeBiasesThisLayer.shape))
				printFunc("shape-decodeWeightsThisLayer " + str(decodeWeightsThisLayer.shape))
				printFunc("shape-decodeBiasesThisLayer " + str(decodeBiasesThisLayer.shape))
			else:
				printFunc("shape-encodeWeightsThisLayer " + str(encodeWeightsThisLayer.get_shape()))
				printFunc("shape-encodeBiasesThisLayer " + str(encodeBiasesThisLayer.get_shape()))
				printFunc("shape-decodeWeightsThisLayer " + str(decodeWeightsThisLayer.get_shape()))
				printFunc("shape-decodeBiasesThisLayer " + str(decodeBiasesThisLayer.get_shape()))
				
			printFunc("-" * 80)
		printFunc("-" * 80)
	printFunc("-" * 80)
	
############################################################################################################################################################

bPrintAllShapes1 = True
bPrintSomeValues1 = True
if bPrintAllShapes1 or bPrintSomeValues1:
	LocalPrintAndLogFunc("*" * 80)
	numLayers = len(initEncodeWeightsDict.keys())
	LocalPrintAndLogFunc("numLayers=" + str(numLayers))
	for hLayerIndex in range(numLayers):
		LocalPrintAndLogFunc("hLayerIndex=" + str(hLayerIndex))
		hKeyStr = "h" + str(hLayerIndex+1)
		bKeyStr = "b" + str(hLayerIndex+1)
		if bPrintAllShapes1:
			LocalPrintAndLogFunc("shape-encodeWeights " + str(initEncodeWeightsDict[hKeyStr].shape))
			LocalPrintAndLogFunc("shape-encodeBiases " + str(initEncodeBiasesDict[bKeyStr].shape))
			LocalPrintAndLogFunc("shape-decodeWeights " + str(initDecodeWeightsDict[hKeyStr].shape))
			LocalPrintAndLogFunc("shape-decodeBiases " + str(initDecodeBiasesDict[bKeyStr].shape))
		if bPrintSomeValues1:
			LocalPrintAndLogFunc("encodeWeights[0:2, 0:2] " + str(initEncodeWeightsDict[hKeyStr][0:2, 0:2]))
			LocalPrintAndLogFunc("encodeBiases[0:2] " + str(initEncodeBiasesDict[bKeyStr][0:2]))
			LocalPrintAndLogFunc("decodeWeights[0:2, 0:2] " + str(initDecodeWeightsDict[hKeyStr][0:2, 0:2]))
			LocalPrintAndLogFunc("decodeBiases[0:2] " + str(initDecodeBiasesDict[bKeyStr][0:2]))
		LocalPrintAndLogFunc("*" * 80)
	LocalPrintAndLogFunc("*" * 80)

############################################################################################################################################################
# block: train sdae
############################################################################################################################################################
	
LocalPrintAndLogFunc("Before training all SDAE layers")
numHiddenLayers = len(hiddenDims)

tf.reset_default_graph()

# initEncodeWeightsDict never changes
# finalEncodeWeightsDict is changed through the iterative layer training

for hLayerIndex in range(numHiddenLayers):
	tf.reset_default_graph() # need this: before, the number of variables grew as hLayerIndex grew; now, it does not
	
	thisLayerHiddenDim = hiddenDims[hLayerIndex]
	layerAboveDim = inputDim # layerAboveDim used to be called inputToThisLayerDim
	if hLayerIndex > 0:
		layerAboveDim = hiddenDims[hLayerIndex-1]
	numLayersToPropagate = hLayerIndex # numLayersToPropagate is num hiddens before this
	LocalPrintAndLogFunc("Before training HiddenLayer {0} which has dim {1}, layerAboveDim {2}".format(hLayerIndex, thisLayerHiddenDim, layerAboveDim))
	
	hKeyStr = "h" + str(hLayerIndex+1)
	bKeyStr = "b" + str(hLayerIndex+1)
	print("hKeyStr [{0}], bKeyStr [{1}]".format(hKeyStr, bKeyStr))
	assert hKeyStr in finalEncodeWeightsDict.keys() and hKeyStr in finalDecodeWeightsDict.keys()
	assert bKeyStr in finalEncodeBiasesDict.keys() and bKeyStr in finalDecodeBiasesDict.keys()
	
	encodeWeightsThisLayerVar = tf.Variable(finalEncodeWeightsDict[hKeyStr], name = "encodeWeightsThisLayerVar_layer_" + str(hLayerIndex), dtype=tf.float32)
	encodeBiasesThisLayerVar = tf.Variable(finalEncodeBiasesDict[bKeyStr], name = "encodeBiasesThisLayerVar_layer_" + str(hLayerIndex), dtype=tf.float32)
	if bTiedWeights:
		decodeWeightsThisLayerVar = tf.transpose(encodeWeightsThisLayerVar)
	else:
		decodeWeightsThisLayerVar = tf.Variable(finalDecodeWeightsDict[hKeyStr], name = "decodeWeightsThisLayerVar_layer_" + str(hLayerIndex), dtype=tf.float32)
	decodeBiasesThisLayerVar = tf.Variable(finalDecodeBiasesDict[bKeyStr], name = "decodeBiasesThisLayerVar_layer_" + str(hLayerIndex), dtype=tf.float32)
	
	xPureInput = tf.placeholder(dtype=tf.float32, shape=[None, layerAboveDim], name='xPureInput')
	xNoisyInput = tf.placeholder(dtype=tf.float32, shape=[None, layerAboveDim], name='xNoisyInput')
	
	encodeOp = TfLib.ActivationOp(tf.add(tf.matmul(xNoisyInput, encodeWeightsThisLayerVar), encodeBiasesThisLayerVar), hiddenEncodeFuncName)
	decodeOp = TfLib.ActivationOp(tf.add(tf.matmul(encodeOp, decodeWeightsThisLayerVar), decodeBiasesThisLayerVar), hiddenDecodeFuncName)
	
	# define loss
	if lossName == 'rmse':
		loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(xPureInput, decodeOp))))
	elif lossName == 'cross-entropy':
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = decodeOp, labels = xPureInput))
	
	trainOp = tf.train.AdamOptimizer(learnRate).minimize(loss)
	
	sess = tf.Session()
	LocalPrintAndLogFunc("After starting session")
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	LocalPrintAndLogFunc("After global_variables_initializer")
	
	TfLib.PrintTrainableVariables(sess, LocalPrintAndLogFunc)
	
	PrintNetParamsSummary1("Before training layer {0}".format(hLayerIndex), LocalPrintAndLogFunc, sess, 
		finalEncodeWeightsDict, finalEncodeBiasesDict, finalDecodeWeightsDict, finalDecodeBiasesDict, printValues = True, printShape = False)
	
	for epoch in range(nTrainEpochs):
		for batchIndex in range(numTrainBatches):
			if bTrainDataBatchesRandom:
				randIndices = np.random.choice(trainCovariates.shape[0], batchSize, replace=False)
				originalInputData = trainCovariates[randIndices]
			else:
				originalInputData, originalLabelsData = mnist.train.next_batch(batchSize)
			
			# dataInputForThisLayer is the zero-noise input to this layer
			dataInputForThisLayer = FeedForwardHiddensPartial_SessionEval(sess, originalInputData, numLayersToPropagate, finalEncodeWeightsDict, finalEncodeBiasesDict, hiddenEncodeFuncName)
			
			# add noise
			noisyDataInputForThisLayer = NumberLib.ApplyMaskingNoise(dataInputForThisLayer, maskNoiseFraction)
		
			sess.run(trainOp, feed_dict={xPureInput: dataInputForThisLayer, xNoisyInput: noisyDataInputForThisLayer})
			
			if batchIndex == 0:
				print("originalInputData.shape", originalInputData.shape)
				print("dataInputForThisLayer.shape", dataInputForThisLayer.shape)
				print("noisyDataInputForThisLayer.shape", noisyDataInputForThisLayer.shape)
			
			if (batchIndex + 1) % printBatchFreq == 0:
				currentLoss = sess.run(loss, feed_dict={xPureInput: dataInputForThisLayer, xNoisyInput: noisyDataInputForThisLayer})
				LocalPrintAndLogFunc("batch {0}, epoch {1}: global loss = {2}".format(batchIndex, epoch, currentLoss))
				
		if (epoch % printEpochFreq) == 0:
			LocalPrintAndLogFunc("epoch {0}, hiddenLayer {1}".format(epoch, hLayerIndex))
			
	# retrieve learnt params from tf.Variables, store np.ndarrays in dict
	finalEncodeWeightsDict[hKeyStr] = encodeWeightsThisLayerVar.eval(session=sess)
	finalEncodeBiasesDict[bKeyStr] = encodeBiasesThisLayerVar.eval(session=sess)
	finalDecodeWeightsDict[hKeyStr] = decodeWeightsThisLayerVar.eval(session=sess)
	finalDecodeBiasesDict[bKeyStr] = decodeBiasesThisLayerVar.eval(session=sess)
	print("shape-finalEncodeWeightsDict[hKeyStr]", finalEncodeWeightsDict[hKeyStr].shape)
	print("shape-finalEncodeBiasesDict[bKeyStr]", finalEncodeBiasesDict[bKeyStr].shape)
	print("shape-finalDecodeWeightsDict[hKeyStr]", finalDecodeWeightsDict[hKeyStr].shape)
	print("shape-finalDecodeBiasesDict[bKeyStr]", finalDecodeBiasesDict[bKeyStr].shape)
	
	PrintNetParamsSummary1("After training layer {0}".format(hLayerIndex), LocalPrintAndLogFunc, sess, 
		finalEncodeWeightsDict, finalEncodeBiasesDict, finalDecodeWeightsDict, finalDecodeBiasesDict, printValues = True, printShape = False)
			
	sess.close()
	LocalPrintAndLogFunc("After closing session")
			
	LocalPrintAndLogFunc("After training HiddenLayer {0} which has dim {1}, layerAboveDim {2}".format(hLayerIndex, thisLayerHiddenDim, layerAboveDim))
	
LocalPrintAndLogFunc("After training all SDAE layers")



saveResultDict = dict()
saveResultDict["totalDims"] = totalDims
saveResultDict["hiddenDims"] = hiddenDims
saveResultDict["inputDim"] = inputDim
saveResultDict["hiddenEncodeFuncName"] = hiddenEncodeFuncName
saveResultDict["hiddenDecodeFuncName"] = hiddenDecodeFuncName
saveResultDict["lossName"] = lossName
saveResultDict["bTiedWeights"] = bTiedWeights
saveResultDict["encodeWeightsDict"] = finalEncodeWeightsDict
saveResultDict["encodeBiasesDict"] = finalEncodeBiasesDict
saveResultDict["decodeWeightsDict"] = finalDecodeWeightsDict
saveResultDict["decodeBiasesDict"] = finalDecodeBiasesDict
saveResultDict["initEncodeWeightsDict"] = initEncodeWeightsDict
saveResultDict["initEncodeBiasesDict"] = initEncodeBiasesDict
saveResultDict["initDecodeWeightsDict"] = initDecodeWeightsDict
saveResultDict["initDecodeBiasesDict"] = initDecodeBiasesDict

sortedSrdKeys = list(saveResultDict.keys()); sortedSrdKeys.sort()
LocalPrintAndLogFunc("saveResultDict-sortedSrdKeys" + str(sortedSrdKeys))

for hLayerIndex in range(len(hiddenDims)):
	hKeyStr = "h" + str(hLayerIndex+1)
	bKeyStr = "b" + str(hLayerIndex+1)
	LocalPrintAndLogFunc("encodeWeightsDict-[" + str(hLayerIndex) + "].shape" + str(saveResultDict["encodeWeightsDict"][hKeyStr].shape))
	LocalPrintAndLogFunc("encodeBiasesDict-[" + str(hLayerIndex) + "].shape" + str(saveResultDict["encodeBiasesDict"][bKeyStr].shape))
	LocalPrintAndLogFunc("decodeWeightsDict-[" + str(hLayerIndex) + "].shape" + str(saveResultDict["decodeWeightsDict"][hKeyStr].shape))
	LocalPrintAndLogFunc("decodeBiasesDict-[" + str(hLayerIndex) + "].shape" + str(saveResultDict["decodeBiasesDict"][bKeyStr].shape))
LocalPrintAndLogFunc("-"*80)
	
tf.reset_default_graph()
sess = tf.Session()
TfLib.PrintTrainableVariables(sess, LocalPrintAndLogFunc)
PrintNetParamsSummary1("After all training, initParams", LocalPrintAndLogFunc, sess, 
	initEncodeWeightsDict, initEncodeBiasesDict, initDecodeWeightsDict, initDecodeBiasesDict, printValues = True, printShape = True)
PrintNetParamsSummary1("After all training, finalParams", LocalPrintAndLogFunc, sess, 
	finalEncodeWeightsDict, finalEncodeBiasesDict, finalDecodeWeightsDict, finalDecodeBiasesDict, printValues = True, printShape = True)
sess.close()
	
# save learnt parameters to file
# eg for reuse eg by MLP
srdFilename = logResult['path'] + 'saveResultDict.p'
srdFilehandler = open(srdFilename, 'wb')
pickle.dump(saveResultDict, srdFilehandler)
srdFilehandler.close()
LocalPrintAndLogFunc("After saving saveResultDict to file")




############################################################################################################################################################
# block, use case B: use the SDAE-learnt features to improve the performance of a regular classifier (LogReg)
############################################################################################################################################################

#bClassifyUseCaseB = False
bClassifyUseCaseB = True
LocalPrintAndLogFunc("bClassifyUseCaseB=" + str(bClassifyUseCaseB))
# use case B: logistic regression
if bClassifyUseCaseB:
	classifTrainBatchSize = 50 * 100
	ctRandIndices = np.random.choice(trainCovariates.shape[0], classifTrainBatchSize, replace=False)
	ctTrainImages = trainCovariates[ctRandIndices]
	ctTrainLabels_OhMatrix = trainLabels[ctRandIndices]
	ctTrainLabels = NumberLib.OneHotMatrixToIntVector(ctTrainLabels_OhMatrix)
	testLabelsIntVector = NumberLib.OneHotMatrixToIntVector(testLabels)
	
	numHiddenLayers = len(hiddenDims)
	ctRecodedTrainImages = FeedForwardHiddensPartial_SessionEval(sess, ctTrainImages, numHiddenLayers, saveResultDict["encodeWeightsDict"], saveResultDict["encodeBiasesDict"], hiddenEncodeFuncName)
	ctRecodedTestImages = FeedForwardHiddensPartial_SessionEval(sess, testCovariates, numHiddenLayers, saveResultDict["encodeWeightsDict"], saveResultDict["encodeBiasesDict"], hiddenEncodeFuncName)
	
	# note, no noise applied here
	ctOrigAndRecodedTrainImages = np.concatenate( (ctTrainImages, ctRecodedTrainImages), axis=1)
	ctOrigAndRecodedTestImages = np.concatenate( (testCovariates, ctRecodedTestImages), axis=1)
	
	# three LogReg: rawData, recoded data, both

	from sklearn.linear_model import LogisticRegression
	lrClassifier_rawData = LogisticRegression()
	trainedLogr_rawData = lrClassifier_rawData.fit(ctTrainImages, ctTrainLabels)
	LocalPrintAndLogFunc("after fitting LogReg to rawData")
	lrClassifier_recodedData = LogisticRegression()
	trainedLogr_recodedData = lrClassifier_recodedData.fit(ctRecodedTrainImages, ctTrainLabels)
	LocalPrintAndLogFunc("after fitting LogReg to recodedData")
	lrClassifier_bothData = LogisticRegression()
	trainedLogr_bothData = lrClassifier_bothData.fit(ctOrigAndRecodedTrainImages, ctTrainLabels)
	LocalPrintAndLogFunc("after fitting LogReg to bothData")

	# eval with testData: testCovariates, testLabels
	import ClassifyLib

	lr_rawData_ypredTest = trainedLogr_rawData.predict(testCovariates)
	lr_recodedData_ypredTest = trainedLogr_recodedData.predict(ctRecodedTestImages)
	lr_bothData_ypredTest = trainedLogr_bothData.predict(ctOrigAndRecodedTestImages)
	LocalPrintAndLogFunc("lr_rawData_ypredTest-shape=" + str(lr_rawData_ypredTest.shape))
	LocalPrintAndLogFunc("lr_recodedData_ypredTest-shape=" + str(lr_recodedData_ypredTest.shape))
	LocalPrintAndLogFunc("lr_bothData_ypredTest-shape=" + str(lr_bothData_ypredTest.shape))

	errorRate_lr_rawData = ClassifyLib.CalcErrorRate(lr_rawData_ypredTest, testLabelsIntVector)
	errorCount_lr_rawData = ClassifyLib.CalcErrorCount(lr_rawData_ypredTest, testLabelsIntVector)
	errorRate_lr_recodedData = ClassifyLib.CalcErrorRate(lr_recodedData_ypredTest, testLabelsIntVector)
	errorCount_lr_recodedData = ClassifyLib.CalcErrorCount(lr_recodedData_ypredTest, testLabelsIntVector)
	errorRate_lr_bothData = ClassifyLib.CalcErrorRate(lr_bothData_ypredTest, testLabelsIntVector)
	errorCount_lr_bothData = ClassifyLib.CalcErrorCount(lr_bothData_ypredTest, testLabelsIntVector)
	numTestExamples = testLabels.shape[0]
	lr_rawData_reportStr = "LogReg-rawData, errorRate {0}, errorCount {1}, numTestExamples {2}".format(errorRate_lr_rawData, errorCount_lr_rawData, numTestExamples)
	lr_recodedData_reportStr = "LogReg-recodedData, errorRate {0}, errorCount {1}, numTestExamples {2}".format(errorRate_lr_recodedData, errorCount_lr_recodedData, numTestExamples)
	lr_bothData_reportStr = "LogReg-bothData, errorRate {0}, errorCount {1}, numTestExamples {2}".format(errorRate_lr_bothData, errorCount_lr_bothData, numTestExamples)
	LocalPrintAndLogFunc(lr_rawData_reportStr)
	LocalPrintAndLogFunc(lr_recodedData_reportStr)
	LocalPrintAndLogFunc(lr_bothData_reportStr)
	LocalPrintAndLogFunc(("*" * 80) + "\n")



	
############################################################################################################################################################
# block, use case C: take original images, corrupt them with noise, give to SDAE-network, see what it generates
############################################################################################################################################################

#bClassifyUseCaseC = False
bClassifyUseCaseC = True
LocalPrintAndLogFunc("bClassifyUseCaseC=" + str(bClassifyUseCaseC))
# use case C: generate data
if bClassifyUseCaseC:
	bSaveInterimLayerGraphs = True # False
	LocalPrintAndLogFunc("bSaveInterimLayerGraphs=" + str(bSaveInterimLayerGraphs))
	nGenerateEpochs = 1 # 10
	LocalPrintAndLogFunc("nGenerateEpochs=" + str(nGenerateEpochs))
	printGenerateFreq = 1 # nGenerateEpochs / 10
	LocalPrintAndLogFunc("printGenerateFreq=" + str(printGenerateFreq))
	numToSavePerEpoch = 10 # 3 # 5
	LocalPrintAndLogFunc("numToSavePerEpoch=" + str(numToSavePerEpoch))
	
	tf.reset_default_graph()
	sess = tf.Session()
	TfLib.PrintTrainableVariables(sess, LocalPrintAndLogFunc)
	LocalPrintAndLogFunc("UseCaseC, After starting session")
	
	for epoch in range(nGenerateEpochs):
		randIndices = np.random.choice(trainCovariates.shape[0], batchSize, replace=False)
		if epoch == 0:
			randIndices = np.array(range(0, batchSize))
		originalInputData = trainCovariates[randIndices]
		originalLabelOhm = trainLabels[randIndices]
		originalLabelIntVec = NumberLib.OneHotMatrixToIntVector(originalLabelOhm)
		
		numHiddenLayers = len(hiddenDims)
		networkFinalLayerData = FeedForwardHiddensPartial_SessionEval(sess, originalInputData, numHiddenLayers, saveResultDict["encodeWeightsDict"], saveResultDict["encodeBiasesDict"], hiddenEncodeFuncName)
		# note, no noise applied here
		
		netGeneratedInput = GenerateInputBackwardHiddensPartial_SessionEval(sess, networkFinalLayerData, numHiddenLayers, saveResultDict["decodeWeightsDict"], saveResultDict["decodeBiasesDict"], hiddenDecodeFuncName)
		
		# also generate images from interim layers
		if numHiddenLayers > 1:
			networkInterimData_layer1 = FeedForwardHiddensPartial_SessionEval(sess, originalInputData, 1, saveResultDict["encodeWeightsDict"], saveResultDict["encodeBiasesDict"], hiddenEncodeFuncName)
			netGeneratedInput_onlyLayer1 = GenerateInputBackwardHiddensPartial_SessionEval(sess, networkInterimData_layer1, 1, saveResultDict["decodeWeightsDict"], saveResultDict["decodeBiasesDict"], hiddenDecodeFuncName)
		if numHiddenLayers > 2:
			networkInterimData_layer2 = FeedForwardHiddensPartial_SessionEval(sess, originalInputData, 2, saveResultDict["encodeWeightsDict"], saveResultDict["encodeBiasesDict"], hiddenEncodeFuncName)
			netGeneratedInput_onlyLayer2 = GenerateInputBackwardHiddensPartial_SessionEval(sess, networkInterimData_layer2, 2, saveResultDict["decodeWeightsDict"], saveResultDict["decodeBiasesDict"], hiddenDecodeFuncName)
		
		# store both original and generated as images, to compare the likeness
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import PlotLib
		import os
		imageSubdir = logResult['path'] + "UseCaseC_images/"
		os.makedirs(imageSubdir)
		printImageFreq = 2
		for saveIndex in range(0, numToSavePerEpoch):
			singleOriginalImage = originalInputData[saveIndex]
			singleGeneratedImage = netGeneratedInput[saveIndex]
			singleLabelInt = originalLabelIntVec[saveIndex]
			singleOriginalFilename = imageSubdir + "Original_Epoch_{0}_Index_{1}.jpg".format(epoch, saveIndex)
			singleGeneratedFilename = imageSubdir + "Generated_Epoch_{0}_Index_{1}.jpg".format(epoch, saveIndex)
			
			if numHiddenLayers > 1:
				singleGeneratedImage_onlyLayer1 = netGeneratedInput_onlyLayer1[saveIndex]
				singleGeneratedLayer1Filename = imageSubdir + "Layer1_Generated_Epoch_{0}_Index_{1}.jpg".format(epoch, saveIndex)
			if numHiddenLayers > 2:
				singleGeneratedImage_onlyLayer2 = netGeneratedInput_onlyLayer2[saveIndex]
				singleGeneratedLayer2Filename = imageSubdir + "Layer2_Generated_Epoch_{0}_Index_{1}.jpg".format(epoch, saveIndex)
			
			PlotLib.CreatePlotOnedMnistFormatImage(singleOriginalImage, singleLabelInt)
			plt.savefig(singleOriginalFilename)
			PlotLib.CreatePlotOnedMnistFormatImage(singleGeneratedImage, singleLabelInt)
			plt.savefig(singleGeneratedFilename)
			if bSaveInterimLayerGraphs:
				if numHiddenLayers > 1:
					PlotLib.CreatePlotOnedMnistFormatImage(singleGeneratedImage_onlyLayer1, singleLabelInt)
					plt.savefig(singleGeneratedLayer1Filename)
				if numHiddenLayers > 2:
					PlotLib.CreatePlotOnedMnistFormatImage(singleGeneratedImage_onlyLayer2, singleLabelInt)
					plt.savefig(singleGeneratedLayer2Filename)
			
			if (saveIndex % printImageFreq) == 0:
				LocalPrintAndLogFunc("saveIndex {0}".format(saveIndex))
		
		if (epoch + 1) % printGenerateFreq == 0:
			LocalPrintAndLogFunc("generate-epoch {0}".format(epoch))
			
	sess.close()
	LocalPrintAndLogFunc("UseCaseC, After closing session")
	
	
	
	
############################################################################################################################################################
# block, cleanup
############################################################################################################################################################
	
harnessEndTime = datetime.datetime.utcnow()
harnessDuration = harnessEndTime - harnessStartTime
LocalPrintAndLogFunc("SdaeTwo harnessDuration " + str(harnessDuration))
LocalPrintAndLogFunc("SdaeTwo duration as str " + DatetimeLib.DurationToString(harnessDuration))

############################################################################################################################################################

