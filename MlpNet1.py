############################################################################################################################################################

import collections
import datetime
import numpy as np
import os
import pickle
import platform
import random
import scipy.stats
import sys
import tensorflow as tf

import DatetimeLib
import FileLib
import GlobalSettings
import LogLib
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

############################################################################################################################################################

LocalPrintAndLogFunc = LogLib.MakeLogBothFunc(logResult['file'])

############################################################################################################################################################

LocalPrintAndLogFunc("MlpNet1 harnessStartTime=" + str(harnessStartTime))
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
LocalPrintAndLogFunc("shape-testImages=" + str(testCovariates.shape))
LocalPrintAndLogFunc("shape-testLabels=" + str(testLabels.shape))

############################################################################################################################################################
# block: set overall parameters [todo-minor, use single dictionary overallParams instead of many globals]
############################################################################################################################################################

boolIsWindows = (platform.system() == 'Windows')
LocalPrintAndLogFunc("boolIsWindows=" + str(boolIsWindows))

# live
inputDim = trainCovariates.shape[1]
outputDim = trainLabels.shape[1]
hiddenDims = (256, 128, 64) # 3-layer
#hiddenDims = (256, 128) # 2-layer
#hiddenDims = (256,) # 1-layer
#hiddenDims = tuple() # 0-layer perceptron
if len(hiddenDims) > 0:
	finalHiddenDim = hiddenDims[len(hiddenDims)-1]
else:
	finalHiddenDim = inputDim

totalDimsList = list(); totalDimsList.append(inputDim); totalDimsList.extend(hiddenDims); totalDimsList.append(outputDim); totalDims = tuple(totalDimsList)
LocalPrintAndLogFunc("inputDim=" + str(inputDim))
LocalPrintAndLogFunc("outputDim=" + str(outputDim))
LocalPrintAndLogFunc("finalHiddenDim=" + str(finalHiddenDim))
LocalPrintAndLogFunc("hiddenDims=" + str(hiddenDims))
LocalPrintAndLogFunc("numHiddenLayers=" + str(len(hiddenDims)))
FileLib.WriteTextFileForParam(logResult['path'], "numHiddenLayers", str(len(hiddenDims)))
LocalPrintAndLogFunc("totalDims=" + str(totalDims))

# combinations
# combo1: hiddenEncodeFunc=relu, outputEncodeFunc=affine[linear], loss=cross-entropy
hiddenEncodeFuncName = "relu"
LocalPrintAndLogFunc("hiddenEncodeFuncName=" + str(hiddenEncodeFuncName))

outputEncodeFuncName = "linear"
LocalPrintAndLogFunc("outputEncodeFuncName=" + str(outputEncodeFuncName))

lossName = "cross-entropy"
LocalPrintAndLogFunc("lossName=" + str(lossName))

learnRate = 0.003
LocalPrintAndLogFunc("learnRate=" + str(learnRate))

nTrainEpochs = 10
LocalPrintAndLogFunc("nTrainEpochs=" + str(nTrainEpochs))

printEpochFreq = 1 # nTrainEpochs / 5
LocalPrintAndLogFunc("printEpochFreq=" + str(printEpochFreq))

batchSize = 100
LocalPrintAndLogFunc("batchSize " + str(batchSize) + " out of " + str(trainCovariates.shape[0]))

numBatchDatasets = int(trainCovariates.shape[0] / batchSize) # int(mnist.train.num_examples / batchSize)
LocalPrintAndLogFunc("numBatchDatasets=" + str(numBatchDatasets))

# Here: decide whether to use loadedWeights (weights loaded from file, eg learnt by SDAE)
# or not (hence random initial weights)
#loadWeightsFilename = None
if boolIsWindows:
	loadWeightsFilename = "d:/temp/Python_Results/Tf_Results/2016-11-07/2016-11-07___17-34-05-529859/saveResultDict.p"
else:
	loadWeightsFilename = "/media/sf_Python_Results/Tf_Results/2016-11-07/2016-11-07___17-34-05-529859/saveResultDict.p"
bLoadWeights = loadWeightsFilename is not None
if bLoadWeights:
	assert os.path.exists(loadWeightsFilename)
LocalPrintAndLogFunc("bLoadWeights=" + str(bLoadWeights))
LocalPrintAndLogFunc("loadWeightsFilename=[" + str(loadWeightsFilename) + "]")

initWbTruncNormal = True
LocalPrintAndLogFunc("initWbTruncNormal=" + str(initWbTruncNormal))
print("*" * 80)

############################################################################################################################################################
# block: create network transform data (weights and biases)
############################################################################################################################################################

# the encodeWeights
# 	dataGenerator options: tf.zeros, tf.linspace, tf.truncated_normal, tf.random_normal

if initWbTruncNormal:
	if len(hiddenDims) > 0: initialEncodeWeights1 = scipy.stats.truncnorm.rvs(-2, 2, size=inputDim * hiddenDims[0]).reshape(inputDim, hiddenDims[0])
	if len(hiddenDims) > 1: initialEncodeWeights2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0] * hiddenDims[1]).reshape(hiddenDims[0], hiddenDims[1])
	if len(hiddenDims) > 2: initialEncodeWeights3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1] * hiddenDims[2]).reshape(hiddenDims[1], hiddenDims[2])
	initialEncodeWeightsOut = scipy.stats.truncnorm.rvs(-2, 2, size=finalHiddenDim * outputDim).reshape(finalHiddenDim, outputDim)
else:
	if len(hiddenDims) > 0:
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
	npMatOut = np.arange(0, finalHiddenDim * outputDim).reshape(finalHiddenDim, outputDim)
	npMatOut = npMatOut / np.max(npMatOut)
	initialEncodeWeightsOut = npMatOut

# encodeBiases
if initWbTruncNormal:
	if len(hiddenDims) > 0: initialEncodeBiases1 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[0])
	if len(hiddenDims) > 1: initialEncodeBiases2 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[1])
	if len(hiddenDims) > 2: initialEncodeBiases3 = scipy.stats.truncnorm.rvs(-2, 2, size=hiddenDims[2])
	initialEncodeBiasesOut = scipy.stats.truncnorm.rvs(-2, 2, size=outputDim)
else:
	if len(hiddenDims) > 0: initialEncodeBiases1 = np.arange(0, hiddenDims[0]) / hiddenDims[0]
	if len(hiddenDims) > 1: initialEncodeBiases2 = np.arange(0, hiddenDims[1]) / hiddenDims[1]
	if len(hiddenDims) > 2: initialEncodeBiases3 = np.arange(0, hiddenDims[2]) / hiddenDims[2]
	initialEncodeBiasesOut = np.arange(0, outputDim) / outputDim
	
############################################################################################################################################################
# block, optionally load weights
# ie read transformData (weights and biases) from the values stored in the file
############################################################################################################################################################

gLoadedWeights = None
if bLoadWeights:
	LocalPrintAndLogFunc("Before loading weights")
	fHandle = open(loadWeightsFilename, "rb")
	loadedWeights = pickle.load(fHandle)
	fHandle.close()
	assert isinstance(loadedWeights, dict)
	gLoadedWeights = loadedWeights
	lwSortedKeysList = list(gLoadedWeights.keys()); lwSortedKeysList.sort()
	LocalPrintAndLogFunc("sortedKeys-gLoadedWeights" + str(lwSortedKeysList))
	loadedInputDim = gLoadedWeights["inputDim"]
	loadedHiddenDims = gLoadedWeights["hiddenDims"]
	assert inputDim == loadedInputDim, "loadedInputDim differs"
	assert hiddenDims == loadedHiddenDims, "loadedHiddenDims differs"
	# the SDAE supplies params for three hiddens, but not for output
	if len(hiddenDims) > 0: 
		initialEncodeWeights1 = gLoadedWeights["encodeWeightsDict"]["h1"]
		initialEncodeBiases1 = gLoadedWeights["encodeBiasesDict"]["b1"]
	if len(hiddenDims) > 1: 
		initialEncodeWeights2 = gLoadedWeights["encodeWeightsDict"]["h2"]
		initialEncodeBiases2 = gLoadedWeights["encodeBiasesDict"]["b2"]
	if len(hiddenDims) > 2: 
		initialEncodeWeights3 = gLoadedWeights["encodeWeightsDict"]["h3"]
		initialEncodeBiases3 = gLoadedWeights["encodeBiasesDict"]["b3"]
	
	loaded_hiddenEncodeFuncName = gLoadedWeights["hiddenEncodeFuncName"]
	hiddenEncodeFuncName = loaded_hiddenEncodeFuncName
	LocalPrintAndLogFunc("changing from loadedWeights, newValue: hiddenEncodeFuncName=" + str(hiddenEncodeFuncName))
	
	LocalPrintAndLogFunc("After loading weights")
	


############################################################################################################################################################
# block, create tensorflow graph (set up variables)
############################################################################################################################################################
	
if len(hiddenDims) > 0: weights1 = tf.Variable(initialEncodeWeights1, name="weights1", dtype=tf.float32)
if len(hiddenDims) > 1: weights2 = tf.Variable(initialEncodeWeights2, name="weights2", dtype=tf.float32)
if len(hiddenDims) > 2: weights3 = tf.Variable(initialEncodeWeights3, name="weights3", dtype=tf.float32)
weightsOut = tf.Variable(initialEncodeWeightsOut, name="weightsOut", dtype=tf.float32)

if len(hiddenDims) > 0: biases1 = tf.Variable(initialEncodeBiases1, name="biases1", dtype=tf.float32)
if len(hiddenDims) > 1: biases2 = tf.Variable(initialEncodeBiases2, name="biases2", dtype=tf.float32)
if len(hiddenDims) > 2: biases3 = tf.Variable(initialEncodeBiases3, name="biases3", dtype=tf.float32)
biasesOut = tf.Variable(initialEncodeBiasesOut, name="biasesOut", dtype=tf.float32)

############################################################################################################################################################

weightsDict = { "hOut": weightsOut }
biasesDict = { "bOut": biasesOut }
if len(hiddenDims) > 0:
	weightsDict["h1"] = weights1
	biasesDict["b1"] = biases1
if len(hiddenDims) > 1:
	weightsDict["h2"] = weights2
	biasesDict["b2"] = biases2
if len(hiddenDims) > 2:
	weightsDict["h3"] = weights3
	biasesDict["b3"] = biases3

initWeightsDict = { "hOut": initialEncodeWeightsOut }
initBiasesDict = { "bOut": initialEncodeBiasesOut }
if len(hiddenDims) > 0:
	initWeightsDict["h1"] = initialEncodeWeights1
	initBiasesDict["b1"] = initialEncodeBiases1
if len(hiddenDims) > 1:
	initWeightsDict["h2"] = initialEncodeWeights2
	initBiasesDict["b2"] = initialEncodeBiases2
if len(hiddenDims) > 2:
	initWeightsDict["h3"] = initialEncodeWeights3
	initBiasesDict["b3"] = initialEncodeBiases3
if len(hiddenDims) > 0: assert isinstance(initWeightsDict["h1"], np.ndarray)
if len(hiddenDims) > 1: assert isinstance(initWeightsDict["h2"], np.ndarray)
if len(hiddenDims) > 2: assert isinstance(initWeightsDict["h3"], np.ndarray)
assert isinstance(initWeightsDict["hOut"], np.ndarray)

############################################################################################################################################################

bPrintAllShapes1 = True
if bPrintAllShapes1:
	print("*" * 80)
	if len(hiddenDims) > 0: print("shape-weights1", weights1.get_shape())
	if len(hiddenDims) > 1: print("shape-weights2", weights2.get_shape())
	if len(hiddenDims) > 2: print("shape-weights3", weights3.get_shape())
	print("shape-weightsOut", weightsOut.get_shape())
	if len(hiddenDims) > 0: print("shape-biases1", biases1.get_shape())
	if len(hiddenDims) > 1: print("shape-biases2", biases2.get_shape())
	if len(hiddenDims) > 2: print("shape-biases3", biases3.get_shape())
	print("shape-biasesOut", biasesOut.get_shape())
	print("*" * 80)
	
############################################################################################################################################################
	
# x is input state; weightDict, biasDict are encode
# whole net
def FeedForwardOutput(x, weightDict, biasDict, hiddenEncodeFuncNameStr:str, outputEncodeFuncNameStr:str):
	numKeys = len(weightsDict.keys())
	numHiddenLayers = numKeys-1 # always have weightsOut
	
	if numHiddenLayers > 0:
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h1']), biasDict['b1']), hiddenEncodeFuncNameStr)
	
	if numHiddenLayers > 1:
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h2']), biasDict['b2']), hiddenEncodeFuncNameStr)
		
	if numHiddenLayers > 2:
		x = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['h3']), biasDict['b3']), hiddenEncodeFuncNameStr)
	
	layer_out = TfLib.ActivationOp(tf.add(tf.matmul(x, weightDict['hOut']), biasDict['bOut']), outputEncodeFuncNameStr)
	return layer_out
	
def PrintNetParamsSummary1(initialReportStr, printFunc, weightDict, biasDict, printShape:bool = False):
	printFunc(initialReportStr)
	numKeys = len(weightDict.keys())
	numHiddenLayers = numKeys-1 # always have weightsOut
	printFunc("numHiddenLayers=" + str(numHiddenLayers))
	for hLayerIndex in range(numHiddenLayers):
		printFunc("hLayerIndex=" + str(hLayerIndex))
		hKeyStr = "h" + str(hLayerIndex+1)
		bKeyStr = "b" + str(hLayerIndex+1)
		assert hKeyStr in weightDict.keys()
		assert bKeyStr in biasDict.keys()
		encodeWeightsThisLayer = weightDict[hKeyStr]
		encodeBiasesThisLayer = biasDict[bKeyStr]
	
		if isinstance(encodeWeightsThisLayer, np.ndarray):
			printFunc("encodeWeightsThisLayer [0,0] " + str(encodeWeightsThisLayer[0,0]))
			printFunc("encodeBiasesThisLayer [0] " + str(encodeBiasesThisLayer[0]))
		else:
			assert False
		printFunc("-" * 80)
		if printShape:
			if isinstance(encodeWeightsThisLayer, np.ndarray):
				printFunc("shape-encodeWeightsThisLayer " + str(encodeWeightsThisLayer.shape))
				printFunc("shape-encodeBiasesThisLayer " + str(encodeBiasesThisLayer.shape))
			else:
				assert False
			printFunc("-" * 80)
	# print outParams
	encodeWeightsOutLayer = weightDict["hOut"]
	encodeBiasesOutLayer = biasDict["bOut"]
	if isinstance(encodeWeightsOutLayer, np.ndarray):
		printFunc("encodeWeightsOutLayer [0,0] " + str(encodeWeightsOutLayer[0,0]))
		printFunc("encodeBiasesOutLayer [0] " + str(encodeBiasesOutLayer[0]))
	else:
		assert False
	printFunc("-" * 80)
	if printShape:
		if isinstance(encodeWeightsOutLayer, np.ndarray):
			printFunc("shape-encodeWeightsOutLayer " + str(encodeWeightsOutLayer.shape))
			printFunc("shape-encodeBiasesOutLayer " + str(encodeBiasesOutLayer.shape))
		else:
			assert False
		printFunc("-" * 80)
	printFunc("-" * 80)
	
	
	
############################################################################################################################################################
# block, train MLP
############################################################################################################################################################
	
LocalPrintAndLogFunc("Before training MlpNet1")
numHiddenLayers = len(hiddenDims)

# tf Graph 
xInputData = tf.placeholder("float", [None, inputDim])
yLabels = tf.placeholder("float", [None, outputDim])

netPrediction = FeedForwardOutput(xInputData, weightsDict, biasesDict, hiddenEncodeFuncName, outputEncodeFuncName)

if lossName == "cross-entropy":
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = netPrediction, labels = yLabels))
else:
	assert False, "unknown lossName"
	
trainOp = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(cost)

initOp = tf.global_variables_initializer()

saveResultDict = dict()
saveResultDict["totalDims"] = totalDims
saveResultDict["hiddenDims"] = hiddenDims
saveResultDict["inputDim"] = inputDim
saveResultDict["outputDim"] = outputDim
saveResultDict["hiddenEncodeFuncName"] = hiddenEncodeFuncName
saveResultDict["outputEncodeFuncName"] = outputEncodeFuncName
saveResultDict["lossName"] = lossName
saveResultDict["initWeightsDict"] = initWeightsDict
saveResultDict["initBiasesDict"] = initBiasesDict

# Training
with tf.Session() as sess:
	sess.run(initOp)
	LocalPrintAndLogFunc("After initialising all variables")
	
	TfLib.PrintTrainableVariables(sess, LocalPrintAndLogFunc)
	
	for epoch in range(nTrainEpochs):
		avg_cost = 0.
		
		# Loop over all batches: progressive training
		for i in range(numBatchDatasets):
			batch_x, batch_y = mnist.train.next_batch(batchSize)
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([trainOp, cost], feed_dict={xInputData: batch_x, yLabels: batch_y})
			# Compute average loss
			avg_cost += c / numBatchDatasets
			
		# Logging
		if epoch % printEpochFreq == 0:
			reportStr = "Epoch {0}, cost {1}".format(epoch+1, avg_cost)
			LocalPrintAndLogFunc(reportStr)
				
	LocalPrintAndLogFunc("After training")

	# Test model
	correct_prediction = tf.equal(tf.argmax(netPrediction, 1), tf.argmax(yLabels, 1))
	
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	resultAccuracy = accuracy.eval({xInputData: mnist.test.images, yLabels: mnist.test.labels})
	LocalPrintAndLogFunc("Accuracy: " + str(resultAccuracy))
	
	if len(hiddenDims) > 0: finalWeights_h1 = weightsDict["h1"].eval(session=sess)
	if len(hiddenDims) > 1: finalWeights_h2 = weightsDict["h2"].eval(session=sess)
	if len(hiddenDims) > 2: finalWeights_h3 = weightsDict["h3"].eval(session=sess)
	finalWeights_hOut = weightsDict["hOut"].eval(session=sess)
	if len(hiddenDims) > 0: finalBiases_b1 = biasesDict["b1"].eval(session=sess)
	if len(hiddenDims) > 1: finalBiases_b2 = biasesDict["b2"].eval(session=sess)
	if len(hiddenDims) > 2: finalBiases_b3 = biasesDict["b3"].eval(session=sess)
	finalBiases_bOut = biasesDict["bOut"].eval(session=sess)
	
	finalWeightsDict = { "hOut": finalWeights_hOut }
	finalBiasesDict = { "bOut": finalBiases_bOut }
	if len(hiddenDims) > 0:
		finalWeightsDict["h1"] = finalWeights_h1
		finalBiasesDict["b1"] = finalBiases_b1
	if len(hiddenDims) > 1:
		finalWeightsDict["h2"] = finalWeights_h2
		finalBiasesDict["b2"] = finalBiases_b2
	if len(hiddenDims) > 2:
		finalWeightsDict["h3"] = finalWeights_h3
		finalBiasesDict["b3"] = finalBiases_b3
	
LocalPrintAndLogFunc("After training MlpNet1")

saveResultDict["weightsDict"] = finalWeightsDict
saveResultDict["biasesDict"] = finalBiasesDict

sortedSrdKeys = list(saveResultDict.keys()); sortedSrdKeys.sort()
LocalPrintAndLogFunc("saveResultDict-sortedSrdKeys" + str(sortedSrdKeys))

# show weights before and after
PrintNetParamsSummary1("After all training, initParams", LocalPrintAndLogFunc, initWeightsDict, initBiasesDict, printShape = True)
PrintNetParamsSummary1("After all training, finalParams", LocalPrintAndLogFunc, finalWeightsDict, finalBiasesDict, printShape = True)

# save learnt parameters to file
srdFilename = logResult['path'] + 'saveResultDict.p'
srdFilehandler = open(srdFilename, 'wb')
pickle.dump(saveResultDict, srdFilehandler)
srdFilehandler.close()
LocalPrintAndLogFunc("After saving saveResultDict to file")
	
	
	
	
############################################################################################################################################################
# block, cleanup
############################################################################################################################################################	
	
harnessEndTime = datetime.datetime.utcnow()
harnessDuration = harnessEndTime - harnessStartTime
LocalPrintAndLogFunc("MlpNet1 harnessDuration " + str(harnessDuration))
LocalPrintAndLogFunc("MlpNet1 duration as str " + DatetimeLib.DurationToString(harnessDuration))

############################################################################################################################################################

