############################################################################################################################################################

import numpy as np

############################################################################################################################################################

# x is np.ndarray, 1d or 2d
# frac is float in [0,1]
# value is same type as x
def ApplyMaskingNoise(x:np.ndarray, frac:float):
	assert isinstance(x, np.ndarray)
	assert isinstance(frac, float)
	assert len(x.shape) in (1, 2)
	
	temp = np.copy(x)
	if len(x.shape) == 2:
		temp = temp.flatten()
	indices = np.random.choice(len(temp), round(frac * len(temp)), replace=False)
	temp[indices] = 0
	if len(x.shape) == 2:
		temp = temp.reshape(x.shape)
	return temp
	
############################################################################################################################################################

# value: scalar integer
def OneHotVectorToInt (ohVector):
	assert len(ohVector.shape) == 1
	numVals = ohVector.shape[0]
	assert np.sum(ohVector) == 1
	oneValues = ohVector[ohVector == 1]
	assert len(oneValues) == 1
	zeroValues = ohVector[ohVector == 0]
	assert len(zeroValues) == numVals-1
	for i in range(0, numVals):
		if ohVector[i] == 1:
			return int(i)
	return int(-1)
	
# value: np.array, 1d; intVector
def OneHotMatrixToIntVector (ohMatrix):
	assert len(ohMatrix.shape) == 2
	numRows = ohMatrix.shape[0]
	result = np.repeat(0, numRows)
	#result = np.zeros(numRows)
	for i in range(0, numRows):
		thisOhVec = ohMatrix[i,:]
		result[i] = OneHotVectorToInt(thisOhVec)
	return result
	
############################################################################################################################################################

