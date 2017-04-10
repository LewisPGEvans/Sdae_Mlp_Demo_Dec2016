import numpy as np
import random

############################################################################################################################################################

# value: scalar-int
def CalcErrorCount (predClasses, trueClasses):
	assert predClasses.shape == trueClasses.shape
	errorCount = (trueClasses != predClasses).sum()
	return errorCount

# value: scalar-float	
def CalcErrorRate (predClasses, trueClasses):
	assert predClasses.shape == trueClasses.shape
	errorRate = CalcErrorCount(predClasses, trueClasses) / (predClasses.shape[0])
	return errorRate
		
############################################################################################################################################################

