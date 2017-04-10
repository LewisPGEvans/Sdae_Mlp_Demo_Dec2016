############################################################################################################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import platform
import random

############################################################################################################################################################
	
# Note: scales-255 by default
# value: None
def CreatePlotOnedMnistFormatImage (onedArray, labelInt:int = -1, multiplyDataBy255:bool = True, verbose:bool = False):
	import numpy as np
	assert len(onedArray.shape) == 1
	assert onedArray.shape[0] == 784
	assert onedArray.dtype == "float32"
	if verbose:
		print("onedArray: type {0}, dtype {1}, shape {2}".format(type(onedArray), onedArray.dtype, onedArray.shape))
		print("min-onedArray", np.min(onedArray))
		print("max-onedArray", np.max(onedArray))
	if multiplyDataBy255:
		onedArray = onedArray * 255
	imgAsUint8 = np.array(onedArray, dtype='uint8')
	imgAsUint8 = imgAsUint8.reshape(28,28)
	
	# show label as title
	if labelInt != -1:
		plt.title('Label {label}'.format(label=labelInt))
	
	if verbose:
		print("imgAsUint8: type {0}, dtype {1}, shape {2}".format(type(imgAsUint8), imgAsUint8.dtype, imgAsUint8.shape))
		print("min-imgAsUint8", np.min(imgAsUint8))
		print("max-imgAsUint8", np.max(imgAsUint8))
	plt.imshow(imgAsUint8, cmap="gray")
		
############################################################################################################################################################

