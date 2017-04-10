import os
import platform

############################################################################################################################################################

# These settings are for my local setup (windows, or linux via VM, or hpc-linux)
# to change directory, just alter this code

############################################################################################################################################################
	
# value: string, ending in /
def GlobalSettings_GetRootResultsDir (verbose=False):
	boolIsLinux = (platform.system() == 'Linux')
	if boolIsLinux:
		baseDir = "/media/sf_Python_Results/"
		if not os.path.exists(baseDir):
			baseDir = "/work/lpe10/Python_Results/" # hpc
		if not os.path.exists(baseDir):
			baseDir = "../Python_Results/" # mcluster
	else:
		baseDir = "d:/temp/Python_Results/" # assuming Windows
    
	if verbose:
		print("results-baseDir", baseDir)
  
	# DirectoryExists
	baseDirMinusFinalSlash = baseDir[0:(len(baseDir)-1)]
	if verbose:
		print("results-baseDirMinusFinalSlash", baseDirMinusFinalSlash)
	assert os.path.exists(baseDirMinusFinalSlash)

	return baseDir

############################################################################################################################################################

