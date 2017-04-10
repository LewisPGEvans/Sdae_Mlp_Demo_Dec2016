############################################################################################################################################################

import csv
import datetime
import glob
import os

############################################################################################################################################################

# value: string
def GetPathEndingInSlash(path:str):
	result = path
	if not path.endswith("/"):
		result = path + "/"
	return result

# no value
# does open-write-close
# assumes file does Not exist yet; throws if exists
def WriteStringToFile (argFilename:str, argString:str):
	assert isinstance(argFilename, str)
	assert isinstance(argString, str)
	assert not os.path.exists(argFilename)
	file1 = open(argFilename, 'w')
	file1.write(argString)
	file1.close()
	
# no value
# does open-write-close	
def AppendStringToFile (argFilename:str, argString:str):
	assert isinstance(argFilename, str)
	assert os.path.exists(argFilename)
	assert isinstance(argString, str)
	file1 = open(argFilename, 'a')
	file1.write(argString)
	file1.close()
	
# value: string
def FixFilenameStr(argStr:str, replaceSpaces=True):
	assert isinstance(argStr, str)
	fixedStr = argStr
	fixedStr = str.replace(fixedStr, ".", "_")
	fixedStr = str.replace(fixedStr, ",", "_") 
	fixedStr = str.replace(fixedStr, ":", "_")
	fixedStr = str.replace(fixedStr, "|", "_")
  
	if replaceSpaces:
		fixedStr = str.replace(fixedStr, " ", "_")
	return fixedStr
	
# use this when paramValue can be part of a valid filename, along with paramName
# no value
def WriteTextFileForParam(path:str, paramName:str, paramValue):
	assert isinstance(path, str)
	assert isinstance(paramName, str)
	assert os.path.exists(path) == True
	
	filenameStem = "z_[" + paramName + "]_[" + str(paramValue) + "]"
	filenameStem = FixFilenameStr(filenameStem) 
	pathEndingSlash = GetPathEndingInSlash(path)
	relativeFilename = filenameStem + ".txt"
	fullFilename = os.path.join(pathEndingSlash, relativeFilename)
	
	utcNow = datetime.datetime.utcnow()
	localFormatStr = "%Y-%m-%d %H:%M:%S.%f"
	utcNowStr = utcNow.strftime(localFormatStr)
	
	filenameStem_andTime = filenameStem + "_time_" + utcNowStr
	WriteStringToFile(fullFilename, filenameStem_andTime)
	
############################################################################################################################################################
		
