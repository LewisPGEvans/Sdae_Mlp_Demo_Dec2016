import datetime
import os

import DatetimeLib
import FileLib

############################################################################################################################################################

# global; its value is set by StartUniqueLog
gLogResult = None

############################################################################################################################################################

# assumes that baseDir ends in forwardSlash
# value: dict (path, file, utcNow)
# Path will end in /

def StartUniqueLog(baseDir:str):
	assert isinstance(baseDir, str)
	assert baseDir.endswith("/")
	utcNow = datetime.datetime.utcnow()
	utcNowDateOnlyStr = DatetimeLib.DateToStringUnderscored(utcNow)
	utcNowStrUnderscore = DatetimeLib.DateTimeToStringUnderscored(utcNow)
	
	# use pbsJobId or processId to reduce collisions of same-dir
	pbsJobId = os.environ.get("PBS_JOBID")
	extendPathStr = ""
	if not pbsJobId is None and len(pbsJobId) > 0:
		extendPathStr = "_jbid_" + str(pbsJobId)
	else:
		# no pbsJobId, use processId
		processId = os.getpid()
		extendPathStr = "_pid_" + str(processId)
	utcNowStrUnderscore += extendPathStr
	path = baseDir + utcNowDateOnlyStr + "/" + utcNowStrUnderscore + "/"
		
	assert os.path.exists(path) == False
	os.makedirs(path)
	pathRelativeFilename = "log.txt"
	logFileName = os.path.join(path, pathRelativeFilename)
	logResult = { 'path':path, 'file':logFileName, 'utcNow':utcNow}
	FileLib.WriteStringToFile(logFileName, "Log start, written by LogLib.StartUniqueLog at " + utcNowStrUnderscore + "\n")
	global gLogResult
	assert gLogResult is None #  prevent second duplicate-non-unique log
	gLogResult = logResult
	return logResult
	
# no value
def LogBothFileAndTerminal (logfile:str, msg:str):
	assert isinstance(logfile, str)
	assert os.path.exists(logfile)
	assert isinstance(msg, str)
	
	print(msg)
	if not logfile is None:
		FileLib.AppendStringToFile(logfile, msg + "\n")
		
############################################################################################################################################################
		
# value: func (singleStringArg=msg)
"""
usage: 
LocalPrintAndLogFunc = LogLib.MakeLogBothFunc(logResult['file'])
LocalPrintAndLogFunc(msg)
"""
def MakeLogBothFunc (logfile:str): 
	assert isinstance(logfile, str)
	return lambda msg: LogBothFileAndTerminal(logfile, msg)
	
############################################################################################################################################################

