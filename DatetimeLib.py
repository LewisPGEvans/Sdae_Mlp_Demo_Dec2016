import datetime
import string
import time

############################################################################################################################################################

def DateTimeToStringUnderscored (dtObj):
	localFormatStr = "%Y-%m-%d %H:%M:%S.%f"
	dateTimeStr = dtObj.strftime(localFormatStr)
	dateTimeStr = str.replace(dateTimeStr, " ", "___")
	dateTimeStr = str.replace(dateTimeStr, ":", "-")
	dateTimeStr = str.replace(dateTimeStr, ".", "-")
	return dateTimeStr
	
def DateToStringUnderscored (dtObj):
	localFormatStr = "%Y-%m-%d"
	dateStr = dtObj.strftime(localFormatStr)
	dateStr = str.replace(dateStr, " ", "___")
	dateStr = str.replace(dateStr, ":", "-")
	dateStr = str.replace(dateStr, ".", "-")
	return dateStr

def DurationToString(duration, secondsRoundDp:int = 6):
	totalSeconds = duration.seconds
	seconds = totalSeconds % 60
	nextDur = (totalSeconds - seconds) / 60
	minutes = nextDur % 60
	nextDur = (nextDur - minutes) / 60
	hours = nextDur % 60
	days = duration.days
	millisecs = duration.microseconds / 1000
	
	secsAndMillisecs = seconds + (millisecs / 1000)
	durationStr = str(round(secsAndMillisecs, secondsRoundDp)) + "s"
	if minutes > 0:
		durationStr = str(minutes) + "m " + durationStr
	if hours > 0:
		durationStr = str(hours) + "h " + durationStr
	if days > 0:
		durationStr = str(days) + "d " + durationStr
  
	return durationStr
	
############################################################################################################################################################

