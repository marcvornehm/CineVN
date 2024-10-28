# based on https://github.com/mrirecon/bart/blob/master/python/wslsupport.py @ commit 1d8bf95
# changes are marked with MV

import string
import os

def PathCorrection(inData):
	outData=inData
	for i in string.ascii_lowercase: #Replace drive letters with /mnt/
		outData=outData.replace(i+':','/mnt/'+i) #if drive letter is supplied in lowercase
		outData=outData.replace(i.upper()+':','/mnt/'+i) #if drive letter is supplied as uppercase
	outData=outData.replace(os.path.sep, '/') #Change windows filesep to linux filesep

	return outData