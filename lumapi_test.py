import sys, os
#default path for current release 
sys.path.append("/Machintosh HD/Applications/Lumerical v211/Contents/API/Python") 
sys.path.append(os.path.dirname(__file__)) #Current directory

import lumapi

session = lumapi.FDTD()