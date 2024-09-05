import importlib.util
#default path for current release 
spec_lin = importlib.util.spec_from_file_location('lumapi', "/opt/lumerical/v242/api/python/lumapi.py")
#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_lin)
spec_lin.loader.exec_module(lumapi)