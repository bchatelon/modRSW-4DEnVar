#!/usr/bin/env python
#######################################################################
# Perturbed obs. EnKF for 1.5D SWEs with rain variable and topography
#######################################################################

'''
> truth generated outside of outer loop as this is the same for all experiments 
> uses subroutine <subr_enkf_modRSW_p> that parallelises ensemble forecasts using multiprocessing module
> Data saved to automatically-generated directories and subdirectories with accompanying readme.txt file.
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import sys
import importlib.util

# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from subr_4DEnVar_cycle import run_4DEnVar

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
config_string = sys.argv[1]
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir

### LOAD TRUTH, OBSERVATIONS AND OBSERVATION OPERATOR ###

f_obs_name = str(outdir+'/Y_obs.npy')
f_H_name = str(outdir+'/H.npy')

try:
    H = np.load(f_H_name)
except:
    print('Failed to load the observation operator H:  run create_truth+obs.py first')

try:
    Y_obs = np.load(f_obs_name)
except:
    print('Failed to load the observations: run create_truth+obs.py first')

##################################################################    
# EnKF: outer loop 
##################################################################
print('--------- LAUNCH EXPERIMENT ---------')
if __name__ == '__main__': 
    run_4DEnVar(Y_obs, H, outdir, config_string)

print(' ')   
print('--------- END OF EXPERIMENT ---------')  

##################################################################    
#                        END OF PROGRAM                          #
##################################################################
