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

from f_modRSW import make_grid

#################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE                     #
#################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_tr = config.Nk_tr
Nk_fc = config.Nk_fc
L = config.L
cfl_tr = config.cfl_tr
Neq = config.Neq
H0 = config.H0
V = config.V
Hc = config.Hc
Hr = config.Hr
Ro = config.Ro
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
tmax = config.tmax
Nmeas = config.Nmeas
Nforec = config.Nforec
dtmeasure = config.dtmeasure
assim_lenght = config.assim_lenght
dres = config.dres

#################################################################
# truth creation function
#################################################################

def generate_truth(U_tr_array, B_tr, Neq, Nk_tr, tr_grid, cfl_tr, dtmeasure, tmax, f_path_name, Hc, Hr, cc2, beta, alpha2, g):
    
    from f_modRSW import time_step, step_forward_topog

    Kk_tr = tr_grid[0] 
    
    tn = 0.0
    tmeasure = dtmeasure
    
    U_tr = U_tr_array[:,:,0]
    
    print(' ')
    print('Integrating forward from t =', tn, 'to', tmax,'...')
    print(' ')
    
    index = 1 # for U_tr_array (start from 1 as 0 contains IC).
    while tn < tmax:
        dt = time_step(U_tr,Kk_tr,cfl_tr,cc2,beta,g)
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12

        U_tr = step_forward_topog(U_tr,B_tr,dt,Neq,Nk_tr,Kk_tr,Hc,Hr,cc2,beta,alpha2,g)

        if tn > tmeasure:
            U_tr_array[:,:,index] = U_tr
            print('*** STORE TRUTH at observing time = ',tmeasure,' ***')
            tmeasure = tmeasure + dtmeasure
            index = index + 1
            
    np.save(f_path_name,U_tr_array)
    
    print('* DONE: truth array saved to:', f_path_name, ' with shape:', np.shape(U_tr_array), ' *')
        
    return U_tr_array

if __name__=='__main__' :

    ##################################################################    
    # Mesh generation and IC for truth 
    ##################################################################
    tr_grid =  make_grid(Nk_tr,L) # truth
    x_tr = tr_grid[1]

    ### Truth ic
    B_tr = config.topog(make_grid(Nk_tr,L)[1],Nk_tr)
    U0_tr = config.ic(x_tr,Nk_tr,Neq,H0,B_tr)

    U_tr_array = np.empty([Neq,Nk_tr,Nmeas*assim_lenght+Nforec+1])
    U_tr_array[:,:,0] = U0_tr

    f_path_name = str(outdir+'/U_tr.npy')

    try:
        U_tr_array = np.load(f_path_name)
        print(' *** Loading truth trajectory... *** ')
    except:
        print(' *** Generating truth trajectory... *** ')
        U_tr_array = generate_truth(U_tr_array, B_tr, Neq, Nk_tr, tr_grid, cfl_tr, dtmeasure, tmax, f_path_name, Hc, Hr, cc2, beta, alpha2, g)

    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc,Nmeas*assim_lenght+Nforec+1])
    for i in range(0,Nk_fc):
        U_tmp[:,i,:] = U_tr_array[:,i*dres,:]

    # [h,hu,hr] -> [h,u,r]
    U_tmp[1:,:,:] = U_tmp[1:,:,:]/U_tmp[0,:,:]

    X_tr = np.empty([Neq*Nk_fc,Nmeas*assim_lenght+Nforec+1])
    for i in range(Nmeas*assim_lenght+Nforec+1) :
        X_tr[:,i] = U_tmp[:,:,i].flatten()
    np.save(f'{outdir}/X_tr.npy',X_tr)

#### END OF THE PROGRAM ####