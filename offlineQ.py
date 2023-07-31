#######################################################################
### This script generates a Q matrix according to the parameters in the configuration file

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import importlib
import sys
from f_modRSW import make_grid, step_forward_topog, time_step

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_fc = config.Nk_fc
L = config.L
V = config.V
Nmeas = config.Nmeas
Neq = config.Neq
dres = config.dres
cfl_fc = config.cfl_fc
Hc = config.Hc
Hr = config.Hr
Ro = config.Ro
assim_lenght = config.assim_lenght
dtmeasure = config.dtmeasure
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
model_noise = config.model_noise
Nhr = config.Nhr
Q_FUNC = config.Q_FUNC
rMODNOISE = config.rMODNOISE
hMODNOISE = config.hMODNOISE

# Q-matrix from pre-defined model noise (diagonal)
def Q_predef():
    var_h = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[0]
    var_u = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[1]
    var_r = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[2]
    Q = np.diag(np.concatenate((var_h,var_u,var_r)))

    return Q

#################################################################
# Cycle over truth trajectory times.  Use each value as the initial condition
# for both the truth (obtained from X_tr at the next timestep) and a
# lower-resolution forecast.
def Q_nhr():
    B = np.load(str(outdir + '/B.npy')) # topog
    U_tr = np.load(str(outdir + '/U_tr.npy')) # truth
    fc_grid = make_grid(Nk_fc, L) # forecast
    Kk_fc = fc_grid[0]
    nsteps = Nmeas*assim_lenght
    X = np.zeros([Neq * Nk_fc, nsteps])
    U = np.copy(U_tr[:, 0::dres, :]) # Sample truth at forecast resolution
    tn = 0
    for T in range(nsteps):
        tmeasure = tn+Nhr*dtmeasure
        print('*** Integrating between time:', tn, ' and: ', tmeasure, ' ***')
        U_fc = np.copy(U[:, :, T])
        while tn < tmeasure:
            dt = time_step(U_fc, Kk_fc, cfl_fc, cc2, beta, g) # compute stable time step
            tn = tn + dt
            if tn > tmeasure:
                dt = dt - (tn - tmeasure) + 1e-12
            U_fc = step_forward_topog(U_fc, B, dt, Neq, Nk_fc, Kk_fc, Hc, Hr, cc2, beta, alpha2, g)
        print('*** Storing the error at time: ', tmeasure, ' ***')
        X[:, T] = U[:, :, T+Nhr].flatten() - U_fc.flatten()
        tmeasure+=dtmeasure
    print("Means of proxy error components = ", np.mean(U[:, :, 1:(nsteps+1)] - np.repeat(U_fc, nsteps).reshape(Neq, Nk_fc, nsteps), axis=(1, 2)))

    # Compute the covariance matrix.
    Q = np.cov(X, bias=False)
    # Extrapolate diagonal of Q
    Q_diag = np.array(Q.diagonal())
    # Pose the covariance of r to 0
    if(rMODNOISE==0):
        Q_diag[2*Nk_fc:] = 0.0
    if(hMODNOISE==0):
        Q_diag[:Nk_fc] = 0.0

    # Return a diagonal matrix
    Q = np.diag(Q_diag)

    return Q

##################################################################

f_path_Q = str(outdir+'/Qmatrix.npy') 

# Load or generate Q matrix according to the choice made in the config file 
try:
    Q = np.load(f_path_Q)
    print(' *** Loading Q matrix *** ')
except:
    print(' *** Generating Q matrix *** ')
    Q = eval(Q_FUNC)
    print(("Min. and max. variances", np.amin(Q.diagonal()), np.amax(Q.diagonal())))
    np.save(str(outdir + '/Qmatrix'), Q)
