#######################################################################
# Create readme.txt file for summarising each experiment, output saved in 
# <dirn> to accompany outputted data from main run script and EnKF subroutine.
#######################################################################

from datetime import datetime
import importlib

def create_readme(dirn, config_file):

    ################################################################
    # IMPORT PARAMETERS FROM CONFIGURATION FILE
    ################################################################

    spec = importlib.util.spec_from_file_location("config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    Nk_tr = config.Nk_tr
    Nk_fc = config.Nk_fc
    H0 = config.H0
    Ro = config.Ro
    Fr = config.Fr
    Hc = config.Hc
    Hr = config.Hr
    alpha2 = config.alpha2
    beta = config.beta
    cc2 = config.cc2
    cfl_fc = config.cfl_fc
    cfl_tr = config.cfl_tr
    n_ens = config.n_ens
    sig_ic = config.sig_ic
    NIAU = config.NIAU
    n_obs = config.n_obs
    ob_noise = config.ob_noise
    add_inf = config.add_inf
        
    fname = str(dirn+'/readme.txt')

    f = open(fname,'w')
    print(' ------------- FILENAME ------------- ', file=f) 
    print(fname, file=f)   
    print(' ', file=f)   
    print('Created: ', str(datetime.now()), file=f)   
    print(' ', file=f)   
    print(' -------------- SUMMARY: ------------- ', file=f)  
    print(' ', file=f) 
    print('Dynamics:', file=f)
    print(' ', file=f) 
    print('Ro =', Ro, file=f)  
    print('Fr = ', Fr, file=f)
    print('(H_0 , H_c , H_r) =', [H0, Hc, Hr], file=f) 
    print('(alpha, beta, c2) = ', [alpha2, beta, cc2], file=f)
    print('(cfl_fc, cfl_tr) = ', [cfl_fc, cfl_tr], file=f)
    print('IC noise for initial ens. generation: ', sig_ic, file=f)
    print(' ', file=f) 
    print('Assimilation:', file=f)
    print(' ', file=f) 
    print('Forecast resolution (number of gridcells) =', Nk_fc, file=f)
    print('Truth resolution (number of gridcells) =', Nk_tr, file=f)   
    print(' ')  
    print('Number of ensembles =', n_ens, file=f)
    print('i.e., total no. of obs. =', n_obs, file=f)
    print('Observation noise =', ob_noise, file=f)
    print('Additive inflation factor =', add_inf, file=f)
    print(' ', file=f)   
    print('Duration of the Incremental Analysis Update into the forecast (in number of cycles) = ', NIAU, file=f)
    print(' ----------- END OF SUMMARY: ---------- ', file=f)  
    print(' ', file=f)  
    f.close()
