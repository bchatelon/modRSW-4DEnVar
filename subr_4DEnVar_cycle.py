#######################################################################
# 4DEnVar for modRSW with topography
#######################################################################
''' 
explainations
'''

def run_4DEnVar(Y_obs, H, dirname, config_file):

    # GENERIC MODULES REQUIRED
    import numpy as np
    import os
    import multiprocessing as mp
    import importlib.util
    import sys

    # CUSTOM FUNCTIONS AND MODULES REQUIRED
    from f_modRSW import make_grid, ens_forecast_topog
    from f_4DEnVar import analysis_step_4DEnVar
    from create_readme import create_readme

    # IMPORT PARAMETERS FROM CONFIGURATION FILE
    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # see config.py for details
    Nk_fc = config.Nk_fc
    L = config.L
    Neq = config.Neq
    n_d = config.n_d
    Hc = config.Hc
    Hr = config.Hr
    alpha2 = config.alpha2
    beta = config.beta
    cc2 = config.cc2
    g = config.g
    cfl_fc = config.cfl_fc
    n_ens = config.n_ens
    TIMEOUT_PAR = config.TIMEOUT_PAR
    dtmeasure = config.dtmeasure
    Nmeas = config.Nmeas
    Nforec = config.Nforec
    NIAU = config.NIAU
    assim_lenght = config.assim_lenght
    add_inf = config.add_inf
    
    # LOADING FILES    
    Q = np.load(f'{dirname}/Qmatrix.npy')   # Model error covariance matrix
    B = np.load(f'{dirname}/B.npy')         # Topography
    U0ens = np.load(f'{dirname}/U0ens.npy') # Ensemble initial conditions
    R = np.load(f'{dirname}/Rmatrix.npy')   # Observation error covariance matrix

    # Mesh generation for forecasts
    Kk_fc = make_grid(Nk_fc,L)[0]
    
    # Define arrays for outputting data
    X_an = np.empty([n_d,n_ens,Nmeas])
    X_fc = np.empty([n_d,n_ens,Nmeas+1,Nforec])
    q2store = np.empty([n_d,n_ens,Nmeas+1,Nforec])

    # create readme file for exp
    create_readme(dirname, config_file)
    
    # list of fixed parameters given to assim func
    PARS = [Nk_fc, n_ens, n_d, Neq, assim_lenght]

    ##################################################################
    #  Integrate ensembles forward in time until assim time

    print('--------- CYCLED FORECAST-ASSIMILATION SYSTEM ---------')
    
    # INIT CYCLE VARIABLES AND PARAMETERS
    U = U0ens
    index = 0 # to step through assim times
    num_cores_use = os.cpu_count() # get number of available cpu for parallel computing

    while True :
             
        try:

            forec_T = 0

            print('')
            print('========= FORECAST =========')

            while forec_T < Nforec: 

                # ADDITIVE INFLATION
                q = add_inf * np.random.multivariate_normal(np.zeros(n_d), Q, n_ens)
                q[0,:] = 0. # no additive inflation for control member 
                q_ave = np.mean(q,axis=0)
                q = q - q_ave
                q = q.T
                if forec_T > NIAU: q[:,:] = 0.0 # stop additive inflation after NIAU *dtmeasure*
                q2store[:,:,index,forec_T] = np.copy(q)
                
                # Integrating ensemble
                print('Ensemble integration from =', round((index*assim_lenght+forec_T)*dtmeasure,3) ,' to', round((index*assim_lenght+forec_T+1)*dtmeasure,3))
                pool = mp.Pool(processes=num_cores_use)
                mp_out = [pool.apply_async(ens_forecast_topog, args=(N, U, B, q, Neq, Nk_fc, Kk_fc, cfl_fc, dtmeasure, Hc, Hr, cc2, beta, alpha2, g)) for N in range(0,n_ens)]
                U = [p.get(timeout=TIMEOUT_PAR) for p in mp_out]
                pool.close()
                pool.join()

                U = np.swapaxes(U,0,1)
                U = np.swapaxes(U,1,2)

                # Saving X from U : (h,hu,hr)->(h,u,r) and flattening
                U_forec_tmp = np.copy(U)
                U_forec_tmp[1:,:,:] = U_forec_tmp[1:,:,:]/U_forec_tmp[0,:,:]
                for N in range(n_ens):
                    X_fc[:,N,index,forec_T] = U_forec_tmp[:,:,N].flatten()
                
                forec_T+=1

            print('============================')

            if index >= Nmeas : break

            # calculate analysis at observing time
            print('')
            print('========= ANALYSIS =========')
            print('Assimilation time = ', round((index+1)*dtmeasure*assim_lenght,3))
            X_b = X_fc[:,:,index,assim_lenght-1:2*assim_lenght-1] # background forecast over DA window
            Y = Y_obs[:,(index+1)*assim_lenght:(index+2)*assim_lenght] # trim relevant obs for DA window
            U_an, X_an[:,:,index] = analysis_step_4DEnVar(X_b, Y, H, R, PARS)
            U = np.copy(U_an) # update U with analysis ensembles for next integration
            print('============================')

            index += 1
            
        except (RuntimeWarning, mp.TimeoutError) as err:
            pool.terminate()
            pool.join()
            print(err)
            print('--------- FORECAST FAILED! ---------')
            break

    print()
    print('End of assimilation cyle')

    # saving outputs
    np.save(str(dirname+'/X_an'),X_an)
    np.save(str(dirname+'/X_fc'),X_fc[:,:,:-1,:])
    np.save(str(dirname+'/qstored'),q2store)