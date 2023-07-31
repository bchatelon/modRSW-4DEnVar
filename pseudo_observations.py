if __name__=='__main__' :

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

    #################################################################
    # IMPORT PARAMETERS FROM CONFIGURATION FILE                     #
    #################################################################
    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    outdir = config.outdir
    Nk_fc = config.Nk_fc
    Nmeas = config.Nmeas
    Nforec = config.Nforec
    assim_lenght = config.assim_lenght
    n_obs = config.n_obs
    dres = config.dres
    n_d = config.n_d
    ob_noise = config.ob_noise
    obs_h_d = config.obs_h_d
    obs_u_d = config.obs_u_d
    obs_r_d = config.obs_r_d
    n_obs_h = config.n_obs_h
    n_obs_u = config.n_obs_u
    n_obs_r = config.n_obs_r
    h_obs_mask = config.h_obs_mask
    hr_obs_mask = config.hr_obs_mask

    print('Total no. of obs. =', n_obs)

    try:
        f_H_name = str(outdir+'/H.npy')
        H = np.load(f_H_name)
    except FileNotFoundError :
        print('Computing and saving observation operator')
        # observation operator
        H = np.zeros([n_obs, n_d])
        row_vec_h = list() #range(obs_h_d, Nk_fc+1, obs_h_d))
        row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
        row_vec_r = list(range(2*Nk_fc+obs_r_d, 3*Nk_fc+1, obs_r_d))
        row_vec = row_vec_h+row_vec_u+row_vec_r
        for i in range(0, n_obs):
            H[i, row_vec[i]-1] = 1
        np.save(f_H_name,H)

    try:
        f_obs_name = str(outdir+'/Y_obs.npy')
        Y_obs = np.load(f_obs_name)
    except FileNotFoundError :
        try :
            f_path_name = str(outdir+'/X_tr.npy')
            X_tr = np.load(f_path_name)
        except FileNotFoundError :
            print('ERROR : please run truth first')
        # create imperfect observations by adding the same observation noise to each member's perfect observation.
        print('Creating and saving pseudo-observations generated from noised truth')
        Y_obs = np.empty([n_obs, assim_lenght*Nmeas+Nforec+1])
        ob_noise_format = np.repeat(ob_noise, [n_obs_h,n_obs_u,n_obs_r])
        for T in range(assim_lenght*Nmeas+Nforec+1):
            Y_mod = np.dot(H, X_tr[:,T])
            obs_pert = ob_noise_format * np.random.randn(n_obs)
            Y_obs[:, T] = Y_mod.flatten() + obs_pert

            # Reset pseudo-observations with negative h or r to zero.
            if(n_obs_h!=0 and n_obs_r!=0): mask = np.append(h_obs_mask[np.array(Y_obs[h_obs_mask, T] < 0.0)], hr_obs_mask[np.array(Y_obs[hr_obs_mask, T] < 0.0)])
            elif(n_obs_r!=0 and n_obs_h==0): mask = hr_obs_mask[np.array(Y_obs[hr_obs_mask, T] < 0.0)]
            elif(n_obs_r==0 and n_obs_h!=0): mask = h_obs_mask[np.array(Y_obs[h_obs_mask, T] < 0.0)]
            Y_obs[mask, T] = 0.0
        np.save(f_obs_name, Y_obs)

    try :
        Rmatrix_path = f'{outdir}/Rmatrix.npy'
        R = np.load(Rmatrix_path)
    except FileNotFoundError :
        print('Computing and saving observation error covariance matrix R')
        Ri = np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_r])*np.repeat(ob_noise,[n_obs_h,n_obs_u,n_obs_r])*np.identity(n_obs) # obs cov matrix
        R = np.kron(np.eye(assim_lenght),Ri) # block matrix of Ri
        np.save(Rmatrix_path, R)