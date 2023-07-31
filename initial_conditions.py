### INITIAL CONDITION ###

import numpy as np

def sine_topog_ic(x,Nk,Neq,H0,B):
#    superposition of cosines
    ic1 = H0*np.ones(len(x))
    ic2=1/ic1 # for hu = 1:
    ic3 = np.zeros(len(x))

    U0 = np.zeros((Neq,Nk))
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    
    return U0

if __name__ == '__main__' :

    from f_modRSW import make_grid
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config) 

    Nk_fc = config.Nk_fc
    L = config.L
    Neq = config.Neq
    H0 = config.H0
    n_ens = config.n_ens
    sig_ic = config.sig_ic
    outdir = config.outdir

    fc_grid =  make_grid(Nk_fc,L) # forecast
    Kk_fc = fc_grid[0]
    x_fc = fc_grid[1]

    try : 
        B = np.load(f'{outdir}/B.npy')
    except FileNotFoundError as err :
        print(err)
        print("Please run topography first")

    U0_fc = config.ic(x_fc,Nk_fc,Neq,H0,B)

    U0ens = np.empty([Neq,Nk_fc,n_ens])

    # Generate initial ensemble
    U0ens[:,:,0] = U0_fc #control member
    for jj in range(0,Neq):
        for N in range(1,n_ens):
            # add sig_ic to EACH GRIDPOINT
            U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(Nk_fc)

    # if hr < 0, set to zero:
    hr = U0ens[2, :, :]
    hr[hr < 0.] = 0.
    U0ens[2, :, :] = hr

    # if h < 0, set to epsilon:
    h = U0ens[0, :, :]
    h[h < 0.] = 1e-3
    U0ens[0, :, :] = h    

    np.save(f'{outdir}/U0ens.npy',U0ens)