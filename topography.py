### topography

import numpy as np

def sines_topog(x,Nk) :
    k = 2*np.pi
    xp = 0.1
    waven = [2,4,6]
    A = [0.2, 0.1, 0.2]
    B = A[0]*(1+np.cos(k*(waven[0]*(x-xp)-0.5)))+ A[1]*(1+np.cos(k*(waven[1]*(x-xp)-0.5)))+    A[2]*(1+np.cos(k*(waven[2]*(x-xp)-0.5)))
    B = 0.5*B
    index = np.where(B<=np.min(B)+1e-10)
    index = index[0]
    B[:index[0]] = 0
    B[index[-1]:] = 0
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    return B

if __name__=='__main__' :
        
    import importlib.util
    import sys
    from f_modRSW import make_grid

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    Nk_fc = config.Nk_fc
    L = config.L
    outdir = config.outdir

    B = config.topog(make_grid(Nk_fc,L)[1],Nk_fc)
    np.save(f'{outdir}/B.npy', B)