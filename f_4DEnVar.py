
### IMPORTS ###

from numpy.linalg import inv
import numpy as np
import scipy.optimize as spop

### COST FUNC AND GRADIENT ###

def j_4DEnVar(w,HX_ens_b_pert,HX_ens_b_bar,Y_obs,R_inv) :
    j_b = np.dot(w.T,w)/2
    tmp = np.dot(HX_ens_b_pert,w) + HX_ens_b_bar - Y_obs
    j_o = np.dot(np.dot(tmp.T, R_inv), tmp)/2
    return j_b + j_o

def grad_j_4DEnVar(w,HX_ens_b_pert,HX_ens_b_bar,Y_obs,R_inv) :
    tmp = np.dot(HX_ens_b_pert,w) + HX_ens_b_bar - Y_obs
    return w + np.dot(np.dot(HX_ens_b_pert.T, R_inv), tmp)

##################################################################
#'''------------------ ANALYSIS STEP ------------------'''
##################################################################

def analysis_step_4DEnVar(X, Y_obs, H, R, pars):
    '''        
        INPUTS
        X: ensemble trajectories
        '''

    Nk_fc = pars[0]
    n_ens = pars[1]
    n_d = pars[2]
    Neq = pars[3]
    assim_lenght = pars[4]

    h_mask = list(range(0,Nk_fc))
    hu_mask = list(range(Nk_fc,2*Nk_fc))
    hr_mask = list(range(2*Nk_fc,3*Nk_fc))
    
    print('--------- 4DEnVar ---------')

    ### INVERTING OBSERVATION ERROR COVARIANCE MATRIX ###

    R_inv = inv(R)

    ### COMPUTE BACKGROUND PERTURBATION ###

    # exlude control member from pertubation computing and backgound covariance sampling
    X = X[:,1:,:]
    n_ens-=1

    # perturbation matrix
    X_ens_b_bar = X.mean(axis=1)

    X_ens_b_pert = np.empty_like(X)
    for n_member in range(n_ens) :
        X_ens_b_pert[:,n_member] = (X[:,n_member] - X_ens_b_bar)/np.sqrt(n_ens-1)

    # map perturbation to observation space and reshaping
    HX_ens_b_bar = np.concatenate([np.matmul(H, X_ens_b_bar[:,lead_index]) for lead_index in range(assim_lenght)])
    HX_ens_b_pert = np.concatenate([np.matmul(H, X_ens_b_pert[:,:,lead_index]) for lead_index in range(assim_lenght)])

    # reshaping obs and obs-err-cov-matrix
    Y_obs = Y_obs.flatten('F')

    ### MINIMIZATION ###

    minimizer = spop.minimize(j_4DEnVar, np.zeros((n_ens)),args=(HX_ens_b_pert,HX_ens_b_bar,Y_obs,R_inv), method='L-BFGS-B', jac=grad_j_4DEnVar, options={'gtol': 1e-16,'maxiter':100})
    print(f'minimizer success : {minimizer.success} ({minimizer.message})')
    X_an_4DEnVar = X_ens_b_bar[:,0] + np.dot(X_ens_b_pert[:,:,0], minimizer.x) # applying control to the beginning of the window

    ### POSTERIOR ENSEMBLE
    work1 = np.dot(R_inv,HX_ens_b_pert)
    work2 = np.dot(HX_ens_b_pert.T,work1)+np.identity(n_ens)
    work3 = np.linalg.pinv(work2)
    
    # EQUATION 15 in 4DEnVar notes:
    Wa = np.linalg.cholesky(work3)
    print(np.diag(Wa), '\n', np.trace(Wa))
    # EQUATION 14 in 4DEnVar notes:
    Xa_dash = np.dot(X_ens_b_pert[:,:,0],4*Wa)
    # EQUATION 16 in 4DEnVar notes:
    xan = np.repeat(X_an_4DEnVar, n_ens).reshape(n_d,n_ens)
    Xan_ens = np.sqrt(n_ens-1)*Xa_dash + xan
    # Xan_ens = np.sqrt(n_ens-1)*X_ens_b_pert[:,:,0] + xan

    # add most accurate analysis for control member
    n_ens+=1
    Xan = np.zeros((n_d,n_ens))
    Xan[:,0] = X_an_4DEnVar
    Xan[:,1:] = Xan_ens

    # transform from X to U for next integration (in h, hu, hr coordinates)
    U_an = np.empty((Neq,Nk_fc,n_ens))
    Xan[hu_mask,:] = Xan[hu_mask,:] * Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] * Xan[h_mask,:]
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan[:,N].reshape(Neq,Nk_fc)
    
    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[hu_mask,:] = Xan[hu_mask,:] / Xan[h_mask,:]
    Xan[hr_mask,:] = Xan[hr_mask,:] / Xan[h_mask,:]

    return U_an, Xan