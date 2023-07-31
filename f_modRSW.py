#######################################################################
# FUNCTIONS REQUIRED FOR NUMERICAL INTEGRATION OF THE modRSW MODEL WITH TOPOGRAPHY
#######################################################################

'''
Module contains numerous functions for the numerical inegration of the modRSW model:
> make_grid            : makes mesh for given length and gridcell number
> NCPflux_topog         : calcualtes flux for flow over topography
> time_step            : calculates stable time step for integration
> heaviside             : vector-aware implementation of heaviside (also works for scalars).
> ens_forecast_topog    : for use in parallel ensemble forecasting
'''
    
import math as m
import numpy as np
from numba import jit, vectorize, float64
#from parameters import *

##################################################################
#'''-------------- Create mesh at given resolution --------------'''
##################################################################  

# domain [0,L]
def make_grid(Nk,L):
    Kk = L/Nk                   # length of cell
    x = np.linspace(0, L, Nk+1)         # edge coordinates
    xc = np.linspace(Kk/2,L-Kk/2,Nk) # cell center coordinates
    grid = [Kk, x, xc]
    return grid

##################################################################
#'''--------------- NCP flux function with topog -------------'''#
##################################################################

@jit(nopython=True, cache=True)
def NCPflux_topog(UL,UR,BL,BR,Hr,Hc,c2,beta,g):

    ### INPUT ARGS:
    # UL: left state 3-vector       e.g.: UL = np.array([1,2,0])
    # UR: right state 3-vector      e.g.: UR = np.array([1.1,1.5,0])
    # BL: left B value
    # BR: right B balue
    # Hr: threshold height r        e.g.: Hr = 1.15 (note that Hc < Hr)
    # Hc: threshold height c        e.g.: Hc = 1.02
    # c2: constant for rain geop.   e.g.: c2 = 9.8*Hr/400
    # beta: constant for hr eq.     e.g.: beta = 1
    # g: scaled gravity             e.g.: g = 1/(Fr**2)

    ### OUTPUT:
    # Flux: 4-vector of flux values between left and right states
    # VNC : 4-vector of VNC values due to NCPs

    # isolate variables for flux calculation
    hL = UL[0]
    hR = UR[0]
    
    if hL < 1e-9:
        uL = 0
        rL = 0
    else:
        uL = UL[1]/hL
        rL = UL[2]/hL

    if hR < 1e-9:
        uR = 0
        rR = 0
    else:
        uR = UR[1]/hR
        rR = UR[2]/hR

    zL = hL + BL
    zR = hR + BR

    # compute left and right wave speeds (eq. 17)
    SL = min(uL - m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zL-Hr) + g*hL*heaviside(Hc-zL)),uR - m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zR-Hr) + g*hR*heaviside(Hc-zR)))
    SR = max(uL + m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zL-Hr) + g*hL*heaviside(Hc-zL)),uR + m.sqrt(c2*beta*heaviside(uL-uR)*heaviside(zR-Hr) + g*hR*heaviside(Hc-zR)))

    # For calculating NCP components
    a = zR-zL
    b = zL-Hr

    # compute the integrals as per the theory
    if a==0:
        Ibeta = beta*heaviside(uL-uR)*heaviside(b)
        Itaubeta = 0.5*beta*heaviside(uL-uR)*heaviside(b)
    else:
        d = (a+b)/a
        e = (a**2 + b**2)/a**2
        ee = (a**2 - b**2)/a**2
        Ibeta = beta*heaviside(uL-uR)*(d*heaviside(a+b) - (b/a)*heaviside(b))
        Itaubeta = 0.5*beta*heaviside(uL-uR)*(ee*heaviside(a+b) + (b**2/a**2)*heaviside(b))

    VNC1 = 0
    VNC2 = -c2*(rL-rR)*0.5*(hL+hR)
    VNC3 = -heaviside(uL-uR)*(uL-uR)*(hR*Ibeta - (hL-hR)*Itaubeta)

    VNC = np.array([VNC1, VNC2, VNC3])

    if SL > 0:
        PhL = 0.5*g*(hL**2 + ((Hc-BL)**2 - hL**2)*heaviside(zL-Hc))
        FluxL = np.array([hL*uL, hL*uL**2 + PhL, hL*uL*rL])
        Flux = FluxL - 0.5*VNC
    elif SR < 0:
        PhR = 0.5*g*(hR**2 + ((Hc-BR)**2 - hR**2)*heaviside(zR-Hc))
        FluxR = np.array([hR*uR, hR*uR**2 + PhR, hR*uR*rR])
        Flux = FluxR + 0.5*VNC
    elif SL < 0 and SR > 0:
        PhL = 0.5*g*(hL**2 + ((Hc-BL)**2 - hL**2)*heaviside(zL-Hc))
        PhR = 0.5*g*(hR**2 + ((Hc-BR)**2 - hR**2)*heaviside(zR-Hc))
        FluxR = np.array([hR*uR, hR*uR**2 + PhR, hR*uR*rR])
        FluxL = np.array([hL*uL, hL*uL**2 + PhL, hL*uL*rL])
        FluxHLL = (FluxL*SR - FluxR*SL + SL*SR*(UR - UL))/(SR-SL)
        Flux = FluxHLL - (0.5*(SL+SR)/(SR-SL))*VNC
    else:
        Flux = np.zeros(3)


    return Flux, SL, SR, VNC

##################################################################
#'''----------------- Heaviside step function -----------------'''
##################################################################
@vectorize([float64(float64)])
def heaviside(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return 1 * (x >= 0)

##################################################################
#'''--------- Compute stable timestep ---------'''
##################################################################

@jit(nopython=True)
def time_step(U,Kk,cfl,cc2,beta,g):
### INPUT ARGS:
# U: array of variarible values at t
# Kk: grid size

### OUTPUT:
# dt: stable timestep (h>0 only)

    # signal velocties (calculated from eigenvalues)
    lam1 = np.abs(U[1,:]/U[0,:] - np.sqrt(cc2*beta + g*U[0,:]))
    lam2 = np.abs(U[1,:]/U[0,:] + np.sqrt(cc2*beta + g*U[0,:]))
    denom = np.maximum(lam1,lam2)
    
    dt = cfl*min(Kk/denom)

    return dt

##################################################################
# NON_ZERO TOPOGRAPHY: integrate forward one time step
##################################################################

@jit(nopython=True)
def sel_loop(arr, arr_index) :
    ### Allow advanced indexation (w/ array ...) to comply to numba implementation
    # Input :
    # arr : 1-D array
    # arr_index : 1D array used as selection indices on arr

    sel_array = np.zeros((np.shape(arr_index)[0]))
    for i,j in enumerate(arr_index) :
        sel_array[i]=arr[j]
    return sel_array

##################################################################

@jit(nopython=True)
def step_forward_topog(U,B,dt,Neq,Nk,Kk,Hc,Hr,cc2,beta,alpha2,g):
    ### INPUT ARGS:
    # U: array of variable at t, size (Neq,Nk)
    # B: bottom topography
    # dt: stable time step
    # Nk, Kk: mesh info

    left = np.arange(0,Nk,1)
    right = np.roll(left,-1)
    
    h = U[0,:]
    h[h<1e-9] =0
    hu = U[1,:]
    hu[h<1e-9] = 0
    U[1,:] = hu

    Bstar = np.maximum(sel_loop(B,left),sel_loop(B,right))
    hminus = np.maximum(sel_loop(U[0],left) + sel_loop(B,left) - Bstar,0)
    hplus = np.maximum(sel_loop(U[0],right) + sel_loop(B,right) - Bstar,0)
    
    uminus = sel_loop(U[1],left)/sel_loop(U[0],left)
    uminus[np.isnan(uminus)] = 0
    uplus = sel_loop(U[1],right)/sel_loop(U[0],right)
    uplus[np.isnan(uplus)] = 0
    
    U[2,:] = np.maximum(U[2,:],0)
    rminus = sel_loop(U[2],left)/sel_loop(U[0],left)
    rminus[np.isnan(rminus)] = 0
    rplus = sel_loop(U[2],right)/sel_loop(U[0],right)
    rplus[np.isnan(rplus)] = 0
    
    huminus = hminus*uminus
    huminus = np.append(huminus[-1], huminus)
    huplus = hplus*uplus
    huplus = np.append(huplus[-1], huplus)
    
    hrminus = hminus*rminus
    hrminus = np.append(hrminus[-1], hrminus)
    hrplus = hplus*rplus
    hrplus = np.append(hrplus[-1], hrplus)
   
    hminus = np.append(hminus[-1], hminus)
    hplus = np.append(hplus[-1], hplus)
    
    Bminus = np.append(B[-1],B)
    Bplus = np.append(B[0],B[right])
    
    # reconstructed states
    Uminus = np.zeros((3,np.shape(hminus)[0]))
    Uminus[0],Uminus[1],Uminus[2] = hminus, huminus, hrminus
    Uplus = np.zeros((3,np.shape(hplus)[0]))
    Uplus[0],Uplus[1],Uplus[2] = hplus, huplus, hrplus

    
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
    SL = np.empty(Nk+1)
    SR = np.empty(Nk+1)
    S = np.zeros((Neq,Nk))
    Sb = np.zeros((Neq,Nk))
    UU = np.empty(np.shape(U))
    
    # determine fluxes ...
    for j in range(0,Nk+1):
        Flux[:,j], SL[j], SR[j], VNC[:,j] = NCPflux_topog(Uminus[:,j],Uplus[:,j],Bminus[j],Bplus[j],Hr,Hc,cc2,beta,g)
    
    # compute topographic terms as per Audusse et al...
    for jj in range(0,Nk):
        
        zminus = Uminus[0,jj+1] + Bminus[jj+1]
        zplus = Uplus[0,jj] + Bplus[jj]
        
        if zminus <= Hc and zplus <= Hc:
            Sb[1,jj] = 0.5*g*(Uminus[0,jj+1]**2 - Uplus[0,jj]**2)
        elif zminus <= Hc and zplus > Hc:
            Sb[1,jj] = 0.5*g*(Uminus[0,jj+1]**2 - (Hc - Bplus[jj])**2)
        elif zminus > Hc and zplus <= Hc:
            Sb[1,jj] = 0.5*g*((Hc - Bminus[jj+1])**2 - Uplus[0,jj]**2)
        elif zminus > Hc and zplus > Hc:
            Sb[1,jj] = 0.5*g*((Hc - Bminus[jj+1])**2 - (Hc - Bplus[jj])**2)
    
    # compute extraneous forcing terms
    S[2,:] = -alpha2*U[2,:]
    
    # DG flux terms
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
   
    #integrate forward to next time level
    BC = 1
    if BC == 1: #PERIODIC
        
        UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*Sb/Kk + dt*S

    return UU

##################################################################
# PARALLEL COMPUTING using multiprocessing
##################################################################

@jit(nopython=True, cache=True)
def ens_forecast_topog(N, U, B, q, Neq, Nk_fc, Kk_fc, cfl_fc, dtmeasure, Hc, Hr, cc2, beta, alpha2, g):

    tn=0

    while tn < dtmeasure:
        
        dt = time_step(U[:,:,N],Kk_fc,cfl_fc,cc2,beta,g) # compute stable time step
        tn = tn + dt
        
        if tn > dtmeasure:
            dt = dt - (tn - dtmeasure) + 1e-12

        # Inject a portion of the additive noise each timestep using an
        # Incremental Analysis Update approach (Bloom et al., 1996).
        U[:,:,N] += (q[:,N].reshape(Neq, Nk_fc)) * dt / dtmeasure
        # if hr < 0, set to zero:
        hr = U[2, :, :]
        # hr[hr < 0.] = 0.
        hr = np.where(hr<0., 0., hr)
        U[2, :, :] = hr
   
        # if h < 0, set to epsilon:
        h = U[0, :, :]
        # h[h < 0.] = 1e-3
        h = np.where(h<0., 1e-3, h)
        U[0, :, :] = h
 
        U[:,:,N] = step_forward_topog(U[:,:,N],B,dt,Neq,Nk_fc,Kk_fc,Hc,Hr,cc2,beta,alpha2,g)
    
    return U[:,:,N]

##################################################################