####################################################################
##  FILE CONTAINING CONFIGURATION PARAMENTERS            	  ##
####################################################################
'''
List of fixed parameters for model integration and EnKF.
'''
import __main__
import numpy as np

'''Output directory'''
outdir = 'output'

''' MODEL PARAMETERS '''

Neq = 3     # number of equations in system (3 with topography, 4 with rotation)
L = 1.0     # length of domain (non-dimensional)

Nk_fc = 200                         # forecast resolution
dres = 2                            # refinement factor for truth gridsize
Nk_tr = dres*Nk_fc                  # truth resolution
n_d = Neq * Nk_fc                   # total number of variables (dgs of freedom)

cfl_fc = 0.5 # Courant Friedrichs Lewy number for time stepping (forecast)
cfl_tr = 0.5 # Courant Friedrichs Lewy number for time stepping (truth)

Ro = 'Inf'          # Rossby number, i.e. Ro ~ V0/(f*L0)
Fr = 1.1            # Froude number, i.e. Fr ~ V0/sqrt(gH)
g = Fr**(-2) 	# effective gravity, determined by scaling.
A = 0.1
V = 1.

### TOPOGRAPHY
try :
    import topography
    topog = topography.sines_topog
except ModuleNotFoundError :
    if 'viz' not in __main__.__file__ : print("WARNING : topography has not been loaded") 

### INITIAL CONDITION ###
try :
    import initial_conditions
    ic = initial_conditions.sine_topog_ic
except ModuleNotFoundError :
    if 'viz' not in __main__.__file__ : print("WARNING : initial_conditions has not been loaded")
sig_ic = [0.1,0.05,0.0]                 # initial perturbation (st. dev) to generate initial conditions [h,u,r]

# FLUID THRESHOLDS
H0 = 1.0
Hc = 1.02 # convection fluid threshold
Hr = 1.05 # rain fluid threshold

# RAIN AND CONVECTION PARAMETERS
beta = 0.2 # coefficient controlling rain production
alpha2 = 10 # coefficient controlling rain removal
cc2 = 0.1*g*Hr # coefficient controlling suppressing of convection

''' FILTER PARAMETERS '''
dtmeasure = 0.144  # duration of forecast step in non-dimensional time units [1h = 1/((L/U)/3600) if U in m/s and L in m]
n_ens = 19                              # number of ensemble members (the 0th is a control member)
TIMEOUT_PAR = n_ens*4			# time to wait until all forecasts running in parallel are over
Nmeas = 4#48                             # number of cycles
Nforec = 12				#  duration of the *dtmeasure*-long forecast launched at the end of each analysis step
NIAU = 1000				# suppress injection of additional inflation with IAU for the first NIAU hours since assimilation 
# spin_up = 12  # spin-up duration for plotting in *dtmeasure*: data during this period are discarded
assim_lenght = 6 # duration of data assimiliation window, in *dtmeasure*
tmax = (assim_lenght*Nmeas+Nforec)*dtmeasure   # maximum duration of truth trajectory, from initial condition to end of last forecast
assim_time = np.linspace(dtmeasure*assim_lenght,Nmeas*dtmeasure*assim_lenght,Nmeas) # array of times of analysis steps

### OBSERVING SYSTEM PARAMETERS ###
ob_noise = [0.05,0.02,0.003]            # ob noise for [h,u,r]
obs_h_d = np.inf #25 # observation spacing for h (in grid points)
obs_u_d = 20 # observation spacing for u (in grid points)
obs_r_d = 20 # observation spacing for r (in grid points)
n_obs_h = 0 #Nk_fc // obs_h_d # number of h observations
n_obs_u = Nk_fc // obs_u_d # number of u observations
n_obs_r = Nk_fc // obs_r_d # number of r observations
n_obs = n_obs_h + n_obs_u + n_obs_r # total number of observations
# Mask for u observations in the observation vector
h_obs_mask = np.array(list(range(n_obs_h)))
# Mask for r observations in the observation vector
hu_obs_mask = np.array(list(range(n_obs_h,n_obs_h+n_obs_u)))
# Mask for r observations in the observation vector
hr_obs_mask = np.array(list(range(n_obs_h+n_obs_u,n_obs_h+n_obs_u+n_obs_r)))


### Q MATRIX GENERATION ###
model_noise = [0.0,0.0,0.0]     # user-specified model noise to create model error covariance matrix Q 
Nhr = 1                             # Duration of model forecasts used to compute Q
Q_FUNC = 'Q_nhr()'                  # Q_predef(): user-specified error covariance matrix Q, Q_nhr(): error covariance matrix Q computed online
rMODNOISE = 0                       # binary parameter to control the presence of additive inflation in r (1=yes,0=no)
hMODNOISE = 1                    # binary parameter to control the presence of additive inflation in sigma (1=yes,0=no)

'''
### OUTER LOOP PARAMETERS (main_p.py) ###
Parameters for outer loop are specified in main_p.py 
add_inf : additive infaltaion factor
'''
# MUST BE FLOATING POINT
add_inf = 0.3

##################################################################
#			END OF PROGRAM				 #
##################################################################
