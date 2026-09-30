"""
Tauchen.py

Purpose:
    Create a discretized grid and markov chain for a stochastic proces following an AR(1)
"""
###########################################################
### Imports
import numpy as np
from scipy.stats import norm

###########################################################
### Functions
###########################################################

###########################################################
### mMarkov, vZ = tauchen(dRho, dSigmaeps, iNumStates, iM)
#@njit
def tauchen(dRho, dSigmaeps, iNumStates, iM, time_increment):
    """Returns the markov chain and grid of an AR(1) proces using the Tauchen method

    Args:
        dRho        (float): Persistence of the AR(1)
        dSigmaeps   (float): Standard deviation of the error term in the AR(1)
        iNumStates  (int):   Number of grid points for the discrete state space
        iM          (in):    an integer that parameterizes the width of the state space

    Returns:
        mMarkov (matrix, float): Markov chain
        vZ     (vector, float): Grid
    """
    # Initialize
    mMarkov = np.zeros(shape = (iNumStates, iNumStates), dtype = float)
    
    # Create the grid
    dSD_Z = np.sqrt(dSigmaeps**2 / (1 - dRho**2))

    dZ_max = iM * dSD_Z
    dZ_min = -dZ_max
    
    dDist = (dZ_max-dZ_min)/(iNumStates-1)

    vZ = np.linspace(dZ_min, dZ_max, iNumStates)
    
    # Fill the matrix
    
    for i in range(iNumStates):
        for j in range(iNumStates):
            if j == 0: # lowest state
                mMarkov[i, j] = norm.cdf((vZ[j] - dRho * vZ[i] + 0.5 * dDist) / dSigmaeps)
            elif j == iNumStates-1: # highest state
                mMarkov[i, j] = 1 - norm.cdf((vZ[j] - dRho * vZ[i] - 0.5 * dDist) / dSigmaeps)
            else: # interior states
                upper_bound = (vZ[j] - dRho * vZ[i] + 0.5 * dDist) / dSigmaeps
                lower_bound = (vZ[j] - dRho * vZ[i] - 0.5 * dDist) / dSigmaeps
                mMarkov[i, j] = norm.cdf(upper_bound) - norm.cdf(lower_bound)
                
    mMarkov = np.linalg.matrix_power(mMarkov,time_increment)      
      
    return mMarkov, vZ

def initial_dist(par, vE):   
    vPi_E=np.zeros((vE.size))
    dPi_cum=0
    for i in range(par.iNumStates):
        if i<par.iNumStates-1:
            vPi_E[i]=norm.cdf((0.5*vE[i]+0.5*vE[i+1])/par.sd_income_initial)-dPi_cum
            dPi_cum+=vPi_E[i]
        else:
            vPi_E[i]=1-dPi_cum
    return vPi_E

def weight_matrix(par, vE, vPi_E, mMarkov):
    mPi_E=np.zeros((par.iNj, vE.size))
    mPi_E[0,:]=vPi_E
    for j in range(1,par.j_ret):
        for i in range(par.iNumStates):
            for k in range(par.iNumStates): 
                mPi_E[j,i]+=mPi_E[j-1,k]*mMarkov[k,i]
    for j in range(par.j_ret, par.iNj):
        mPi_E[j,:]=mPi_E[j-1,:]
    return mPi_E
    

def lifecycle(par,J_ret):
    
    chi=np.zeros((J_ret))
    for j in range(J_ret):
        age=21+par.time_increment*j
        chi[j]=par.beta0_age+par.beta1_age*age+par.beta2_age*age**2+par.beta3_age*age**3+par.beta4_age*age**4
    
    return chi 

def median_inc(chi, z, z_dist):
    
    mIncome = np.zeros((chi.size, z.size))
    mDens = np.zeros((chi.size,z.size))
    
    for j in range(chi.size):
        for z_idx in range(z.size):
            mIncome[j,z_idx] = np.exp(chi[j] + z[z_idx])
            mDens[j,z_idx] = (1/chi.size * z_dist[j, z_idx])
            
    mIncome_flattened = mIncome.flatten()
    mDens_flattened = mDens.flatten()
    
    sorted_indices = np.argsort(mIncome_flattened)
    mIncome_sort = mIncome_flattened[sorted_indices]
    mDens_sorted = mDens_flattened[sorted_indices]
    
    cumulative_density = np.cumsum(mDens_sorted) / np.sum(mDens_sorted)
    
    median = np.interp(0.5, cumulative_density, mIncome_sort)
    
    return median

def invar_dist(mMarkov):
    evals, evecs = np.linalg.eig(mMarkov.T)
    evec1 = (evecs[:,np.isclose(evals, 1)])[:,0]
    vStationary_Y = evec1 / evec1.sum()
    
    return vStationary_Y

def combine_vectors(vE, vE_trans, vE_stationary, vE_trans_stationary):
    n1 = vE.size
    n2 = vE_trans.size
    total_size = n1 * n2

    # Allocate result arrays
    sum_vec = np.empty(total_size, dtype=np.float64)
    prob_vec = np.empty(total_size, dtype=np.float64)

    idx = 0
    for i in range(n1):
        for j in range(n2):
            sum_vec[idx] = vE[i] + vE_trans[j]
            prob_vec[idx] = vE_stationary[i] * vE_trans_stationary[j]
            idx += 1

    return sum_vec, prob_vec


###########################################################
### main#
#def main():
    # Magic numbers
    
    ## Income proces
#    dRho = 0.96
##    dSigmaeps = 0.12
#    iNumStates = 5
#    iM = 3
#    J = 15

    # Initialisation
    
    
    # Estimation
#    mMarkov, vZ = tauchen(dRho, dSigmaeps, iNumStates, iM)
#    vStationary_Y = invar_dist(mMarkov)
#    chi = lifecycle(J)
#    median = median_inc(chi, vZ, vStationary_Y)

###########################################################
### start main
#if __name__ == "__main__":
#    main()
