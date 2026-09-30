
"""
stayer_problem.py
"""

import numpy as np
from numba import njit
import utility_epsilons as ut
import grids as gridsfun

@njit
def solve(par, grids, j_index, k_index, g_index, mW_next, mQ_next, mW_next_wf, config):
    
       
    mVt = np.zeros((grids.vM.size, grids.vH.size, grids.vL.size))*np.nan
    mVt_wf = np.zeros((grids.vM.size, grids.vH.size, grids.vL.size))
    mC_pol = np.empty((grids.vM.size, grids.vH.size, grids.vL.size))
    mB_pol = np.empty((grids.vM.size, grids.vH.size, grids.vL.size))
    mQt = np.empty((grids.vM.size, grids.vH.size, grids.vL.size))
  
    
    c_pol_max=0.5*(grids.vM[-1])
    vC_nodes=gridsfun.nonlinspace_jit(0.01,c_pol_max,100,par.nonlingrid_big)
    j=j_index
    g=grids.vG[g_index]
    vM_grid = grids.vM
    vB_grid = grids.vB
    
    for h_index in range(grids.vH.size): 
        h=grids.vH[h_index]
        vU_c_nodes=ut.u_c(j,vC_nodes,h,g+par.dOmega, par)
        for l_index in range(grids.vL.size):   
       
            vM_endog=np.zeros((vB_grid.size))
            
            vM_endog_max=0
            vM_endog_max_index=0
            vC_endog=np.zeros((vB_grid.size))        
            # inverse interpolation of marginal utility function
            mQ_slice = mQ_next[:, h_index, l_index]
            vC_endog = np.interp(mQ_slice, vU_c_nodes[::-1], vC_nodes[::-1])
            
            
            vM_endog=vB_grid+vC_endog
            
            for b_prime_index in range(vB_grid.size):
                # corner solution
                if vM_endog[b_prime_index]>vM_endog_max:
                    vM_endog_max=vM_endog[b_prime_index]
                    vM_endog_max_index=b_prime_index
           

            
            # check no borrowing constraint
            for m_index in range(vM_grid.size):
                if vM_grid[m_index]<=vM_endog[0]:
                    mC_pol[m_index,h_index,l_index]=vM_grid[m_index]
                    mVt[m_index,h_index,l_index]=-1/(ut.u(j,mC_pol[m_index,h_index,l_index],h,g+par.dOmega, par)+mW_next[0,h_index, l_index])
                    mQt[m_index,h_index,l_index]=-1/ut.u_c(j,mC_pol[m_index,h_index,l_index],h,g+par.dOmega, par)      
                    mB_pol[m_index,h_index,l_index]=vB_grid[0]
                    if config.welfare == True:
                        mVt_wf[m_index,h_index,l_index] = -1/(ut.u(j,mC_pol[m_index,h_index,l_index],h,g+par.dOmega, par)+mW_next_wf[0,h_index, l_index])
                
            # upper envelope due to value function not being concave (FOC not sufficient, only necessary)
            for b_prime_index in range(grids.vB.size-1):
                for m_index in range(grids.vM.size):
                    if ((vM_endog[b_prime_index]<grids.vM[m_index]) and (grids.vM[m_index]<=vM_endog[b_prime_index+1])) or ((vM_endog[b_prime_index]>=grids.vM[m_index]) and (grids.vM[m_index]>vM_endog[b_prime_index+1])):
                       C_candidate=vC_endog[b_prime_index]+(vC_endog[b_prime_index+1]-vC_endog[b_prime_index])/(vM_endog[b_prime_index+1]-vM_endog[b_prime_index])*(grids.vM[m_index]-vM_endog[b_prime_index])
                       Vt_candidate=-1/(ut.u(j,C_candidate,h,g+par.dOmega, par)+mW_next[b_prime_index,h_index, l_index]+(mW_next[b_prime_index+1,h_index, l_index]-mW_next[b_prime_index,h_index, l_index])/(grids.vB[b_prime_index+1]-grids.vB[b_prime_index])*((grids.vM[m_index]-C_candidate)-grids.vB[b_prime_index]))
                       if (Vt_candidate>mVt[m_index,h_index,l_index]) or (np.isnan(mVt[m_index,h_index,l_index])):
                           mVt[m_index,h_index,l_index]=Vt_candidate
                           mC_pol[m_index,h_index,l_index]=C_candidate
                           mQt[m_index,h_index,l_index]=-1/ut.u_c(j,mC_pol[m_index,h_index,l_index],h,g+par.dOmega, par)
                           mB_pol[m_index,h_index,l_index]=grids.vM[m_index]-mC_pol[m_index,h_index,l_index]
                           if config.welfare == True:
                               mVt_wf[m_index,h_index,l_index] =-1/(ut.u(j,C_candidate,h,g+par.dOmega, par)+mW_next_wf[b_prime_index,h_index, l_index]+(mW_next_wf[b_prime_index+1,h_index, l_index]-mW_next_wf[b_prime_index,h_index, l_index])/(grids.vB[b_prime_index+1]-grids.vB[b_prime_index])*((grids.vM[m_index]-C_candidate)-grids.vB[b_prime_index]))
            
            # handle corner case properly (no extrapolation)
            for m_index in range(grids.vM.size):                
                if (grids.vM[m_index]>vM_endog_max):
                   C_candidate=grids.vM[m_index]-grids.vB[vM_endog_max_index]
                   Vt_candidate=-1/(ut.u(j,C_candidate,h,g+par.dOmega, par)+mW_next[vM_endog_max_index,h_index, l_index])
                   if (Vt_candidate>mVt[m_index,h_index,l_index]) or (np.isnan(mVt[m_index,h_index,l_index])):
                       mVt[m_index,h_index,l_index]=Vt_candidate
                       mC_pol[m_index,h_index,l_index]=C_candidate
                       mQt[m_index,h_index,l_index]=-1/ut.u_c(j,mC_pol[m_index,h_index,l_index],h,g+par.dOmega, par)
                       mB_pol[m_index,h_index,l_index]=grids.vB[vM_endog_max_index]
                       if config.welfare == True:
                           mVt_wf[m_index,h_index,l_index] = -1/(ut.u(j,C_candidate,h,g+par.dOmega, par)+mW_next_wf[vM_endog_max_index,h_index, l_index])
                    

    assert np.all(mC_pol>0)  
    assert np.isnan(mQt).sum()==0
    
    return mVt, mC_pol, mQt, mB_pol, mVt_wf