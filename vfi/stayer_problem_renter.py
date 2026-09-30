

"""
stayer_problem.py
"""

import numpy as np
from numba import njit
import utility_epsilons as ut
import grids as gridsfun
import interp as interpfun

                
@njit
def solve(par, grids, j_index, k_index, g_index, t_index, dP_C, dP_NC, dP_C_prime, dP_NC_prime, mW_next, mQ_next, mW_next_wf, config):

    
    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))    
    rental_price_c=par.dPsi+dP_C-(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime 
    g_renter_c=grids.vG[g_index]         
         
    rental_price_nc=par.dPsi+dP_NC-(1-par.dDelta)/(1+par.r)*dP_NC_prime
    g_renter_nc=1

    mVt_c, mQt_c, mB_pol_c, mVt_wf_c=renter_solver(par, grids, j_index, k_index, g_index, t_index, dP_C, dP_NC, dP_C_prime, dP_NC_prime, mW_next, mQ_next, mW_next_wf, rental_price_c, g_renter_c, config)
    mVt_nc, mQt_nc, mB_pol_nc, mVt_wf_nc=renter_solver(par, grids, j_index, k_index, g_index, t_index, dP_C, dP_NC, dP_C_prime, dP_NC_prime, mW_next, mQ_next, mW_next_wf, rental_price_nc, g_renter_nc, config)

    mVt    = np.where(mVt_c >= mVt_nc, mVt_c,    mVt_nc)
    mQt    = np.where(mVt_c >= mVt_nc, mQt_c,    mQt_nc)
    mB_pol = np.where(mVt_c >= mVt_nc, mB_pol_c, mB_pol_nc)
    mVt_wf = np.where(mVt_c >= mVt_nc, mVt_wf_c, mVt_wf_nc)

    return mVt, mQt, mB_pol, mVt_wf

@njit
def renter_solver(par, grids, j_index, k_index, g_index, t_index, dP_C, dP_NC, dP_C_prime, dP_NC_prime, mW_next, mQ_next, mW_next_wf, rental_price, g_renter, config):
    j=j_index  
    h_share,c_share,w=ut.renter_solve(par,rental_price,g_renter)       
    mVt = np.zeros((grids.vX.size))*np.nan  
    mVt_wf = np.zeros((grids.vX.size))
    mQt = np.empty((grids.vX.size))
    mB_pol = np.empty((grids.vX.size))
    
    vX_endog=np.zeros((grids.vB.size))
    vX_endog_max=0
    vX_endog_max_index=0
    vExp_endog=np.zeros((grids.vB.size))
    c_pol_max=0.5*(grids.vX[-1]-rental_price*grids.vH_renter[0])
    vC_nodes=gridsfun.nonlinspace_jit(0.01,c_pol_max,100,par.nonlingrid_big)
    
        
    vU_c_nodes=ut.u_c(j,vC_nodes,grids.vH_renter[-1],g_renter, par)
     
    for b_prime_index in range(grids.vB.size):
        q_next=mQ_next[b_prime_index]
        b=grids.vB[b_prime_index]
        expenditures=(q_next/(par.vAgeEquiv[j]*w))**(-1/par.dSigma)
        if (h_share/rental_price)*expenditures<grids.vH_renter[0]:     
            consumption_endog=interpfun.interp_1d(vU_c_nodes[::-1], vC_nodes[::-1], q_next)
            expenditures=consumption_endog+rental_price*grids.vH_renter[0]          
        elif (h_share/rental_price)*expenditures>grids.vH_renter[-1]:     
            consumption_endog=interpfun.interp_1d(vU_c_nodes[::-1], vC_nodes[::-1], q_next)
            expenditures=consumption_endog+rental_price*grids.vH_renter[-1]                  
        vX_endog[b_prime_index]=b+expenditures 
        vExp_endog[b_prime_index]=expenditures
        assert vExp_endog[b_prime_index]>0
        if vX_endog[b_prime_index]>vX_endog_max:
            vX_endog_max=vX_endog[b_prime_index]
            vX_endog_max_index=b_prime_index
        
    for x_index in range(grids.vX.size):
        if grids.vX[x_index]<=vX_endog[0]:
            h_pol=max(min(h_share/rental_price*grids.vX[x_index],grids.vH_renter[-1]),grids.vH_renter[0])
            C_candidate=grids.vX[x_index]-h_pol*rental_price
            Vt_candidate=-1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next[0])
            if (Vt_candidate>mVt[x_index]) or (np.isnan(mVt[x_index])):
                mVt[x_index]=Vt_candidate
                mC_pol=C_candidate
                assert not np.isnan(mC_pol) and mC_pol>0
                mQt[x_index]=-1/ut.u_c(j,mC_pol,h_pol,g_renter, par)                 
                mB_pol[x_index]=grids.vB[0]
                if config.welfare == True:
                    mVt_wf[x_index] = -1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next_wf[0])
                
    for b_prime_index in range(grids.vB.size-1):  
        for x_index in range(grids.vX.size):
            if ((vX_endog[b_prime_index]<grids.vX[x_index]) and (grids.vX[x_index]<=vX_endog[b_prime_index+1])) or  ((vX_endog[b_prime_index]>=grids.vX[x_index]) and (grids.vX[x_index]>vX_endog[b_prime_index+1])):
                Exp_candidate=vExp_endog[b_prime_index]+(vExp_endog[b_prime_index+1]-vExp_endog[b_prime_index])/(vX_endog[b_prime_index+1]-vX_endog[b_prime_index])*(grids.vX[x_index]-vX_endog[b_prime_index])
                h_pol=max(min(h_share/rental_price*Exp_candidate,grids.vH_renter[-1]),grids.vH_renter[0])
                C_candidate=Exp_candidate-rental_price*h_pol
                Vt_candidate=-1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next[b_prime_index]+(mW_next[b_prime_index+1]-mW_next[b_prime_index])/(grids.vB[b_prime_index+1]-grids.vB[b_prime_index])*((grids.vX[x_index]-Exp_candidate)-grids.vB[b_prime_index]))
                if (Vt_candidate>mVt[x_index]) or (np.isnan(mVt[x_index])):
                    mVt[x_index]=Vt_candidate
                    mC_pol=C_candidate
                    assert not np.isnan(mC_pol) and mC_pol>0
                    mQt[x_index]=-1/ut.u_c(j,mC_pol,h_pol,g_renter, par)   
                    mB_pol[x_index]=grids.vX[x_index]-Exp_candidate
                    if config.welfare == True:
                        mVt_wf[x_index] = -1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next_wf[b_prime_index]+(mW_next_wf[b_prime_index+1]-mW_next_wf[b_prime_index])/(grids.vB[b_prime_index+1]-grids.vB[b_prime_index])*((grids.vX[x_index]-Exp_candidate)-grids.vB[b_prime_index]))
                    
    for x_index in range(grids.vX.size):                
        if (grids.vX[x_index]>vX_endog_max):
            Exp_candidate=grids.vX[x_index]-grids.vB[vX_endog_max_index]
            h_pol=max(min(h_share/rental_price*Exp_candidate,grids.vH_renter[-1]),grids.vH_renter[0])
            C_candidate=Exp_candidate-rental_price*h_pol           
            Vt_candidate=-1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next[vX_endog_max_index])
            if (Vt_candidate>mVt[x_index]) or (np.isnan(mVt[x_index])):
                mVt[x_index]=Vt_candidate
                mC_pol=C_candidate
                assert not np.isnan(mC_pol) and mC_pol>0
                mQt[x_index]=-1/ut.u_c(j,mC_pol,h_pol,g_renter, par)   
                mB_pol[x_index]=grids.vX[x_index]-Exp_candidate
                if config.welfare == True:
                    mVt_wf[x_index] = -1/(ut.u(j,C_candidate,h_pol,g_renter, par)+mW_next_wf[vX_endog_max_index])
            
    
    assert np.isnan(mVt).sum() == 0
    
    return mVt, mQt, mB_pol, mVt_wf