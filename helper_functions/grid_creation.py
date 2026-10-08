import misc_functions as misc    
import tauchen as tauch
import pandas as pd
import numpy as np
import grids as grid
from scipy.optimize import brentq

def create(par, start_2026=False, mortgage_premium = False):
       
    # create grids
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    vPi_E=tauch.initial_dist(par, vE)
    mPi_E=tauch.weight_matrix(par, vE, vPi_E, mMarkov)
    
    vE_trans=np.zeros(1)
    vE_combined=vE
    vChi = tauch.lifecycle(par,par.j_ret)
    
    #We do not use fully transitory income shocks in the end, but have a structure that allows for it below
    mMarkov_trans=np.ones((1,1))    
    #mMarkov_trans, vE_trans = tauch.tauchen(0, par.dSigmaeps_trans, par.iNumTrans, par.iM)     
    #vStationary_E = tauch.invar_dist(mMarkov)
    #vE_combined, vStationary_E_combined = tauch.combine_vectors(vE, vE_trans, vStationary_E, mMarkov_trans[0,:])    
    
    median_inc_pretax = tauch.median_inc(vChi, vE, mPi_E)  
    #For simplicity, and to prevent the income normalisation from changing with endogenous model outcomes, we ignore the mortgage rebate in the median income calculation
    median_inc=median_inc_pretax-par.tau_0*(median_inc_pretax)**(1-par.tau_1)
    min_inc=(np.exp(vChi[0] + vE[0])-par.tau_0*np.exp(vChi[0] + vE[0])**(1-par.tau_1))/median_inc
   
    #construct lti vector as function of age
    mPTI_C=np.zeros((par.iNj, vE.size))
    mPTI_NC=np.zeros((par.iNj, vE.size))
    if mortgage_premium == False:
        mortgage_rate_c = par.r_m_c
    elif mortgage_premium == True:
        mortgage_rate_c = par.r_m_c_experiment 
        
    for j_index in range(par.iNj-1):
        for e_index in range(vE.size):
            mPTI_C[j_index,e_index]=max_mortgage_size(par, j_index, e_index, vChi, vE, mortgage_rate_c)/median_inc
            mPTI_NC[j_index,e_index]=max_mortgage_size(par, j_index, e_index, vChi, vE, par.r_m_nc)/median_inc
        
    
    

    

  
    vZ= np.array([1, 0.95, 0.60, 0.1])
    vPDF_z= np.array([1, 0.5, 0.35, 0.15])
    
    #WARNING - RETHINK FOR INCREASING CURVE
    vPi_L=np.ones(len(par.vPi_S_median))*par.vPi_S_median[0]

    vL_sim=np.linspace(0, 1.5, 35)
    vH=  np.array([1.50, 1.92, 2.46, 3.15, 4.03, 5.15])
    #vH=np.linspace(1.50,par.h_max,3)
    #vH_renter=np.array([1.17, 1.92])
    vH_renter=np.array([0.1,par.h_max])
   
    
    
    #We want to have equal and narrow grid spacing over the range where discrete decisions vary with income. 
    #Beyond the point where the biggest house can be bought with no mortgage and value functions are essentially linear, wider grid spacing is fine
    #We use a wider cut-off than vH[-1] because of the possibility of holding cash in hand and mortgage debt simultaneously (house price does not exceed 1 under normal calibration)
    
    """
    if par.iBmax>15:
        ##Make sure the part of the value function with high curvature is adequately covered
        vX_left_tail=np.linspace(0.01,1,par.iNb_left_tail)            
        vX_left_remainder=np.linspace(1,15,par.iNb_left)  
        vX_left=np.append(vX_left_tail[:-1], vX_left_remainder)
        
        #Beyond income 15, no need for narrow grid spacing
        max_income=np.exp(np.max(vChi)+np.max(vE_combined)-np.log(median_inc))
        gridspace_left=(15-1)/(par.iNb_left-1)
        phi_right=np.log((par.iBmax+max_income-15)/gridspace_left)/np.log(par.iNb_right-1)
        vX_right=grid.nonlinspace_jit(15,par.iBmax+max_income, par.iNb_right, phi_right)
        vX=np.append(vX_left,vX_right[1:])
        vM=vX
        
        vX_sim_left_tail=np.linspace(0,1,par.iNb_left_tail*2)      
        vX_sim_left_remainder=np.linspace(1,15,par.iNb_left*2)  
        vX_sim_left=np.append(vX_sim_left_tail[:-1], vX_sim_left_remainder)        

        gridspace_left=(15-1)/(par.iNb_left*2-1)
        phi_right=np.log((par.iBmax+max_income-15)/gridspace_left)/np.log(par.iNb_right*2-1)
        vX_sim_right=grid.nonlinspace_jit(15,par.iBmax+max_income, par.iNb_right*2, phi_right)
        vX_sim=np.append(vX_sim_left, vX_sim_right[1:])
        vM_sim=vX_sim
                
        vB_left_tail=np.linspace(0,1,par.iNb_left_tail)      
        vB_left_remainder=np.linspace(1,15,par.iNb_left)  
        vB_left=np.append(vB_left_tail[:-1], vB_left_remainder)      

        gridspace_left=(15)/(par.iNb_left-1)
        phi_right=np.log((par.iBmax-15)/gridspace_left)/np.log(par.iNb_right-1)
    
        vB_right=grid.nonlinspace_jit(15,par.iBmax, par.iNb_right,phi_right)
        vB=np.append(vB_left,vB_right[1:])
    else:
        vB_left_tail=np.linspace(0,1,par.iNb_left_tail)    
        vB_remainder=grid.nonlinspace_jit(par.iBmin, par.iBmax, par.iNb_left+par.iNb_right, par.nonlingrid)
        vB=np.append(vB_left_tail[:-1], vB_remainder)
        
        vX_left_tail=np.linspace(0.01,1,par.iNb_left_tail)  
        vX_remainder=grid.nonlinspace_jit(0.01, par.iBmax*(1+par.r)+np.exp(np.max(vChi)+np.max(vE_combined)-np.log(median_inc)), par.iNb_left+par.iNb_right, par.nonlingrid_big)
        vX=np.append(vX_left_tail[:-1], vX_remainder)
        vM=vX
        
        vX_sim_left_tail=np.linspace(0,1,par.iNb_left_tail*2)  
        vX_sim_remainder=grid.nonlinspace_jit(1, par.iBmax*(1+par.r), (par.iNb_left+par.iNb_right)*2, par.nonlingrid_big)
        vX_sim=np.append(vX_sim_left_tail[:-1], vX_sim_remainder)
        vM_sim=vX_sim       
    """
    max_income=np.exp(np.max(vChi)+np.max(vE_combined)-np.log(median_inc))
    #vX=grid.nonlinspace_jit(min_inc, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    #The lowest value of vX should be s.t. the smallest rental unit remains affordable with pos consumption. 
    vX=grid.nonlinspace_jit((1-(1-par.dDelta)/(1+par.r))*vH_renter[0]+par.dPsi, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    #vX=grid.nonlinspace_jit((1-(1-par.dDelta)/(1+par.r))*vH_renter[0]+par.dPsi, 10, par.iNb, 1.4)
    vM=grid.nonlinspace_jit(0.01, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    vB=grid.nonlinspace_jit(0, par.iBmax, par.iNb, 1.4)
    vX_sim=grid.nonlinspace_jit(0, par.iBmax, par.iNb*2, 1)
    vM_sim=vX_sim #vX_sim and vM_sim are clunkily named savings grids for simulation with twice the grid points
    
    mTypes = np.ones((len(par.vPi_S_median), 2))
    concern_vec = pd.read_excel("concern_forecast.xlsx", usecols=[1], header=None).to_numpy().ravel()
    mTypes[0:len(concern_vec)-1,0]=concern_vec[0:-1] #Explanatory note: we force jump to no sceptics in 2100; hence >=2100 entry of concern_vec overwritten with 1
    mTypes[:,1]= 1-mTypes[:,0]  
    grids_dict = {'vB': vB,
                  "vH":  vH,
                  "vH_renter":  vH_renter,
                  'vX':vX,
                  'vX_sim':vX_sim,
                  'vM':vM,
                  'vM_sim':vM_sim,
                  "vK": np.array([0,1]), # 0 is realist, 1 is optimist,
                  'vG': np.linspace(par.dXi_min, par.dXi_max, par.iXin),
                  'vL': np.linspace(0, 1.3, 20),
                  'vL_sim': vL_sim,
                  'vE': vE,
                  'vE_trans': vE_trans, 
                  'median_inc': median_inc,
                  'median_inc_pretax': median_inc_pretax,
                  'vTime': np.arange(0,len(par.vPi_S_median)),
                  'vZ': vZ,   
                  'vPDF_z':vPDF_z,
                  'vChi': vChi,
                  'min_inc': min_inc,
                  'mMarkov': mMarkov,
                  #'mIncome': np.exp(vChi[:, np.newaxis, np.newaxis] + vE[np.newaxis, :, np.newaxis] + vE_trans[np.newaxis, np.newaxis, :]) /median_inc,
                  #'mIncome_pers': mIncome_pers,
                  #'vIncome_ret': mIncome_pers[par.j_ret-1,:]*0.7,
                  'median_inc': median_inc,
                  'mMarkov_trans': mMarkov_trans[0,:],
                  'mPTI_C': mPTI_C,
                  'mPTI_NC': mPTI_NC,
                  'vPi_S_median': par.vPi_S_median,
                  'mTypes': mTypes,
                  'max_ltv': par.max_ltv, 
                  'vPi_E': vPi_E,
                  'vPi_L': vPi_L}

    grids = misc.construct_jitclass(grids_dict)
    
    return grids

def net_payment_frac(mortgage_size, par, j, e_index, vChi, vE, mortgage_rate):
    if j<par.j_ret:
        pretax_income_pers=np.exp(vChi[j] + vE[e_index])
    else:
        pretax_income_pers=0.7*np.exp(vChi[par.j_ret-1] + vE[e_index])   
    posttax_income=pretax_income_pers-par.tau_0*(max(pretax_income_pers-mortgage_rate*mortgage_size,0))**(1-par.tau_1)
    mortgage_rebate=par.tau_0*(pretax_income_pers)**(1-par.tau_1)-(posttax_income-pretax_income_pers)
    payment_next = (mortgage_rate*(1+mortgage_rate)**(par.iNj-(j+1))/((1+mortgage_rate)**(par.iNj-(j+1))-1))*mortgage_size-mortgage_rebate
    pti_gap=payment_next/posttax_income-par.lambda_pti
    return pti_gap


def max_mortgage_size(par, j, e_index, vChi, vE, mortgage_rate,
                      m_min=0.0, m_max=10000000):
    """
    Finds the maximum mortgage size such that
    net_payment_frac(...) <= par.pti
    """

    # If even zero mortgage violates the constraint, return 0
    if net_payment_frac(m_min, par, j, e_index, vChi, vE, mortgage_rate) > 0:
        return 0.0

    # Ensure upper bound actually violates the constraint
    if net_payment_frac(m_max, par, j, e_index, vChi, vE, mortgage_rate) < 0:
        raise ValueError("m_max too small — increase the upper search bound.")

    # Solve pti(m) = par.pti
    m_star = brentq(net_payment_frac, m_min, m_max, args=(par, j, e_index, vChi, vE, mortgage_rate))

    return m_star