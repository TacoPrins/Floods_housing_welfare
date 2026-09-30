import numpy as np
import math
from numba import njit

@njit
def norm_cdf(x, mu, sigma):
    return 0.5 * (1.0 + math.erf((x-mu)/(sigma*math.sqrt(2.0))))

@njit 
def initial_joint(par, grids, bequest):
    
    mPi_joint=np.zeros((grids.vX_sim.size,par.iNumStates))
    
    cov_inc_poswealth=par.corr_poswealth_income*par.sd_income_initial*np.sqrt(par.var_poswealth) #Covariance between logged income and logged initial wealth given wealth>0
    cond_var_poswealth=par.var_poswealth-cov_inc_poswealth**2/(par.sd_income_initial**2)
    
    vnowealth_frac=np.zeros((grids.vE.size))
    

    for e_index in range(grids.vE.size):
        log_inc_frommedian=grids.vE[e_index]
        logit_input=par.beta0_nowealth+log_inc_frommedian/par.sd_income_initial*par.beta1_nowealth
        vnowealth_frac[e_index]=1/(1+np.exp(-logit_input))
    
    
    dnowealth_frac=np.dot(grids.vPi_E, vnowealth_frac)

    #Obtain expected log wealth conditional on each level of log income
    vcond_mean_poswealth=np.log(bequest/(1-dnowealth_frac)) + cov_inc_poswealth/(par.sd_income_initial**2)*grids.vE
    
    #PROBLEM - SPREAD OF INITIAL WEALTH SEEMS UNACCEPTABLY HIGH 
    for e_index in range(grids.vE.size): 
        dPi_cum=0
        for x_index_sim in range(1,grids.vX_sim.size):            
            if x_index_sim<grids.vX_sim.size-1:
                mPi_joint[x_index_sim,e_index]=norm_cdf(0.5*np.log(grids.vX_sim[x_index_sim])+0.5*np.log(grids.vX_sim[x_index_sim+1]), vcond_mean_poswealth[e_index], np.sqrt(cond_var_poswealth))-dPi_cum
                dPi_cum+=mPi_joint[x_index_sim, e_index]
                #CUT OFF FAR RIGHT TAIL USING EITHER CDF OR 95TH PERCENTILE EMPIRICAL VALUE OF RATIO INITIAL WEALTH/INITIAL MEDIAN INC
                if dPi_cum>=0.95 or grids.vX_sim[x_index_sim]/np.exp(grids.vChi[0]-np.log(grids.median_inc_pretax))>5.014401:
                    mPi_joint[:,e_index]=mPi_joint[:,e_index]/dPi_cum
                    break
            else:
                mPi_joint[x_index_sim, e_index]=1-dPi_cum
        #Scale by fraction with positive wealth and in income bin
        mPi_joint[:,e_index]=mPi_joint[:,e_index]*(1-vnowealth_frac[e_index])*grids.vPi_E[e_index]
        
    #Paste no wealth observations    
    mPi_joint[0,: ]+=grids.vPi_E*vnowealth_frac
    
    #VISUAL CHECK
    #gridbounds_right=0.5*np.log(grids.vX_sim[2:])+0.5*np.log(grids.vX_sim[1:-1])
    #gridsize_exceptails=gridbounds_right[1:]-gridbounds_right[:-1]
    #gridsize_last=np.log(grids.vX_sim[-1])-gridbounds_right[-1]
    #gridsizes=np.append(gridsize_exceptails,gridsize_last)
    #for e_index in range(grids.vE.size):
    #    plt.plot(np.log(grids.vX_sim[2:-1]),mPi_joint[e_index,2:-1]/(grids.vPi_E[e_index]*gridsize_exceptails),label=f'{e_index}')
    #plt.legend()
    #plt.show()
    
        
    return mPi_joint


