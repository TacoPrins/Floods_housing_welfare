
import numpy as np
import misc_functions as misc
import interp as interp
from numba import njit

@njit(fastmath=True)
def solve(par,grids,vt_stay_input,j,h,e, dZ, dP,mortgage_start,max_ltv_cache_choice_index_left, min_payment, ltv_cache_minpay_index_left, selected_values_stay, mortgage_rate):
        
    #Initialise value from paying off more than the minimum and refinancing
    vt_stay=np.ones((grids.vM_sim.size))*-1e12
    ltv_out=np.empty((grids.vM_sim.size))
    m_out=np.empty((grids.vM_sim.size))
    
    #Some helper variables to prevent unnecessary computations    
    mortgage_withint=(1+mortgage_rate)*mortgage_start
    house_value=h*dP
    depreciation=((1-dZ)+par.dDelta)*house_value
    
    #cash in hand after housing costs
    m_0=grids.vM[0]   
    
    if ltv_cache_minpay_index_left>max_ltv_cache_choice_index_left:
        max_ltv_cache_choice_index_left=ltv_cache_minpay_index_left
        
        
    
    if j<par.iNj-1:
        l_index_l=0
        for l_choice_index in range(0,max_ltv_cache_choice_index_left+1):
            ltv_cache = grids.vL_sim[l_choice_index]
            l_index_l=misc.binary_search_sim(l_index_l, grids.vL.size, grids.vL,ltv_cache) 
            if l_choice_index>ltv_cache_minpay_index_left:
                payment=(mortgage_withint-ltv_cache*house_value*(1-par.dZeta)) + par.dZeta_fixed
            else: 
                payment=mortgage_withint-ltv_cache*house_value
            m_index_l=0
            for m_index_sim in selected_values_stay:            
                stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
                if stayer_cih>m_0:
                    m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih) 
                    candidate_value=misc._interp_2d(grids.vM, grids.vL, vt_stay_input, stayer_cih, ltv_cache,m_index_l, l_index_l)
                    if candidate_value>vt_stay[m_index_sim]:    
                        m_out[m_index_sim]=stayer_cih
                        ltv_out[m_index_sim] = ltv_cache
                        vt_stay[m_index_sim]=candidate_value 
        ltv_cache =(mortgage_withint-min_payment)/(house_value)    
        l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv_cache) 
        payment=mortgage_withint-ltv_cache*house_value
        m_index_l=0
        for m_index_sim in selected_values_stay:
            stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
            if stayer_cih>m_0:
                m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih) 
                candidate_value=misc._interp_2d(grids.vM, grids.vL, vt_stay_input, stayer_cih, ltv_cache,m_index_l, l_index_l)
                if candidate_value>vt_stay[m_index_sim]:    
                    m_out[m_index_sim]=stayer_cih
                    ltv_out[m_index_sim] = ltv_cache
                    vt_stay[m_index_sim]=candidate_value
    else:
        payment=mortgage_withint
        m_index_l=0
        for m_index_sim in selected_values_stay:
            stayer_cih=grids.vM_sim[m_index_sim]+e-depreciation-payment
            m_index_l=misc.binary_search_sim(m_index_l, grids.vM.size, grids.vM,stayer_cih) 
            if stayer_cih>m_0:             
                candidate_value=interp._interp_1d(grids.vM, vt_stay_input[:,0], stayer_cih,m_index_l)
                if candidate_value>vt_stay[m_index_sim]:    
                    m_out[m_index_sim]=stayer_cih
                    ltv_out[m_index_sim] = 0
                    vt_stay[m_index_sim]=candidate_value
    
                 
    return vt_stay, ltv_out, m_out