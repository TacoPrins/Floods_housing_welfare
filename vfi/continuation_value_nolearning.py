

"""
continuation_value.py
"""

import numpy as np
import misc_functions as misc
from numba import njit
import utility_epsilons as ut
import interp as interp

@njit
def solve_last_period_owners_C(par, grids,  vPi_S, dPi_L, k_index, dP_C_prime,mortgage_size_C, config):
    
    if config.mortgage_premium == False:
        mortgage_rate = par.r_m_c
    elif config.mortgage_premium == True:
        mortgage_rate = par.r_m_c_experiment 
    
    mW = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size))
    mW_wf = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size))
    mQ = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size))
        
        
    
    for h_index in range(grids.vH.size):
        h=grids.vH[h_index]
        for b_index in range(grids.vB.size):
            b=grids.vB[b_index]     
            for l_index in range(grids.vL.size):
                mortgage_start=mortgage_size_C[h_index,l_index]
                for damage_index in range(grids.vZ.size):
                    dZ = grids.vZ[damage_index]
                    if k_index==0:
                        if damage_index==0:
                            prob_dZ=(1-vPi_S)
                            prob_dZ_true = (1-vPi_S)
                        else:                                    
                            prob_dZ = vPi_S*grids.vPDF_z[damage_index] 
                            prob_dZ_true = vPi_S*grids.vPDF_z[damage_index] 
                    else:
                        if damage_index==0:
                            prob_dZ=(1-dPi_L)
                            prob_dZ_true = (1-vPi_S)
                        else:                                    
                            prob_dZ = dPi_L*grids.vPDF_z[damage_index] 
                            prob_dZ_true = vPi_S*grids.vPDF_z[damage_index] 
                    mW[b_index,h_index,l_index]+=prob_dZ*ut.W_bequest(par,(1+par.r)*b+(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C_prime-(1+mortgage_rate)*mortgage_start)
                    mQ[b_index,h_index,l_index]+=prob_dZ*ut.Q_bequest(par, (1+par.r)*b+(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C_prime-(1+mortgage_rate)*mortgage_start)
                    if config.welfare==True:
                        mW_wf[b_index,h_index,l_index]+=prob_dZ_true*ut.W_bequest(par,(1+par.r)*b+(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C_prime-(1+mortgage_rate)*mortgage_start)
                  
                   
   

    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0

    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW_wf

@njit
def solve_last_period_owners_NC(par, grids, k_index, dP_NC_prime,mortgage_size_NC, config): 
 
    mW = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size))
    mQ= np.zeros((grids.vB.size, grids.vH.size, grids.vL.size))
 
    for h_index in range(grids.vH.size):
        h=grids.vH[h_index]
        for b_index in range(grids.vB.size):
            b=grids.vB[b_index]
            for l_index in range(grids.vL.size):
                mortgage_start=mortgage_size_NC[h_index,l_index]                
                mW[b_index,h_index,l_index]=ut.W_bequest(par,(1+par.r)*b+(1-par.dDelta-par.dKappa_sell)*h*dP_NC_prime-(1+par.r_m_nc)*mortgage_start)
                mQ[b_index,h_index,l_index]=ut.Q_bequest(par, (1+par.r)*b+(1-par.dDelta-par.dKappa_sell)*h*dP_NC_prime-(1+par.r_m_nc)*mortgage_start)
                  
      
    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0
    
    #Return mW for final argument, not mW_f, since flood risk does not affect terminal value
    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW

@njit
def solve_last_period_renters(par, grids):
    
    mW = np.zeros((grids.vB.size))
    mQ = np.zeros((grids.vB.size))   

    for b_index in range(grids.vB.size):
        
        b=grids.vB[b_index]
        
        mW[b_index]=ut.W_bequest(par, (1+par.r)*b)
        mQ[b_index]=ut.Q_bequest(par, (1+par.r)*b)
   
    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0
      
    #Return mW for final argument, not mW_f, since flood risk does not affect terminal value
    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW

@njit
def solve_owners_C(par, grids, j_index, k_index, vPi_S, dPi_L, coastal_stayer_inputs,coastal_mover_inputs, dP_C_prime,mortgage_size_C, config):
    if config.mortgage_premium == False:
        mortgage_rate = par.r_m_c
    elif config.mortgage_premium == True:
        mortgage_rate = par.r_m_c_experiment 
   
    vt_stay_c_input = coastal_stayer_inputs['vt_stay_c_input']
    vt_renter_input = coastal_mover_inputs['vt_renter_input']
    vt_buy_c_input = coastal_mover_inputs['vt_buy_c_input']
    vt_buy_nc_input = coastal_mover_inputs['vt_buy_nc_input']
    
    vt_stay_c_input_wf = coastal_stayer_inputs['vt_stay_c_input_wf']
    vt_renter_input_wf = coastal_mover_inputs['vt_renter_input_wf']
    vt_buy_c_input_wf = coastal_mover_inputs['vt_buy_c_input_wf']
    vt_buy_nc_input_wf = coastal_mover_inputs['vt_buy_nc_input_wf']
    
    qt_stay_c_input = coastal_stayer_inputs['qt_stay_c_input']
    qt_renter_input = coastal_mover_inputs['qt_renter_input']
    qt_buy_c_input = coastal_mover_inputs['qt_buy_c_input']
    qt_buy_nc_input = coastal_mover_inputs['qt_buy_nc_input']            


    if j_index<par.j_ret:   
        mW_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size))       
        mW_wf_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size))
        mQ_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size))

    else:
        mW_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))
        mW_wf_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))
        mQ_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))
    mW = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
    mW_wf = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
    mQ = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
 
  
    for h_index in range(grids.vH.size):
        h=grids.vH[h_index]       
        house_value=h*dP_C_prime          
        for l_index in range(grids.vL.size):  
            if l_index==0:
                mortgage_start=0
                mortgage_withint=0
                min_payment=0
                ltv_minpay=0
                ltv_minpay_index=0
            elif l_index>0:               
                mortgage_start=mortgage_size_C[h_index,l_index]
                mortgage_withint=(1+mortgage_rate)*mortgage_start
                min_payment = (mortgage_rate*(1+mortgage_rate)**(par.iNj-j_index)/((1+mortgage_rate)**(par.iNj-j_index)-1))*mortgage_start
                ltv_minpay=(mortgage_withint-min_payment)/(house_value) 
                ltv_minpay_index=misc.binary_search(0,grids.vL.size,grids.vL,ltv_minpay)                
                   
            for e_prime_index in range(grids.vE.size):     
                max_mortgage_pti=grids.mPTI_C[j_index,e_prime_index]            
                    
                if grids.max_ltv<max_mortgage_pti/(house_value):
                    max_ltv_choice=grids.max_ltv
                else: 
                    max_ltv_choice=max_mortgage_pti/(house_value)
                max_ltv_choice_index=misc.binary_search(0,grids.vL.size,grids.vL,max_ltv_choice)
                    
                for e_trans_index in range(grids.vE_trans.size):
                    e_prime, mortgage_rebate=misc.net_income(par, grids, j_index, e_prime_index, e_trans_index, mortgage_start, mortgage_rate)
                    if e_trans_index>0:
                        continue
                    for b_index in range(grids.vB.size):
                        b=grids.vB[b_index]
                        savings=(1+par.r)*b
                        defaulter_cih=savings+e_prime-mortgage_rebate
                        # flood scenarions
                        for damage_index in range(grids.vZ.size):
                            dZ = grids.vZ[damage_index]
                            if k_index==0:
                                if damage_index==0:
                                    prob_dZ=(1-vPi_S)
                                    prob_dZ_true = (1-vPi_S)
                                else:                                    
                                    prob_dZ = vPi_S*grids.vPDF_z[damage_index] 
                                    prob_dZ_true = vPi_S*grids.vPDF_z[damage_index] 
                            else:
                                if damage_index==0:
                                    prob_dZ=(1-dPi_L)
                                    prob_dZ_true = (1-vPi_S)
                                else:                                    
                                    prob_dZ = dPi_L*grids.vPDF_z[damage_index] 
                                    prob_dZ_true = vPi_S*grids.vPDF_z[damage_index] 
                            sell_value=(dZ-par.dDelta-par.dKappa_sell)*h*dP_C_prime       
                            depreciation=(1-dZ+par.dDelta)*h*dP_C_prime  
                            max_payment = savings-depreciation+e_prime-grids.vM[0]
                            stayer_cih_beforem=savings-depreciation+e_prime
                            seller_cih=savings+sell_value+e_prime-mortgage_withint
                            #Initialise choices that require optimal choice of mortgage 
                            stay_paymore  = -1e12
                            stay_refinance = -1e12
                            if min_payment > max_payment:
                                stay = -1e12
                            else:
                                stay = misc.interp_2d(grids.vM, grids.vL,vt_stay_c_input[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                            
                            if j_index<par.iNj-1 and mortgage_start>0:
                                for l_choice_index in range(0,ltv_minpay_index+1):
                                    payment=(mortgage_withint-grids.vL[l_choice_index]*house_value)
                                    if payment>max_payment:
                                        continue
                                    assert payment>=min_payment
                                    candidate_value=interp.interp_1d(grids.vM, vt_stay_c_input[:,h_index, l_choice_index, e_prime_index],stayer_cih_beforem-payment)
                                    if candidate_value>stay_paymore:
                                        payment_paymore = payment
                                        l_index_paymore = l_choice_index
                                        stay_paymore=candidate_value
                            

                            
                            if max_ltv_choice_index>ltv_minpay_index and j_index<par.iNj-1:
                                for l_choice_index in range(ltv_minpay_index+1,max_ltv_choice_index+1):
                                    payment=(mortgage_withint-grids.vL[l_choice_index]*house_value*(1-par.dZeta))+par.dZeta_fixed
                                    if payment>max_payment:
                                        continue
                                    assert payment-par.dZeta*grids.vL[l_choice_index]*house_value-par.dZeta_fixed<=min_payment
                                    candidate_value=interp.interp_1d(grids.vM, vt_stay_c_input[:,h_index, l_choice_index, e_prime_index],stayer_cih_beforem-payment)
                                    if candidate_value>stay_refinance:
                                        payment_refinance = payment
                                        l_index_refinance = l_choice_index
                                        stay_refinance=candidate_value
                           
                            
                            if mortgage_withint>sell_value:
                                default = -1/(-1/interp.interp_1d(grids.vX, vt_renter_input[:, e_prime_index],defaulter_cih)-par.dXi_foreclosure)
                            else:
                                default =-1e12
                                                                                         
                            if seller_cih>0:
                                stay_renter = interp.interp_1d(grids.vX, vt_renter_input[:, e_prime_index],seller_cih)  
                            else:
                                stay_renter = -1e12
                            
                            buyC = interp.interp_1d(grids.vX, vt_buy_c_input[:, e_prime_index],seller_cih)
                            buyNC = interp.interp_1d(grids.vX, vt_buy_nc_input[:, e_prime_index],seller_cih)
                
                              
                            if (stay > stay_renter) and (stay > buyC) and (stay > buyNC) and (stay >= stay_paymore) and (stay >= stay_refinance) and (stay > default):
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ * -1/stay
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] +=  prob_dZ *-1/misc.interp_2d(grids.vM, grids.vL, qt_stay_c_input[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/misc.interp_2d(grids.vM, grids.vL,vt_stay_c_input_wf[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                                
                                
                            elif (stay_paymore > stay_renter) and (stay_paymore > buyC) and (stay_paymore > buyNC) and (stay_paymore > stay_refinance) and (stay_paymore > default):
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/stay_paymore
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/interp.interp_1d(grids.vM, qt_stay_c_input[ :,h_index, l_index_paymore, e_prime_index],stayer_cih_beforem-payment_paymore)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/interp.interp_1d(grids.vM, vt_stay_c_input_wf[ :,h_index, l_index_paymore, e_prime_index],stayer_cih_beforem-payment_paymore)
                                
                                
                            elif (stay_refinance > stay_renter) and (stay_refinance > buyC) and (stay_refinance > buyNC) and (stay_refinance > default): 
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/stay_refinance
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/interp.interp_1d(grids.vM,qt_stay_c_input[ :,h_index, l_index_refinance, e_prime_index],stayer_cih_beforem-payment_refinance)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/interp.interp_1d(grids.vM,vt_stay_c_input_wf[  :,h_index, l_index_refinance, e_prime_index],stayer_cih_beforem-payment_refinance)
                           
                           
                            elif (default > stay_renter) and (default > buyC) and (default > buyNC): 
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/default
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/interp.interp_1d(grids.vX, qt_renter_input[:, e_prime_index],defaulter_cih)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * (-1/interp.interp_1d(grids.vX, vt_renter_input_wf[:, e_prime_index],defaulter_cih)-par.dXi_foreclosure)
                               
                                
                            elif (stay_renter > buyC) and (stay_renter > buyNC):
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ * -1/stay_renter
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ * -1/interp.interp_1d(grids.vX, qt_renter_input[:, e_prime_index],seller_cih)  
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/interp.interp_1d(grids.vX, vt_renter_input_wf[:, e_prime_index],seller_cih)

                            elif buyC > buyNC:
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/buyC
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/interp.interp_1d(grids.vX, qt_buy_c_input[:, e_prime_index],seller_cih)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/interp.interp_1d(grids.vX, vt_buy_c_input_wf[:, e_prime_index],seller_cih)

                            else: 
                                mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/buyNC
                                mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ *-1/interp.interp_1d(grids.vX, qt_buy_nc_input[:, e_prime_index],seller_cih)
                                if config.welfare == True:
                                    mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] += prob_dZ_true * -1/interp.interp_1d(grids.vX, vt_buy_nc_input_wf[:, e_prime_index],seller_cih)
                                    
    if j_index<par.j_ret:  
        for e_index in range(grids.vE.size):
            for e_prime_index in range(grids.vE.size):
                for e_trans_index in range(grids.vE_trans.size):        
                    prob_weight_e=grids.mMarkov_trans[e_trans_index]*grids.mMarkov[e_index, e_prime_index]
                    mW[:, :, :, e_index] +=prob_weight_e * mW_inner[:, :, :, e_prime_index, e_trans_index] 
                    mQ[:, :, :, e_index] +=prob_weight_e * mQ_inner[:, :, :, e_prime_index, e_trans_index]                         
                    if config.welfare == True:
                        mW_wf[:, :, :, e_index] +=prob_weight_e * mW_wf_inner[:, :, :, e_prime_index, e_trans_index] 
    else:
        mW[:, :, :, :]=mW_inner[:, :, :, :, 0]
        mQ[:, :, :, :]=mQ_inner[:, :, :, :, 0]
        if config.welfare == True:
            mW_wf[:, :, :, :]=mW_wf_inner[:, :, :, :, 0]


    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0

    
    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW_wf, mW_wf_inner[:, :, :, :, 0] 

@njit
def solve_owners_NC(par, grids, j_index, k_index, noncoastal_stayer_inputs,noncoastal_mover_inputs, dP_NC_prime, mortgage_size_NC, config):
        
    vt_stay_nc_input = noncoastal_stayer_inputs['vt_stay_nc_input']
    vt_renter_input = noncoastal_mover_inputs['vt_renter_input']
    vt_buy_c_input = noncoastal_mover_inputs['vt_buy_c_input']
    vt_buy_nc_input = noncoastal_mover_inputs['vt_buy_nc_input']
    
    vt_stay_nc_input_wf = noncoastal_stayer_inputs['vt_stay_nc_input_wf']
    vt_renter_input_wf = noncoastal_mover_inputs['vt_renter_input_wf']
    vt_buy_c_input_wf = noncoastal_mover_inputs['vt_buy_c_input_wf']
    vt_buy_nc_input_wf = noncoastal_mover_inputs['vt_buy_nc_input_wf']
    
    qt_stay_nc_input = noncoastal_stayer_inputs['qt_stay_nc_input']
    qt_renter_input = noncoastal_mover_inputs['qt_renter_input']
    qt_buy_c_input = noncoastal_mover_inputs['qt_buy_c_input']
    qt_buy_nc_input = noncoastal_mover_inputs['qt_buy_nc_input']    

 
    if j_index<par.j_ret:   
        mW_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size))       
        mW_wf_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size))
        mQ_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, grids.vE_trans.size)) 

    else:
        mW_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))       
        mW_wf_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))
        mQ_inner = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size, 1))
    mW = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
    mW_wf = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
    mQ = np.zeros((grids.vB.size, grids.vH.size, grids.vL.size, grids.vE.size))
    
    mortgage_rate=par.r_m_nc
       
    for h_index in range(grids.vH.size):
        h=grids.vH[h_index]         
        sell_value=(1-par.dDelta-par.dKappa_sell)*h*dP_NC_prime   
        depreciation=(par.dDelta)*h*dP_NC_prime        
        house_value=h*dP_NC_prime

        for l_index in range(grids.vL.size):
            if l_index==0:
                mortgage_start=0
                mortgage_withint=0
                min_payment=0
                ltv_minpay=0
                ltv_minpay_index=0
            elif l_index>0: 
                mortgage_start=mortgage_size_NC[h_index,l_index]
                min_payment = (par.r_m_nc*(1+par.r_m_nc)**(par.iNj-j_index)/((1+par.r_m_nc)**(par.iNj-j_index)-1))*mortgage_start
                mortgage_withint=(1+par.r_m_nc)*mortgage_start
                ltv_minpay=(mortgage_withint-min_payment)/(house_value) 
                ltv_minpay_index=misc.binary_search(0,grids.vL.size,grids.vL,ltv_minpay)


            for e_prime_index in range(grids.vE.size):
                max_mortgage_pti=grids.mPTI_NC[j_index,e_prime_index]                         
                if grids.max_ltv<max_mortgage_pti/(house_value):
                    max_ltv_choice=grids.max_ltv
                else: 
                    max_ltv_choice=max_mortgage_pti/(house_value)
                max_ltv_choice_index=misc.binary_search(0,grids.vL.size,grids.vL,max_ltv_choice)


                
                for e_trans_index in range(grids.vE_trans.size):    
                    e_prime, mortgage_rebate =misc.net_income(par, grids, j_index, e_prime_index, e_trans_index, mortgage_start, mortgage_rate)
                    if e_trans_index>0:
                        continue            
                    for b_index in range(grids.vB.size):
                        b=grids.vB[b_index]
                        savings=(1+par.r)*b
                        max_payment = savings-depreciation+e_prime-grids.vM[0]
                        stayer_cih_beforem=savings-depreciation+e_prime
                        seller_cih=savings+sell_value+e_prime-mortgage_withint
                        defaulter_cih=savings+e_prime-mortgage_rebate
                        #Initialise choices that require optimal choice of mortgage 
                        stay_paymore  = -1e12     
                        stay_refinance = -1e12

                        
                        if min_payment>max_payment:
                            stay = -1e12
                        else:
                            stay = misc.interp_2d(grids.vM, grids.vL, vt_stay_nc_input[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                        
                                                                
                        if j_index<par.iNj-1 and mortgage_start>0:
                            for l_choice_index in range(0,ltv_minpay_index+1):
                                payment=(mortgage_withint-grids.vL[l_choice_index]*house_value)
                                if payment>max_payment:
                                    continue
                                assert payment>=min_payment
                                candidate_value=interp.interp_1d(grids.vM, vt_stay_nc_input[ :,h_index, l_choice_index, e_prime_index],stayer_cih_beforem-payment)
                                if candidate_value>stay_paymore:
                                    payment_paymore = payment
                                    l_index_paymore = l_choice_index
                                    stay_paymore=candidate_value
                            
                                                       
                        if max_ltv_choice_index>ltv_minpay_index and j_index<par.iNj-1:
                            for l_choice_index in range(ltv_minpay_index+1,max_ltv_choice_index+1):
                                payment=(mortgage_withint-grids.vL[l_choice_index]*house_value*(1-par.dZeta))+par.dZeta_fixed
                                if payment>max_payment:
                                    continue
                                assert payment-par.dZeta*grids.vL[l_choice_index]*house_value-par.dZeta_fixed<=min_payment
                                candidate_value=interp.interp_1d(grids.vM, vt_stay_nc_input[ :,h_index, l_choice_index, e_prime_index],stayer_cih_beforem-payment)
                                if candidate_value>stay_refinance:
                                    payment_refinance = payment
                                    l_index_refinance = l_choice_index
                                    stay_refinance=candidate_value
                        
                                                    
                        if mortgage_withint>sell_value:
                            default = -1/(-1/interp.interp_1d(grids.vX, vt_renter_input[:, e_prime_index],defaulter_cih)-par.dXi_foreclosure)
                        else:
                            default =-1e12
                                                     
                        if seller_cih>0:
                            stay_renter = interp.interp_1d(grids.vX, vt_renter_input[:, e_prime_index],seller_cih)  
                        else:
                            stay_renter=-1e12    
                       
                        buyC = interp.interp_1d(grids.vX, vt_buy_c_input[:, e_prime_index],seller_cih)        
                        buyNC = interp.interp_1d(grids.vX, vt_buy_nc_input[:, e_prime_index],seller_cih)
                                                                                
                        
                        if (stay > stay_renter) and (stay > buyC) and (stay > buyNC) and (stay >= stay_paymore) and (stay >= stay_refinance) and (stay>default):
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/stay
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/misc.interp_2d(grids.vM, grids.vL, qt_stay_nc_input[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/misc.interp_2d(grids.vM, grids.vL, vt_stay_nc_input_wf[ :,h_index, :, e_prime_index],stayer_cih_beforem-min_payment, ltv_minpay)
                                
                        elif (stay_paymore > stay_renter) and (stay_paymore > buyC) and (stay_paymore > buyNC) and (stay_paymore > stay_refinance) and (stay_paymore>default):
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/stay_paymore
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vM, qt_stay_nc_input[ :,h_index, l_index_paymore, e_prime_index],stayer_cih_beforem-payment_paymore)
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vM, vt_stay_nc_input_wf[ :,h_index, l_index_paymore, e_prime_index],stayer_cih_beforem-payment_paymore)

                        elif (stay_refinance > stay_renter) and (stay_refinance > buyC) and (stay_refinance > buyNC) and (stay_refinance>default): 
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/stay_refinance
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vM, qt_stay_nc_input[ :,h_index, l_index_refinance, e_prime_index],stayer_cih_beforem-payment_refinance)
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vM, vt_stay_nc_input_wf[ :,h_index, l_index_refinance, e_prime_index],stayer_cih_beforem-payment_refinance)

                        
                        elif (stay_renter > buyC) and (stay_renter > buyNC) and (stay_renter>default):
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/stay_renter
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_renter_input[:, e_prime_index],seller_cih)  
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_renter_input_wf[:, e_prime_index],seller_cih)
  
                        elif (buyC > buyNC) and (buyC>default):
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/buyC
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_buy_c_input[:, e_prime_index],seller_cih)
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_buy_c_input_wf[:, e_prime_index],seller_cih)

                        elif buyNC>default: 
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/buyNC
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_buy_nc_input[:, e_prime_index],seller_cih) 
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_buy_nc_input_wf[:, e_prime_index],seller_cih) 

                        else: 
                            mW_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/default
                            mQ_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_renter_input[:, e_prime_index],defaulter_cih)  
                            if config.welfare == True:
                                mW_wf_inner[ b_index,h_index,l_index, e_prime_index, e_trans_index] = (-1/interp.interp_1d(grids.vX, vt_renter_input_wf[:, e_prime_index],defaulter_cih)-par.dXi_foreclosure)
                                
    if j_index<par.j_ret:  
        for e_index in range(grids.vE.size):
            for e_prime_index in range(grids.vE.size):
                for e_trans_index in range(grids.vE_trans.size):        
                    prob_weight_e=grids.mMarkov_trans[e_trans_index]*grids.mMarkov[e_index, e_prime_index]
                    mW[:, :, :, e_index] +=prob_weight_e * mW_inner[:, :, :, e_prime_index, e_trans_index] 
                    mQ[:, :, :, e_index] +=prob_weight_e * mQ_inner[:, :, :, e_prime_index, e_trans_index]                         
                    if config.welfare == True:
                        mW_wf[:, :, :, e_index] +=prob_weight_e * mW_wf_inner[:, :, :, e_prime_index, e_trans_index] 
    else:
        mW[:, :, :, :]=mW_inner[:, :, :, :, 0]
        mQ[:, :, :, :]=mQ_inner[:, :, :, :, 0]
        if config.welfare == True:
            mW_wf[:, :, :, :]=mW_wf_inner[:, :, :, :, 0]
            

    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0
    
    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW_wf, mW_wf_inner[:, :, :, :, 0] 


@njit
def solve_renters(par, grids, j_index, k_index, renter_inputs, config):
      
  
    vt_renter_input = renter_inputs['vt_renter_input']
    vt_buy_c_input = renter_inputs['vt_buy_c_input']
    vt_buy_nc_input = renter_inputs['vt_buy_nc_input']
    
    vt_renter_input_wf = renter_inputs['vt_renter_input_wf']
    vt_buy_c_input_wf = renter_inputs['vt_buy_c_input_wf']
    vt_buy_nc_input_wf = renter_inputs['vt_buy_nc_input_wf']    
   
    qt_renter_input = renter_inputs['qt_renter_input']
    qt_buy_c_input = renter_inputs['qt_buy_c_input']
    qt_buy_nc_input = renter_inputs['qt_buy_nc_input']    
   
    
    if j_index<par.j_ret:   
        mW_inner = np.zeros((grids.vB.size, grids.vE.size, grids.vE_trans.size))       
        mW_wf_inner = np.zeros((grids.vB.size, grids.vE.size, grids.vE_trans.size))
        mQ_inner = np.zeros((grids.vB.size, grids.vE.size, grids.vE_trans.size))
    else:
        mW_inner = np.zeros((grids.vB.size, grids.vE.size, 1))       
        mW_wf_inner = np.zeros((grids.vB.size, grids.vE.size, 1))
        mQ_inner = np.zeros((grids.vB.size, grids.vE.size, 1))
    mW = np.zeros((grids.vB.size, grids.vE.size))
    mW_wf = np.zeros((grids.vB.size, grids.vE.size))
    mQ = np.zeros((grids.vB.size, grids.vE.size))

     
   
    for e_prime_index in range(grids.vE.size):
        for e_trans_index in range(grids.vE_trans.size):                                                
            e_prime, mortgage_rebate =misc.net_income(par, grids, j_index, e_prime_index, e_trans_index, 0, 0)
            if e_trans_index>0:
                continue                     
            for b_index in range(grids.vB.size):
                b=grids.vB[b_index]
                savings=(1+par.r)*b        
                renter_cih = savings+e_prime
                stay_renter = interp.interp_1d(grids.vX, vt_renter_input[:, e_prime_index],renter_cih)  
                buyC = interp.interp_1d(grids.vX, vt_buy_c_input[:, e_prime_index],renter_cih)
                buyNC = interp.interp_1d(grids.vX, vt_buy_nc_input[:, e_prime_index],renter_cih)
                    
                if (stay_renter > buyC) and (stay_renter > buyNC):
                    mW_inner[ b_index, e_prime_index, e_trans_index] = -1/stay_renter
                    mQ_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_renter_input[:, e_prime_index],renter_cih)  
                    if config.welfare == True:
                        mW_wf_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_renter_input_wf[:, e_prime_index],renter_cih)  

                elif buyC > buyNC:
                    mW_inner[ b_index, e_prime_index, e_trans_index] = -1/buyC
                    mQ_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_buy_c_input[:, e_prime_index],renter_cih)
                    if config.welfare == True:
                        mW_wf_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_buy_c_input_wf[:, e_prime_index],renter_cih)  

                else: 
                    mW_inner[ b_index, e_prime_index, e_trans_index] = -1/buyNC
                    mQ_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, qt_buy_nc_input[:, e_prime_index],renter_cih)
                    if config.welfare == True:
                        mW_wf_inner[ b_index, e_prime_index, e_trans_index] = -1/interp.interp_1d(grids.vX, vt_buy_nc_input_wf[:, e_prime_index],renter_cih)  
    
    
    if j_index<par.j_ret:  
        for e_index in range(grids.vE.size):
            for e_prime_index in range(grids.vE.size):
                for e_trans_index in range(grids.vE_trans.size):        
                    prob_weight_e=grids.mMarkov_trans[e_trans_index]*grids.mMarkov[e_index, e_prime_index]
                    mW[:, e_index] +=prob_weight_e * mW_inner[:,e_prime_index, e_trans_index] 
                    mQ[:, e_index] +=prob_weight_e * mQ_inner[:,e_prime_index, e_trans_index]                         
                    if config.welfare == True:
                        mW_wf[:, e_index] +=prob_weight_e * mW_wf_inner[:,e_prime_index, e_trans_index] 
    else:
        mW[:, :]=mW_inner[:, :, 0]
        mQ[:, :]=mQ_inner[:, :, 0]
        if config.welfare == True:
            mW_wf[:, :]=mW_wf_inner[:, :, 0]
        
    assert np.isnan(mW).sum() == 0
    assert np.isnan(mQ).sum() == 0
          
    return par.dBeta*mW, par.dBeta*(1+par.r)*mQ, par.dBeta*mW_wf, mW_wf_inner[:, :, 0]


@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left