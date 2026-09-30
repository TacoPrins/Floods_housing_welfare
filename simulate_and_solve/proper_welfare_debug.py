# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:00:10 2026

@author: tprins
"""

import numpy as np
import misc_functions as misc
import household_problem_epsilons_nolearning as household_problem  
import simulation as sim
from numba import njit
import LoM_epsilons as lom

@njit
def find_expenditure_equiv_EK_SLR(par, grids, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_in, vCoeff_NC_in,mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, vcoastal_beq, vnoncoastal_beq, vsavings_beq, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, solve_initial_ss_, config):
    dP_C_lom=lom.LoM_path(par,grids,vCoeff_C_in, config)
    dP_NC_lom=lom.LoM_path(par,grids,vCoeff_NC_in, config)
    if config.sceptics == False:
        k_dim=1
    else:
        k_dim=grids.vK.size      
        
    
    # price_history, _, _, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf,_,_,_=equil.generate_pricepath(grids, par, vCoeff_C_in,vCoeff_NC_in, dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, config)
    v_nonowner_wf_expanded_SLR=grid_adjust_rentshape(par,grids,v_nonowner_wf)
    v_owner_c_wf_expanded_SLR=grid_adjust(par,grids,v_owner_c_wf)
    v_owner_nc_wf_expanded_SLR=grid_adjust(par,grids,v_owner_nc_wf)
    

    wf_loss = np.linspace(-0.02, 0.05,10)
    
    ce_C  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_NC = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_newborns = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
    wf_SLR_c=np.zeros((k_dim,grids.vE.size))
    wf_SLR_nc=np.zeros((k_dim,grids.vE.size))
    wf_SLR_rent=np.zeros((k_dim,grids.vE.size))
    wf_SS_newborns=np.zeros((wf_loss.size,k_dim,grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):
            wf_SLR_c[k_index,e_index] = np.sum(mDist1_c_SS[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_SLR[0,1:,k_index, :, :,:,:,e_index])
            wf_SLR_nc[k_index,e_index] = np.sum(mDist1_nc_SS[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_SLR[0,1:,k_index, :, :,:,:,e_index])
            wf_SLR_rent[k_index,e_index] = np.sum(mDist1_renter_SS[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_SLR[0,:,k_index, :, :,e_index])

    for wf_idx in range(wf_loss.size):
        par.wf_wedge[0] = wf_loss[wf_idx]
        _, _, _, _, _, _, v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS = household_problem.solve_ss(grids, par, vCoeff_C_initial[0], vCoeff_NC_initial[0], solve_initial_ss_)
        v_nonowner_wf_expanded_SS=grid_adjust_rentshape(par,grids,v_nonowner_wf_SS)
        v_owner_c_wf_expanded_SS=grid_adjust(par,grids,v_owner_c_wf_SS)
        v_owner_nc_wf_expanded_SS=grid_adjust(par,grids,v_owner_nc_wf_SS)
        
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                wf_SS_c = np.sum(mDist1_c_SS[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_SS[0,1:,k_index, :, :,:,:,e_index])
                wf_SS_nc = np.sum(mDist1_nc_SS[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_SS[0,1:,k_index, :, :,:,:,e_index])
                wf_SS_rent = np.sum(mDist1_renter_SS[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_SS[0,:,k_index, :, :,e_index])

                ce_C[wf_idx,k_index, e_index] = wf_SS_c - wf_SLR_c[k_index,e_index]
                ce_NC[wf_idx,k_index, e_index] = wf_SS_nc - wf_SLR_nc[k_index,e_index]
                ce_renter[wf_idx,k_index, e_index] = wf_SS_rent - wf_SLR_rent[k_index,e_index]
               
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                wf_SS_newborns[wf_idx, k_index,e_index] = np.sum(mDist1_renter_SS[0,k_index, :, :,e_index]* v_nonowner_wf_expanded_SS[0,0,k_index, :, :, e_index])/np.sum(mDist1_renter_SS[0,k_index, :, :,e_index])
        # print('wf_SS_newborns', wf_idx, 'gives k,e matrix', wf_SS_newborns[wf_idx,:])
     
    wf_SLR_newborns=np.zeros((grids.vTime.size,k_dim,grids.vE.size)) 
     
    for t_index in range(grids.vTime.size-1):
        newborn_dist = sim.gen_initial_dist(par, grids, t_index, dP_C_lom[t_index], dP_NC_lom[t_index], vcoastal_beq[t_index], vnoncoastal_beq[t_index], vsavings_beq[t_index], config.sceptics)
        # weight
        for k_index in range(k_dim):
            # print('time', t_index, 'has mass for k_index', k_index, 'equal to', np.sum(newborn_dist[k_index,]))
            for e_index in range(grids.vE.size):
                if np.sum(newborn_dist[k_index,:,:,e_index]) > 0:                    
                    wf_SLR_newborns[t_index, k_index, e_index]  = np.sum(newborn_dist[k_index,:,:,e_index]* v_nonowner_wf_expanded_SLR[t_index,0,k_index, :, :,e_index])/np.sum(newborn_dist[k_index,:,:,e_index])
                else: 
                    wf_SLR_newborns[t_index, k_index, e_index] = np.nan
                # print('t,k,e', t_index, k_index, e_index, 'has wf_SLR_newborns:', wf_SLR_newborns)
                for wf_idx in range(wf_loss.size):
                    ce_renter_newborns[wf_idx,t_index, k_index, e_index] = wf_SS_newborns[wf_idx, k_index,e_index] - wf_SLR_newborns[t_index, k_index, e_index] 
            
    par.wf_wedge[0] = 0.0
    
    
    tax_equiv_C = np.zeros((k_dim, grids.vE.size))
    tax_equiv_NC = np.zeros((k_dim, grids.vE.size))
    tax_equiv_renter = np.zeros((k_dim, grids.vE.size))
    tax_equiv_newborns = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):            
            
            tax_equiv_C[k_index, e_index] = find_zero_linear(wf_loss, ce_C[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
            tax_equiv_NC[k_index, e_index] = find_zero_linear(wf_loss, ce_NC[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
            tax_equiv_renter[k_index, e_index] = find_zero_linear(wf_loss, ce_renter[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero
            for t_index in range(grids.vTime.size-1):
                tax_equiv_newborns[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns[:,t_index, k_index, e_index])
    
    return tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns, wf_SLR_newborns


# @njit
# def find_expenditure_equiv_EK_policy(par, grids, vCoeff_C_in, vCoeff_NC_in,vCoeff_C_pol_in, vCoeff_NC_pol_in,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vTotal_bequest_HE, config_nopol, config_pol):
#     """
#     New: import the value functions at the time of policy introduction as well as the distribution:
#     I call it the 'business as usual' (BAU) value functions
        
    
#     dist_c_2026: J x K x G x M x H x L x E (2026 distribution)
#     dist_nc_2026: J x K x G x M x H x L x E 
#     dist_renter_2026: J x K x G x M x E 
    
#     Compute >2026 welfare for policy case.
#     Within tax_wf, calculate wf for no-policy (BAU)
#     Find tax_wf that eqaulizes
#     """
    
#     if config_pol.sceptics == False:
#         k_dim=1
#     else:
#         k_dim=grids.vK.size        

#     # get value functions over transition with experiment
#     t_pol = int((par.experiment_year-par.starting_year)/par.time_increment)                                                                                                                                                                                                                                                        
#     _, _, _, _, _, _, v_owner_c_wf_pol, v_owner_nc_wf_pol, v_nonowner_wf_pol = household_problem.solve(grids, par, vCoeff_C_pol_in,vCoeff_NC_pol_in, config_pol)
    
#     v_owner_c_wf_expanded_pol=grid_adjust(par,grids,v_owner_c_wf_pol)
#     v_owner_nc_wf_expanded_pol=grid_adjust(par,grids,v_owner_nc_wf_pol)
#     v_nonowner_wf_expanded_pol=grid_adjust_rentshape(par,grids,v_nonowner_wf_pol)
    
#     wf_loss = np.linspace(-0.10, 0.15,25)
    
#     ce_C  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
#     ce_NC = np.zeros((wf_loss.size,k_dim, grids.vE.size))
#     ce_renter = np.zeros((wf_loss.size,k_dim, grids.vE.size))
#     ce_renter_newborns = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
#     wf_pol_c=np.zeros((k_dim,grids.vE.size))
#     wf_pol_nc=np.zeros((k_dim,grids.vE.size))
#     wf_pol_rent=np.zeros((k_dim,grids.vE.size))
#     wf_pol_newborns=np.zeros((grids.vTime.size,k_dim,grids.vE.size))       
   
    
#     for k_index in range(k_dim):
#         for e_index in range(grids.vE.size):
#             wf_pol_c[k_index,e_index] = np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_pol[t_pol,1:,k_index, :, :,:,:,e_index])
#             wf_pol_nc[k_index,e_index] = np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_pol[t_pol,1:,k_index, :, :,:,:,e_index])
#             wf_pol_rent[k_index,e_index] = np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_pol[t_pol,:,k_index, :, :,e_index])

    
#     for t_index in range(t_pol, grids.vTime.size):
#         total_bequest = vTotal_bequest_HE[t_index]
#         mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
#         # weight
#         for k_index in range(k_dim):
#             for e_index in range(grids.vE.size):
#                 if config_pol.sceptics==True:
#                     newborn_dist = (1/par.iNj)*(1/grids.vG.size)*grids.mTypes[t_index,k_index]*mPi_joint[:,e_index]        
#                 else:
#                     newborn_dist = (1/par.iNj)*(1/grids.vG.size)*mPi_joint[:,e_index]     
#                 # NOTE HERE: Size wf_pol_newborns is grids.vTime.size. It starts fillling here from 14, until grids.vTime.size
#                 wf_pol_newborns[t_index,k_index,e_index] = np.sum(newborn_dist * v_nonowner_wf_expanded_pol[t_index,0,k_index, :, :,e_index])

#     for wf_idx in range(wf_loss.size):
#         par.wf_wedge[0] = wf_loss[wf_idx]
#         print(par.wf_wedge[0])
#         _, _, _, _, _, _, v_owner_c_wf_BAU, v_owner_nc_wf_BAU, v_nonowner_wf_BAU = household_problem.solve(grids, par, vCoeff_C_in,vCoeff_NC_in, config_nopol)
#         v_owner_c_wf_expanded_BAU=grid_adjust(par,grids,v_owner_c_wf_BAU)
#         v_owner_nc_wf_expanded_BAU=grid_adjust(par,grids,v_owner_nc_wf_BAU)
#         v_nonowner_wf_expanded_BAU=grid_adjust_rentshape(par,grids,v_nonowner_wf_BAU)
#         for k_index in range(k_dim):
#             for e_index in range(grids.vE.size):
#                 ce_C[wf_idx,k_index,e_index]=np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_pol_c[k_index,e_index]
#                 ce_NC[wf_idx,k_index,e_index]=np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_pol_nc[k_index,e_index]
#                 ce_renter[wf_idx,k_index,e_index]=np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_BAU[t_pol,:,k_index, :, :,e_index])-wf_pol_rent[k_index,e_index]
        
#         for t_index in range(t_pol, grids.vTime.size):
#             total_bequest = vTotal_bequest_HE[t_index]
#             mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
#             for k_index in range(k_dim):
#                 for e_index in range(grids.vE.size):
#                     if config_pol.sceptics==True:
#                         newborn_dist= (1/par.iNj)*(1/grids.vG.size)*grids.mTypes[t_index,k_index]*mPi_joint[:,e_index]        
#                     else:
#                         newborn_dist= (1/par.iNj)*(1/grids.vG.size)*mPi_joint[:,e_index]     
#                     ce_renter_newborns[wf_idx,t_index,k_index,e_index] = np.sum(newborn_dist* v_nonowner_wf_expanded_BAU[t_index,0,k_index, :, :,e_index])-wf_pol_newborns[t_index,k_index,e_index]

#     par.wf_wedge[0] = 0.0
    
#     tax_equiv_C = np.zeros((k_dim, grids.vE.size))
#     tax_equiv_NC = np.zeros((k_dim, grids.vE.size))
#     tax_equiv_renter = np.zeros((k_dim, grids.vE.size))
#     tax_equiv_newborns = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    
#     for k_index in range(k_dim):
#         for e_index in range(grids.vE.size):            
            
#             tax_equiv_C[k_index, e_index] = find_zero_linear(wf_loss, ce_C[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
#             tax_equiv_NC[k_index, e_index] = find_zero_linear(wf_loss, ce_NC[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
#             tax_equiv_renter[k_index, e_index] = find_zero_linear(wf_loss, ce_renter[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero
#             for t_index in range(t_pol, grids.vTime.size):
#                 tax_equiv_newborns[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns[:,t_index, k_index, e_index])
                    

#     return tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns


def find_expenditure_equiv_EK_policy(par, grids, vCoeff_C_in, vCoeff_NC_in,vCoeff_C_MP_in, vCoeff_NC_MP_in,vCoeff_C_BR_in, vCoeff_NC_BR_in,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, config_nopol, config_MP, config_BR):
    """
    New: import the value functions at the time of policy introduction as well as the distribution:
    I call it the 'business as usual' (BAU) value functions
        
    
    dist_c_2026: J x K x G x M x H x L x E (2026 distribution)
    dist_nc_2026: J x K x G x M x H x L x E 
    dist_renter_2026: J x K x G x M x E 
    
    Compute >2026 welfare for policy case.
    Within tax_wf, calculate wf for no-policy (BAU)
    Find tax_wf that eqaulizes
    """
    dP_C_lom=lom.LoM_path(par,grids,vCoeff_C_in, config_nopol)
    dP_NC_lom=lom.LoM_path(par,grids,vCoeff_NC_in, config_nopol)
    if config_BR.sceptics == False:
        k_dim=1
    else:
        k_dim=grids.vK.size       
        
    par.wf_wedge[0] = 0.0

    # get value functions over transition with experiment
    t_pol = int((par.experiment_year-par.starting_year)/par.time_increment)                                                                                                                                                                                                                                                        

    

    
    wf_loss = np.linspace(-0.1, 0.15,10)
    
    ce_C_BR  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_NC_BR = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_BR = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_newborns_BR = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
    ce_C_MP  = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_NC_MP = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_MP = np.zeros((wf_loss.size,k_dim, grids.vE.size))
    ce_renter_newborns_MP = np.zeros((wf_loss.size,grids.vTime.size, k_dim, grids.vE.size))
    
    wf_MP_pol_c  =np.zeros((k_dim,grids.vE.size))
    wf_MP_pol_nc =np.zeros((k_dim,grids.vE.size))
    wf_MP_pol_rent=np.zeros((k_dim,grids.vE.size))
    wf_MP_pol_newborns=np.zeros((grids.vTime.size,k_dim,grids.vE.size))    
    wf_BR_pol_c   =np.zeros((k_dim,grids.vE.size))
    wf_BR_pol_nc  =np.zeros((k_dim,grids.vE.size))
    wf_BR_pol_rent=np.zeros((k_dim,grids.vE.size))
    wf_BR_pol_newborns=np.zeros((grids.vTime.size,k_dim,grids.vE.size))   
    
    _, _, _, _, _, _, v_owner_c_wf_MP, v_owner_nc_wf_MP, v_nonowner_wf_MP = household_problem.solve(grids, par, vCoeff_C_MP_in,vCoeff_NC_MP_in, config_MP)
    v_owner_c_wf_expanded_MP=grid_adjust(par,grids,v_owner_c_wf_MP)
    del v_owner_c_wf_MP
    v_owner_nc_wf_expanded_MP=grid_adjust(par,grids,v_owner_nc_wf_MP)
    del v_owner_nc_wf_MP
    v_nonowner_wf_expanded_MP=grid_adjust_rentshape(par,grids,v_nonowner_wf_MP)
    del v_nonowner_wf_MP
   
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):
            wf_MP_pol_c[k_index,e_index] = np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_MP[t_pol,1:,k_index, :, :,:,:,e_index])
            wf_MP_pol_nc[k_index,e_index] = np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_MP[t_pol,1:,k_index, :, :,:,:,e_index])
            wf_MP_pol_rent[k_index,e_index] = np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_MP[t_pol,:,k_index, :, :,e_index])
    del v_owner_c_wf_expanded_MP, v_owner_nc_wf_expanded_MP
    
    for t_index in range(t_pol, grids.vTime.size-1):
        newborn_dist = sim.gen_initial_dist(par, grids, t_index, dP_C_lom[t_index], dP_NC_lom[t_index], vcoastal_beq[t_index], vnoncoastal_beq[t_index], vsavings_beq[t_index], config_nopol.sceptics)
        # weight
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                # NOTE HERE: Size wf_pol_newborns is grids.vTime.size. It starts fillling here from 14, until grids.vTime.size
                wf_MP_pol_newborns[t_index,k_index,e_index] = np.sum(newborn_dist[k_index,:,:,e_index] * v_nonowner_wf_expanded_MP[t_index,0,k_index, :, :,e_index])                
    del v_nonowner_wf_expanded_MP

    _, _, _, _, _, _, v_owner_c_wf_BR, v_owner_nc_wf_BR, v_nonowner_wf_BR = household_problem.solve(grids, par, vCoeff_C_BR_in,vCoeff_NC_BR_in, config_BR)
    v_owner_c_wf_expanded_BR=grid_adjust(par,grids,v_owner_c_wf_BR)
    del v_owner_c_wf_BR
    v_owner_nc_wf_expanded_BR=grid_adjust(par,grids,v_owner_nc_wf_BR)
    del v_owner_nc_wf_BR
    v_nonowner_wf_expanded_BR=grid_adjust_rentshape(par,grids,v_nonowner_wf_BR)
    del v_nonowner_wf_BR
        
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):
            wf_BR_pol_c[k_index,e_index] = np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_BR[t_pol,1:,k_index, :, :,:,:,e_index])
            wf_BR_pol_nc[k_index,e_index] = np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_BR[t_pol,1:,k_index, :, :,:,:,e_index])
            wf_BR_pol_rent[k_index,e_index] = np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_BR[t_pol,:,k_index, :, :,e_index])
    del v_owner_c_wf_expanded_BR, v_owner_nc_wf_expanded_BR
    
    for t_index in range(t_pol, grids.vTime.size-1):
        newborn_dist = sim.gen_initial_dist(par, grids, t_index, dP_C_lom[t_index], dP_NC_lom[t_index], vcoastal_beq[t_index], vnoncoastal_beq[t_index], vsavings_beq[t_index], config_nopol.sceptics)
        # weight
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                # NOTE HERE: Size wf_pol_newborns is grids.vTime.size. It starts fillling here from 14, until grids.vTime.size                
                wf_BR_pol_newborns[t_index,k_index,e_index] = np.sum(newborn_dist[k_index,:,:,e_index] * v_nonowner_wf_expanded_BR[t_index,0,k_index, :, :,e_index])
    del v_nonowner_wf_expanded_BR

    
    for wf_idx in range(wf_loss.size):
        par.wf_wedge[0] = wf_loss[wf_idx]
        _, _, _, _, _, _, v_owner_c_wf_BAU, v_owner_nc_wf_BAU, v_nonowner_wf_BAU = household_problem.solve(grids, par, vCoeff_C_in,vCoeff_NC_in, config_nopol)
        v_owner_c_wf_expanded_BAU=grid_adjust(par,grids,v_owner_c_wf_BAU)
        del v_owner_c_wf_BAU
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                ce_C_MP[wf_idx,k_index,e_index]=np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_MP_pol_c[k_index,e_index]
                ce_C_BR[wf_idx,k_index,e_index]=np.sum(mDist1_c_2026[1:,k_index, :, :,:,:,e_index]* v_owner_c_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_BR_pol_c[k_index,e_index]
        del v_owner_c_wf_expanded_BAU        
        
        v_owner_nc_wf_expanded_BAU=grid_adjust(par,grids,v_owner_nc_wf_BAU)
        del v_owner_nc_wf_BAU
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                ce_NC_MP[wf_idx,k_index,e_index]=np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_MP_pol_nc[k_index,e_index]
                ce_NC_BR[wf_idx,k_index,e_index]=np.sum(mDist1_nc_2026[1:,k_index, :, :,:,:,e_index]* v_owner_nc_wf_expanded_BAU[t_pol,1:,k_index, :, :,:,:,e_index])-wf_BR_pol_nc[k_index,e_index]        
        del v_owner_nc_wf_expanded_BAU
       
        v_nonowner_wf_expanded_BAU=grid_adjust_rentshape(par,grids,v_nonowner_wf_BAU)
        del v_nonowner_wf_BAU
        for k_index in range(k_dim):
            for e_index in range(grids.vE.size):
                ce_renter_MP[wf_idx,k_index,e_index]=np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_BAU[t_pol,:,k_index, :, :,e_index])-wf_MP_pol_rent[k_index,e_index]
                ce_renter_BR[wf_idx,k_index,e_index]=np.sum(mDist1_renter_2026[:,k_index, :, :,e_index]* v_nonowner_wf_expanded_BAU[t_pol,:,k_index, :, :,e_index])-wf_BR_pol_rent[k_index,e_index]
        
        for t_index in range(t_pol, grids.vTime.size-1):
            newborn_dist = sim.gen_initial_dist(par, grids, t_index, dP_C_lom[t_index], dP_NC_lom[t_index], vcoastal_beq[t_index], vnoncoastal_beq[t_index], vsavings_beq[t_index], config_nopol.sceptics)
            for k_index in range(k_dim):
                for e_index in range(grids.vE.size):
                    ce_renter_newborns_MP[wf_idx,t_index,k_index,e_index] = np.sum(newborn_dist[k_index,:,:,e_index]* v_nonowner_wf_expanded_BAU[t_index,0,k_index, :, :,e_index])-wf_MP_pol_newborns[t_index,k_index,e_index]
                    ce_renter_newborns_BR[wf_idx,t_index,k_index,e_index] = np.sum(newborn_dist[k_index,:,:,e_index]* v_nonowner_wf_expanded_BAU[t_index,0,k_index, :, :,e_index])-wf_BR_pol_newborns[t_index,k_index,e_index]
        del v_nonowner_wf_expanded_BAU
        
        
    par.wf_wedge[0] = 0.0
    
    tax_equiv_C_MP = np.zeros((k_dim, grids.vE.size))
    tax_equiv_NC_MP = np.zeros((k_dim, grids.vE.size))
    tax_equiv_renter_MP = np.zeros((k_dim, grids.vE.size))
    tax_equiv_newborns_MP = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    tax_equiv_C_BR = np.zeros((k_dim, grids.vE.size))
    tax_equiv_NC_BR = np.zeros((k_dim, grids.vE.size))
    tax_equiv_renter_BR = np.zeros((k_dim, grids.vE.size))
    tax_equiv_newborns_BR = np.zeros((grids.vTime.size, k_dim, grids.vE.size))
    
    for k_index in range(k_dim):
        for e_index in range(grids.vE.size):            
            
            tax_equiv_C_MP[k_index, e_index] = find_zero_linear(wf_loss, ce_C_MP[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
            tax_equiv_NC_MP[k_index, e_index] = find_zero_linear(wf_loss, ce_NC_MP[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
            tax_equiv_renter_MP[k_index, e_index] = find_zero_linear(wf_loss, ce_renter_MP[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero
            tax_equiv_C_BR[k_index, e_index] = find_zero_linear(wf_loss, ce_C_BR[:,k_index, e_index]) # find wf_loss that makes ce_C[:,k_index, g_index, e_index] equal to zero
            tax_equiv_NC_BR[k_index, e_index] = find_zero_linear(wf_loss, ce_NC_BR[:,k_index, e_index])  # find wf_loss that makes ce_NC[:,k_index, g_index, e_index] equal to zero
            tax_equiv_renter_BR[k_index, e_index] = find_zero_linear(wf_loss, ce_renter_BR[:,k_index, e_index])  # find wf_loss that makes ce_renter[:,k_index, g_index, e_index] equal to zero
            for t_index in range(t_pol, grids.vTime.size-1):
                tax_equiv_newborns_MP[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns_MP[:,t_index, k_index, e_index])
                tax_equiv_newborns_BR[t_index, k_index, e_index] = find_zero_linear(wf_loss, ce_renter_newborns_BR[:,t_index, k_index, e_index])
                    
    return tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP,tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR

@njit
def find_zero_linear(xgrid, ygrid):
    n = xgrid.size

    # exact hit
    for i in range(n):
        if ygrid[i] == 0.0:
            return xgrid[i]

    # sign change between adjacent points
    for i in range(n - 1):
        y0 = ygrid[i]
        y1 = ygrid[i + 1]

        if (y0 < 0.0 and y1 > 0.0) or (y0 > 0.0 and y1 < 0.0):
            x0 = xgrid[i]
            x1 = xgrid[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)

    # no crossing found
    print('no crossing found')
    return np.nan

@njit 
def grid_adjust_rentshape(par,grids,v_nonowner_in):
    T,J,K,G,X,E= v_nonowner_in.shape 
    v_rentshape_expanded=np.zeros((T,J,K,G, grids.vX_sim.size, E))
    for x_index_sim in range(grids.vX_sim.size):
        x=grids.vX_sim[x_index_sim]/(1+par.r) #Need to correct for inconsistency between VFI and simulation in how we record savings (excluding interest rate in VFI, including interest rate in simulation)
        x_index_l=misc.binary_search(0, grids.vX.size, grids.vX,x) 
        x_weight_left=compute_p_left(grids.vX, x, x_index_l)
        v_rentshape_expanded[:,:,:,:,x_index_sim,:]+=x_weight_left*v_nonowner_in[:,:,:,:,x_index_l,:]
        v_rentshape_expanded[:,:,:,:,x_index_sim,:]+=(1-x_weight_left)*v_nonowner_in[:,:,:,:,x_index_l+1,:]  
    return v_rentshape_expanded


@njit 
def grid_adjust(par,grids,v_owner_in):
    T,J,K,G,M,H,L,E= v_owner_in.shape
    v_stay_expanded=np.zeros((T,J,K,G, grids.vM_sim.size, H, grids.vL_sim.size, E))

    for m_index_sim in range(grids.vM_sim.size):
        m=grids.vM_sim[m_index_sim]/(1+par.r) #Need to correct for inconsistency between VFI and simulation in how we record savings (excluding interest rate in VFI, including interest rate in simulation)
        m_index_l=misc.binary_search(0, grids.vM.size, grids.vM,m) 
        m_weight_left=compute_p_left(grids.vM, m, m_index_l)
        for l_index_sim in range(grids.vL_sim.size):
            ltv=grids.vL_sim[l_index_sim]
            l_index_l=misc.binary_search(0, grids.vL.size, grids.vL,ltv) 
            l_weight_left=compute_p_left(grids.vL, ltv, l_index_l)
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*l_weight_left*v_owner_in[:,:,:,:,m_index_l,:,l_index_l,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*l_weight_left*v_owner_in[:,:,:,:,m_index_l+1,:,l_index_l,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=m_weight_left*(1-l_weight_left)*v_owner_in[:,:,:,:,m_index_l,:,l_index_l+1,:]
            v_stay_expanded[:,:,:,:,m_index_sim,:,l_index_sim,:]+=(1-m_weight_left)*(1-l_weight_left)*v_owner_in[:,:,:,:,m_index_l+1,:,l_index_l+1,:] 
    return v_stay_expanded

@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left

