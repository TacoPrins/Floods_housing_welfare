"""
simulation.py

Purpose:
    Simulate the economy for a given law of motion
"""
import numpy as np
from numba import njit
import interp as interp
import LoM_epsilons as lom
import misc_functions as misc
import buyer_problem_simulation as buy_sim
import numba as nb
import utility_epsilons as ut
import mortgage_choice_simulation as mortgage_sim
import simulate_initial_joint as initial_joint_sim
import tauchen as tauch


   
@njit
def stat_dist_finder(par, grids, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vCoeff_in_C,vCoeff_in_NC, config, bequest_guess=np.zeros((3))):

   if config.initial==True:
       t_index=0
   else:
       t_index=grids.vTime.size-1

   coastal_beq=bequest_guess[0]
   noncoastal_beq=bequest_guess[1]
   savings_beq=bequest_guess[2] 
   
   dP_C_lom=lom.LoM(par,grids,0, vCoeff_in_C)
   dP_NC_lom=lom.LoM(par,grids,0, vCoeff_in_NC)
   dP_C_lag=dP_C_lom
   dP_NC_lag=dP_NC_lom
   
   max_it=20
   vcoastal_beq=np.zeros((max_it))
   vnoncoastal_beq=np.zeros((max_it))
   vsavings_beq=np.zeros((max_it))   
   
   #Allow for immediate convergence if guess is correct
   vcoastal_beq[0]=coastal_beq
   vnoncoastal_beq[0]=noncoastal_beq
   vsavings_beq[0]=savings_beq
   
   if config.sceptics==False:
       k_dim=1 
   else:
       k_dim=2
       
   mDist0_c = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
   mDist0_nc = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
   mDist0_renter = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vX_sim.size, grids.vE.size))
   mDist1_c = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
   mDist1_nc = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
   mDist1_renter = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vX_sim.size, grids.vE.size))       
   
       
   for it_outer in range(1,max_it):
       mDist0_c[:,:,:,:,:,:,:] = 0
       mDist0_nc[:,:,:,:,:,:,:] = 0
       mDist0_renter[:,:,:,:,:] = 0
       mDist1_c[:,:,:,:,:,:,:] = 0
       mDist1_nc[:,:,:,:,:,:,:] = 0
       mDist1_renter[:,:,:,:,:] = 0
       rental_stock_C_out=0 
       rental_stock_NC_out=0
       for it in range(0,par.iNj):               
           if it>0:
               mDist0_c = mDist1_c
               mDist0_nc = mDist1_nc
               mDist0_renter = mDist1_renter
           stationary = True         
           mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, _,_= update_dist_continuous(grids, par, t_index, mDist0_c, mDist0_nc, mDist0_renter, dP_C_lom, dP_NC_lom, vt_stay_c, vt_stay_nc,  vt_renter, b_stay_c, b_stay_nc, b_renter,  coastal_beq, noncoastal_beq, savings_beq,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag, config, stationary, it)
           rental_stock_C_out+=stock_demand_rental_C
           rental_stock_NC_out+=stock_demand_rental_NC
       vcoastal_beq[it_outer]=coastal_beq
       vnoncoastal_beq[it_outer]=noncoastal_beq
       vsavings_beq[it_outer]=savings_beq

       if np.abs(vcoastal_beq[it_outer]-vcoastal_beq[it_outer-1])*par.iNj<0.01 and np.abs(vnoncoastal_beq[it_outer]-vnoncoastal_beq[it_outer-1])*par.iNj<0.01 and np.abs(vsavings_beq[it_outer]-vsavings_beq[it_outer-1])*par.iNj<0.01:
           #print("Steady state found in iteration:", it_outer)
           break
       elif it_outer==max_it-1:
           print("No steady state convergence")
  
 
   return mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, no_beq
  
@njit
def excess_demand_continuous(use_stock_clearing, grids, par, t_index, mDist0_c, mDist0_nc, mDist0_renter, dP_C, dP_NC,  vt_stay_c, vt_stay_nc, vt_renter, b_stay_c,  b_stay_nc, b_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag, config):
    
    #print("Price in:", dP_C, dP_NC)
    demand_C=0
    demand_NC=0
    stayer_demand_C=0
    stayer_demand_NC=0
    supply_NC = 0
    supply_C = 0
    stock_demand_rental_C = 0.0
    stock_demand_rental_NC = 0.0
    depreciation_C = 0
    depreciation_NC = 0
    default_stock_C=0
    default_stock_NC=0
    dPi_S=grids.vPi_S_median[t_index]   
    
    if config.building_rest == True:
        cost_shifter = par.building_rest_cost
    else:
        cost_shifter = 0
    
    if config.sceptics==False:
        k_dim=1 
    else:
        k_dim=2
          
    dP_C_growth_lom=lom.LoM(par,grids,t_index,vCoeff_in_C)-lom.LoM(par,grids,max(t_index-1,0), vCoeff_in_C)
    dP_NC_growth_lom=lom.LoM(par,grids,t_index, vCoeff_in_NC)-lom.LoM(par,grids,max(t_index-1,0), vCoeff_in_NC)
    
    dP_C_growth_prime_lom=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_C)-lom.LoM(par,grids,t_index,vCoeff_in_C)
    dP_NC_growth_prime_lom=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_NC)-lom.LoM(par,grids,t_index,vCoeff_in_NC)
    
    dP_C_lom=dP_C_lag+dP_C_growth_lom
    dP_NC_lom=dP_NC_lag+dP_NC_growth_lom
    
    dP_C_prime=dP_C+dP_C_growth_prime_lom
    dP_NC_prime=dP_NC+dP_NC_growth_prime_lom
    
    dP_C_prime_lom=dP_C_lom+dP_C_growth_prime_lom
    dP_NC_prime_lom=dP_NC_lom+dP_NC_growth_prime_lom
    
    
    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
    
    rental_price_C=par.dPsi+max(dP_C-(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime,0)
    rental_price_NC=par.dPsi+max(dP_NC-(1-par.dDelta)/(1+par.r)*dP_NC_prime,0)
    
    rental_price_lom_C=par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0)
    rental_price_lom_NC=par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0)
    
    mortgage_rate_nc=par.r_m_nc
    if config.mortgage_premium == False:
        mortgage_rate_c = par.r_m_c
    elif config.mortgage_premium == True:
        mortgage_rate_c = par.r_m_c_experiment 
    
    if config.building_rest == True:
        t_index_start=int((par.experiment_year-par.starting_year)/par.time_increment)
        arrival_rate_discount = (1-par.dDelta)**grids.vTime[t_index-t_index_start] + (1-(1-par.dDelta)**grids.vTime[t_index-t_index_start])*(1-par.dMitigation_rate)
        # print('arrival_rate_discount sim,', arrival_rate_discount)
    else:
        arrival_rate_discount = 1 
    
    
    minpay_matrix_C, ltv_minpay_index_left_C, minpay_matrix_NC, ltv_minpay_index_left_NC, max_ltv_C,max_ltv_NC, max_ltv_index_C, max_ltv_index_NC=mortgage_matrix_solve(par, grids, dP_C_lag, dP_NC_lag, dP_C, dP_NC, mortgage_rate_c)
    
    mDist_age_0_renter=gen_initial_dist(par, grids, t_index, dP_C, dP_NC, coastal_beq, noncoastal_beq, savings_beq, config.sceptics)
    mDist0_c[0,:,:,:,:,:,:]=0
    mDist0_nc[0,:,:,:,:,:,:]=0
    mDist0_renter[0,:,:,:,:]=mDist_age_0_renter
    
   

    mass=np.empty((grids.vM_sim.size))
    for j in range(par.iNj):     
        for e_index in range(grids.vE.size):
            # coastal homeowners        
            for k_index in range(k_dim):     
                for g_index in range(grids.vG.size):
                    h_share_lom, w_lom, h_share, w, rental_price_lom, rental_price, coastal_rent_share, g_renter_lom, g_renter=renter_solve(par, grids, g_index, rental_price_lom_C, rental_price_lom_NC, rental_price_C, rental_price_NC)
                    vt_stay_c_input, vt_stay_nc_input = vt_stay_c[j, k_index, g_index, :, :, :, e_index],vt_stay_nc[j, k_index, g_index,  :, :, :, e_index]
                    vt_renter_input, b_renter_input = vt_renter[j, k_index, g_index, :, e_index], b_renter[j, k_index, g_index,  :, e_index]
                    for e_trans_index in range(grids.vE_trans.size):
                        if j<par.j_ret:                        
                           income_mass=grids.mMarkov_trans[e_trans_index]
                        else:
                            income_mass=1
                            if e_trans_index>0:
                                continue                     
                        for h_index in range(grids.vH.size):
                            h = grids.vH[h_index]                        
                            for l_index_sim in range(grids.vL_sim.size):
                                ltv=grids.vL_sim[l_index_sim]
                                #Prices to calculate beginning of period mortgage
                                
                                mortgage_start=ltv*h*dP_C_lag                                 
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start, mortgage_rate_c) 

                                for damage_index in range(grids.vZ.size):
                                    dZ = grids.vZ[damage_index]                               
                                    if damage_index==0:
                                        prob_dZ=(1-dPi_S*arrival_rate_discount)
                                    else:                                    
                                        prob_dZ = dPi_S*arrival_rate_discount*grids.vPDF_z[damage_index] 
                                    for m_index_sim in range(grids.vM_sim.size):                                              
                                        mass[m_index_sim] = income_mass*prob_dZ*mDist0_c[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                    mass_pos_idx=np.where(mass>0)[0]                               
                                    #For stayers, interpolate value function at cash in hand 
                                    vt_stay_c_sim,_,_= mortgage_sim.solve(par,grids,vt_stay_c_input[:,h_index,:],j,h, e, dZ, dP_C,mortgage_start,max_ltv_index_C[j,h_index,e_index], minpay_matrix_C[j, h_index, l_index_sim], ltv_minpay_index_left_C[j, h_index, l_index_sim],mass_pos_idx, mortgage_rate_c) 
                                    x_sell_vec=grids.vM_sim+e+(1-(1-dZ)-par.dDelta-par.dKappa_sell)*h*dP_C-(1+mortgage_rate_c)*mortgage_start
                                    vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                    vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                    
                                    vt_renter_sim, h_renter = renter_sim_demand(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, x_sell_vec, mass_pos_idx)
                                    if (1+mortgage_rate_c)*mortgage_start>(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C:
                                        vt_default, h_default = renter_sim_demand(True, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                    else:
                                        vt_default=np.ones(grids.vM_sim.size)*-1e12
                                        h_default=np.zeros(grids.vM_sim.size)                                     
               
                                         
                                    mass_stay,mass_rent,mass_buyc,mass_buync,mass_default = continuous_decide(grids,vt_stay_c_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default,mass)
                                                      
                                    for m_index_sim in mass_pos_idx:
                                        depreciation_C += par.dDelta*mass[m_index_sim]*h
                                        stayer_demand_C += mass_stay[m_index_sim]*h
                                        if mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share<0 or mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share<0:
                                            assert mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share>0 and mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share>0
                                        
                                        stock_demand_rental_C += mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share + mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share
                                        stock_demand_rental_NC += mass_rent[m_index_sim]*h_renter[m_index_sim]*(1-coastal_rent_share) + mass_default[m_index_sim]*h_default[m_index_sim]*(1-coastal_rent_share)
                                        default_stock_C+=mass_default[m_index_sim]*h               
                                        demand_C += mass_buyc[m_index_sim]*h_pol_C[m_index_sim]                                  
                                        demand_NC += mass_buync[m_index_sim]*h_pol_NC[m_index_sim]
                                        supply_C += (mass[m_index_sim]-mass_stay[m_index_sim])*h-par.dDelta_default*mass_default[m_index_sim]*h  
                                 
                        # non coastal homeowners                             
                        for h_index in range(grids.vH.size):
                            h = grids.vH[h_index]                        
                            for l_index_sim in range(grids.vL_sim.size):
                                # ltv=grids.vL_sim[l_index_sim]
                                ltv=grids.vL_sim[l_index_sim]
                             
                                mortgage_start=ltv*h*dP_NC_lag                                    
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start, mortgage_rate_nc) 

                                for m_index_sim in range(grids.vM_sim.size):                                            
                                    mass[m_index_sim] = income_mass*mDist0_nc[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                mass_pos_idx=np.where(mass>0)[0]
                                vt_stay_nc_sim,_,_= mortgage_sim.solve(par,grids,vt_stay_nc_input[:,h_index,:],j,h, e,1, dP_NC,mortgage_start,max_ltv_index_NC[j,h_index,e_index], minpay_matrix_NC[j, h_index, l_index_sim], ltv_minpay_index_left_NC[j, h_index, l_index_sim],mass_pos_idx, mortgage_rate_nc)
                                x_sell_vec = grids.vM_sim+e+(1-par.dDelta-par.dKappa_sell)*h*dP_NC-(1+par.r_m_nc)*mortgage_start
                                vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                             
                                vt_renter_sim, h_renter = renter_sim_demand(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter,x_sell_vec, mass_pos_idx)
                                if (1+par.r_m_nc)*mortgage_start>(1-par.dDelta-par.dKappa_sell)*h*dP_NC:
                                    vt_default, h_default = renter_sim_demand(True, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                else:
                                    vt_default=np.ones(grids.vM_sim.size)*-1e12
                                    h_default=np.zeros(grids.vM_sim.size)
                                mass_stay,mass_rent,mass_buyc,mass_buync,mass_default = continuous_decide(grids,vt_stay_nc_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default,mass)
                                                          
                                for m_index_sim in mass_pos_idx:
                                    depreciation_NC += par.dDelta*mass[m_index_sim]*h
                                    stayer_demand_NC += mass_stay[m_index_sim]*h
                                    if mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share<0 or mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share<0:
                                        assert mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share>0 and mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share>0

                                    
                                    stock_demand_rental_C += mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share + mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share
                                    stock_demand_rental_NC += mass_rent[m_index_sim]*h_renter[m_index_sim]*(1-coastal_rent_share) + mass_default[m_index_sim]*h_default[m_index_sim]*(1-coastal_rent_share)
                                    default_stock_NC+=mass_default[m_index_sim]*h                               
                                    demand_C += mass_buyc[m_index_sim]*h_pol_C[m_index_sim]   
                                    demand_NC += mass_buync[m_index_sim]*h_pol_NC[m_index_sim]
                                    supply_NC += (mass[m_index_sim]-mass_stay[m_index_sim])*h-par.dDelta_default*mass_default[m_index_sim]*h

    
                # renters      
                        for x_index_sim in range(grids.vX_sim.size):                             
                            mass[x_index_sim] = income_mass*mDist0_renter[j,k_index,g_index,x_index_sim,e_index]
                        mass_pos_idx=np.where(mass>0)[0]                    
                        e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, 0, 0) 

                        vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                        vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                        vt_renter_sim, h_renter = renter_sim_demand(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vX_sim+e, mass_pos_idx)
                                
                        mass_rent,mass_buyc,mass_buync = continuous_decide_renter(grids,vt_buy_c, vt_buy_nc, vt_renter_sim, mass)
              
                                            
                        for x_index_sim in mass_pos_idx:
                            stock_demand_rental_C += mass_rent[x_index_sim]*h_renter[x_index_sim]*coastal_rent_share 
                            stock_demand_rental_NC += mass_rent[x_index_sim]*h_renter[x_index_sim]*(1-coastal_rent_share) 
                            demand_C += mass_buyc[x_index_sim]*h_pol_C[x_index_sim]
                            demand_NC += mass_buync[x_index_sim]*h_pol_NC[x_index_sim]

                            
    net_demand_C=demand_C-supply_C
    net_demand_NC=demand_NC-supply_NC
    

    investment_C = (par.dTheta*dP_C)**(par.dTheta/(1-par.dTheta))*par.dC_frac*(par.dL)*(1+cost_shifter)**(par.dTheta/(par.dTheta-1))        
    investment_NC = (par.dTheta*dP_NC)**(par.dTheta/(1-par.dTheta))*par.dNC_frac*par.dL
 
      
    stock_demand_NC=(stayer_demand_NC+demand_NC)*par.dDelta+stock_demand_rental_NC*par.dDelta_deprec_rental+default_stock_NC*par.dDelta_default 
    stock_demand_C=(stayer_demand_C+demand_C)*par.dDelta+stock_demand_rental_C*par.dDelta_deprec_rental+default_stock_C*par.dDelta_default 
    excess_demand_C_stock = stock_demand_C - investment_C
    excess_demand_NC_stock = stock_demand_NC - investment_NC     
    
    excess_demand_C_flow=net_demand_C+depreciation_C-investment_C-coastal_beq*(1-par.dDelta)+stock_demand_rental_C-rental_stock_C*(1-par.dDelta_deprec_rental)
    excess_demand_NC_flow=net_demand_NC+depreciation_NC-investment_NC-noncoastal_beq*(1-par.dDelta)+stock_demand_rental_NC-rental_stock_NC*(1-par.dDelta_deprec_rental)
    
    

   
    # return excess_demand_C, excess_demand_NC
    if use_stock_clearing:
        print("prices:",dP_C, dP_NC) 
        print("Rental prices:", rental_price_C, rental_price_NC)
        print("Excess demands:",excess_demand_C_stock, excess_demand_NC_stock)        
        print("Rental demands:",stock_demand_rental_C, stock_demand_rental_NC)
        return excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental_C, rental_stock_C, stock_demand_rental_NC, rental_stock_NC
    else:
        print("prices:",dP_C, dP_NC)   
        print("Rental prices:", rental_price_C, rental_price_NC)
        print("Excess demands:",excess_demand_C_flow, excess_demand_NC_flow)        
        print("Rental demands:",stock_demand_rental_C, stock_demand_rental_NC)
        return excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental_C, rental_stock_C, stock_demand_rental_NC, rental_stock_NC


@njit
def update_dist_continuous(grids, par, t_index, mDist0_c, mDist0_nc, mDist0_renter, dP_C, dP_NC, vt_stay_c, vt_stay_nc,  vt_renter, b_stay_c, b_stay_nc, b_renter,  coastal_beq, noncoastal_beq, savings_beq,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag, config, stationary=False, it=0):
    

    stock_demand_rental_C=0.0
    stock_demand_rental_NC=0.0
    dPi_S=grids.vPi_S_median[t_index]
    
    if config.sceptics==False:
        k_dim=1 
    else:
        k_dim=2
        
    default_mass_C=np.zeros((k_dim))
    default_mass_NC=np.zeros((k_dim))
    
    if stationary==True:
        mDist1_c = mDist0_c
        mDist1_nc = mDist0_nc
        mDist1_renter = mDist0_renter          
    else:
        if config.sceptics==True:
            k_dim=2
        else:
            k_dim=1
        mDist1_c = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size, grids.vE.size))
        mDist1_nc = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size, grids.vE.size))
        mDist1_renter = np.zeros((par.iNj, k_dim, grids.vG.size, grids.vX_sim.size, grids.vE.size))
           
    dP_C_growth_lom=lom.LoM(par,grids,t_index,vCoeff_in_C)-lom.LoM(par,grids,max(t_index-1,0), vCoeff_in_C)
    dP_NC_growth_lom=lom.LoM(par,grids,t_index, vCoeff_in_NC)-lom.LoM(par,grids,max(t_index-1,0), vCoeff_in_NC)
    
    dP_C_growth_prime_lom=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_C)-lom.LoM(par,grids,t_index,vCoeff_in_C)
    dP_NC_growth_prime_lom=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_NC)-lom.LoM(par,grids,t_index,vCoeff_in_NC)
    
    dP_C_lom=dP_C_lag+dP_C_growth_lom
    dP_NC_lom=dP_NC_lag+dP_NC_growth_lom
    
    dP_C_prime=dP_C+dP_C_growth_prime_lom
    dP_NC_prime=dP_NC+dP_NC_growth_prime_lom
    
    dP_C_prime_lom=dP_C_lom+dP_C_growth_prime_lom
    dP_NC_prime_lom=dP_NC_lom+dP_NC_growth_prime_lom
    
    

    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
    rental_price_C=par.dPsi+max(dP_C-(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime,0)
    rental_price_NC=par.dPsi+max(dP_NC-(1-par.dDelta)/(1+par.r)*dP_NC_prime,0)
    
    rental_price_lom_C=par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0)
    rental_price_lom_NC=par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0)   
  
        
    if stationary==False or it==0:
        mDist_age_0_renter=gen_initial_dist(par, grids, t_index, dP_C, dP_NC, coastal_beq, noncoastal_beq, savings_beq, config.sceptics)
        mDist0_c[0,:,:,:,:,:,:]=0
        mDist0_nc[0,:,:,:,:,:,:]=0
        mDist0_renter[0,:,:,:,:]=mDist_age_0_renter
        
        
    coastal_beq = 0
    noncoastal_beq = 0
    savings_beq = 0
    no_beq=0    
      
    mortgage_rate_nc=par.r_m_nc
    if config.mortgage_premium == False:
        mortgage_rate_c = par.r_m_c
    elif config.mortgage_premium == True:
        mortgage_rate_c = par.r_m_c_experiment 
        
    minpay_matrix_C, ltv_minpay_index_left_C, minpay_matrix_NC, ltv_minpay_index_left_NC, max_ltv_C,max_ltv_NC, max_ltv_index_C, max_ltv_index_NC=mortgage_matrix_solve(par, grids, dP_C_lag, dP_NC_lag, dP_C, dP_NC, mortgage_rate_c)

    if config.building_rest == True:
        t_index_start=int((par.experiment_year-par.starting_year)/par.time_increment)
        arrival_rate_discount = (1-par.dDelta)**grids.vTime[t_index-t_index_start] + (1-(1-par.dDelta)**grids.vTime[t_index-t_index_start])*(1-par.dMitigation_rate)
    else:
        arrival_rate_discount = 1     

    #Initialise since may not be filled in the function
    mass=np.empty((grids.vM_sim.size))  
    
    for j in range(par.iNj):
        if stationary==True and j!=it:
            continue        
        #Find the expenditure shares of the final good and housing for renters
     
        for e_index in range(grids.vE.size):
            for k_index in range(k_dim):
                if k_index==1 and stationary==False and grids.mTypes[t_index,1]>1e-8:
                    belief_switching_frac=-(grids.mTypes[t_index+1,1]-grids.mTypes[t_index,1])/grids.mTypes[t_index,1]
                    # print(belief_switching_frac)
                else:
                    belief_switching_frac=0.
                for g_index in range(grids.vG.size):
                    h_share_lom, w_lom, h_share, w, rental_price_lom, rental_price, coastal_rent_share, g_renter_lom, g_renter=renter_solve(par, grids, g_index, rental_price_lom_C, rental_price_lom_NC, rental_price_C, rental_price_NC)
                    b_stay_c_input, b_stay_nc_input = b_stay_c[j, k_index, g_index,  :, :, :, e_index], b_stay_nc[j, k_index, g_index,  :, :, :, e_index]
                    vt_stay_c_input, vt_stay_nc_input = vt_stay_c[j, k_index, g_index, :, :, :, e_index],vt_stay_nc[j, k_index, g_index,  :, :, :, e_index]
                    vt_renter_input, b_renter_input = vt_renter[j, k_index, g_index,  :, e_index], b_renter[j, k_index, g_index, :, e_index]
                    for e_trans_index in range(grids.vE_trans.size):
                        if j<par.j_ret:    
                            income_mass=grids.mMarkov_trans[e_trans_index]
                        else:
                            income_mass=1
                            if e_trans_index>0:
                                #mDist0_c[j,k_index,g_index,:,:,:,e_index]=0
                                continue
                        for h_index in range(grids.vH.size):
                            h = grids.vH[h_index]
                            for l_index_sim in range(grids.vL_sim.size):
                                ltv=grids.vL_sim[l_index_sim]
                                #Prices to calculate beginning of period mortgage
                                mortgage_start=ltv*h*dP_C_lag
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start, mortgage_rate_c) 

                                for damage_index in range(grids.vZ.size):
                                    dZ = grids.vZ[damage_index]                                    
                                    if damage_index==0:
                                        prob_dZ=(1-dPi_S*arrival_rate_discount)
                                    else:                                    
                                        prob_dZ = dPi_S*arrival_rate_discount*grids.vPDF_z[damage_index] 
                                    
                                    for m_index_sim in range(grids.vM_sim.size):                                              
                                        mass[m_index_sim] = income_mass*prob_dZ*mDist0_c[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                    mass_pos_idx=np.where(mass>0)[0]                               
                                    #For stayers, interpolate value function at cash in hand 
                                                                        
                                    vt_stay_c_sim,ltv_stay_c_sim,m_out= mortgage_sim.solve(par,grids,vt_stay_c_input[:,h_index,:],j,h, e, dZ,dP_C,mortgage_start,max_ltv_index_C[j,h_index,e_index], minpay_matrix_C[j, h_index, l_index_sim], ltv_minpay_index_left_C[j, h_index, l_index_sim],mass_pos_idx, mortgage_rate_c) 
                                                                      
                                    
                                    x_sell_vec=grids.vM_sim+e+(1-(1-dZ)-par.dDelta-par.dKappa_sell)*h*dP_C-(1+mortgage_rate_c)*mortgage_start

                                    vt_buy_c, _,h_pol_C_index,ltv_pol_C_max, ltv_pol_C_index =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                    vt_buy_nc, _, h_pol_NC_index,ltv_pol_NC_max, ltv_pol_NC_index =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                                   
                                    vt_renter_sim = renter_sim(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, x_sell_vec, mass_pos_idx)
                                    if (1+mortgage_rate_c)*mortgage_start>(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C:
                                        vt_default = renter_sim(True, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                    else:
                                        vt_default=np.ones(grids.vM_sim.size)*-1e12
                                    mass_stay,mass_rent,mass_buyc,mass_buync,mass_default = continuous_decide(grids,vt_stay_c_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default,mass)
                                    assert not (np.sum(mass_stay)+np.sum(mass_rent)+np.sum(mass_buyc)+np.sum(mass_buync)+np.sum(mass_default)-np.sum(mass))>1e-10 
                                    
                                    for m_index_sim in mass_pos_idx:    
                                        x_sell = x_sell_vec[m_index_sim]
                                        if mass_stay[m_index_sim]>0:
                                            b_stay_c_sim = misc.interp_2d(grids.vM, grids.vL, b_stay_c_input[:,h_index,:], m_out[m_index_sim], ltv_stay_c_sim[m_index_sim])
                                            if j<par.j_ret-1:
                                                mDist1_c, coastal_beq, savings_beq = simulate_stay(par, grids, mDist1_c, mass_stay[m_index_sim], h_index,e_index, k_index, g_index, j, b_stay_c_sim, ltv_stay_c_sim[m_index_sim], coastal_beq, savings_beq,belief_switching_frac)
                                            else:
                                                mDist1_c, coastal_beq, savings_beq = simulate_stay_ret(par, grids, mDist1_c, mass_stay[m_index_sim], h_index,e_index, k_index, g_index, j, b_stay_c_sim, ltv_stay_c_sim[m_index_sim], coastal_beq, savings_beq,belief_switching_frac)
                                      
                                        if mass_buyc[m_index_sim]>0:
                                            mDist1_c, coastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_c, dP_C, mass_buyc[m_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_C_index[m_index_sim], max_ltv_C[j,h_pol_C_index[m_index_sim],e_index], ltv_pol_C_max[m_index_sim], ltv_pol_C_index[m_index_sim], b_stay_c_input, x_sell, coastal_beq, savings_beq,belief_switching_frac)

                                        if mass_buync[m_index_sim]>0:
                                            mDist1_nc, noncoastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_nc, dP_NC, mass_buync[m_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_NC_index[m_index_sim], max_ltv_NC[j,h_pol_NC_index[m_index_sim],e_index], ltv_pol_NC_max[m_index_sim], ltv_pol_NC_index[m_index_sim], b_stay_nc_input, x_sell, noncoastal_beq, savings_beq,belief_switching_frac)

                                        if mass_rent[m_index_sim]>0:
                                            mDist1_renter, h_renter_sim, savings_beq, no_beq= simulate_rent_outer(par, grids, mDist1_renter, mass_rent[m_index_sim], j,  k_index, g_index, e_index, savings_beq, b_renter_input, x_sell, h_share, rental_price, no_beq,belief_switching_frac)                   
                                            stock_demand_rental_C+=mass_rent[m_index_sim]*h_renter_sim*coastal_rent_share
                                            stock_demand_rental_NC+=mass_rent[m_index_sim]*h_renter_sim*(1-coastal_rent_share)                                        

                                        if mass_default[m_index_sim]>0:
                                            m = grids.vM_sim[m_index_sim]+e-mortgage_rebate
                                            mDist1_renter, h_renter_sim, savings_beq, no_beq= simulate_rent_outer(par, grids, mDist1_renter, mass_default[m_index_sim], j,  k_index, g_index, e_index, savings_beq, b_renter_input, m, h_share, rental_price, no_beq,belief_switching_frac)                   
                                            stock_demand_rental_C+=mass_default[m_index_sim]*h_renter_sim*coastal_rent_share
                                            stock_demand_rental_NC+=mass_default[m_index_sim]*h_renter_sim*(1-coastal_rent_share)                                            
                                            default_mass_C[k_index]+=mass_default[m_index_sim]
                                          
                                       

                    # noncoastal homeowners

                                mortgage_start=ltv*h*dP_NC_lag
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start, mortgage_rate_nc) 

                                for m_index_sim in range(grids.vM_sim.size):                                            
                                    mass[m_index_sim] = income_mass*mDist0_nc[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                mass_pos_idx=np.where(mass>0)[0]
                                vt_stay_nc_sim,ltv_stay_nc_sim,m_out= mortgage_sim.solve(par,grids,vt_stay_nc_input[:,h_index,:],j,h, e, 1, dP_NC,mortgage_start,max_ltv_index_NC[j,h_index,e_index], minpay_matrix_NC[j, h_index, l_index_sim], ltv_minpay_index_left_NC[j, h_index, l_index_sim],mass_pos_idx, mortgage_rate_nc)
                                                               
                                
                                x_sell_vec = grids.vM_sim+e+(1-par.dDelta-par.dKappa_sell)*h*dP_NC-(1+par.r_m_nc)*mortgage_start
                                vt_buy_c, _, h_pol_C_index,ltv_pol_C_max, ltv_pol_C_index=buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                vt_buy_nc, _, h_pol_NC_index,ltv_pol_NC_max,ltv_pol_NC_index =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                           
                                vt_renter_sim = renter_sim(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter,x_sell_vec, mass_pos_idx)
                                if (1+par.r_m_nc)*mortgage_start>(1-par.dDelta-par.dKappa_sell)*h*dP_NC:
                                    vt_default = renter_sim(True, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                else:
                                    vt_default=np.ones(grids.vM_sim.size)*-1e12

                                mass_stay,mass_rent,mass_buyc,mass_buync, mass_default = continuous_decide(grids,vt_stay_nc_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default, mass)
                                assert not (np.sum(mass_stay)+np.sum(mass_rent)+np.sum(mass_buyc)+np.sum(mass_buync)+np.sum(mass_default)-np.sum(mass))>1e-10 
                            
                                for m_index_sim in mass_pos_idx:
                                    x_sell = x_sell_vec[m_index_sim]
                                    if mass_stay[m_index_sim]>0:
                                        b_stay_nc_sim = misc.interp_2d(grids.vM, grids.vL, b_stay_nc_input[:,h_index,:], m_out[m_index_sim], ltv_stay_nc_sim[m_index_sim]) 
                                        if j<par.j_ret-1:
                                            mDist1_nc, noncoastal_beq, savings_beq = simulate_stay(par, grids, mDist1_nc, mass_stay[m_index_sim], h_index,e_index, k_index, g_index, j, b_stay_nc_sim, ltv_stay_nc_sim[m_index_sim], noncoastal_beq, savings_beq,belief_switching_frac)
                                        else:
                                            mDist1_nc, noncoastal_beq, savings_beq = simulate_stay_ret(par, grids, mDist1_nc, mass_stay[m_index_sim], h_index,e_index, k_index, g_index, j, b_stay_nc_sim, ltv_stay_nc_sim[m_index_sim], noncoastal_beq, savings_beq,belief_switching_frac)
   
                                    if mass_buyc[m_index_sim]>0:
                                        mDist1_c, coastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_c, dP_C, mass_buyc[m_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_C_index[m_index_sim], max_ltv_C[j,h_pol_C_index[m_index_sim],e_index], ltv_pol_C_max[m_index_sim], ltv_pol_C_index[m_index_sim], b_stay_c_input, x_sell, coastal_beq, savings_beq,belief_switching_frac)

                                    if mass_buync[m_index_sim]>0:
                                        mDist1_nc, noncoastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_nc, dP_NC, mass_buync[m_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_NC_index[m_index_sim], max_ltv_NC[j,h_pol_NC_index[m_index_sim],e_index], ltv_pol_NC_max[m_index_sim], ltv_pol_NC_index[m_index_sim], b_stay_nc_input, x_sell, noncoastal_beq, savings_beq,belief_switching_frac)

                                    if mass_rent[m_index_sim]>0:
                                        mDist1_renter, h_renter_sim, savings_beq, no_beq= simulate_rent_outer(par, grids, mDist1_renter, mass_rent[m_index_sim], j,  k_index, g_index, e_index, savings_beq, b_renter_input, x_sell, h_share, rental_price, no_beq,belief_switching_frac)                   
                                        stock_demand_rental_C+=mass_rent[m_index_sim]*h_renter_sim*coastal_rent_share
                                        stock_demand_rental_NC+=mass_rent[m_index_sim]*h_renter_sim*(1-coastal_rent_share)

                                    if mass_default[m_index_sim]>0:
                                        m = grids.vM_sim[m_index_sim]+e-mortgage_rebate
                                        mDist1_renter, h_renter_sim, savings_beq, no_beq= simulate_rent_outer(par, grids, mDist1_renter, mass_default[m_index_sim], j,  k_index, g_index, e_index, savings_beq, b_renter_input, m, h_share, rental_price, no_beq,belief_switching_frac)                   
                                        stock_demand_rental_C+=mass_default[m_index_sim]*h_renter_sim*coastal_rent_share
                                        stock_demand_rental_NC+=mass_default[m_index_sim]*h_renter_sim*(1-coastal_rent_share)                                        
                                        default_mass_NC[k_index]+=mass_default[m_index_sim]

                    
               # renters      
                        e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, 0, 0) 

                        for x_index_sim in range(grids.vX_sim.size):                             
                            mass[x_index_sim] = income_mass*mDist0_renter[j,k_index,g_index,x_index_sim,e_index]
                        mass_pos_idx=np.where(mass>0)[0]     
                        vt_buy_c, _, h_pol_C_index,ltv_pol_C_max, ltv_pol_C_index =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                        vt_buy_nc, _, h_pol_NC_index,ltv_pol_NC_max,ltv_pol_NC_index =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                
                        
                        vt_renter_sim = renter_sim(False, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vX_sim+e, mass_pos_idx)
                        mass_rent,mass_buyc,mass_buync = continuous_decide_renter(grids,vt_buy_c, vt_buy_nc, vt_renter_sim,mass)        
                        assert not (np.sum(mass_rent)+np.sum(mass_buyc)+np.sum(mass_buync)-np.sum(mass))>1e-10 
                   
                        
                        for x_index_sim in mass_pos_idx:                   
                            x = grids.vX_sim[x_index_sim]+e
                            if mass_buyc[x_index_sim]>0:
                                mDist1_c, coastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_c, dP_C, mass_buyc[x_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_C_index[x_index_sim], max_ltv_C[j,h_pol_C_index[x_index_sim],e_index], ltv_pol_C_max[x_index_sim], ltv_pol_C_index[x_index_sim], b_stay_c_input, x, coastal_beq, savings_beq,belief_switching_frac)

                            if mass_buync[x_index_sim]>0:
                                mDist1_nc, noncoastal_beq, savings_beq = simulate_buy_outer(par, grids, mDist1_nc, dP_NC, mass_buync[x_index_sim],  j, k_index, g_index, h_index,e_index, h_pol_NC_index[x_index_sim], max_ltv_NC[j,h_pol_NC_index[x_index_sim],e_index], ltv_pol_NC_max[x_index_sim], ltv_pol_NC_index[x_index_sim], b_stay_nc_input, x, noncoastal_beq, savings_beq,belief_switching_frac)

                            if mass_rent[x_index_sim]>0:
                                mDist1_renter, h_renter_sim, savings_beq, no_beq= simulate_rent_outer(par, grids, mDist1_renter, mass_rent[x_index_sim], j,  k_index, g_index, e_index, savings_beq, b_renter_input, x, h_share, rental_price, no_beq,belief_switching_frac)                   
                                stock_demand_rental_C+=mass_rent[x_index_sim]*h_renter_sim*coastal_rent_share
                                stock_demand_rental_NC+=mass_rent[x_index_sim]*h_renter_sim*(1-coastal_rent_share)

  
    
    return mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, default_mass_C, default_mass_NC


#############################################################################
### Simulate functions
#############################################################################

@njit 
def simulate_buy_ret(par, grids, mDist1, d_mass_buy, h_index,e_index, k_index, g_index, j, h_pol_index, b_buy, max_ltv, ltv_pol_max, ltv_pol_index, housing_beq, savings_beq,belief_switching_frac):
    h1 = h_pol_index
    b = b_buy
    m1 = construct_m1(grids, par, j, b)
    if grids.vM_sim[0] <= m1 < grids.vM_sim[-1]:
        m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,m1)
        p_left = compute_p_left(grids.vM_sim, m1, m1_index)
    elif m1 >= grids.vM_sim[-1]: 
        m1_index = grids.vM_sim.size-2
        p_left = 0
        
    if ltv_pol_max:
        p_left_ltv = compute_p_left(grids.vL_sim, max_ltv, ltv_pol_index)
        assert 0<=p_left_ltv<=1
    else:
        p_left_ltv = 1    
    assert max_ltv<=par.max_ltv 
        
    if j < par.iNj-1:
        mDist1[j+1,k_index,g_index,m1_index,h1,ltv_pol_index,e_index] += (1-belief_switching_frac)*d_mass_buy * p_left* p_left_ltv      
        mDist1[j+1,k_index,g_index,m1_index+1,h1,ltv_pol_index,e_index] += (1-belief_switching_frac)*d_mass_buy *  (1-p_left)* p_left_ltv          
        if ltv_pol_max:
            mDist1[j+1,k_index,g_index,m1_index,h1,ltv_pol_index+1,e_index] +=  (1-belief_switching_frac)*d_mass_buy *  p_left * (1-p_left_ltv)            
            mDist1[j+1,k_index,g_index,m1_index+1,h1,ltv_pol_index+1,e_index] += (1-belief_switching_frac)*d_mass_buy *  (1-p_left) * (1-p_left_ltv)
        if k_index==1 and belief_switching_frac>0:
            mDist1[j+1,k_index-1,g_index,m1_index,h1,ltv_pol_index,e_index] += belief_switching_frac*d_mass_buy * p_left* p_left_ltv      
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,ltv_pol_index,e_index] += belief_switching_frac*d_mass_buy *  (1-p_left)* p_left_ltv          
            if ltv_pol_max:
                mDist1[j+1,k_index-1,g_index,m1_index,h1,ltv_pol_index+1,e_index] +=  belief_switching_frac*d_mass_buy *  p_left * (1-p_left_ltv)            
                mDist1[j+1,k_index-1,g_index,m1_index+1,h1,ltv_pol_index+1,e_index] += belief_switching_frac*d_mass_buy *  (1-p_left) * (1-p_left_ltv)
            
    else:
        housing_beq += d_mass_buy *grids.vH[h1]
        savings_beq += d_mass_buy*b
        
    assert 0<=p_left<=1

    return mDist1, housing_beq, savings_beq

@njit 
def simulate_buy(par, grids, mDist1, d_mass_buy, h_index,e_index, k_index, g_index, j, h_pol_index, b_buy, max_ltv, ltv_pol_max, ltv_pol_index, housing_beq, savings_beq,belief_switching_frac):
    h1 = h_pol_index
    b = b_buy
    m1 = construct_m1(grids, par, j, b)
    if grids.vM_sim[0] <= m1 < grids.vM_sim[-1]:
        m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,m1)
        p_left = compute_p_left(grids.vM_sim, m1, m1_index)
    elif m1 >= grids.vM_sim[-1]: 
        m1_index = grids.vM_sim.size-2
        p_left = 0
        
    if ltv_pol_max:
        p_left_ltv = compute_p_left(grids.vL_sim, max_ltv, ltv_pol_index)
        assert 0<=p_left_ltv<=1
    else:
        p_left_ltv = 1                  
    

    for e1 in range(grids.vE.size):   
        p_e = grids.mMarkov[e_index,e1]
        mDist1[j+1,k_index,g_index,m1_index,h1,ltv_pol_index,e1] += (1-belief_switching_frac)*d_mass_buy * p_e * p_left* p_left_ltv      
        mDist1[j+1,k_index,g_index,m1_index+1,h1,ltv_pol_index,e1] += (1-belief_switching_frac)*d_mass_buy * p_e * (1-p_left)* p_left_ltv     
        if ltv_pol_max:
            mDist1[j+1,k_index,g_index,m1_index,h1,ltv_pol_index+1,e1] +=  (1-belief_switching_frac)*d_mass_buy * p_e * p_left * (1-p_left_ltv)            
            mDist1[j+1,k_index,g_index,m1_index+1,h1,ltv_pol_index+1,e1] += (1-belief_switching_frac)*d_mass_buy * p_e * (1-p_left) * (1-p_left_ltv)
        if k_index==1 and belief_switching_frac>0:
            mDist1[j+1,k_index-1,g_index,m1_index,h1,ltv_pol_index,e1] += belief_switching_frac*d_mass_buy * p_e * p_left* p_left_ltv      
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,ltv_pol_index,e1] += belief_switching_frac*d_mass_buy * p_e * (1-p_left)* p_left_ltv     
            if ltv_pol_max:
                mDist1[j+1,k_index-1,g_index,m1_index,h1,ltv_pol_index+1,e1] +=  belief_switching_frac*d_mass_buy * p_e * p_left * (1-p_left_ltv)            
                mDist1[j+1,k_index-1,g_index,m1_index+1,h1,ltv_pol_index+1,e1] += belief_switching_frac*d_mass_buy * p_e * (1-p_left) * (1-p_left_ltv)

        
    assert 0<=p_left<=1

    return mDist1, housing_beq, savings_beq

@njit
def simulate_rent(par, grids, mDist1_renter, d_mass_rent, e_index, k_index, g_index, j, B_pol_rent, savings_beq,belief_switching_frac):
    b = B_pol_rent
    x1 = (1+par.r)*b 
    if grids.vX_sim[0] <= x1 < grids.vX_sim[-1]:
        x1_index = misc.binary_search(0, grids.vX_sim.size, grids.vX_sim,x1)
        p_left = compute_p_left(grids.vX_sim, x1, x1_index)
    elif x1 >= grids.vX_sim[-1]: 
        x1_index = grids.vX_sim.size-2
        p_left = 0

    assert 0<=p_left<=1

    for e1 in range(grids.vE.size):   
        p_e = grids.mMarkov[e_index,e1]        
        mDist1_renter[j+1,k_index,g_index,x1_index,e1] +=  (1-belief_switching_frac)*d_mass_rent * p_e * p_left         
        mDist1_renter[j+1,k_index,g_index,x1_index+1,e1] +=  (1-belief_switching_frac)*d_mass_rent * p_e * (1-p_left)     
        if k_index==1 and belief_switching_frac>0:
            mDist1_renter[j+1,k_index-1,g_index,x1_index,e1] +=  belief_switching_frac*d_mass_rent * p_e * p_left         
            mDist1_renter[j+1,k_index-1,g_index,x1_index+1,e1] +=  belief_switching_frac*d_mass_rent * p_e * (1-p_left)   
            
        

    return mDist1_renter, savings_beq

@njit
def simulate_rent_ret(par, grids, mDist1_renter, d_mass_rent, e_index, k_index, g_index, j, B_pol_rent, savings_beq, no_beq,belief_switching_frac):
    b = B_pol_rent
    x1 = (1+par.r)*b 
    if grids.vX_sim[0] <= x1 < grids.vX_sim[-1]:
        x1_index = misc.binary_search(0, grids.vX_sim.size, grids.vX_sim,x1)
        p_left = compute_p_left(grids.vX_sim, x1, x1_index)
    elif x1 >= grids.vX_sim[-1]: 
        x1_index = grids.vX_sim.size-2
        p_left = 0
   
    assert 0<=p_left<=1
    if j<par.iNj-1:
        mDist1_renter[j+1,k_index,g_index,x1_index,e_index] +=  (1-belief_switching_frac)*d_mass_rent * p_left         
        mDist1_renter[j+1,k_index,g_index,x1_index+1,e_index] +=  (1-belief_switching_frac)*d_mass_rent * (1-p_left)  
        if k_index==1 and belief_switching_frac>0:
            mDist1_renter[j+1,k_index-1,g_index,x1_index,e_index] +=  belief_switching_frac*d_mass_rent * p_left         
            mDist1_renter[j+1,k_index-1,g_index,x1_index+1,e_index] +=  belief_switching_frac*d_mass_rent * (1-p_left)  
    else:
        savings_beq += d_mass_rent*b
        if b<1e-4:
            no_beq += d_mass_rent*par.iNj
    return mDist1_renter, savings_beq, no_beq
    
@njit
def simulate_stay(par, grids, mDist1, d_mass_stay, h_index,e_index, k_index, g_index, j, b_stay_sim, ltv_stay_sim, housing_beq, savings_beq,belief_switching_frac):
 
    h1 = h_index
    b = b_stay_sim
    
    if grids.vL_sim[0] <= ltv_stay_sim < grids.vL_sim[-1]:
        l_index = misc.binary_search(0, grids.vL_sim.size, grids.vL_sim,ltv_stay_sim)
        p_left_ltv = compute_p_left(grids.vL_sim, ltv_stay_sim, l_index)
    assert ltv_stay_sim <= grids.vL_sim[-1]
    
    assert 0<=p_left_ltv<=1 
    m1 = construct_m1(grids, par, j, b)
    if grids.vM_sim[0] <= m1 < grids.vM_sim[-1]:
        m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,m1)
        p_left_m = compute_p_left(grids.vM_sim, m1, m1_index)
    elif m1 >= grids.vM_sim[-1]: 
        m1_index = grids.vM_sim.size-2
        p_left_m = 0 
    
    assert m1>=grids.vM_sim[0]
    
    assert 0<=p_left_m<=1

    for e1 in range(grids.vE.size):
        p_e = grids.mMarkov[e_index,e1]  
        mDist1[j+1,k_index,g_index,m1_index,h1,l_index, e1] += (1-belief_switching_frac)*d_mass_stay * p_e * p_left_m *p_left_ltv           
        mDist1[j+1,k_index,g_index,m1_index+1,h1,l_index, e1] += (1-belief_switching_frac)*d_mass_stay * p_e * (1-p_left_m) *p_left_ltv       
        mDist1[j+1,k_index,g_index,m1_index,h1,l_index+1, e1] += (1-belief_switching_frac)*d_mass_stay * p_e * p_left_m *(1-p_left_ltv)           
        mDist1[j+1,k_index,g_index,m1_index+1,h1,l_index+1, e1] += (1-belief_switching_frac)*d_mass_stay * p_e * (1-p_left_m) *(1-p_left_ltv)
        if k_index==1 and belief_switching_frac>0:
            mDist1[j+1,k_index-1,g_index,m1_index,h1,l_index, e1] += belief_switching_frac*d_mass_stay * p_e * p_left_m *p_left_ltv           
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,l_index, e1] += belief_switching_frac*d_mass_stay * p_e * (1-p_left_m) *p_left_ltv       
            mDist1[j+1,k_index-1,g_index,m1_index,h1,l_index+1, e1] += belief_switching_frac*d_mass_stay * p_e * p_left_m *(1-p_left_ltv)           
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,l_index+1, e1] += belief_switching_frac*d_mass_stay * p_e * (1-p_left_m) *(1-p_left_ltv)
    
        
    return mDist1, housing_beq, savings_beq

@njit
def simulate_stay_ret(par, grids, mDist1, d_mass_stay, h_index,e_index, k_index, g_index, j, b_stay_sim, ltv_stay_sim, housing_beq, savings_beq,belief_switching_frac):
 
    h1 = h_index
    b = b_stay_sim
    
    if grids.vL_sim[0] <= ltv_stay_sim < grids.vL_sim[-1]:
        l_index = misc.binary_search(0, grids.vL_sim.size, grids.vL_sim,ltv_stay_sim)
        p_left_ltv = compute_p_left(grids.vL_sim, ltv_stay_sim, l_index)
    elif ltv_stay_sim >= grids.vL_sim[-1]: 
        l_index = grids.vL_sim.size-2
        p_left_ltv = 0
    assert 0<=p_left_ltv<=1 
    m1 = construct_m1(grids, par, j, b)
    if grids.vM_sim[0] <= m1 < grids.vM_sim[-1]:
        m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,m1)
        p_left_m = compute_p_left(grids.vM_sim, m1, m1_index)
    elif m1 >= grids.vM_sim[-1]: 
        m1_index = grids.vM_sim.size-2
        p_left_m = 0 
    assert 0<=p_left_m<=1
    if j<par.iNj-1:
        mDist1[j+1,k_index,g_index,m1_index,h1,l_index, e_index] += (1-belief_switching_frac)*d_mass_stay * p_left_m *p_left_ltv           
        mDist1[j+1,k_index,g_index,m1_index+1,h1,l_index, e_index] += (1-belief_switching_frac)*d_mass_stay *  (1-p_left_m) *p_left_ltv       
        mDist1[j+1,k_index,g_index,m1_index,h1,l_index+1, e_index] += (1-belief_switching_frac)*d_mass_stay *  p_left_m *(1-p_left_ltv)           
        mDist1[j+1,k_index,g_index,m1_index+1,h1,l_index+1, e_index] += (1-belief_switching_frac)*d_mass_stay * (1-p_left_m) *(1-p_left_ltv) 
        if k_index==1 and belief_switching_frac>0:
            mDist1[j+1,k_index-1,g_index,m1_index,h1,l_index, e_index] += belief_switching_frac*d_mass_stay * p_left_m *p_left_ltv           
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,l_index, e_index] += belief_switching_frac*d_mass_stay *  (1-p_left_m) *p_left_ltv       
            mDist1[j+1,k_index-1,g_index,m1_index,h1,l_index+1, e_index] += belief_switching_frac*d_mass_stay *  p_left_m *(1-p_left_ltv)           
            mDist1[j+1,k_index-1,g_index,m1_index+1,h1,l_index+1, e_index] += belief_switching_frac*d_mass_stay * (1-p_left_m) *(1-p_left_ltv) 
    else:
        housing_beq += d_mass_stay *grids.vH[h1]
        savings_beq += d_mass_stay*b
            
        
    return mDist1, housing_beq, savings_beq


@njit
def simulate_rent_outer(par, grids, mDist1_renter, d_mass_rent, j,  k_index, g_index, e_index, savings_beq, b_renter_input, x_renter, h_share, rental_price, no_beq,belief_switching_frac):
    B_pol=interp.interp_1d(grids.vX,b_renter_input, x_renter)
    expenditures=x_renter-B_pol
    h_renter_sim=max(min((h_share/rental_price)*expenditures,grids.vH_renter[-1]),grids.vH_renter[0])
    if j<par.j_ret-1:
        mDist1_renter, savings_beq = simulate_rent(par, grids, mDist1_renter, d_mass_rent, e_index, k_index, g_index, j, B_pol, savings_beq,belief_switching_frac)
    else:
        mDist1_renter, savings_beq, no_beq = simulate_rent_ret(par, grids, mDist1_renter, d_mass_rent, e_index, k_index, g_index, j, B_pol, savings_beq, no_beq,belief_switching_frac)
    return mDist1_renter, h_renter_sim, savings_beq, no_beq  

@njit 
def simulate_buy_outer(par, grids, mDist1, dP, d_mass_buy,  j, k_index, g_index, h_index,e_index, h_pol_index, max_ltv, ltv_pol_max, ltv_pol_index, b_stay_input, x_buy, housing_beq, savings_beq,belief_switching_frac):
    if ltv_pol_max:
        ltv_buy=max_ltv
    else:
        ltv_buy=grids.vL_sim[ltv_pol_index]
    if ltv_pol_index>0:
        origination_cost=par.dZeta_fixed
    else:
        origination_cost=0.                                        
    m_buy = x_buy - (dP*(1+par.dKappa_buy-ltv_buy*(1-par.dZeta)))*grids.vH[h_pol_index]-origination_cost
    b_buy = misc.interp_2d(grids.vM, grids.vL, b_stay_input[:,h_pol_index,:], m_buy, ltv_buy)  
    if j<par.j_ret-1:
       mDist1, housing_beq, savings_beq = simulate_buy(par, grids, mDist1, d_mass_buy, h_index,e_index, k_index, g_index, j, h_pol_index, b_buy, max_ltv, ltv_pol_max, ltv_pol_index, housing_beq, savings_beq,belief_switching_frac)
    else:
       mDist1, housing_beq, savings_beq = simulate_buy_ret(par, grids, mDist1, d_mass_buy, h_index,e_index, k_index, g_index, j, h_pol_index, b_buy, max_ltv, ltv_pol_max, ltv_pol_index, housing_beq, savings_beq,belief_switching_frac)
    return mDist1, housing_beq, savings_beq
        
#############################################################################
### Boolean functions
#############################################################################
@njit
def continuous_decide(grids,vt_stay_sim, vt_buy_c, vt_buy_nc, vt_renter_sim, vt_default, mass):
    

    mass_stay=np.empty((grids.vM_sim.size))
    mass_rent=np.empty((grids.vM_sim.size))
    mass_buyc=np.empty((grids.vM_sim.size))
    mass_buync=np.empty((grids.vM_sim.size))
    mass_default=np.empty((grids.vM_sim.size))
    
    output_mass=np.zeros((grids.vM_sim.size,5))
    
    mass[0]=2*mass[0]
    mass[-1]=2*mass[-1]    
 
    value_vector=np.zeros((2))
    value_plus_vector=np.zeros((2))
    cross_point_vector=np.zeros((grids.vM_sim.size-1))
    choice_vector=np.zeros((grids.vM_sim.size),dtype=np.int32)
    
    
    if mass[0]>0:
        if (vt_stay_sim[0] > vt_renter_sim[0]) and (vt_stay_sim[0] > vt_buy_c[0]) and (vt_stay_sim[0] > vt_buy_nc[0]) and (vt_stay_sim[0] > vt_default[0]):
            value_vector[0]=vt_stay_sim[0]
            value_plus_vector[0]=vt_stay_sim[1]     
            choice_vector[0]=0          
        elif (vt_renter_sim[0] > vt_buy_c[0]) and (vt_renter_sim[0] > vt_buy_nc[0]) and (vt_renter_sim[0] > vt_default[0]):
            value_vector[0]=vt_renter_sim[0]
            value_plus_vector[0]=vt_renter_sim[1]   
            choice_vector[0]=1 
        elif (vt_buy_c[0] > vt_buy_nc[0]) and (vt_buy_c[0] > vt_default[0]):
            value_vector[0]=vt_buy_c[0]
            value_plus_vector[0]=vt_buy_c[1]
            choice_vector[0]=2 
        elif (vt_buy_nc[0] > vt_default[0]): 
            value_vector[0]=vt_buy_nc[0]
            value_plus_vector[0]=vt_buy_nc[1]   
            choice_vector[0]=3 
        else:
            value_vector[0]=vt_default[0]
            value_plus_vector[0]=vt_default[1]
            choice_vector[0]=4      
       
    for m_index_sim in range(1,grids.vM_sim.size-1):
        if mass[m_index_sim]>0:
            if (vt_stay_sim[m_index_sim] > vt_renter_sim[m_index_sim]) and (vt_stay_sim[m_index_sim] > vt_buy_c[m_index_sim]) and (vt_stay_sim[m_index_sim] > vt_buy_nc[m_index_sim]) and (vt_stay_sim[m_index_sim] > vt_default[m_index_sim]):
                value_vector[1]=vt_stay_sim[m_index_sim]
                value_plus_vector[1]=vt_stay_sim[m_index_sim+1]
                minus_value=vt_stay_sim[m_index_sim-1]
                choice_vector[m_index_sim]=0          
            elif (vt_renter_sim[m_index_sim] > vt_buy_c[m_index_sim]) and (vt_renter_sim[m_index_sim] > vt_buy_nc[m_index_sim]) and (vt_renter_sim[m_index_sim] > vt_default[m_index_sim]):
                value_vector[1]=vt_renter_sim[m_index_sim]
                value_plus_vector[1]=vt_renter_sim[m_index_sim+1]
                minus_value=vt_renter_sim[m_index_sim-1]
                choice_vector[m_index_sim]=1 
            elif (vt_buy_c[m_index_sim] > vt_buy_nc[m_index_sim]) and (vt_buy_c[m_index_sim] > vt_default[m_index_sim]):
                value_vector[1]=vt_buy_c[m_index_sim]
                value_plus_vector[1]=vt_buy_c[m_index_sim+1]
                minus_value=vt_buy_c[m_index_sim-1]
                choice_vector[m_index_sim]=2
            elif (vt_buy_nc[m_index_sim] > vt_default[m_index_sim]): 
                value_vector[1]=vt_buy_nc[m_index_sim]
                value_plus_vector[1]=vt_buy_nc[m_index_sim+1]
                minus_value=vt_buy_nc[m_index_sim-1]
                choice_vector[m_index_sim]=3 
            else:
                value_vector[1]=vt_default[m_index_sim]
                value_plus_vector[1]=vt_default[m_index_sim+1]
                minus_value=vt_default[m_index_sim-1]
                choice_vector[m_index_sim]=4                 
            if choice_vector[m_index_sim-1]!=choice_vector[m_index_sim] and mass[m_index_sim-1]>0:
                cross_point_vector[m_index_sim-1]=(value_vector[0]-minus_value)/(value_vector[1]-minus_value-value_plus_vector[0]+value_vector[0])
        value_vector[0]=value_vector[1]
        value_plus_vector[0]=value_plus_vector[1]
        
    
    if mass[-1]>0:
        if (vt_stay_sim[-1] > vt_renter_sim[-1]) and (vt_stay_sim[-1] > vt_buy_c[-1]) and (vt_stay_sim[-1] > vt_buy_nc[-1]) and (vt_stay_sim[-1] > vt_default[-1]):
            value_vector[1]=vt_stay_sim[-1]           
            minus_value=vt_stay_sim[-2]
            choice_vector[-1]=0          
        elif (vt_renter_sim[-1] > vt_buy_c[-1]) and (vt_renter_sim[-1] > vt_buy_nc[-1]) and (vt_renter_sim[-1] > vt_default[-1]):
            value_vector[1]=vt_renter_sim[-1]      
            minus_value=vt_renter_sim[-2]
            choice_vector[-1]=1 
        elif (vt_buy_c[-1] > vt_buy_nc[-1]) and (vt_buy_c[-1] > vt_default[-1]):
            value_vector[1]=vt_buy_c[-1]   
            minus_value=vt_buy_c[-2]
            choice_vector[-1]=2 
        elif (vt_buy_nc[-1] > vt_default[-1]): 
            value_vector[1]=vt_buy_nc[-1]   
            minus_value=vt_buy_nc[-2]
            choice_vector[-1]=3 
        else:
            value_vector[1]=vt_default[-1]
            minus_value=vt_default[-2]
            choice_vector[-1]=4                
        if choice_vector[-2]!=choice_vector[-1] and mass[-2]>0:
            cross_point_vector[-1]=(value_vector[0]-minus_value)/(value_vector[1]-minus_value-value_plus_vector[0]+value_vector[0])

    for m_index_sim in range(0,grids.vM_sim.size-1):
        if mass[m_index_sim]==0 and mass[m_index_sim+1]==0:
            continue
        elif mass[m_index_sim]==0:            
            for idx in range(5):
                if choice_vector[m_index_sim+1]==idx:
                    output_mass[m_index_sim+1,idx]+=0.5*mass[m_index_sim+1]
        elif mass[m_index_sim+1]==0:
            for idx in range(5):
                if choice_vector[m_index_sim]==idx:
                    output_mass[m_index_sim,idx]+=0.5*mass[m_index_sim]
        elif choice_vector[m_index_sim]==choice_vector[m_index_sim+1]:
            for idx in range(5):
                if choice_vector[m_index_sim]==idx:
                    output_mass[m_index_sim,idx]+=0.5*mass[m_index_sim]
                    output_mass[m_index_sim+1,idx]+=0.5*mass[m_index_sim+1]
        elif cross_point_vector[m_index_sim]<0.5:
            for idx in range(5):
                if choice_vector[m_index_sim+1]==idx:
                    output_mass[m_index_sim+1,idx]+=0.5*mass[m_index_sim+1]
                    for jdx in range(5):
                        if choice_vector[m_index_sim]==jdx:
                            output_mass[m_index_sim,idx]+=(0.5-cross_point_vector[m_index_sim])*mass[m_index_sim]
                            output_mass[m_index_sim,jdx]+=cross_point_vector[m_index_sim]*mass[m_index_sim]
        else:
            for idx in range(5):
                if choice_vector[m_index_sim]==idx:
                    output_mass[m_index_sim,idx]+=0.5*mass[m_index_sim]
                    for jdx in range(5):
                        if choice_vector[m_index_sim+1]==jdx:
                            output_mass[m_index_sim+1,idx]+=(cross_point_vector[m_index_sim]-0.5)*mass[m_index_sim+1]
                            output_mass[m_index_sim+1,jdx]+=(1-cross_point_vector[m_index_sim])*mass[m_index_sim+1]

           
    mass[0]=0.5*mass[0]
    mass[-1]=0.5*mass[-1]
    
    mass_stay=output_mass[:,0]
    mass_rent=output_mass[:,1]
    mass_buyc=output_mass[:,2]
    mass_buync=output_mass[:,3]
    mass_default=output_mass[:,4]
    
    #Check that no mass was spilled
    total = mass_stay + mass_rent + mass_buyc + mass_buync + mass_default
    tol = 0.01 * np.maximum(mass, 1e-10)  
    if not np.all((np.abs(total - mass) < tol)) or not np.all(output_mass>=0):
        print(vt_stay_sim, vt_buy_c, vt_buy_nc, vt_renter_sim, vt_default)
    
    assert np.all((np.abs(total - mass) < tol)) and np.all(output_mass>=0)

    return mass_stay,mass_rent,mass_buyc,mass_buync, mass_default

    # return mass_stay,mass_rent,mass_buyc,mass_buync, mass_default

@njit
def continuous_decide_renter(grids,vt_buy_c, vt_buy_nc, vt_renter_sim, mass):

    mass_rent=np.zeros((grids.vX_sim.size))
    mass_buyc=np.zeros((grids.vX_sim.size))
    mass_buync=np.zeros((grids.vX_sim.size))
 
       
    mass[0]=2*mass[0]
    mass[-1]=2*mass[-1]
       
    for x_index_sim in range(grids.vX_sim.size-1):
        if mass[x_index_sim]==0:
            if (vt_renter_sim[x_index_sim+1] > vt_buy_c[x_index_sim+1]) and (vt_renter_sim[x_index_sim+1] > vt_buy_nc[x_index_sim+1]):
                mass_rent[x_index_sim+1]+= 0.5*mass[x_index_sim+1]
            elif (vt_buy_c[x_index_sim+1] > vt_buy_nc[x_index_sim+1]):
                mass_buyc[x_index_sim+1]+= 0.5*mass[x_index_sim+1]                
            else:
                mass_buync[x_index_sim+1]+= 0.5*mass[x_index_sim+1]                
        elif (vt_renter_sim[x_index_sim] > vt_buy_c[x_index_sim]) and (vt_renter_sim[x_index_sim] > vt_buy_nc[x_index_sim]):
            if ((vt_renter_sim[x_index_sim+1] > vt_buy_c[x_index_sim+1]) and (vt_renter_sim[x_index_sim+1] > vt_buy_nc[x_index_sim+1])) or mass[x_index_sim+1]==0:
                mass_rent[x_index_sim] += 0.5*mass[x_index_sim]
                mass_rent[x_index_sim+1] += 0.5*mass[x_index_sim+1]
            elif vt_buy_c[x_index_sim+1] > vt_buy_nc[x_index_sim+1]:
                cross_point=(vt_renter_sim[x_index_sim]-vt_buy_c[x_index_sim])/(vt_buy_c[x_index_sim+1]-vt_buy_c[x_index_sim]-vt_renter_sim[x_index_sim+1]+vt_renter_sim[x_index_sim])
                assert 0<=cross_point<=1
                mass_rent[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_rent[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_buyc[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim]
                mass_buyc[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
            else:
                cross_point=(vt_renter_sim[x_index_sim]-vt_buy_nc[x_index_sim])/(vt_buy_nc[x_index_sim+1]-vt_buy_nc[x_index_sim]-vt_renter_sim[x_index_sim+1]+vt_renter_sim[x_index_sim])
                assert 0<=cross_point<=1
                mass_rent[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_rent[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_buync[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim]   
                mass_buync[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
        elif (vt_buy_c[x_index_sim] > vt_buy_nc[x_index_sim]):
            if ((vt_buy_c[x_index_sim+1] > vt_renter_sim[x_index_sim+1]) and (vt_buy_c[x_index_sim+1] > vt_buy_nc[x_index_sim+1])) or mass[x_index_sim+1]==0:
                mass_buyc[x_index_sim] += 0.5*mass[x_index_sim]
                mass_buyc[x_index_sim+1] += 0.5*mass[x_index_sim+1]
            elif vt_renter_sim[x_index_sim+1] > vt_buy_nc[x_index_sim+1]:
                cross_point=(vt_buy_c[x_index_sim]-vt_renter_sim[x_index_sim])/(vt_renter_sim[x_index_sim+1]-vt_renter_sim[x_index_sim]-vt_buy_c[x_index_sim+1]+vt_buy_c[x_index_sim])
                assert 0<=cross_point<=1
                mass_buyc[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_buyc[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_rent[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim]
                mass_rent[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
            else:
                cross_point=(vt_buy_c[x_index_sim]-vt_buy_nc[x_index_sim])/(vt_buy_nc[x_index_sim+1]-vt_buy_nc[x_index_sim]-vt_buy_c[x_index_sim+1]+vt_buy_c[x_index_sim])
                assert 0<=cross_point<=1
                mass_buyc[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_buyc[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_buync[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim]     
                mass_buync[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
        else: 
            if ((vt_buy_nc[x_index_sim+1] > vt_renter_sim[x_index_sim+1]) and (vt_buy_nc[x_index_sim+1] > vt_buy_c[x_index_sim+1])) or mass[x_index_sim+1]==0:
                mass_buync[x_index_sim] += 0.5*mass[x_index_sim]
                mass_buync[x_index_sim+1] += 0.5*mass[x_index_sim+1]
            elif vt_renter_sim[x_index_sim+1] > vt_buy_c[x_index_sim+1]:
                cross_point=(vt_buy_nc[x_index_sim]-vt_renter_sim[x_index_sim])/(vt_renter_sim[x_index_sim+1]-vt_renter_sim[x_index_sim]-vt_buy_nc[x_index_sim+1]+vt_buy_nc[x_index_sim])
                assert 0<=cross_point<=1
                mass_buync[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_buync[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_rent[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim]
                mass_rent[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
            else:
                cross_point=(vt_buy_c[x_index_sim]-vt_buy_nc[x_index_sim])/(vt_buy_nc[x_index_sim+1]-vt_buy_nc[x_index_sim]-vt_buy_c[x_index_sim+1]+vt_buy_c[x_index_sim])
                assert 0<=cross_point<=1
                mass_buync[x_index_sim] += np.min(np.array([0.5,cross_point]))*mass[x_index_sim]
                mass_buync[x_index_sim+1] += np.max(np.array([0,cross_point-0.5]))*mass[x_index_sim+1]
                mass_buyc[x_index_sim] += np.max(np.array([0,0.5-cross_point]))*mass[x_index_sim] 
                mass_buyc[x_index_sim+1] += np.min(np.array([0.5,1-cross_point]))*mass[x_index_sim+1]
    
    mass[0]=0.5*mass[0]
    mass[-1]=0.5*mass[-1]
    
    #Check that no mass was spilled
    total = mass_rent + mass_buyc + mass_buync
    tol = 0.01 * np.maximum(mass, 1e-10)  # to avoid issues when mass == 0
    assert np.all((np.abs(total - mass) < tol)) and np.all(mass_rent>=0) and np.all(mass_buyc>=0) and np.all(mass_buync>=0) 
    
    return mass_rent,mass_buyc,mass_buync


#############################################################################
### Other functions
#############################################################################
@njit
def construct_m1(grids, par, j, b):


    m1 = (1+par.r)*b
    
    return m1


@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left


@njit(fastmath=True) 
def mortgage_matrix_solve(par, grids, dP_C_lag, dP_NC_lag, dP_C, dP_NC, mortgage_rate_c):
    minpay_matrix_C = np.empty((par.iNj, grids.vH.size,grids.vL_sim.size))
    ltv_minpay_index_left_C = np.empty((par.iNj, grids.vH.size,grids.vL_sim.size),dtype=np.int64)
    minpay_matrix_NC = np.empty((par.iNj, grids.vH.size,grids.vL_sim.size))
    ltv_minpay_index_left_NC = np.empty((par.iNj, grids.vH.size,grids.vL_sim.size),dtype=np.int64)   
    max_ltv_index_C= np.zeros((par.iNj,grids.vH.size,grids.vE.size),dtype=np.int64)
    max_ltv_index_NC= np.zeros((par.iNj,grids.vH.size,grids.vE.size),dtype=np.int64)
    max_ltv_C= np.zeros((par.iNj,grids.vH.size,grids.vE.size),dtype=np.float64)
    max_ltv_NC= np.zeros((par.iNj,grids.vH.size,grids.vE.size),dtype=np.float64)
    
    for j in range(par.iNj):
        for h_index in range(grids.vH.size):
            h = grids.vH[h_index]
            for l_index_sim in range(grids.vL_sim.size):
                ltv = grids.vL_sim[l_index_sim]
                mortgage_start_C=ltv*h*dP_C_lag
                minpay_matrix_C[j, h_index, l_index_sim] = (mortgage_rate_c*(1+mortgage_rate_c)**(par.iNj-j)/((1+mortgage_rate_c)**(par.iNj-j)-1))*mortgage_start_C
                ltv_minpay_C=((1+mortgage_rate_c)*mortgage_start_C-minpay_matrix_C[j,h_index, l_index_sim])/(dP_C*h)     
                ltv_minpay_index_left_C[j, h_index, l_index_sim]=misc.binary_search(0,grids.vL_sim.size,grids.vL_sim,ltv_minpay_C)
                
                mortgage_start_NC=ltv*h*dP_NC_lag
                minpay_matrix_NC[j, h_index, l_index_sim] = (par.r_m_nc*(1+par.r_m_nc)**(par.iNj-j)/((1+par.r_m_nc)**(par.iNj-j)-1))*mortgage_start_NC
                ltv_minpay_NC=((1+par.r_m_nc)*mortgage_start_NC-minpay_matrix_NC[j,h_index, l_index_sim])/(dP_NC*h)     
                ltv_minpay_index_left_NC[j, h_index, l_index_sim]=misc.binary_search(0,grids.vL_sim.size,grids.vL_sim,ltv_minpay_NC)

    
    for j in range(par.iNj-1):
        #Last period not allowed to take out mortgages
        for e_index in range(grids.vE.size):
             max_mortgage_pti_C=grids.mPTI_C[j,e_index]
             max_mortgage_pti_NC=grids.mPTI_NC[j,e_index]
                 # coastal homeowners
             for h_index in range(grids.vH.size):
                 h = grids.vH[h_index]
                 if grids.max_ltv<max_mortgage_pti_C/(dP_C*h):
                     max_ltv_C[j,h_index,e_index]=grids.max_ltv
                 else: 
                     max_ltv_C[j,h_index,e_index]=max_mortgage_pti_C/(dP_C*h)                
                 max_ltv_index_C[j,h_index,e_index] = misc.binary_search(0,grids.vL_sim.size,grids.vL_sim,max_ltv_C[j,h_index,e_index]) 
                 if grids.max_ltv<max_mortgage_pti_NC/(dP_NC*h):
                     max_ltv_NC[j,h_index,e_index]=grids.max_ltv
                 else: 
                     max_ltv_NC[j,h_index,e_index]=max_mortgage_pti_NC/(dP_NC*h)                  
                 max_ltv_index_NC[j,h_index,e_index] = misc.binary_search(0,grids.vL_sim.size,grids.vL_sim,max_ltv_NC[j,h_index,e_index]) 
             
    return minpay_matrix_C, ltv_minpay_index_left_C, minpay_matrix_NC, ltv_minpay_index_left_NC, max_ltv_C,max_ltv_NC, max_ltv_index_C, max_ltv_index_NC

@njit
def renter_sim(default, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, x_renter_vec, idx_list):    
    vt_renter_out=np.ones((grids.vM_sim.size))*-1e12
    for idx in idx_list:
        x_renter=x_renter_vec[idx]
        if x_renter>max(rental_price,rental_price_lom)*grids.vH_renter[0]:
            if default:
                utility_penalty=par.dXi_foreclosure
            else:
                utility_penalty=0
            vt_renter_lom=-1/(-1/interp.interp_1d(grids.vX,vt_renter_input, x_renter) -utility_penalty)
            vt_renter_out[idx]=vt_renter_lom
        else:
            vt_renter_out[idx]=-1e12

    return vt_renter_out

@njit
def renter_sim_demand(default, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, x_renter_vec, idx_list):    
    vt_renter_out=np.ones((grids.vM_sim.size))*-1e12
    h_renter_out=np.empty((grids.vM_sim.size))
    for idx in idx_list:
        x_renter=x_renter_vec[idx]
        if x_renter>max(rental_price,rental_price_lom)*grids.vH_renter[0]:    
            if default:
                utility_penalty=par.dXi_foreclosure
            else:
                utility_penalty=0
            vt_renter_lom=-1/(-1/interp.interp_1d(grids.vX,vt_renter_input, x_renter) -utility_penalty)
            B_pol=interp.interp_1d(grids.vX,b_renter_input, x_renter)
            expenditures=x_renter-B_pol
            h_renter_out[idx]=max(min(h_share/rental_price*expenditures,grids.vH_renter[-1]),grids.vH_renter[0])            
            vt_renter_out[idx]=vt_renter_lom 
            assert h_renter_out[idx]>0 and not np.isnan(vt_renter_out[idx])
        else:
            vt_renter_out[idx]=-1e12
            h_renter_out[idx]=0.          
        
        
        
        
    return vt_renter_out, h_renter_out

@njit
def renter_solve(par, grids, g_index, rental_price_lom_C, rental_price_lom_NC, rental_price_C, rental_price_NC):
    
    g=grids.vG[g_index]    
    g_indiff_lom=rental_price_lom_C/rental_price_lom_NC
    g_indiff=rental_price_C/rental_price_NC 
        
    grid_spacing=0.5*(grids.vG[-1]-grids.vG[0])/(grids.vG.size-1)
    right_bound=min(g+grid_spacing,grids.vG[-1])
    left_bound=max(g-grid_spacing,grids.vG[0])
    
    if g_indiff_lom<left_bound:
        rental_price_lom=rental_price_lom_C
        h_share_lom,c_share,w_lom=ut.renter_solve(par,rental_price_lom,g)
        g_renter_lom=g
    elif g_indiff_lom>right_bound:
        rental_price_lom=rental_price_lom_NC
        h_share_lom,c_share,w_lom=ut.renter_solve(par,rental_price_lom,1)   
        g_renter_lom=1   
    else:
        weight_C=(right_bound-g_indiff_lom)/(right_bound-left_bound)   
        rental_price_lom=weight_C*rental_price_lom_C+(1-weight_C)*rental_price_lom_NC 
        h_share_lom_C,_,w_lom_C=ut.renter_solve(par,rental_price_lom_C,right_bound)
        h_share_lom_NC,_,w_lom_NC=ut.renter_solve(par,rental_price_lom_NC,1)
        h_share_lom=weight_C*h_share_lom_C+(1-weight_C)*h_share_lom_NC
        w_lom=weight_C*w_lom_C+(1-weight_C)*w_lom_NC
        g_renter_lom=weight_C*g+(1-weight_C)

            
    if g_indiff<left_bound:
        rental_price=rental_price_C        
        h_share,c_share,w=ut.renter_solve(par,rental_price,g)  
        g_renter=g
        coastal_rent_share=1 
    elif g_indiff>right_bound:
        rental_price=rental_price_NC
        h_share,c_share,w=ut.renter_solve(par,rental_price,1)  
        g_renter=1     
        coastal_rent_share=0
    else:
        weight_C=(right_bound-g_indiff)/(right_bound-left_bound)   
        rental_price=weight_C*rental_price_C+(1-weight_C)*rental_price_NC 
        h_share_C,_,w_C=ut.renter_solve(par,rental_price_C,right_bound)
        h_share_NC,_,w_NC=ut.renter_solve(par,rental_price_NC,1)
        h_share=weight_C*h_share_C+(1-weight_C)*h_share_NC
        w=weight_C*w_C+(1-weight_C)*w_NC
        g_renter=weight_C*g+(1-weight_C)
        coastal_rent_share=weight_C
    

    assert h_share>0 and coastal_rent_share>=0 and coastal_rent_share<=1
    
    return h_share_lom, w_lom, h_share, w, rental_price_lom, rental_price, coastal_rent_share, g_renter_lom, g_renter

@njit
def gen_initial_dist(par, grids, t_index, dP_C, dP_NC, coastal_beq, noncoastal_beq, savings_beq, sceptics):
    
    if sceptics:
        k_dim=2 
    else:
        k_dim=1  
    mDist_age_0_renter=np.zeros((k_dim,grids.vG.size, grids.vX_sim.size, grids.vE.size))
    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
    housing_bequest=coastal_beq*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq*(1-par.dDelta)*dP_NC
    total_bequest = (housing_bequest+savings_beq*(1+par.r))*par.iNj
    mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
    for k_index in range(k_dim):
        if sceptics==True:
            k_weight=grids.mTypes[t_index,k_index]
        else:
            k_weight=1.
        for g_index in range(grids.vG.size):
            mDist_age_0_renter[k_index,g_index,:,:]= (1/par.iNj)*k_weight*(1/grids.vG.size)*mPi_joint
    

    
    return mDist_age_0_renter 