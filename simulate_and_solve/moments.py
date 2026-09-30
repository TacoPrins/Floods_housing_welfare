"""
moments.py
"""


import numpy as np
import numba as nb
import misc_functions as misc
from numba import njit
from numba import prange
import interp as interpfun
import LoM_epsilons as lom

@njit
def calc_moments(par, grids, t_index, mDist_c, mDist_nc,mDist_renter, vCoeff_C, vCoeff_NC):
    
    
    ##Filling the coastal and non-coastal renter matrices
    mDist_renter_C = np.zeros((par.iNj, grids.vK.size, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    mDist_renter_NC = np.zeros((par.iNj, grids.vK.size, grids.vG.size, grids.vX_sim.size, grids.vE.size))
      
       
    dP_C = lom.LoM(par,grids,t_index,vCoeff_C)
    dP_NC = lom.LoM(par,grids,t_index,vCoeff_NC)
    
    dP_C_prime=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_C)
    dP_NC_prime=lom.LoM(par,grids,min(t_index+1,grids.vTime.size-1),vCoeff_NC)
    #rental_price_c=(par.dPsi+dP_C-dPi_S*((1-par.dDelta-(1-par.dZ))/(1+par.r))*dP_C_prime_flood-
    #(1-dPi_S)*((1-par.dDelta)/(1+par.r))*dP_C_prime_noflood)
    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
    
    rental_price_C=par.dPsi+dP_C-(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime
    rental_price_NC=par.dPsi+dP_NC-(1-par.dDelta)/(1+par.r)*dP_NC_prime
    
    g_indiff=rental_price_C/rental_price_NC
    if grids.vG[0]<g_indiff and g_indiff<grids.vG[-1]:
        g_still_nc=misc.binary_search(0,grids.vG.size, grids.vG,g_indiff)    
        for g_index in range(0,g_still_nc+1):
            mDist_renter_NC[:,:,g_index,:,:]=mDist_renter[:,:,g_index,:,:]       
        for g_index in range(g_still_nc+1,grids.vG.size):
            mDist_renter_C[:,:,g_index,:,:]=mDist_renter[:,:,g_index,:,:]
    elif grids.vG[0]>g_indiff: 
       mDist_renter_C=mDist_renter  
    else:
       mDist_renter_NC=mDist_renter 
    
    #homeownership shares
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death = homeowner_renter_shares(grids, g_indiff, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC)
    
    """
    (k0g0_HO_C_share, k0g0_HO_NC_share, k0g0_R_C_share, k0g0_R_NC_share,
    k1g0_HO_C_share, k1g0_HO_NC_share, k1g0_R_C_share, k1g0_R_NC_share,
    k0g1_HO_C_share, k0g1_HO_NC_share, k0g1_R_C_share, k0g1_R_NC_share,
    k1g1_HO_C_share, k1g1_HO_NC_share, k1g1_R_C_share, k1g1_R_NC_share,
    k0g2_HO_C_share, k0g2_HO_NC_share, k0g2_R_C_share, k0g2_R_NC_share,
    k1g2_HO_C_share, k1g2_HO_NC_share, k1g2_R_C_share, k1g2_R_NC_share) = type_amenity_perhousetype(mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC)
    """
    # NW Dynamics 
    #agg_NW, NW_HO_C, NW_HO_NC, NW_R_C, NW_R_NC, NW_HO, NW_R  = aggregate_NW(par, grids, dP_C, dP_NC, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC, HO_C_share, HO_NC_share, R_C_share, R_NC_share)
    total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted  = end_of_life_NW(par, grids, dP_C, dP_NC, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC,HO_C_share, HO_NC_share, R_C_share, R_NC_share)
    
    """
    moments = {
        'HO_C_share': HO_C_share,
        'HO_NC_share': HO_NC_share,
        'R_C_share': R_C_share,
        'R_NC_share': R_NC_share,
        'agg_NW': agg_NW,
        'NW_HO_C': NW_HO_C,
        'NW_HO_NC': NW_HO_NC,
        'NW_R_C': NW_R_C,
        'NW_R_NC': NW_R_NC,
        'NW_HO': NW_HO,
        'NW_R':NW_R,
        'total_10':total_10,
        'total_end':total_end,
        'median_10':median_10,
        'median_end':median_end,
        'thirtythree_percentile_end':thirtythree_percentile_end,
        'sixtyseven_percentile_end':sixtyseven_percentile_end,
        'tenth_percentile_housing':tenth_percentile_housing,
        'median_housing': median_housing,
        'ninetieth_percentile_housing': ninetieth_percentile_housing
    }
    

    'k0g0_HO_C_share': k0g0_HO_C_share,
    'k0g0_HO_NC_share': k0g0_HO_NC_share,
    'k0g0_R_C_share': k0g0_R_C_share,
    'k0g0_R_NC_share': k0g0_R_NC_share,
    
    'k1g0_HO_C_share': k1g0_HO_C_share,
    'k1g0_HO_NC_share': k1g0_HO_NC_share,
    'k1g0_R_C_share': k1g0_R_C_share,
    'k1g0_R_NC_share': k1g0_R_NC_share,
    
    'k0g1_HO_C_share': k0g1_HO_C_share,
    'k0g1_HO_NC_share': k0g1_HO_NC_share,
    'k0g1_R_C_share': k0g1_R_C_share,
    'k0g1_R_NC_share': k0g1_R_NC_share,
    
    'k1g1_HO_C_share': k1g1_HO_C_share,
    'k1g1_HO_NC_share': k1g1_HO_NC_share,
    'k1g1_R_C_share': k1g1_R_C_share,
    'k1g1_R_NC_share': k1g1_R_NC_share,
    
    'k0g2_HO_C_share': k0g2_HO_C_share,
    'k0g2_HO_NC_share': k0g2_HO_NC_share,
    'k0g2_R_C_share': k0g2_R_C_share,
    'k0g2_R_NC_share': k0g2_R_NC_share,
    
    'k1g2_HO_C_share': k1g2_HO_C_share,
    'k1g2_HO_NC_share': k1g2_HO_NC_share,
    'k1g2_R_C_share': k1g2_R_C_share,
    'k1g2_R_NC_share': k1g2_R_NC_share,
    """
    
    return HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted  
@njit
def homeowner_renter_shares(grids, g_indiff, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC):
    """
    Calculate shares of households in different houses
    """
    HO_C_share = np.sum(mDist_c)
    HO_NC_share = np.sum(mDist_nc)
    renters_total=1-HO_C_share-HO_NC_share
    #CAREFUL: WE ASSUME UNIFORM (CONDITIONAL) DISTRIBUTION OVER G HERE
    renters_coastal_share=min(max((grids.vG[-1]-g_indiff)/(grids.vG[-1]-grids.vG[0]),0),1)
    
    R_C_share = renters_coastal_share*renters_total
    R_NC_share =(1-renters_coastal_share)*renters_total
    
    HO_C_share_before35 = np.sum(mDist_c[:8,])/(8/mDist_c.shape[0])
    HO_NC_share_before35 = np.sum(mDist_nc[:8,])/(8/mDist_c.shape[0])
    
    HO_C_share_death = np.sum(mDist_c[-1,])/(1/mDist_c.shape[0])
    HO_NC_share_death = np.sum(mDist_nc[-1,])/(1/mDist_c.shape[0]) # or equivalently * 30 for equal sized cohorts.
    
    # for j in range(30):
        # print('sum of coastal homeowners in age:',j, 'is', np.sum(mDist_c[j,]))
        # print('sum of inland homeowners in age:',j, 'is', np.sum(mDist_nc[j,]))
    
    return HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death

"""
@njit
def type_amenity_perhousetype(mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC):
    
    O_C_mass=np.sum(mDist_c)
    O_NC_mass=np.sum(mDist_nc)
    R_C_mass=np.sum(mDist_renter_C)
    R_NC_mass=np.sum(mDist_renter_NC)
    
    k0g0_HO_C_share  = safe_share(mDist_c[:,0,0,:,:,:,:],  O_C_mass)
    k0g0_HO_NC_share = safe_share(mDist_nc[:,0,0,:,:,:,:], O_NC_mass)
    k0g0_R_C_share   = safe_share(mDist_renter_C[:,0,0,:,:], R_C_mass)
    k0g0_R_NC_share  = safe_share(mDist_renter_NC[:,0,0,:,:], R_NC_mass)
    
    k1g0_HO_C_share  = safe_share(mDist_c[:,1,0,:,:,:,:],  O_C_mass)
    k1g0_HO_NC_share = safe_share(mDist_nc[:,1,0,:,:,:,:], O_NC_mass)
    k1g0_R_C_share   = safe_share(mDist_renter_C[:,1,0,:,:], R_C_mass)
    k1g0_R_NC_share  = safe_share(mDist_renter_NC[:,1,0,:,:], R_NC_mass)
    
    k0g1_HO_C_share  = safe_share(mDist_c[:,0,1,:,:,:,:],  O_C_mass)
    k0g1_HO_NC_share = safe_share(mDist_nc[:,0,1,:,:,:,:], O_NC_mass)
    k0g1_R_C_share   = safe_share(mDist_renter_C[:,0,1,:,:], R_C_mass)
    k0g1_R_NC_share  = safe_share(mDist_renter_NC[:,0,1,:,:], R_NC_mass)
    
    k1g1_HO_C_share  = safe_share(mDist_c[:,1,1,:,:,:,:],  O_C_mass)
    k1g1_HO_NC_share = safe_share(mDist_nc[:,1,1,:,:,:,:], O_NC_mass)
    k1g1_R_C_share   = safe_share(mDist_renter_C[:,1,1,:,:], R_C_mass)
    k1g1_R_NC_share  = safe_share(mDist_renter_NC[:,1,1,:,:], R_NC_mass)
    
    k0g2_HO_C_share  = safe_share(mDist_c[:,0,2,:,:,:,:],  O_C_mass)
    k0g2_HO_NC_share = safe_share(mDist_nc[:,0,2,:,:,:,:], O_NC_mass)
    k0g2_R_C_share   = safe_share(mDist_renter_C[:,0,2,:,:], R_C_mass)
    k0g2_R_NC_share  = safe_share(mDist_renter_NC[:,0,2,:,:], R_NC_mass)
    
    k1g2_HO_C_share  = safe_share(mDist_c[:,1,2,:,:,:,:],  O_C_mass)
    k1g2_HO_NC_share = safe_share(mDist_nc[:,1,2,:,:,:,:], O_NC_mass)
    k1g2_R_C_share   = safe_share(mDist_renter_C[:,1,2,:,:], R_C_mass)
    k1g2_R_NC_share  = safe_share(mDist_renter_NC[:,1,2,:,:], R_NC_mass)

    
    return (k0g0_HO_C_share, k0g0_HO_NC_share, k0g0_R_C_share, k0g0_R_NC_share,
    k1g0_HO_C_share, k1g0_HO_NC_share, k1g0_R_C_share, k1g0_R_NC_share,
    k0g1_HO_C_share, k0g1_HO_NC_share, k0g1_R_C_share, k0g1_R_NC_share,
    k1g1_HO_C_share, k1g1_HO_NC_share, k1g1_R_C_share, k1g1_R_NC_share,
    k0g2_HO_C_share, k0g2_HO_NC_share, k0g2_R_C_share, k0g2_R_NC_share,
    k1g2_HO_C_share, k1g2_HO_NC_share, k1g2_R_C_share, k1g2_R_NC_share)


@njit
def aggregate_NW(par, grids, dP_C, dP_NC, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC, HO_C_share, HO_NC_share, R_C_share, R_NC_share):

    Calculate aggregate NW, and average NW per type of house

    NW_HO_C = 0
    NW_HO_NC = 0 
    NW_R_C = 0
    NW_R_NC = 0
    
    total_mass = HO_C_share+HO_NC_share+R_C_share+R_NC_share
    total_mass_homeowners =  HO_C_share+HO_NC_share
    total_mass_renters = R_C_share+R_NC_share
    
    total_mass_owners_C=HO_C_share
    total_mass_owners_NC=HO_NC_share
    total_mass_renters_C=R_C_share
    total_mass_renters_NC=R_NC_share
    # For homeowners: b' + P^h * H - m'
    for j in range(par.iNj):
        for k_index in range(grids.vK.size):
            for g_index in range(grids.vG.size):
                for m_index_sim in range(grids.vM_sim.size):
                    m = grids.vM_sim[m_index_sim]
                    for h_index in range(grids.vH.size):
                        h = grids.vH[h_index]
                        for l_index_sim in range(grids.vL_sim.size):
                            ltv = grids.vL_sim[l_index_sim]
                            for e_index in range(grids.vE.size):
                                if total_mass_owners_C>0:
                                    Net_worth_C = m + h * dP_C *(1-par.dDelta) - ltv*(1+par.r_m) * h * dP_C
                                    mass_C = mDist_c[j, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                                    NW_HO_C += Net_worth_C * (mass_C/total_mass_owners_C)
                                
                                if total_mass_owners_NC>0:
                                    Net_worth_NC = m + h * dP_NC *(1-par.dDelta) - ltv*(1+par.r_m) * h * dP_NC
                                    mass_NC = mDist_nc[j, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                                    NW_HO_NC += Net_worth_NC * (mass_NC/total_mass_owners_NC)
                                                 
    for j in range(par.iNj):
        for k_index in range(grids.vK.size):
            for g_index in range(grids.vG.size):
                for x_index_sim in range(grids.vX_sim.size):
                    x=grids.vX_sim[x_index_sim]
                    for e_index in range(grids.vE.size):
                        Net_worth = x
                        
                        if total_mass_renters_C>0:
                            mass_C = mDist_renter_C[j, k_index, g_index, x_index_sim, e_index]
                            NW_R_C += Net_worth * (mass_C/total_mass_renters_C)
                        
                        if total_mass_renters_NC>0:
                            mass_NC = mDist_renter_NC[j, k_index, g_index, x_index_sim, e_index]
                            NW_R_NC += Net_worth * (mass_NC/total_mass_renters_NC)
                        
    NW_HO = NW_HO_C * (total_mass_owners_C/total_mass_homeowners) + NW_HO_NC * (total_mass_owners_NC/total_mass_homeowners)
    NW_R = NW_R_C * (total_mass_renters_C/total_mass_renters) + NW_R_NC * (total_mass_renters_NC/total_mass_renters)
    agg_NW = NW_HO * (total_mass_homeowners/total_mass) + NW_R * (total_mass_renters/total_mass)
    
    return agg_NW, NW_HO_C, NW_HO_NC, NW_R_C, NW_R_NC, NW_HO, NW_R
"""
@njit
def end_of_life_NW(par, grids, dP_C, dP_NC, mDist_c, mDist_nc,mDist_renter_C,mDist_renter_NC,HO_C_share, HO_NC_share, R_C_share, R_NC_share):
    """
    Calculate aggregate NW, and average NW per type of house
    """

    J, K, G, M, H, L, E = mDist_c.shape

    dens_mhl_c_age_27 = np.zeros((M, H, L))
    dens_mhl_nc_age_27 = np.zeros((M, H, L))
    dens_m_r_age_27 = np.zeros((grids.vX_sim.size))
    dens_mhl_c_age_15 = np.zeros((M, H, L))
    dens_mhl_nc_age_15 = np.zeros((M, H, L))
    dens_m_r_age_15 = np.zeros((grids.vX_sim.size))
    dens_mhl_c_age_30 = np.zeros((M, H, L))
    dens_mhl_nc_age_30 = np.zeros((M, H, L))
    dens_m_r_age_30 = np.zeros((grids.vX_sim.size))
    dens_mhl_c_all_ages = np.zeros((M, H, L))
    dens_mhl_nc_all_ages = np.zeros((M, H, L))
    dens_m_r_all_ages = np.zeros((grids.vX_sim.size))
    mNet_worth_c  = np.zeros((M, H, L))
    mNet_worth_nc  = np.zeros((M, H, L))
    mHouse_wealth_share_c  = np.zeros((M, H, L))
    mHouse_wealth_share_nc  = np.zeros((M, H, L))
    
   
    # owners
    for m_index_sim in range(M):
        m = grids.vM_sim[m_index_sim]
        for h_index in range(H):
            h = grids.vH[h_index]
            house_value_C=h*dP_C
            house_value_NC=h*dP_NC
            for l_index_sim in range(L):
                ltv = grids.vL_sim[l_index_sim]

                # Net worth depends only on (m,h,l)
                mNet_worth_c[m_index_sim, h_index, l_index_sim] = m + (1-ltv*(1.0+par.r_m_c)-par.dDelta)*house_value_C
                mNet_worth_nc[m_index_sim, h_index, l_index_sim] = m + (1-ltv*(1.0+par.r_m_nc)-par.dDelta)*house_value_NC
                mHouse_wealth_share_c[m_index_sim, h_index, l_index_sim] = max(((1-ltv*(1.0+par.r_m_c)-par.dDelta)*house_value_C),0)/( m + (1-ltv*(1.0+par.r_m_c)-par.dDelta)*house_value_C)
                mHouse_wealth_share_nc[m_index_sim, h_index, l_index_sim] = max(((1-ltv*(1.0+par.r_m_nc)-par.dDelta)*house_value_NC),0)/(m + (1-ltv*(1.0+par.r_m_nc)-par.dDelta)*house_value_NC)

                # Sum density over (k,g,e)
                acc_c_age_27 = 0.0
                acc_nc_age_27 = 0.0
                acc_c_age_15 = 0.0
                acc_nc_age_15 = 0.0
                acc_c_age_30 = 0.0
                acc_nc_age_30 = 0.0
                acc_c_all_ages = 0.0
                acc_nc_all_ages = 0.0
                for k_index in range(K):
                    for g_index in range(G):
                        for e_index in range(E):
                            acc_c_age_27 += mDist_c[27, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            acc_nc_age_27 += mDist_nc[27, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            acc_c_age_15 += mDist_c[15, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            acc_nc_age_15 += mDist_nc[15, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            acc_c_age_30 += mDist_c[-1, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            acc_nc_age_30 += mDist_nc[-1, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                            for j_index in range(J):                            
                                acc_c_all_ages += mDist_c[j_index, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                                acc_nc_all_ages += mDist_nc[j_index, k_index, g_index, m_index_sim, h_index, l_index_sim, e_index]
                dens_mhl_c_age_27[m_index_sim, h_index, l_index_sim] = acc_c_age_27
                dens_mhl_nc_age_27[m_index_sim, h_index, l_index_sim] = acc_nc_age_27
                dens_mhl_c_age_15[m_index_sim, h_index, l_index_sim] = acc_c_age_15
                dens_mhl_nc_age_15[m_index_sim, h_index, l_index_sim] = acc_nc_age_15
                dens_mhl_c_age_30[m_index_sim, h_index, l_index_sim] = acc_c_age_30
                dens_mhl_nc_age_30[m_index_sim, h_index, l_index_sim] = acc_nc_age_30
                dens_mhl_c_all_ages[m_index_sim, h_index, l_index_sim] = acc_c_all_ages
                dens_mhl_nc_all_ages[m_index_sim, h_index, l_index_sim] = acc_nc_all_ages
    # renters            
    vNet_worth_r = grids.vX_sim   
    for x_index_sim in range(grids.vX_sim.size):
        acc_r_age_27 = 0.0
        acc_r_age_15 = 0.0
        acc_r_age_30 = 0.0
        acc_r_all_ages = 0.0
        for k_index in range(K):
            for g_index in range(G):
                for e_index in range(E):
                    if R_C_share>0 and R_NC_share>0:
                        acc_r_age_27 += mDist_renter_C[27, k_index, g_index, x_index_sim, e_index]+mDist_renter_NC[27, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_15 += mDist_renter_C[15, k_index, g_index, x_index_sim, e_index]+mDist_renter_NC[15, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_30 += mDist_renter_C[-1, k_index, g_index, x_index_sim, e_index]+mDist_renter_NC[-1, k_index, g_index, x_index_sim, e_index]
                        for j_index in range(J):
                            acc_r_all_ages += mDist_renter_C[j_index, k_index, g_index, x_index_sim, e_index]+mDist_renter_NC[j_index, k_index, g_index, x_index_sim, e_index]
                    elif R_C_share>0: 
                        acc_r_age_27 += mDist_renter_C[27, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_15 += mDist_renter_C[15, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_30 += mDist_renter_C[-1, k_index, g_index, x_index_sim, e_index]
                        for j_index in range(J):
                            acc_r_all_ages += mDist_renter_C[j_index, k_index, g_index, x_index_sim, e_index]
                    elif R_NC_share>0:
                        acc_r_age_27 += mDist_renter_NC[27, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_15 += mDist_renter_NC[15, k_index, g_index, x_index_sim, e_index]
                        acc_r_age_30 += mDist_renter_NC[-1, k_index, g_index, x_index_sim, e_index]
                        for j_index in range(J):
                            acc_r_all_ages += mDist_renter_NC[j_index, k_index, g_index, x_index_sim, e_index]

        dens_m_r_age_27[x_index_sim] = acc_r_age_27
        dens_m_r_age_15[x_index_sim] = acc_r_age_15
        dens_m_r_age_30[x_index_sim] = acc_r_age_30
        dens_m_r_all_ages[x_index_sim] = acc_r_all_ages
    
    # Flatten ownership NW matrices and all densities
    mHouse_wealth_share_c_flat = mHouse_wealth_share_c.flatten()
    mHouse_wealth_share_nc_flat = mHouse_wealth_share_nc.flatten()
    mNet_worth_c_flat = mNet_worth_c.flatten()
    mNet_worth_nc_flat = mNet_worth_nc.flatten()
    dens_mhl_c_age_27_flat = dens_mhl_c_age_27.flatten()
    dens_mhl_nc_age_27_flat = dens_mhl_nc_age_27.flatten()
    dens_mhl_c_age_15_flat = dens_mhl_c_age_15.flatten()
    dens_mhl_nc_age_15_flat = dens_mhl_nc_age_15.flatten()
    dens_mhl_c_age_30_flat = dens_mhl_c_age_30.flatten()
    dens_mhl_nc_age_30_flat = dens_mhl_nc_age_30.flatten()
    dens_mhl_c_all_ages_flat = dens_mhl_c_all_ages.flatten()
    dens_mhl_nc_all_ages_flat = dens_mhl_nc_all_ages.flatten()
    
    #Calculate net worth of subgroups
    total_NW_HO_C=np.dot(mNet_worth_c_flat,dens_mhl_c_all_ages_flat)/HO_C_share
    total_NW_HO_NC=np.dot(mNet_worth_nc_flat,dens_mhl_nc_all_ages_flat)/HO_NC_share
    total_NW_R=np.dot(vNet_worth_r,dens_m_r_all_ages)/(R_C_share+R_NC_share)
    
    total_NW_HO=(total_NW_HO_C*HO_C_share+total_NW_HO_NC*HO_NC_share)/(HO_C_share+HO_NC_share)
    # build big matrices
    N_own = mNet_worth_c_flat.size
    N_r   = vNet_worth_r.size
    N_all = 2*N_own + N_r
    
    NW = np.empty(N_all)
    
    dens_age_27 = np.empty(N_all)
    dens_age_15 = np.empty(N_all)
    dens_age_30 = np.empty(N_all)
    dens_all_ages = np.empty(N_all)
    
    # --- OWNERS-ONLY objects for housing wealth share ---
    NW_housing_share = np.empty(2*N_own)
    dens_housing_all_ages = np.empty(2*N_own)
    
    # Owners C
    s = 0
    e = N_own
    NW[s:e] = mNet_worth_c_flat
    dens_age_27[s:e] = dens_mhl_c_age_27_flat
    dens_age_15[s:e] = dens_mhl_c_age_15_flat
    dens_age_30[s:e] = dens_mhl_c_age_30_flat
    dens_all_ages[s:e] = dens_mhl_c_all_ages_flat
    
    # owners-only housing share (C)
    NW_housing_share[0:N_own] = mHouse_wealth_share_c_flat
    dens_housing_all_ages[0:N_own] = dens_mhl_c_all_ages_flat
    
    # Owners NC
    s = e
    e = s + N_own
    NW[s:e] = mNet_worth_nc_flat
    dens_age_27[s:e] = dens_mhl_nc_age_27_flat
    dens_age_15[s:e] = dens_mhl_nc_age_15_flat
    dens_age_30[s:e] = dens_mhl_nc_age_30_flat
    dens_all_ages[s:e] = dens_mhl_nc_all_ages_flat
    
    # owners-only housing share (NC)
    NW_housing_share[N_own:2*N_own] = mHouse_wealth_share_nc_flat
    dens_housing_all_ages[N_own:2*N_own] = dens_mhl_nc_all_ages_flat
    
    # Renters
    s = e
    e = s + N_r
    NW[s:e] = vNet_worth_r
    dens_age_27[s:e] = dens_m_r_age_27
    dens_age_15[s:e] = dens_m_r_age_15
    dens_age_30[s:e] = dens_m_r_age_30
    dens_all_ages[s:e] = dens_m_r_all_ages
            

    
    # Now sort in order to get statistics within the distribution
    order = np.argsort(NW)
    NW_sorted = NW[order]
    dens_age_15_sorted = dens_age_15[order]
    dens_age_27_sorted = dens_age_27[order] 
    dens_age_30_sorted = dens_age_30[order]
    dens_all_ages_sorted=dens_all_ages[order] 
    
    order_housing = np.argsort(NW_housing_share)
    NW_housing_share_sorted = NW_housing_share[order_housing]
    dens_housing_all_ages_sorted = dens_housing_all_ages[order_housing]
    # cumulative densities for interpolation later
    cumdens_age_15 = np.cumsum(dens_age_15_sorted)
    cumdens_age_27 = np.cumsum(dens_age_27_sorted)
    cumdens_age_30 = np.cumsum(dens_age_30_sorted)
    cumdens_all_ages = np.cumsum(dens_all_ages_sorted) 
    
    total_mass_age_15 = cumdens_age_15[-1]    
    total_mass_age_27 = cumdens_age_27[-1]  
    total_mass_age_30 = cumdens_age_30[-1]
    total_mass_all_ages = cumdens_all_ages[-1]
    
    assert np.abs(total_mass_age_27-1/par.iNj)<1e-10
    assert np.abs(total_mass_age_15-1/par.iNj)<1e-10
    assert np.abs(total_mass_age_30-1/par.iNj)<1e-10
    assert np.abs(total_mass_all_ages-1)<1e-10

    total_NW_age_15=np.dot(dens_age_15,NW)/total_mass_age_15
    total_NW_age_27=np.dot(dens_age_27,NW)/total_mass_age_27
    total_NW_all_ages=np.dot(dens_all_ages,NW)/total_mass_all_ages    

    
    median_NW_age_15 = np.interp(0.5*total_mass_age_15, cumdens_age_15, NW_sorted)
    median_NW_age_27 = np.interp(0.5*total_mass_age_27, cumdens_age_27, NW_sorted)
    median_NW_all_ages = np.interp(0.5*total_mass_all_ages, cumdens_all_ages, NW_sorted)
    
    thirtythree_percentile_NW_age_27 = np.interp(0.33*total_mass_age_27, cumdens_age_27, NW_sorted)
    sixtyseven_percentile_NW_age_27 = np.interp(0.67*total_mass_age_27, cumdens_age_27, NW_sorted)
    thirtythree_percentile_NW_age_30 = np.interp(0.33*total_mass_age_30, cumdens_age_30, NW_sorted)
    sixtyseven_percentile_NW_age_30 = np.interp(0.67*total_mass_age_30, cumdens_age_30, NW_sorted)
    
    # ----------------------------------------------------
    # Sort for owners-only housing wealth share stats
    mass_owners = np.sum(dens_housing_all_ages)
    mass_share_zero = np.sum(dens_housing_all_ages[NW_housing_share == 0.0])
    share_zero_frac = mass_share_zero / mass_owners
    print('above 0.1?', share_zero_frac)
    mass_UW = underwater_mass_jhl(par, grids, mDist_c, mDist_nc)

    # Print only cells with positive mass
    print("j_index | h_index | l_index | ltv | mass")
    J, H, L = mass_UW.shape
    for j in range(J):
        for h in range(H):
            for l in range(L):
                m = mass_UW[j, h, l]
                if m > 0.0:
                    print(j, h, l, grids.vL_sim[l], m)
    order_housing = np.argsort(NW_housing_share)
    NW_housing_share_sorted = NW_housing_share[order_housing]
    dens_housing_all_ages_sorted = dens_housing_all_ages[order_housing]
    
    cumdens_housing_all_ages = np.cumsum(dens_housing_all_ages_sorted)
    total_mass_housing_all_ages = cumdens_housing_all_ages[-1]
    
    tenth_percentile_housing = np.interp(0.10*total_mass_housing_all_ages,
                                         cumdens_housing_all_ages, NW_housing_share_sorted)
    median_housing = np.interp(0.50*total_mass_housing_all_ages,
                               cumdens_housing_all_ages, NW_housing_share_sorted)
    ninetieth_percentile_housing = np.interp(0.90*total_mass_housing_all_ages,
                                             cumdens_housing_all_ages, NW_housing_share_sorted)
    
    return total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted
@njit 
def safe_share(numerator, denominator):
    output=0.0    
    
    if denominator>0:
        output=np.sum(numerator)/denominator
    
    return output 

@njit
def underwater_mass_jhl(par, grids, mDist_c, mDist_nc):
    """
    Returns mass_UW[j,h,l] = mass of owners (C+NC) with negative housing wealth
    (i.e. equity_factor < 0 and h > 0), summed over (k,g,m,e).
    """
    J, K, G, M, H, L, E = mDist_c.shape
    mass_UW = np.zeros((J, H, L))

    for l_index in range(L):
        ltv = grids.vL_sim[l_index]
        equity_factor = 1.0 - ltv * (1.0 + par.r_m_c) - par.dDelta  # sign drives underwater if h>0

        if equity_factor < 0.0:
            for h_index in range(H):
                if grids.vH[h_index] > 0.0:
                    for j_index in range(J):
                        acc = 0.0
                        for k_index in range(K):
                            for g_index in range(G):
                                for m_index in range(M):
                                    for e_index in range(E):
                                        acc += mDist_c[j_index, k_index, g_index, m_index, h_index, l_index, e_index]
                                        acc += mDist_nc[j_index, k_index, g_index, m_index, h_index, l_index, e_index]
                        mass_UW[j_index, h_index, l_index] = acc

    return mass_UW