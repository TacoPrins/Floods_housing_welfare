
import numpy as np
import misc_functions as misc
import grid_creation as grid_creation
import moments as find_moments
import copy
import par_epsilons as parfile
import experiment_config
import unpack_configs
import coeff_io
import equilibrium as equil
import time
import nlopt



def f(x,grad):

    # dbeta, dNu, db_bar, dOmega, domega_g, dPsi, dPhi
    dBeta = x[0]
    dNu = x[1]
    db_bar =x[2]
    dOmega =x[3]
    omega_g =x[4]
    dPsi =x[5]
    dPhi = x[6]
    
    cfg = unpack_configs.unpack(experiment_config, misc)
    C = coeff_io.load_with_defaults()
    vCoeff_C_initial=C["vCoeff_C_initial_HE"]
    vCoeff_NC_initial=C["vCoeff_NC_initial_HE"]
    
    
    par_dict = copy.deepcopy(parfile.par_dict)
    par_dict["dBeta"]   = dBeta**par_dict["time_increment"]
    par_dict["dNu"]     = dNu
    par_dict["b_bar"]   = db_bar
    par_dict["dOmega"]  = dOmega
    par_dict["dXi_min"] = 1 - omega_g
    par_dict["dXi_max"] = 1 + omega_g
    par_dict["dPsi"]    = dPsi
    par_dict["dPhi"]    = dPhi

    
    par = misc.construct_jitclass(par_dict)   
    grids=grid_creation.create(par)        
    vCoeff_C_new, vCoeff_NC_new, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq=equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial, vCoeff_NC_initial, cfg.solve_initial_ss_HE)      
    
    dP_C_lom = vCoeff_C_new[0]
    dP_NC_lom = vCoeff_NC_new[0]
        
    # MODEL MOMENTS
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter,  vCoeff_C_new, vCoeff_NC_new)
    total_saving_model = median_NW_all_ages
    NW_decay_model = total_NW_age_27/total_NW_age_15
    bequest_ineq_model = sixtyseven_percentile_NW_age_30/thirtythree_percentile_NW_age_30
    homeownership_model = HO_C_share+HO_NC_share
    price_diff_model = (dP_C_lom-dP_NC_lom)/dP_NC_lom
    homeownership_young_model = HO_C_share_before35 + HO_NC_share_before35
    med_housing_model = median_housing
    
  
    
    # DATA MOMENTS
    total_saving_data = 1.2
    NW_decay_data = 1.51
    bequest_ineq_data = 3.24
    homeownership = 0.66
    price_diff = -0.114
    homeownership_young = 0.39
    med_housing = 0.5
    
    print('total_saving_model', total_saving_model, 'data:' , total_saving_data)
    print('NW_decay_data', NW_decay_model, 'data:', NW_decay_data)
    print('bequest_ineq_data', bequest_ineq_model, 'data:', bequest_ineq_data)
    print('homeownership', homeownership_model, 'data:', homeownership)
    print('total_saving_model', price_diff_model, 'data:', price_diff)
    print('homeownership_young_model', homeownership_young_model, 'data:', homeownership_young)
    print('med_housing_model', med_housing_model, 'data:', med_housing)
    

    
    sq_saving       = ((total_saving_data-total_saving_model)/total_saving_data)**2
    sq_nw           = ((NW_decay_data-NW_decay_model)/NW_decay_data)**2
    sq_ineqnw       = ((bequest_ineq_data-bequest_ineq_model)/bequest_ineq_data)**2
    sq_homeownership = ((homeownership - homeownership_model)/homeownership)**2
    sq_price_diff     = ((price_diff - price_diff_model)/price_diff)**2
    sq_homeownership_young = ((homeownership_young - homeownership_young_model)/homeownership_young)**2
    sq_med_housing    = ((med_housing - med_housing_model)/med_housing)**2
    weights = np.array([1,1,1,1.5,1,1,1])
    
    squaredsum =  weights[0]*sq_saving + weights[1]*sq_nw + weights[2]*sq_ineqnw + weights[3]*sq_homeownership + weights[4]*sq_price_diff +  weights[5]*sq_homeownership_young + weights[6]*sq_med_housing 
    
    return squaredsum

def main():


    t0 = time.time()
     
    # Define bounds for each parameter: # dbeta, dNu, db_bar, dOmega, domega_g, dPsi, dPhi
    lb = [0.94,  10,  0.1, 0,     0.02, 0.003,  0.1]  # Lower bounds
    ub = [0.965, 50, 5, 0.01, 0.06, 0.015, 0.2]   # Upper bounds
    
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 7)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(f)
    opt.set_xtol_rel(1e-3)   
    opt.set_maxeval(48)
    
    # Optimize
    # x = [dbeta, deta, b_bar, dgamma, s_bar,A_g, A_r, A_kf, A_ec, A_ffc]
    x_opt = opt.optimize([9.4132422e-01, 4.02843750e+01, 2.79152344e+00, 2.41390625e-03, 4.32468750e-02, 1.16789062e-02, 1.38917969e-01])
    t1 = time.time()
    print("Calibration time:", t1-t0, ".")
    # Print results
    print("Optimized parameters:", x_opt)
    
###########################################################
### start main
if __name__ == "__main__":
    main()