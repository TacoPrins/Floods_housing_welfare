# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:24:47 2026

@author: tprins
"""

"""
collect_results.py
"""

###########################################################
### Imports
import numpy as np
import equilibrium as equil
import proper_welfare_debug as welfare_stats
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import LoM_epsilons as lom
import unpack_configs
import experiment_config
import misc_functions as misc 
import grid_creation as grid_creation
import par_epsilons as parfile
import coeff_io
import create_plotting_inputs
import pandas as pd

WELFARE_NAMES = ["tax_equiv_C_RE", "tax_equiv_NC_RE", "tax_equiv_renter_RE", "tax_equiv_newborns_RE",
                 "tax_equiv_C", "tax_equiv_NC", "tax_equiv_renter", "tax_equiv_newborns",
                 "tax_equiv_C_BR", "tax_equiv_NC_BR", "tax_equiv_renter_BR", "tax_equiv_newborns_BR",
                 "tax_equiv_C_MP", "tax_equiv_NC_MP", "tax_equiv_renter_MP", "tax_equiv_newborns_MP"]


def write_welfare(names, arrays, path="welfare_results.xlsx", replace_only=False):
    """Write welfare arrays to Excel, one sheet each. replace_only=True overwrites just these
    sheets in the existing file and leaves all other sheets untouched."""
    kw = dict(mode="a", engine="openpyxl", if_sheet_exists="replace") if replace_only else {}
    with pd.ExcelWriter(path, **kw) as writer:
        for name, arr in zip(names, arrays):
            a = np.asarray(arr)
            if a.ndim == 3:                     # newborns: (T, k, E) -> flatten T,k into rows
                T, k, E = a.shape
                a = a.reshape(T * k, E)
                idx = pd.MultiIndex.from_product([range(T), range(k)], names=["t", "k"])
                df = pd.DataFrame(a, index=idx)
            else:                               # (k, E)
                df = pd.DataFrame(a, index=pd.Index(range(a.shape[0]), name="k"))
            df.columns = [f"e{j}" for j in range(df.shape[1])]
            df.to_excel(writer, sheet_name=name[:31])   # Excel caps sheet names at 31 chars
    print(f"saved {path}")


def collect_results(plot_distribution, calculate_welfare, only_RE_welfare=False):
    """only_RE_welfare=True: run only the RE steady state, RE transition and RE welfare of SLR,
    overwrite the four _RE sheets in welfare_results.xlsx (other sheets untouched), then stop."""
    par = misc.construct_jitclass(parfile.par_dict)
    grids=grid_creation.create(par)
    cfg = unpack_configs.unpack(experiment_config, misc)

    # load solved coefficients by label (written by solve_model.py)
    C = coeff_io.load_with_defaults()
    if not only_RE_welfare:                 # HE part, not needed for the RE-only update
        "(1) initial steady state"
    
        dP_C_initial = lom.LoM(par,grids,0,C["vCoeff_C_initial_HE"])
        dP_NC_initial = lom.LoM(par,grids,0,C["vCoeff_NC_initial_HE"])

        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, C["vCoeff_C_initial_HE"][0], C["vCoeff_NC_initial_HE"][0], cfg.solve_initial_ss_HE)
        mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], C["vCoeff_C_initial_HE"],C["vCoeff_NC_initial_HE"], cfg.solve_initial_ss_HE)
    
        del vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter
    
        "(2) baseline transition with sceptics"
        # run generate price path without experiments, with sceptics. save the (collapsed) distributions. Also save the 2026 distributions and welfare value functions
        _, _, _, _, _, _, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, vdefault_mass_C_HE, vdefault_mass_NC_HE =equil.generate_pricepath(grids, par, C["vCoeff_C_HE"],C["vCoeff_NC_HE"], dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, cfg.transition_path)    
    
        if calculate_welfare:
            "WELFARE COSTS OF MISBELIEFS"        
            tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns, wf_SLR_newborns =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, C["vCoeff_C_initial_HE"], C["vCoeff_NC_initial_HE"], C["vCoeff_C_HE"], C["vCoeff_NC_HE"], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS,  vcoastal_beq, vnoncoastal_beq, vsavings_beq, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, cfg.solve_initial_ss_HE, cfg.transition_path)

        del v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf
    
    "(3) baseline transition without sceptics (RE)"
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, C["vCoeff_C_initial_RE"][0], C["vCoeff_NC_initial_RE"][0], cfg.solve_initial_ss_RE)
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], C["vCoeff_C_initial_RE"],C["vCoeff_NC_initial_RE"], cfg.solve_initial_ss_RE)
    
    del vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter
    
    # run generate price path without experiments, without sceptics. save the (collapsed) distributions and welfare value functions
    dP_C_initial_RE = lom.LoM(par,grids,0,C["vCoeff_C_initial_RE"])
    dP_NC_initial_RE = lom.LoM(par,grids,0,C["vCoeff_NC_initial_RE"])
    price_history_RE, _, _, _, _, _, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, _, _, _, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE, vdefault_mass_C_RE, vdefault_mass_NC_RE = equil.generate_pricepath(grids, par, C["vCoeff_C_RE"], C["vCoeff_NC_RE"], dP_C_initial_RE, dP_NC_initial_RE, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, cfg.transition_path_RE)
    
    if calculate_welfare or only_RE_welfare:
        "WELFARE COSTS OF MISBELIEFS"        
        tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE, wf_SLR_newborns =  welfare_stats.find_expenditure_equiv_EK_SLR(par, grids, C["vCoeff_C_initial_RE"], C["vCoeff_NC_initial_RE"], C["vCoeff_C_RE"], C["vCoeff_NC_RE"], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, vcoastal_beq_RE, vnoncoastal_beq_RE, vsavings_beq_RE, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE, cfg.solve_initial_ss_RE, cfg.transition_path_RE)
        
    if only_RE_welfare:                     # update the _RE sheets only, and stop here
        welfare_out = (tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE)
        write_welfare(WELFARE_NAMES[:4], welfare_out, replace_only=True)
        return welfare_out

    del mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, v_owner_c_wf_RE, v_owner_nc_wf_RE, v_nonowner_wf_RE
    
    "(distribution outputs for plotting — plotting itself lives in plot_dist)"
    vTime = np.asarray(grids.vTime)[:full_dist_C_HE.shape[0]]
    if plot_distribution:
        create_plotting_inputs.create(
        (vTime,
         full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE,
         full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE,
         vdefault_mass_C_HE, vdefault_mass_NC_HE,
         vdefault_mass_C_RE, vdefault_mass_NC_RE),
        grids, "plotting_inputs.xlsx")

    del full_dist_C_HE, full_dist_NC_HE, full_dist_renter_HE, full_dist_C_RE, full_dist_NC_RE, full_dist_renter_RE
    

    """############################################################################
    ### Plot 2026 dist
    ############################################################################"""
  
    
    """############################################################################
    ### START EXPERIMENTS
    ############################################################################"""
    if calculate_welfare:
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter,_,_,_ = household_problem.solve_ss(grids, par, C["vCoeff_C_initial_HE"][0], C["vCoeff_NC_initial_HE"][0], cfg.solve_initial_ss_HE)
        mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, no_beq=sim.stat_dist_finder(par, grids, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], C["vCoeff_C_initial_HE"],C["vCoeff_NC_initial_HE"], cfg.solve_initial_ss_HE)

        price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, _, _, _, _, _, _, _, _, _, _, _, _, _, _=equil.generate_pricepath(grids, par, C["vCoeff_C_HE"], C["vCoeff_NC_HE"], dP_C_initial, dP_NC_initial, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, cfg.path_until_experiment)
            
        del vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS
        "(4) + (5) welfare effects of policy: building restrictions and mortgage premium"
        print("start with welfare of policy")
        tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP,tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR = welfare_stats.find_expenditure_equiv_EK_policy(par, grids, C["vCoeff_C_HE"], C["vCoeff_NC_HE"],C["vCoeff_C_MP"], C["vCoeff_NC_MP"], C["vCoeff_C_BR"], C["vCoeff_NC_BR"], mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, cfg.transition_path, cfg.experiment_mortgage_prem, cfg.experiment_building_rest)

        welfare_out = (tax_equiv_C_RE, tax_equiv_NC_RE, tax_equiv_renter_RE, tax_equiv_newborns_RE,
                       tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns,
                       tax_equiv_C_BR, tax_equiv_NC_BR, tax_equiv_renter_BR, tax_equiv_newborns_BR,
                       tax_equiv_C_MP, tax_equiv_NC_MP, tax_equiv_renter_MP, tax_equiv_newborns_MP)
        write_welfare(WELFARE_NAMES, welfare_out)
    else:
        welfare_out = None

    "distribution outputs first, then welfare outputs; each is None when its toggle is off"
    return welfare_out