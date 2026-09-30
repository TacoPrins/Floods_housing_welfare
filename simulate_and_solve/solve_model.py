import equilibrium as equil
import coeff_io
import unpack_configs
import misc_functions as misc
import experiment_config
import grid_creation as grid_creation
import par_epsilons as parfile

def solve():
    
    par = misc.construct_jitclass(parfile.par_dict)
    grids=grid_creation.create(par)
     
    "load previously-solved coefficients (warm start); defaults used if file/label absent"
    _saved = coeff_io.load_coefficients()
    _g = lambda name: coeff_io.get_guess(_saved, name, coeff_io.DEFAULTS[name])

    "incremental saving: write each solve to its own file, and warm-start from them"
    import os, glob
    _outdir = "coeff_checkpoints"
    os.makedirs(_outdir, exist_ok=True)
    def _save(name, coeff):
        coeff_io.save_coefficients({name: coeff}, path=os.path.join(_outdir, f"{name}.xlsx"))
    # merge any checkpoint files from a previous (crashed) run into the warm-start dict
    for _f in glob.glob(os.path.join(_outdir, "*.xlsx")):
        _saved.update(coeff_io.load_coefficients(_f))

    "coefficients 2 different initial steady states, including default values if excel not filled"
    vCoeff_C_initial_HE_guess = _g("vCoeff_C_initial_HE")
    vCoeff_NC_initial_HE_guess = _g("vCoeff_NC_initial_HE")
    vCoeff_C_initial_RE_guess = _g("vCoeff_C_initial_RE")
    vCoeff_NC_initial_RE_guess = _g("vCoeff_NC_initial_RE")
    "coefficients for baseline and rational expectation transitions"
    vCoeff_C_HE_guess=_g("vCoeff_C_HE")
    vCoeff_NC_HE_guess=_g("vCoeff_NC_HE")
    vCoeff_C_RE_guess=_g("vCoeff_C_RE")
    vCoeff_NC_RE_guess=_g("vCoeff_NC_RE")
    vCoeff_C_BR_guess=_g("vCoeff_C_BR")
    vCoeff_NC_BR_guess=_g("vCoeff_NC_BR")
    vCoeff_C_MP_guess=_g("vCoeff_C_MP")
    vCoeff_NC_MP_guess=_g("vCoeff_NC_MP")

    cfg = unpack_configs.unpack(experiment_config, misc)
    ## config solve_initial_ss_RE
    vCoeff_C_initial_RE, vCoeff_NC_initial_RE, mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_RE_guess, vCoeff_NC_initial_RE_guess, cfg.solve_initial_ss_RE)
    _save("vCoeff_C_initial_RE", vCoeff_C_initial_RE); _save("vCoeff_NC_initial_RE", vCoeff_NC_initial_RE)
    dP_C_initial_RE=vCoeff_C_initial_RE[0]
    dP_NC_initial_RE=vCoeff_NC_initial_RE[0]
    
    ## config: find_coef_RE
    _, _, vCoeff_C_RE, vCoeff_NC_RE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_RE_guess, vCoeff_NC_RE_guess,dP_C_initial_RE, dP_NC_initial_RE,mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE, rental_stock_C_initial_RE, rental_stock_NC_initial_RE, coastal_beq_initial_RE, noncoastal_beq_initial_RE, savings_beq_initial_RE,cfg.find_coeff_path_RE)
    _save("vCoeff_C_RE", vCoeff_C_RE); _save("vCoeff_NC_RE", vCoeff_NC_RE)

    #We only simulate the RE distributions forwards without policy experiments, so we don't need to keep the initial distributions 
    del mDist0_c_initial_RE, mDist0_nc_initial_RE, mDist0_renter_initial_RE
    
    ## config solve_initial_ss
    vCoeff_C_initial_HE, vCoeff_NC_initial_HE,  mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_initial_HE_guess, vCoeff_NC_initial_HE_guess, cfg.solve_initial_ss_HE)
    _save("vCoeff_C_initial_HE", vCoeff_C_initial_HE); _save("vCoeff_NC_initial_HE", vCoeff_NC_initial_HE)
    dP_C_initial_HE=vCoeff_C_initial_HE[0]
    dP_NC_initial_HE=vCoeff_NC_initial_HE[0]

    
    ## config: find_coef_baseline  
    _, _, vCoeff_C_HE, vCoeff_NC_HE, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_HE_guess, vCoeff_NC_HE_guess,dP_C_initial_HE, dP_NC_initial_HE,mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE,cfg.find_coeff_path_HE)
    _save("vCoeff_C_HE", vCoeff_C_HE); _save("vCoeff_NC_HE", vCoeff_NC_HE)

        
    "find distribution in 2026 (experiment year) with generate price path using coefficients from baseline"
    price_history, mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, _, _, _, _, _, _,_,_=equil.generate_pricepath(grids, par, vCoeff_C_HE, vCoeff_NC_HE, dP_C_initial_HE, dP_NC_initial_HE, mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE, rental_stock_C_initial_HE, rental_stock_NC_initial_HE, coastal_beq_initial_HE, noncoastal_beq_initial_HE, savings_beq_initial_HE, cfg.path_until_experiment)
    dP_C_2026=price_history[-2,0]
    dP_NC_2026=price_history[-2,1]
    coastal_beq_2026=vcoastal_beq[-1]
    noncoastal_beq_2026=vnoncoastal_beq[-1]
    savings_beq_2026=vsavings_beq[-1]
    
    #From this point, we are simulating forward from the experiment year, so delete initial distributions
    del mDist0_c_initial_HE, mDist0_nc_initial_HE, mDist0_renter_initial_HE 

    
    "find coefficients for two experiments using correct initial distributions (2026)"
    _, _, vCoeff_C_BR, vCoeff_NC_BR, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_BR_guess, vCoeff_NC_BR_guess,dP_C_2026, dP_NC_2026,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, coastal_beq_2026, noncoastal_beq_2026, savings_beq_2026,cfg.find_coeff_buildingrest)
    _save("vCoeff_C_BR", vCoeff_C_BR); _save("vCoeff_NC_BR", vCoeff_NC_BR)
    "find coefficients for two experiments using correct initial distributions (2026)"
    _, _, vCoeff_C_MP, vCoeff_NC_MP, _, _, _, _, _, _, _=equil.find_coefficients(par, grids, vCoeff_C_MP_guess, vCoeff_NC_MP_guess,dP_C_2026, dP_NC_2026,mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026, rental_stock_C_2026, rental_stock_NC_2026, coastal_beq_2026, noncoastal_beq_2026, savings_beq_2026,cfg.find_coeff_mortgageprem)
    _save("vCoeff_C_MP", vCoeff_C_MP); _save("vCoeff_NC_MP", vCoeff_NC_MP)
    
    del mDist1_c_2026, mDist1_nc_2026, mDist1_renter_2026
    
    "find coefficients for 4 different terminal steady states"
    vCoeff_C_terminal_RE_guess = _g("vCoeff_C_terminal_RE")
    vCoeff_NC_terminal_RE_guess = _g("vCoeff_NC_terminal_RE")
    vCoeff_C_terminal_HE_guess = _g("vCoeff_C_terminal_HE")
    vCoeff_NC_terminal_HE_guess = _g("vCoeff_NC_terminal_HE")
    vCoeff_C_terminal_BR_guess = _g("vCoeff_C_terminal_BR")
    vCoeff_NC_terminal_BR_guess = _g("vCoeff_NC_terminal_BR")
    vCoeff_C_terminal_MP_guess = _g("vCoeff_C_terminal_MP")
    vCoeff_NC_terminal_MP_guess = _g("vCoeff_NC_terminal_MP")
    ## config solve_terminal_ss_baseline
    vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, _, _, _, _, _, _, _, _   = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_HE_guess, vCoeff_NC_terminal_HE_guess, cfg.solve_terminal_ss_HE)
    _save("vCoeff_C_terminal_HE", vCoeff_C_terminal_HE); _save("vCoeff_NC_terminal_HE", vCoeff_NC_terminal_HE)

    ## config solve_terminal_ss_RE
    vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_RE_guess, vCoeff_NC_terminal_RE_guess, cfg.solve_terminal_ss_RE)
    _save("vCoeff_C_terminal_RE", vCoeff_C_terminal_RE); _save("vCoeff_NC_terminal_RE", vCoeff_NC_terminal_RE)

    ## config solve_terminal_ss_building_rest
    vCoeff_C_terminal_BR, vCoeff_NC_terminal_BR, _, _, _, _, _, _, _, _  = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_BR_guess, vCoeff_NC_terminal_BR_guess, cfg.solve_terminal_ss_building_rest)
    _save("vCoeff_C_terminal_BR", vCoeff_C_terminal_BR); _save("vCoeff_NC_terminal_BR", vCoeff_NC_terminal_BR)

    ## config solve_terminal_ss_mortgage_premium
    vCoeff_C_terminal_MP, vCoeff_NC_terminal_MP, _, _, _, _, _, _, _, _ = equil.initialise_coefficients_ss(par, grids, vCoeff_C_terminal_MP_guess, vCoeff_NC_terminal_MP_guess, cfg.solve_terminal_ss_mortgage_premium)
    _save("vCoeff_C_terminal_MP", vCoeff_C_terminal_MP); _save("vCoeff_NC_terminal_MP", vCoeff_NC_terminal_MP)
    
    "export every solved coefficient set to Excel (labels match the guess loader)"
    coeff_io.save_coefficients({
        "vCoeff_C_initial_HE": vCoeff_C_initial_HE, "vCoeff_NC_initial_HE": vCoeff_NC_initial_HE,
        "vCoeff_C_initial_RE": vCoeff_C_initial_RE, "vCoeff_NC_initial_RE": vCoeff_NC_initial_RE,
        "vCoeff_C_HE": vCoeff_C_HE, "vCoeff_NC_HE": vCoeff_NC_HE,
        "vCoeff_C_RE": vCoeff_C_RE, "vCoeff_NC_RE": vCoeff_NC_RE,
        "vCoeff_C_BR": vCoeff_C_BR, "vCoeff_NC_BR": vCoeff_NC_BR,
        "vCoeff_C_MP": vCoeff_C_MP, "vCoeff_NC_MP": vCoeff_NC_MP,
        "vCoeff_C_terminal_HE": vCoeff_C_terminal_HE, "vCoeff_NC_terminal_HE": vCoeff_NC_terminal_HE,
        "vCoeff_C_terminal_RE": vCoeff_C_terminal_RE, "vCoeff_NC_terminal_RE": vCoeff_NC_terminal_RE,
        "vCoeff_C_terminal_BR": vCoeff_C_terminal_BR, "vCoeff_NC_terminal_BR": vCoeff_NC_terminal_BR,
        "vCoeff_C_terminal_MP": vCoeff_C_terminal_MP, "vCoeff_NC_terminal_MP": vCoeff_NC_terminal_MP,
    })