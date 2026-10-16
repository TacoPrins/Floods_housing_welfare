
import numpy as np

time_increment=2
vPi_S_median=np.array([0.0194,    0.0198,    0.0202,    0.0206,    0.0210,    0.0214,    0.0218,    0.0222,    0.0226,    0.0230,    0.0234,    0.0239,    0.0243,
                       0.0248,    0.0254,    0.0259,    0.0265,    0.0273,    0.0280,   0.0289,    0.0300,    0.0310,    0.0321,    0.0333,    0.0347,    0.0361,
0.0376,    0.0392,    0.0410,    0.0427,    0.0444,    0.0461,    0.0478,    0.0495,    0.0513,    0.0530,    0.0547,    0.0565,    0.0583,
0.0601,    0.0619,    0.0637,    0.0654,    0.0672,    0.0690,    0.0708,    0.0726,    0.0744,    0.0762,    0.0780,    0.0798,    0.0816,    0.0816,    0.0816,    0.0816,    0.0816,  0.0816,    0.0816])

vPi_S_median=1-(1-vPi_S_median)**time_increment    


par_dict = {"time_increment": time_increment,
          "iNj": 30,
          "j_ret": 23,
          #"dBeta": 9.4132422e-01**time_increment, 
          "dBeta": 9.42388105e-01**time_increment, 
          "dDelta": 1-(1-0.015)**time_increment, 
          #"dPsi": 1.16789062e-02,
          "dPsi": 1.22747018e-02,
          "dDelta_deprec_rental": 1-(1-0.015)**time_increment,
          "dDelta_default": 0,
          "r": 1.03**time_increment-1, 
          "r_m_c": 1.04**time_increment-1, 
          'r_m_c_experiment': 1.06**time_increment-1, 
          "r_m_nc": 1.04**time_increment-1, 
          'vPi_S_median': vPi_S_median,
          "dKappa_sell": 0.07,
          "dKappa_buy": 0,
          "dXi_foreclosure": 0.8,
          #"dNu": 4.02843750e+01,
          "dNu": 4.59859461e+01,
          "dZeta": 0.01, 
          "dZeta_fixed": 1/26, 
          "lambda_pti":0.25,
          "max_ltv": 0.95,
          "damage_states": 3,
          "dLambda": 0.8,
          'wf_wedge': np.array([0.]),
          "dGamma": 1/1.25,
          "dSigma": 2,
          #"b_bar":2.79152344e+00,
          "b_bar":3.69412935e+00,
          #"dPhi":  1.38917969e-01,
          "dPhi": 1.56281058e-01,
          "nonlingrid": 1,
          "nonlingrid_big": 1,  
          "iNb":70,
          "iBmin": 0, 
          "iBmax": 27,
          "dZ":0.8,
          "h_max": 5.15,
          #"dXi_min":1-4.32468750e-02,          
          #"dXi_max": 1+4.32468750e-02,
          "dXi_min":1-4.65839810e-02,          
          "dXi_max": 1+4.65839810e-02,
          #"iXin":7,
          "iXin": 5, 
          "alpha_0": 0.4,
          "dRho": 0.97, 
          "dSigmaeps": 0.20, 
          "dSigmaeps_trans": 0.05, 
          "iNumStates":5, #THIS NUMBER MUST BE UNEVEN
          "iNumTrans":3, 
          "iM":1,
          'vAgeEquiv':np.ones(50),
          'dNC_frac': 0.5,
          'dC_frac': 0.5,
          'dTheta': 1.5/2.5, 
          'dL': 0.311,
          #'dOmega': 2.41390625e-03,
          'dOmega':5.47901061e-03,
          'dMitigation_rate': 0.55,
          'building_rest_cost': 0.14,
          'sd_income_initial': .3228617,
          'beta0_nowealth': -.9910767, #=Constant in logit regression nowealth=1 on log income in lowest age bracket
          'beta1_nowealth': -.3417672, #=Coeff on log income (in terms of sds away from the median) in logit regression nowealth=1 on log income in lowest age bracket
          'var_poswealth': 1.557037, #=Variance log wealth conditional on wealth>0 in lowest age bracket
          'beta0_age': 2.0017337, #=Constant of fitted line age --> mean log income 
          'beta1_age': .78795906  , #=Coeff of age for fitted line age --> mean log income 
          'beta2_age': -.02717023, #=Coeff of age^2 for fitted line age --> mean log income  
          'beta3_age': .00042535, #=Coeff of age^3 for fitted line age --> mean log income 
          'beta4_age':  -2.505727467e-06, #=Coeff of age^4 for fitted line age --> mean log income 
          'corr_poswealth_income': 0.1984, #=Correlation between log income (in terms of sds away from the median) and log wealth conditional on wealth>0 in lowest age bracket
          'tau_0': 0.75,
          'tau_1': 0.151,
          'building_rest': 0.5,
          'starting_year': 1998,
          'experiment_year': 2026,
          'order_polynomial': 4, #The order of the polynomial in the LoM for prices
          }