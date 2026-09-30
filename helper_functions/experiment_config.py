solve_initial_ss_HE = {'sceptics': True,                    
                    'welfare': True,
                    'building_rest': False,
                    'mortgage_premium': False,
                    'initial': True}

solve_initial_ss_RE = {'sceptics': False,                    
                    'welfare': True,
                    'building_rest': False,
                    'mortgage_premium': False,
                    'initial': True}

solve_terminal_ss_HE  = {'sceptics': True,
                    'welfare': False,
                    'building_rest': False,
                    'mortgage_premium': False,
                    'initial': False}

solve_terminal_ss_RE = {'sceptics': False,
                    'welfare': False,
                    'building_rest': False,
                    'mortgage_premium': False,
                    'initial': False}

solve_terminal_ss_building_rest = {'sceptics': True,
                    'welfare': False,
                    'building_rest': True,
                    'mortgage_premium': False,
                    'initial': False}

solve_terminal_ss_mortgage_premium = {'sceptics': True,
                    'welfare': False,
                    'building_rest': False,
                    'mortgage_premium': True,
                    'initial': False}

transition_path = {'sceptics': True,
            'stop_for_experiment': False,
            'welfare': True,
            'record_dist': True,
            'building_rest': False,
            'mortgage_premium': False,
            'run_experiment': False}

path_until_experiment= {'sceptics': True,
            'stop_for_experiment': True,
            'welfare': False,
            'record_dist': False,
            'building_rest': False,
            'mortgage_premium': False,
            'run_experiment': False}

transition_path_RE = {'sceptics': False,
            'stop_for_experiment': False,
            'welfare': True,
            'record_dist': True,
            'building_rest': False,
            'mortgage_premium': False,
            'run_experiment': False}


experiment_building_rest = {'sceptics': True,
                            'stop_for_experiment': False,
                            'welfare': True,
                            'record_dist': True,
                            'building_rest': True,
                            'mortgage_premium': False,
                            'run_experiment': True}

experiment_mortgage_prem = {'sceptics': True,
                            'stop_for_experiment': False,
                            'welfare': True,
                            'record_dist': True,
                            'building_rest': False,
                            'mortgage_premium': True,
                            'run_experiment': True}

find_coeff_path_HE = {'sceptics': True,
                        'stop_for_experiment': False,
                        'welfare': False,
                        'record_dist': False,
                        'building_rest': False,
                        'mortgage_premium': False,
                        'run_experiment': False}

find_coeff_path_RE = {'sceptics': False,
                'stop_for_experiment': False,
                'welfare': False,
                'record_dist': False,
                'building_rest': False,
                'mortgage_premium': False,
                'run_experiment': False}

find_coeff_buildingrest = {'sceptics': True,
                        'stop_for_experiment': False,
                        'welfare': False,
                        'record_dist': False,
                        'building_rest': True,
                        'mortgage_premium': False,
                        'run_experiment': True}

find_coeff_mortgageprem = {'sceptics': True,
                        'stop_for_experiment': False,
                        'welfare': False,
                        'record_dist': False,
                        'building_rest': False,
                        'mortgage_premium': True,
                        'run_experiment': True}
