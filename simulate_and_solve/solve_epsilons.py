"""
solve.py

Purpose:
    Solve the model
"""
###########################################################
### Imports
import solve_model as solve_model
import collect_results
import grid_creation
import par_epsilons as parfile
import misc_functions as misc 
###########################################################
### main
def main():
    plot_distribution=False
    calculate_welfare=True   
    only_RE_welfare=True
    welfare_out = collect_results.collect_results(plot_distribution, calculate_welfare, only_RE_welfare)

  
###########################################################

### start main
if __name__ == "__main__":
    main()
