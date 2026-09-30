"""
grids.py

Purpose:
Create grids (from Druedahl)
"""
###########################################################
### Imports
import numpy as np
from numba import njit

###########################################################
### Functions
###########################################################


#######################################################
### function to create non linear grid
@njit
def nonlinspace_jit(x_min,x_max,n,phi):
    """ like np.linspace. but with unequal spacing

    Args:

    x_min (double): minimum value
    x_max (double): maximum value
    n (int): number of points
    phi (double): phi = 1 -> eqaul spacing, phi up -> more points closer to minimum

    Returns:

    y (list): grid with unequal spacing

    """
        
    y = np.zeros(n)

    y[0] = x_min
    for i in range(1,n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi 
    
    return y

#######################################################
### function to create equidistant log space grid
def equilogspace(x_min,x_max,n):
    """ like np.linspace. but (close to) equidistant in logs

    Args:

        x_min (double): maximum value
        x_max (double): minimum value
        n (int): number of points
    
    Returns:

        y (list): grid with unequal spacing

    """

    pivot = np.abs(x_min) + 0.25
    y = np.geomspace(x_min + pivot, x_max + pivot, n) - pivot
    y[0] = x_min  # make sure *exactly* equal to x_min
    return y
