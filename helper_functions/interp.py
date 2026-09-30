"""
interp.py
"""

import numpy as np
from numba import njit

@njit(fastmath=True)
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(fastmath=True)
def _interp_1d(grid1,value,xi1,j1):
    """ 1d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point
        j1 (int): location in grid 

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right                    
        nom += nom_1*value[j1+k1]

    return nom/denom

@njit(fastmath=True)
def interp_1d(grid1,value,xi1):
    """ 1d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (1d)
        xi1 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)

    return _interp_1d(grid1,value,xi1,j1)

@njit(fastmath=True)
def _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3):
    """ 3d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right               
                nom += nom_1*nom_2*nom_3*value[j1+k1,j2+k2,j3+k3]

    return nom/denom

@njit(fastmath=True)
def interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3):
    """ 3d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)

    return _interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3,j1,j2,j3)

@njit
def fast_interp_all(x, x_grid, y_matrix):
    n_y = y_matrix.shape[1]
    result = np.empty(n_y)
    
    # Find bracketing indices for x
    idx = np.searchsorted(x_grid, x) - 1
    idx = max(0, min(idx, x_grid.size - 2))  # ensure bounds

    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    dx = x1 - x0
    if dx == 0:
        weight = 0.0
    else:
        weight = (x - x0) / dx

    for j in range(n_y):
        y0 = y_matrix[idx, j]
        y1 = y_matrix[idx + 1, j]
        result[j] = y0 + weight * (y1 - y0)

    return result