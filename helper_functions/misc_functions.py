"""
misc_functions.py
"""

import numpy as np
import numba as nb
from numba import njit

###########################################################
### mAH =  DoubleGrid(vA, vH)
@njit
def DoubleGrid(vA, vH):
    """
    Returns an (iNa*iNh) x 2 matrix containing all possible combinations 
    of savings (first column) and house values (second column).
    """
    iNa = vA.shape[0]
    iNh = vH.shape[0]
    N   = iNa * iNh

    # Allocate the output grid
    mAH = np.empty((N, 2), dtype=vA.dtype)

    # Fill it: for each house-value index j and saving-index i,
    # place (vA[i], vH[j]) at row idx = j*iNa + i
    for j in range(iNh):
        for i in range(iNa):
            idx = j * iNa + i
            mAH[idx, 0] = vA[i]
            mAH[idx, 1] = vH[j]

    return mAH

###########################################################
### Interpolation functions
@njit
def lininterp_zero_crossing(x_min, y_min, x_max, y_max):
    # Calculate the x value where the line between (x1, y1) and (x2, y2) crosses y=0
    x_zero = x_min + (- y_min) * (x_max - x_min) / (y_max - y_min)
    return x_zero


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

@njit(fastmath=True)
def _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4):
    """ 4d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid

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

    nom_4_left = grid4[j4+1]-xi4
    nom_4_right = xi4-grid4[j4]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right  
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right  
                    nom += nom_1*nom_2*nom_3*nom_4*value[j1+k1,j2+k2,j3+k3,j4+k4]

    return nom/denom

@njit(fastmath=True)
def interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4):
    """ 4d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)
    j4 = binary_search(0,grid4.size,grid4,xi4)

    return _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4)

#################
# binary search #
#################

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

###########################################################
### vMaxRow, vArgMaxRow= maxRow(mMatrix)
@njit
def maxRow(mMatrix):
    """Returns the MaxRow of a matrix.

    Args:
        mMatrix (matrix, float): input matrix

    Returns:
        vMaxRow (vector, float): Maxrow vector
        vArgMaxRow (vector, integer): argument of max row in each col
    """
    iCols = mMatrix.shape[1]
    vMaxRow = np.zeros(iCols, dtype = nb.float64)
    vArgMaxRow = np.zeros(iCols, dtype = nb.int64)
    for i in range(iCols):
        vMaxRow[i] = np.max(mMatrix[:,i])
        vArgMaxRow[i] = np.argmax(mMatrix[:,i])
    return vMaxRow, vArgMaxRow

###########################################################
### par = construct_jitclass(dictionary):
def construct_jitclass(dictionary):
    """
    Convert dictionary into numba object.

    Args:
        dictionary (dictionary): Dictionary with all model parameters or other objects.

    Returns:
        par (numba object): Numba object with all model parameters or other objects.
    """

    parlist = [(key, nb.typeof(val)) for key, val in dictionary.items()]

    @nb.experimental.jitclass(parlist)
    class ParClass:
        def __init__(self):
            pass

    par = ParClass()
    for key, val in dictionary.items():
        setattr(par, key, val)

    return par

###########################################################
### 2D inteprolation

@njit(fastmath=True)
def _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2):
    """ 2d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right                    
            nom += nom_1*nom_2*value[j1+k1,j2+k2]

    return nom/denom

@njit(fastmath=True)
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ 2d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)

#################
# binary search #
#################

@njit(fastmath=True)
def binary_search_sim(imin,Nx,x,xi):

    # a. checks
    if xi <= x[imin]:
        imin=binary_search(0,Nx,x,xi)
    elif xi >= x[Nx-2]:
        return Nx-2    
    elif xi <= x[imin+1]:
        return imin
    elif xi <= x[imin+2]:
        return imin+1
    else:
        imin=binary_search(0,Nx,x,xi) 
        
    return imin

@njit
def ols_numba(X, Y):
    """Compute OLS estimates using (X'X)^-1 X'Y, Numba optimized."""
    XtX = X.T @ X 
    XtY = X.T @ Y  
    beta = np.linalg.solve(XtX, XtY)  
    return beta


@njit 
def net_income(par, grids, j, e_index, e_trans_index, mortgage_size, mortgage_rate):
    if j<par.j_ret:
        pretax_income=np.exp(grids.vChi[j] + grids.vE[e_index]+ grids.vE_trans[e_trans_index])
    else:
        pretax_income=0.7*np.exp(grids.vChi[par.j_ret-1] + grids.vE[e_index])
    posttax_income=pretax_income-par.tau_0*(max(pretax_income-mortgage_rate*mortgage_size*grids.median_inc,0))**(1-par.tau_1)
    mortgage_rebate=par.tau_0*(pretax_income)**(1-par.tau_1)-(pretax_income-posttax_income)
    ##Normalise with pre-tax median income
    net_income=(posttax_income/grids.median_inc)*(1-par.wf_wedge[0])    
    mortgage_rebate=(mortgage_rebate/grids.median_inc)*(1-par.wf_wedge[0])
        
    return net_income, mortgage_rebate