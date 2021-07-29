# This code is a minor modification of the following script based project
#
#### --> https://github.com/dqsis/parsec-airfoils <-- ####
#
# by Dimitrios Kiousis.
#
# The repository containing the script is posted under the
# Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License 
# http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US 
# As such, this code code is also subject to this license
#
# Only minor modifications have been made, addressing the fact that no script is run
# and that it generates 2xN arrays for airfoil coordinates that end at x/c=1.0

# Import libraries
from math import sqrt, tan, pi
import numpy as np

# User function pcoef
def pcoef(
        xte,yte,rle,
        x_cre,y_cre,d2ydx2_cre,th_cre,
        pressure_side):
    

    # Docstrings
    """evaluate the PARSEC coefficients"""


    # Initialize coefficients
    coef = np.zeros(6)


    # 1st coefficient depends on surface (pressure or suction)
    if pressure_side:
        coef[0] = -sqrt(2*rle)
    else:
        coef[0] = sqrt(2*rle)
 
    # Form system of equations
    A = np.array([
                 [xte**1.5, xte**2.5, xte**3.5, xte**4.5, xte**5.5],
                 [x_cre**1.5, x_cre**2.5, x_cre**3.5, x_cre**4.5, 
                  x_cre**5.5],
                 [1.5*sqrt(xte), 2.5*xte**1.5, 3.5*xte**2.5, 
                  4.5*xte**3.5, 5.5*xte**4.5],
                 [1.5*sqrt(x_cre), 2.5*x_cre**1.5, 3.5*x_cre**2.5, 
                  4.5*x_cre**3.5, 5.5*x_cre**4.5],
                 [0.75*(1/sqrt(x_cre)), 3.75*sqrt(x_cre), 8.75*x_cre**1.5, 
                  15.75*x_cre**2.5, 24.75*x_cre**3.5]
                 ]) 

    B = np.array([
                 [yte - coef[0]*sqrt(xte)],
                 [y_cre - coef[0]*sqrt(x_cre)],
                 [tan(th_cre) - 0.5*coef[0]*(1/sqrt(xte))],
                 [-0.5*coef[0]*(1/sqrt(x_cre))],
                 [d2ydx2_cre + 0.25*coef[0]*x_cre**(-1.5)]
                 ])
    

    # Solve system of linear equations
    X = np.linalg.solve(A,B) 


    # Gather all coefficients
    coef[1:6] = X[0:5,0]


    # Return coefficients
    return coef

"""
Export PARSEC airfoil in plain coordinate format, for use with e.g. XFOIL.
No file saving is performed, that is left to the user.
"""

def ppoints(cf_pre, cf_suc, npts=121, xte=1.0):
    '''
    Takes PARSEC coefficients, number of points, and returns list of
    [x,y] coordinates starting at trailing edge pressure side.
    Assumes trailing edge x position is 1.0 if not specified.
    Returns 121 points if 'npts' keyword argument not specified.
    '''
    # Using cosine spacing to concentrate points near TE and LE,
    # see http://airfoiltools.com/airfoil/naca4digit
    xpts = (1 - np.cos(np.linspace(0.0, 1.0, int(np.ceil(npts/2)))*pi)) / 2
    # Take TE x-position into account
    xpts *= xte

    # Powers to raise coefficients to
    pwrs = (1/2, 3/2, 5/2, 7/2, 9/2, 11/2)
    # Make [[1,1,1,1],[2,2,2,2],...] style array
    xptsgrid = np.meshgrid(np.arange(len(pwrs)), xpts)[1]
    # Evaluate points with concise matrix calculations. One x-coordinate is
    # evaluated for every row in xptsgrid
    evalpts_pre = np.sum(cf_pre*xptsgrid**pwrs, axis=1)
    evalpts_suc = np.sum(cf_suc*xptsgrid**pwrs, axis=1)
    # Move into proper order: start at TE, over bottom, then top
    # Avoid leading edge pt (0,0) being included twice by slicing [1:]
    xcoords = np.r_[xpts[::-1], xpts[1:]]
    ycoords = np.r_[evalpts_pre[::-1], evalpts_suc[1:]]
    # Return 2D list of coordinates [[x,y],[x,y],...] by transposing .T
    return np.c_[xcoords,ycoords].T # Adapted to result in 2xN array

# Assume xte = 1.0
def parsec_airfoil(N,yte, rle,
                   x_pre, y_pre, d2ydx2_pre, th_pre,
                  x_suc, y_suc, d2ydx2_suc, th_suc):
    xte = 1.0
    # Evaluate pressure (lower) surface coefficients
    cf_pre = pcoef(xte, yte, rle,
                      x_pre, y_pre, d2ydx2_pre, th_pre,
                      True)
    # Evaluate suction (upper) surface coefficients
    cf_suc = pcoef(xte, yte, rle,
                      x_suc, y_suc, d2ydx2_suc, th_suc,
                      False)
    pts = ppoints(cf_pre, cf_suc, N, xte=xte)
    return pts