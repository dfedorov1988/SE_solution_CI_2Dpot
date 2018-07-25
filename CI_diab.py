'''
Created on Jul 31, 2017

@author: Dmitry A. Fedorov
'''
#!/usr/local/bin/python3.6
import sys
import os, ctypes
import time
import pdb
from sympy import *
import numpy as np
import math
import mpmath
import scipy.special as sp 
from scipy import linalg as lin
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad, fixed_quad
from numpy.lib.scimath import *
from numpy.ma.bench import m1
from mpmath.libmp.libmpf import h_mask
import matplotlib.pyplot as plt
from misc import transformMatrix, isHermitian, psi, calculateDensity, densityAtSinglePoint, printMatrix, V11, V11_2, V22
    
def mainDiabatic(max_m, max_n, zeros, b, L, c, delta, reducedBasis, representation, stateToCompute):
    """Main module for diabatic rep. Constructs and diagonalizes Hamiltonian"""  
    ini = time.process_time()
    start = time.process_time()
    dim = 2 * max_m * max_n + max_n 
    
    print("\nNumber of basis functions = {}\n".format((max_m+1) * max_n))
    
    T = calcKinEnDiabaticRep(max_m, max_n, b, zeros, dim) # Building kinetic energy matrix
    print("Calculating potential energy:\n")
    # Buidling potential energy matrix elements
    V, transformationMatrix, coeff, cs = calcPotEnDiabaticRep(max_m, max_n, b, zeros, dim, L, c, delta) 
    end = time.process_time()
    print("Potential Energy Matrix Elements were calculated in {0:.2f} s\n".format(end-start))

#    Building Hamiltonian
    H = T + V
    
    if reducedBasis == 0: # Do a calculation using full basis
        finalH = H
        
    elif reducedBasis == 1: # A calculation without the basis function which is 0 at the CI  
        HT =  transformMatrix(transformationMatrix, H)                                     
        finalH = HT
        print(np.shape(H), np.shape(HT))
    E, eigenvecs = lin.eigh(finalH)        
    for eLevel in range(6): print("{0:1.8f}".format(E[eLevel]))
    
#    Density at the intersection point
    rCoordinate = delta / L # the x coordinate of the conical intersection
    phiCoordinate = np.pi

    """Calculating density"""
 
    if (reducedBasis == 1):
        expansionCoefficientsTransformed = np.dot(transformationMatrix, eigenvecs)
        totalDensity = densityAtSinglePoint(rCoordinate, phiCoordinate, b, dim, \
                                      zeros, expansionCoefficientsTransformed, coeff, transformationMatrix, representation, stateToCompute) 
        calculateDensity(b, dim, zeros, expansionCoefficientsTransformed, coeff, transformationMatrix, representation, stateToCompute)
#         """Analysis of the components of the transformed Hamiltonian in terms of 
#         original basis functions"""
# #         np.set_printoptions(precision=2) 
#         coeff1 = np.chararray(2*dim, itemsize = 15)
#         for i in range(2*dim):
#             sign = lambda a: '+' if a > 0 else '-' if a < 0 else ' '
#             coeff1[i] = str(round(np.abs(expansionCoefficientsTransformed[i, stateToCompute]), 3)) + "(" + str(coeff[i][0]) + "," +  str(coeff[i][1]) + ")" + cs[i] 
#         coeff1 = coeff1.decode("utf-8")
#         print("\nThe following m, n contribute to the wave function:")       
#         for j in range(1):
#             significantIndices = np.abs(expansionCoefficientsTransformed[:,j]) > 0.00001
#             string = list((np.unique(coeff1[significantIndices])))
#             joinedString = " ".join(string)
#             print("state {}: {}".format(j, joinedString))
         
    else:
        totalDensity = densityAtSinglePoint(rCoordinate, phiCoordinate, b, dim, \
                                      zeros, eigenvecs, coeff, transformationMatrix, representation, stateToCompute) 
        calculateDensity(b, dim, zeros, eigenvecs, coeff, transformationMatrix, representation, stateToCompute)

#         np.set_printoptions(precision=2) 
        coeff1 = np.chararray(2*dim, itemsize = 21)
        for i in range(2*dim):
            sign = lambda a: '+' if a > 0 else '-' if a < 0 else ' '
            coeff1[i] = str(round(np.abs(eigenvecs[i, stateToCompute]), 3)) + "(" + str(coeff[i][0]) + "," +  str(coeff[i][1]) + ")" + cs[i] 
        coeff1 = coeff1.decode("utf-8")
        print("\nThe following m, n contribute to the wave function:")       
        for j in range(1):
            significantIndices = np.abs(eigenvecs[:,j]) > 0.00001
            string = list((np.unique(coeff1[significantIndices])))
            joinedString = " ".join(string)
            print("state {}: {}".format(j, joinedString))

    print("Total density at CI = {}".format(totalDensity))
    
    last = time.process_time()
    print("\n ----- This calculation time is {0:.2f} s----- ".format(last-ini)) 
    return(E)    

def calcPotEnDiabaticRep (max_m, max_n, b, zeros, dim, L, c, delta):
    """ Potential energy in diabatic representation.
    Integration of complex variables is not trivial using SciPy,
    so the integration of angular part was performed in Mathematica,
    here we only perform integration of the radial part: 
    V11 matrix elements are non-zero in 2 cases:
    1) m1=m2, any n1, n2. In this case the V11 function is used 
    2) m1=m2+1, any n1, n2. V11_2 function is used 
    This function returns two matrices: 
    diagonal potential energy MEs V11=V22,
    off-diagonal potential energy MEs V12"""
            
    def A (n1, n2, b, zeros): 
        """tranformation matrix getting rid of the basis function that is 0 at CI"""
        
        result = 2 / (np.square(b) * sp.jv(1, zeros[0, n1-1]) * sp.jv(1, zeros[0, n2-1]))
        return(result)
    
    V = np.zeros((2*dim, 2*dim), dtype=np.complex128)
    VT = np.zeros((2*dim, 2*dim))
    Reduced_VT = np.zeros ((2*dim, 2*(dim-1)))
    
    """ Low-level callback routine which uses c library to speed up integration using quad
    V11: psi(r, m1, n1) * psi(r, m2, n2) * r * (L^2/4 + r**2) / 2 (substitute for V11)
    V11_offDiag: psi(r, m1, n1) * psi(r, m2, n2) * L * r**2 / 2 (substitute for V11_2)
    V22:psi(r, m1, n1) * psi(r, m2, n2) * r * (L**2 + 4*r**2 - 8*delta ) / 4 / 2 )"""
    
    V11_cfunc = ctypes.CDLL(os.path.abspath('V11.so'))
    V11_cfunc.f.restype = ctypes.c_double
    V11_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    V11_2_cfunc = ctypes.CDLL(os.path.abspath('V11_2.so'))
    V11_2_cfunc.f.restype = ctypes.c_double
    V11_2_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    V22_cfunc = ctypes.CDLL(os.path.abspath('V22.so'))
    V22_cfunc.f.restype = ctypes.c_double
    V22_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    mn1 = ctypes.c_double * 7 # array to pass to the c library (m1, n1, m2, n2, b, L, delta)   
    kk = 0
        
    coeff = [[0] * dim for i in range(2 * dim)] # this array allows to convert k to m,n
    cs = [[] for i in range(2*dim)]
    for m1 in range(-max_m, 1):
        for m2 in range(-max_m, 1):
            for n1 in range(1, (max_n+1)):
                for n2 in range(n1, (max_n+1)):
                    
                    k1 = (m1 + max_m) * max_n + n1
                    k2 = (m2 + max_m) * max_n + n2
                    k11 = k1 + 2 * abs(m1) * max_n
                    k22 = k2 + 2 * abs(m2) * max_n
                    coeff[k1-1][0] = m1
                    coeff[k1-1][1] = n1
                    coeff[k11-1][0] = -m1
                    coeff[k11-1][1] = n1
                    coeff[k1-1+dim][0] = m1
                    coeff[k1-1+dim][1] = n1
                    coeff[k11-1+dim][0] = -m1
                    coeff[k11-1+dim][1] = n1
                    
                    cs[k1-1], cs[k1-1+dim] = 'A', 'B'
                    cs[k11-1], cs[k11-1+dim] = 'A', 'B'
                    
                    mn = mn1(abs(m1), n1, abs(m2), n2, b, L, delta)
                    user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)

                    if m1 == m2: # m1 = m2, using V11 integral (see Mathematica notebook) 
                        if V[k1-1, k2-1] == 0.0:
                            func = LowLevelCallable(V11_cfunc.f, user_data)
                            V[k1-1, k2-1], error = quad (func, 0, b, epsrel=0, \
                                                            epsabs=1e-12, limit=10000, maxp1=500, limlst=500)
#                             V[k1-1, k2-1], error = fixed_quad (V11, 0, b, args=(m1, n1, m2, n2, L, b, zeros), n = 1000)
                            kk += 1
                            V[k2-1, k1-1] = V[k1-1, k2-1]
                            
                            # Due to symmetry the MEs for m and -m are equal
                            if k11 <= dim and k22 <= dim:
                                V[k22-1, k11-1] = V[k1-1, k2-1]
                                V[k11-1, k22-1] = V[k1-1, k2-1]
                            
                            if m1 == 0 and m2 == 0: 
                                # Constructing matrix to throw away the bf which is 0 at CI
                                VT[k1-1, k2-1] = A(n1, n2, b, zeros)
                                VT[k1-1+dim, k2-1+dim] = A(n1, n2, b, zeros)
                                
                                if VT[k2-1, k1-1]==0:
                                    VT[k2-1, k1-1] = VT[k1-1, k2-1]
                                    VT[k2-1+dim, k1-1+dim] = VT[k1-1+dim, k2-1+dim]
                                    
                        if V[k1-1+dim, k2-1+dim] == 0.0:
                            if delta != 0.0:
                                func = LowLevelCallable(V22_cfunc.f, user_data)
                                V[k1-1+dim, k2-1+dim], error = quad (func, 0, b, epsrel = 0, \
                                                                epsabs = 1e-12, limit = 10000, maxp1=500, limlst=500)

                                kk += 1

                            else:
                                V[k1-1+dim, k2-1+dim] = V[k1-1, k2-1]
                            
                            V[k2-1+dim, k1-1+dim] = V[k1-1+dim, k2-1+dim]
                            
                            """Due to symmetry the MEs for m and -m are equal"""
                            if k11 <= dim and k22 <= dim:
                                V[k22-1+dim, k11-1+dim] = V[k1-1+dim, k2-1+dim]
                                V[k11-1+dim, k22-1+dim] = V[k1-1+dim, k2-1+dim]
                                
                    elif abs(m2-m1) == 1: # m2 = m1 + 1
                        if V[k1-1, k2-1] == 0.0:
                            kk = kk+1 
                            func = LowLevelCallable(V11_2_cfunc.f, user_data)
                            V[k1-1, k2-1], error = quad (func, 0, b, epsrel = 0, \
                                                            epsabs = 1e-12, limit = 10000, maxp1=500, limlst=500)
#                             V[k1-1, k2-1], error = fixed_quad (V11_2, 0, b, args=(m1, n1, m2, n2, L, b, zeros), n = 1000)

                            V[k2-1, k1-1] = V[k1-1, k2-1]
                            
                            V[k2-1+dim, k1-1+dim] = -V[k1-1, k2-1]
                            V[k1-1+dim, k2-1+dim] = -V[k1-1, k2-1]
                            
                            """Due to symmetry the MEs for m and -m are equal"""
                            if k11 <= dim and k22 <= dim:
                                V[k22-1, k11-1] = V[k1-1, k2-1]
                                V[k11-1, k22-1] = V[k1-1, k2-1]
                                
                                V[k22-1+dim, k11-1+dim] = -V[k1-1, k2-1]
                                V[k11-1+dim, k22-1+dim] = -V[k1-1, k2-1]
                           
                            """Off-diagonal matrix elements, V12, are nonzero \
                            when m1 = m2+1 or m1 = m2-1 for any n1,n2
                            V12= 2 * c / L i * V11_2, so we don't need to calculate it"""
                            V[k1-1, k2-1+dim] = 2 * 1j * c / L * V[k1-1, k2-1] * (m2-m1)
                            V[k11-1, k22-1+dim] = -V[k1-1, k2-1+dim]
                            V[k22-1, k11-1+dim] = V[k1-1, k2-1+dim]
                            V[k2-1, k1-1+dim] = -V[k1-1, k2-1+dim]
                            
                            V[k2-1+dim, k1-1] = -V[k1-1, k2-1+dim]
                            V[k22-1+dim, k11-1] = -V[k11-1, k22-1+dim]
                            V[k11-1+dim, k22-1] = -V[k22-1, k11-1+dim]
                            V[k1-1+dim, k2-1] = -V[k2-1, k1-1+dim]
                            
                        if V[k1-1+dim, k2-1+dim] == 0.0:
                            V[k1-1+dim, k2-1+dim] = -V[k1-1, k2-1]
                            V[k2-1+dim, k1-1+dim] = V[k1-1+dim, k2-1+dim]
                            
                            # Due to symmetry the MEs for m and -m are equal
                            if k11 <= dim and k22 <= dim:
                                V[k22-1+dim, k11-1+dim] = V[k1-1+dim, k2-1+dim]
                                V[k11-1+dim, k22-1+dim] = V[k1-1+dim, k2-1+dim]
                           
    """The following section has a transformation of basis to get rid of the function 
    that is zero at the intersection. This will allow to show that the density at
    the intersection can be nonzero, and that it lowers the energy of the system"""

    eigenvals, wf = lin.eigh(VT)
    ZeroEigenvalsIndex = np.abs(eigenvals) < 1e-8
    Reduced_VT = wf[:, ZeroEigenvalsIndex]
#     print(eigenvals)
#     printMatrix(wf)
#     print("reduced=")
#     printMatrix(Reduced_VT)
    print('\nTotal number of integrals evaluated = {}'.format(kk))
    return(V, Reduced_VT, coeff, cs)

def calcKinEnDiabaticRep(max_m, max_n, b, zeros, dim):
    """Kinetic energy matrix elements are simply solutions of the Schrodinger equation for 
    a free particle in the cylindrical potential T=jmn^2 / 2b^2,
    jmn - nth zero of the Bessel function of order m"""
    
    T = np.zeros((2*dim, 2*dim))
    for m in range(-max_m, (max_m+1)): 
        for n in range(1, (max_n+1)):
            k = (m+max_m) * ((max_n+1)-1) + n
            T[k-1, k-1] = zeros[abs(m), n-1]**2 / (2 * b**2)
            T[k-1+dim, k-1+dim] = zeros[abs(m), n-1]**2 / (2 * b**2)
#     printMatrix(T)
    return(T)
 
if __name__ == "__calcKinEnDiabaticRep__": calcKinEnDiabaticRep()        
if __name__ == "__calcPotEnDiabaticRep__": calcPotEnDiabaticRep()
if __name__ == "__calculateProbDensityDistribution__": calculateProbDensityDistribution()        
if __name__ == "__mainDiabatic__": mainDiabatic()
if __name__ == "__calculateProbDensityAtSinglePoint__": calcPotEnDiabaticRep()
