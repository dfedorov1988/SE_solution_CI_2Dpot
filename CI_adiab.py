'''
Created on Jul 31, 2017

This program solves the Schrodinger equation at the conical intersection using the basis of
cylindrical basis functions in both diabatic and adiabatic representations. The model system is 
the 2D Jahn-Teller model (similar to C. Xie et al., Phys. Rev. A 95, 022104, (2017))

The main goal of the project is to prove that the density at the intersection doesn't have 
to be 0 no matter what representation is used. More info about the problem is in:
G. Meek and B. Levine J. Chem. Phys. 144, 184109 (2016)

The diabatic part is fully working and agrees with results from the paper by Xie et al.
The adiabatic part is still in progress.

We use polar coordinates to take advantage of the symmetry:
1. Integration of matrix elements over angle is performed in Mathematica (see notebook for 
derivation of integrals).
2. Integration over r is performed in this program. To speed up calculations, the integrand
functions are written and precompiled in c language. Gives a 5X speedup. Requires V11.so,
V2.so, T1.so, etc. in the working directory.

@author: Dmitry A. Fedorov
'''

#!/usr/local/bin/python3.6
import sys
import os, ctypes
import time
import numpy as np
import scipy.special as sp 
from scipy import linalg as lin
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad
from numpy.lib.scimath import *
import matplotlib.pyplot as plt
import CI_diab
from CI_diab import mainDiabatic
#from prettytable import PrettyTable
from misc import transformMatrix, isHermitian, psi, calculateDensity, densityAtSinglePoint, printMatrix,\
    largeBasFuncComp

def main():
    """Main module. Initialization of variables and calls either Diabatic or Adiabatic module"""
    
    absoluteZeroTime = time.process_time()
    max_m = 2   # Max order m of the Bessel basis functions, -m_max<m<max_m
    max_n = 13   # Max number of roots n of the Bessel basis functions, 1 < n < max_n
                  # Basis will contain (m+1)*n total basis functions  
    b = 8        # Limit of integration (at r=b all basis functions are zero)
    L = 6         # Shift of the potential from the origin on x axis 
    c = L/2       # k parameter in V12 = k*y coupling term, 
                  # in adiabatic representation c=L/2 makes equations easier
    delta = 0.00   # shift of one of the parabolas
    reducedBasis = 0 # 0 - full basis, 1 - reduced basis
    representation = "diabatic" 
    zerosToCompute = 500
    eps = 1e-12 # the integration limit for divergent integrals
    iterNumber = 1 # number of iterations to do with different eps
    stateToCompute = 0 # calculate density for the GS
    
    dim = 2 * max_m * max_n + max_n   # calculating dimensionality 
    print('------------------\nGlobal parameters:\n------------------\n\
Maximum m = {}\nMaximum n = {}\nb = {}\
          \nL = {}\nc = {}'.format(max_m, max_n, b, L, c))

    """Array with coordinates of zeros of Bessel functions we need"""         
    zeros = np.zeros((max_m + 1, zerosToCompute + 2))

    """compute all zeros of bessel function for order i"""
    for i in range(0, max_m+1):
        zeros[i, :] = sp.jn_zeros(i, zerosToCompute + 2) 

    if representation == "adiabatic":
        
        for i in range(iterNumber):
            print("\nWorking in adiabatic representation")
            dim = 2 * max_m * max_n + max_n
            print("m = {}\nn = {}".format(max_m, max_n))
            '''Calculating overlap, reduced overlap matrix, transformation matrix'''
            S, S_T, reduced_S_eigenvecs = buildOverlapMatrix(max_m, max_n, b, zeros, L, c, eps)
    
            """Calculating Hamiltonian, coefficients k -> m,n and cs (symmetry of nuclear wf)
            Since the trick with operators A and B didn't work we're trying different approach:
            1. Calculating divergent integrals up to eps instead of zero for 2 values of eps
            2. Subtract Hamiltonians corresponding to 2 values of eps (matrix C)
            3. Diagonalize C and throw away eigenvectors corresponding to nonzero eigenvals
                Seems to be doing what we want"""
                
            H, C, coeff, cs = mainAdiabatic(max_m, max_n, b, zeros, L, c, delta, eps, S, S_T,\
                                reduced_S_eigenvecs)

            C_T = transformMatrix(reduced_S_eigenvecs, C)
            C_T_eigenvals, C_T_eigenvecs = lin.eigh(C_T, b=S_T)
            zeroEigenvalueIndex = np.abs(C_T_eigenvals) < 1e-8
            C_T_eigenvecs_reduced = C_T_eigenvecs[:, zeroEigenvalueIndex]
            
            H_T = transformMatrix(reduced_S_eigenvecs, H)
            H_T2 = transformMatrix(C_T_eigenvecs_reduced, H_T)
            print(np.shape(H_T), np.shape(H_T2))
            S_T2 = transformMatrix(C_T_eigenvecs_reduced, S_T)

            print("Dimensionality of final Hamiltonian: {}".format(np.shape(H_T2)))
            
            print("\nChecking final S and H matrices for Hermiticity:")
            if not (isHermitian(S_T2)):
                sys.exit("Matrix S is not Hermitian! Exiting...")    
            else: print("Matrix S: OK")                  
            if not (isHermitian(H_T2)):
                sys.exit("MatriX H is not Hermitian! Exiting...")
            else: print("Matrix H: OK\n")
        
            eigenvals, eigenvecs = lin.eigh(H_T2, b=S_T2)
            eigenvals.sort()
            print("energies =\n")
            for eLevel in range(6): print("{0:1.8f}".format(eigenvals[eLevel]))
            
            """Transforming to original basis and calculating density"""
            eigenvecs_T = np.dot(C_T_eigenvecs_reduced, eigenvecs)
            eigenvecs_T2 = np.dot(reduced_S_eigenvecs, eigenvecs_T)
            densityAtOrigin = densityAtSinglePoint(0, 0, b, dim, zeros, eigenvecs_T2, coeff, C_T_eigenvecs_reduced, reducedBasis, stateToCompute)
            print("\nDensity at Origin: {0:1.2e}".format(densityAtOrigin))
            calculateDensity(b, dim, zeros, eigenvecs_T2, coeff, C_T_eigenvecs_reduced, reducedBasis, stateToCompute, n_grid=50, limit_plot=6)      

#             largeBasFuncComp(dim, coeff, eigenvecs_T2, cs)

            max_m += 1
            max_n += 4
                        
    if representation == "diabatic":
        
        i = 0
        E = mainDiabatic(max_m + 2*i, max_n + 2*i, zeros, b, L, c, delta, reducedBasis, representation, stateToCompute)
        
    absoluteEndTime = time.process_time()
    print("\nTotal execution time is {0:.2f} s\n".format(absoluteEndTime-absoluteZeroTime))

def mainAdiabatic(max_m, max_n, b, zeros, L, c, delta, eps, S, S_T, reduced_S_eigenvecs):
    """The main module to construct and diagonalize Hamiltonian in adiabatic rep
    Compared to diabatic representation we have double amount of basis functions:
    I, J stand for the electronic state
    c, s stand for cos and sin component of nuclear wf
    """
    
    V, coeff, cs = calcPotEnAdiabaticRep(max_m, max_n, b, zeros, L, c, delta)    
    T1 = calcKinEnAdiabaticRep(max_m, max_n, b, zeros, L, c, eps, coeff)
    T2 = calcKinEnAdiabaticRep(max_m, max_n, b, zeros, L, c, eps/1000, coeff)
    H = T1 + V
    H2 = T2 + V    
    C = T2 - T1
    
    return(H, C, coeff, cs)

def buildOverlapMatrix(max_m, max_n, b, zeros, L, c, eps):
    """Calculates overlap matrix and transformation matrix that throws away
    linearly dependent basis functions"""
    
    def overlap(r, m1, n1, m2, n2, L, b, zeros, c):
        """Expression for the overlap integral"""
        
        result = r/2 * sp.jv(m1, r/b * zeros[np.abs(m1), n1-1]) \
        * sp.jv(m2, r/b * zeros[np.abs(m2), n2-1]) \
        /  (b**2 * sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
        * sp.jv(m2+1, zeros[np.abs(m2), n2-1]))
        return(result)
    
    print("\n---------------------------\nCalculating overlap matrix:\n---------------------------")  
    dim = 2 * max_m * max_n + max_n  
    S = np.zeros((4*dim, 4*dim))
    tmpm = np.zeros((4*dim))
    tmpn = np.zeros((4*dim))

    kk = 0
    coeff = [[0] * 2 for i in range(4*dim)] # this array allows to convert k to m,n
    epsabs = 1e-12
    
    overlap_cfunc = ctypes.CDLL(os.path.abspath('overlap.so'))
    overlap_cfunc.f.restype = ctypes.c_double
    overlap_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    mn1 = ctypes.c_double * 7 # array to pass to the c library (m1, n1, m2, n2, b)   
    
    for m1 in range(-max_m, max_m+1):
        for m2 in range(-max_m, max_m+1):
            for n1 in range(1, max_n+1):
                for n2 in range(1, max_n+1):
    
                    k1 = (m1+max_m) * ((max_n+1)-1) + n1
                    k2 = (m2+max_m) * ((max_n+1)-1) + n2
                    
                    mn = mn1(abs(m1), n1, abs(m2), n2, b, L, c)
                    user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)

                    if m1 == m2:
                        
                        if S[k1-1, k2-1] == 0.0 and n1==n2:       
                                                
                            S[k1-1, k2-1] = 0.5 # IcIc
                            S[k1-1+dim, k2-1+dim] = 0.5 # IsIs
                            S[k1-1+2*dim, k2-1+2*dim] = 0.5 # JcJc
                            S[k1-1+3*dim, k2-1+3*dim] = 0.5 # JsJs
                            
                    if m2 - m1 == 1: # m2 = m1+1
                        
                        if S[k1-1, k2-1] == 0.0:
                            func = LowLevelCallable(overlap_cfunc.f, user_data)
                            S[k1-1, k2-1], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100)

                            kk += 1    
                            
                            S[k1-1+dim, k2-1+dim] = -S[k1-1, k2-1] # IsIs
                            S[k1-1+2*dim, k2-1+2*dim] = S[k1-1, k2-1] # JcJc
                            S[k1-1+3*dim, k2-1+3*dim] = -S[k1-1, k2-1] # JsJs
 
                            S[k1-1, k2-1+dim] = S[k1-1, k2-1] # IcIs                            
                            S[k1-1+dim, k2-1] = -S[k1-1, k2-1] # IsIc                                                        
                            S[k1-1+2*dim, k2-1+3*dim] = S[k1-1, k2-1] # JcJs                            
                            S[k1-1+3*dim, k2-1+2*dim] = -S[k1-1, k2-1] # JsJc

                    if m1 - m2 == 1: # m2 = m1+1
                        
                        if S[k1-1, k2-1] == 0.0:
                            func = LowLevelCallable(overlap_cfunc.f, user_data)
                            S[k1-1, k2-1], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100) #IcIc

                            kk += 1    
                            
                            S[k1-1+dim, k2-1+dim] = -S[k1-1, k2-1] # IsIs
                            S[k1-1+2*dim, k2-1+2*dim] = S[k1-1, k2-1] # JcJc
                            S[k1-1+3*dim, k2-1+3*dim] = -S[k1-1, k2-1] # JsJs
 
                            S[k1-1, k2-1+dim] = -S[k1-1, k2-1] # IcIs                            
                            S[k1-1+dim, k2-1] = S[k1-1, k2-1] # IsIc                                                        
                            S[k1-1+2*dim, k2-1+3*dim] = -S[k1-1, k2-1] # JcJs                            
                            S[k1-1+3*dim, k2-1+2*dim] = S[k1-1, k2-1] # JsJc

    '''Finding eigenvalues of the overlap matrix, throwing away the eigenvectors
    corresponding to the smallest eigenvalues ( due to linearly dependent basis functions).
    Using the remaining eigenvectors obtaining an overlap matrix of reduced dimensionality.
    This matrix will be used to transform Hamiltonian matrix '''
    
    S_eigenvals, S_eigenvecs = lin.eigh(S)
    indicesAboveThreshold = np.abs(S_eigenvals) > 1e-3
#     print(S_eigenvals)
    reduced_S_eigenvecs = S_eigenvecs[:, indicesAboveThreshold]
    reduced_S_eigenvals = S_eigenvals[indicesAboveThreshold]
    S_T = transformMatrix(reduced_S_eigenvecs, S)
    
    np.set_printoptions(precision=8)
    #print("Matrix S condition number is {0:.2e}".format(np.linalg.cond(S)))
    print("{} out of {} basis functions were thrown out of the basis".\
    format(np.shape(S_eigenvals)[0] - np.shape(reduced_S_eigenvals)[0], np.shape(S_eigenvals)[0]))
    print("Smallest eigenvalue of reduced overlap matrix is {0:.4f}".format(min(reduced_S_eigenvals)))
    print("The new overlap matrix condition number is {0:.2f}".format(np.linalg.cond(S_T)))
    print("Total number of {} integrals was evaluated".format(kk)) 
         
    return(S, S_T, reduced_S_eigenvecs) 
                      
def calcKinEnAdiabaticRep(max_m, max_n, b, zeros, L, c, eps, coeff):
    """Calculates kinetic energy matrix elements in adiabatic representation:
    T = 1/2 * (d/dr^2 + 1/4r^2 + 1/r^2 * d/dphi^2)
    Integration over angle is performed analytically in Mathematica (see notebook).
    Since the radial integrals have to be calculated numerically, which takes forever,
    we take advantage of the symmetry and that's why you see endless assignments of 
    matrix elements. 
    The notation is I and J for electronic states, c and s for cos and sin components 
    of nuclear wave function"""  
        
    print("\n-------------------------------------------\
    \nCalculating kinetic energy matrix elements:\n-------------------------------------------")    

    dim = 2 * max_m * max_n + max_n  

    T_CC, T_SS, T_CS, T_SC = np.zeros((4, dim, dim))
    DC_CC, DC_CS, DC_SC, DC_SS = np.zeros((4, dim, dim))
    
    T = np.zeros ((4*dim, 4*dim), dtype=np.complex128)    
    DC = np.zeros ((4*dim, 4*dim))
    
    """The following 2 lowLevelCallable functions are precompiled in c and used in numerical
    integration"""

    T1T2_cfunc = ctypes.CDLL(os.path.abspath('T1T2.so'))
    T1T2_cfunc.f.restype = ctypes.c_double
    T1T2_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

    T3DBOC_cfunc = ctypes.CDLL(os.path.abspath('T3.so'))
    T3DBOC_cfunc.f.restype = ctypes.c_double
    T3DBOC_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

    mn1 = ctypes.c_double * 7  # array to pass into a c function
         
    kk = 0 # number of numerical integrals calculated
    absError = 1e-12 # absolute error for calculating numerical integrals
    intLimit = 1000 # limit of subdivisions in Gauss-Legendre procedure
    
    for m1 in range(-max_m, max_m+1):

        m_left = max(-max_m, m1-1)
        m_right = min(max_m+1, m1+2)

        for m2 in range(m_left, m_right):
            for n1 in range(1, (max_n+1)):
                for n2 in range(1, (max_n+1)):
    
                    k1 = (m1 + max_m) * max_n + n1
                    k2 = (m2 + max_m) * max_n + n2
                    mn = mn1(abs(m1), n1, abs(m2), n2, b, L, c)
                    user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)
                    funcT1T2 = LowLevelCallable(T1T2_cfunc.f, user_data)
                    func3 = LowLevelCallable(T3DBOC_cfunc.f, user_data)

                    if abs(m1 - m2) <= 1:

                        int = quad (func3, eps, b, \
                                   args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                   epsabs = absError, limit = intLimit)[0]                        
                        kk += 1                               
                        
                        intR = quad (funcT1T2, 0, b, \
                                     args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                     epsabs = absError, limit = intLimit)[0]
                        
                    if m1 == m2:
                        """Main diagonal m1 = m2
                        first term is the T1 + T2 operators, second term T3 + DBOC"""
                              
                        T_CC[k1-1, k2-1]= intR + int * (-1/2) * (1 + 2*np.square(m1))
                        T_SS[k1-1, k2-1] = T_CC[k1-1, k2-1]
                        T_CS[k1-1, k2-1] = int * m1
                        T_SC[k1-1, k2-1] = T_CS[k1-1, k2-1]
                         
                        DC_CC[k1-1, k2-1] = int * (-1/2) * m1
                        DC_SS[k1-1, k2-1] = DC_CC[k1-1, k2-1]                                                                   
                        DC_CS[k1-1, k2-1] = int * (1/4)
                        DC_SC[k1-1, k2-1] = DC_CS[k1-1, k2-1]
                                               
                        kk += 2
                                                                         
                    if m2 - m1 == 1: # m2 = m1 + 1
                        """Upper diagonal m2 = m1 + 1
                        first term is the T1 + T2 operators, second term T3 + DBOC"""
                        
                        T_CC[k1-1, k2-1] = intR/2 + int * (-1/2) * (1/2 + np.square(m1) + m1)                
                        T_SS[k1-1, k2-1] = -T_CC[k1-1, k2-1]
                        T_CS[k1-1, k2-1] = T_CC[k1-1, k2-1]
                        T_SC[k1-1, k2-1] = -T_CC[k1-1, k2-1]
 
                        DC_CC[k1-1, k2-1] = int * (-1/8) * (2*m1 + 1)  
                        DC_SS[k1-1, k2-1] = -DC_CC[k1-1, k2-1]  
                        DC_CS[k1-1, k2-1] = DC_CC[k1-1, k2-1]  
                        DC_SC[k1-1, k2-1] = -DC_CC[k1-1, k2-1]                                                                                                         
                                                
                        kk += 2
                        
                    if m1 - m2 == 1: # m2 = m1 - 1
                        """Lower diagonal m2 = m1 - 1
                        first term is the T1 + T2 operators, second term T3 + DBOC"""
                        
                        T_CC[k1-1, k2-1]= intR/2 + int * (-1/2) * (1/2 - m1 + np.square(m1))                     
                        T_SS[k1-1, k2-1] = -T_CC[k1-1, k2-1]
                        T_CS[k1-1, k2-1] = -T_CC[k1-1, k2-1]
                        T_SC[k1-1, k2-1] = T_CC[k1-1, k2-1]
         
                        DC_CC[k1-1, k2-1] = int * (-1/8) * (2*m1 - 1)   
                        DC_SS[k1-1, k2-1] = -DC_CC[k1-1, k2-1]  
                        DC_CS[k1-1, k2-1] = -DC_CC[k1-1, k2-1]  
                        DC_SC[k1-1, k2-1] = DC_CC[k1-1, k2-1]
                        
                        kk += 2

    T_CC = -0.5 * (T_CC)  
    T_SS = -0.5 * (T_SS) 
    T_CS = -0.5 * (T_CS) 
    T_SC = -0.5 * (T_SC) 
    
    DC_CC_T = np.transpose(DC_CC)
    DC_CS_T = np.transpose(DC_CS)
    DC_SC_T = np.transpose(DC_SC)
    DC_SS_T = np.transpose(DC_SS)
    for k1 in range(dim):
        for k2 in range(dim):
            """"Constructing final matrix T"""
            
            # First quarter        
            T[k1, k2] = T_CC[k1, k2] 
            T[k1, k2+dim] = T_CS[k1, k2]
            T[k1+dim, k2] = T_SC[k1, k2]
            T[k1+dim, k2+dim] = T_SS[k1, k2]
                                           
            # Second quarter
            T[k1, k2+2*(dim)] = 1j * DC_CC[k1, k2]  
            T[k1, k2+3*(dim)] = 1j * DC_CS[k1, k2]
            T[k1+dim, k2+2*(dim)] = 1j * DC_SC[k1, k2]
            T[k1+dim, k2+3*(dim)] = 1j * DC_SS[k1, k2]
            
            # Third quarter         
            T[k1+2*(dim), k2] = -1j * DC_CC_T[k1, k2]
            T[k1+2*(dim), k2+dim] = -1j * DC_SC_T[k1, k2]
            T[k1+3*(dim), k2] = -1j * DC_CS_T[k1, k2]
            T[k1+3*(dim), k2+dim] = -1j * DC_SS_T[k1, k2]

            # Fourth quarter
            T[k1+2*(dim), k2+2*(dim)] = T_CC[k1, k2]
            T[k1+2*(dim), k2+3*(dim)] = T_CS[k1, k2]
            T[k1+3*(dim), k2+2*(dim)] = T_SC[k1, k2]
            T[k1+3*(dim), k2+3*(dim)] = T_SS[k1, k2] 
        
    print("Total number of {} integrals was evaluated\n".format(kk)) 
#     diabatizeStates(T, dim)
    
    for k1 in range(4*dim):
        for k2 in range(4*dim):
            if np.real(T[k1-1, k2-1] - T[k2-1, k1-1]) > 1e-7:
                print("m = {}, n = {} and m' = {}, n' = {} elements don't match! dH = {}"\
                      .format(coeff[k1][0], coeff[k1][1], coeff[k2][0], coeff[k2][1],\
                              np.real(T[k1-1, k2-1] - T[k2-1, k1-1])))
    
    return(T)
        
def calcPotEnAdiabaticRep(max_m, max_n, b, zeros, L, c, delta):
    """Calculates potential energy matrix elements in adiabatic basis.
    In general, matrix elements <m1n1|V|m2n2> are in 4 dimensions, but we make a 2D array
    of dimensionality k=m*n. 1D array coeff is stored to convert k into m,n pairs later 
    and cs array contains the labels of states (I/J, c/s), which is used to look at the contributions of
    of individual basis functions to the total wave function"""
    
    print("\n---------------------------------------------\n\
Calculating potential energy matrix elements:\n---------------------------------------------")
    dim = 2 * max_m * max_n + max_n 
       
    V = np.zeros((4*dim, 4*dim))
    V_A, V_B = np.zeros((2, dim, dim))
    
    kk=0
    coeff = [[0] * 2 for i in range(4*dim)] # this array allows to convert k to m,n
    cs = [[] for i in range(4*dim)]

    V_plus_cfunc = ctypes.CDLL(os.path.abspath('V_plus.so'))
    V_plus_cfunc.f.restype = ctypes.c_double
    V_plus_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    V_minus_cfunc = ctypes.CDLL(os.path.abspath('V_minus.so'))
    V_minus_cfunc.f.restype = ctypes.c_double
    V_minus_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    mn1 = ctypes.c_double * 7 # array to pass to the c library (m1, n1, m2, n2, b, L, c)   
    
    for m1 in range(-max_m, 1):
        for m2 in range(-max_m, 1):
            for n1 in range(1, (max_n+1)):
                for n2 in range(1, (max_n+1)):
    
                    k1 = (m1+max_m) * ((max_n+1)-1) + n1
                    k2 = (m2+max_m) * ((max_n+1)-1) + n2
                    k11 = k1 + 2 * abs(m1) * ((max_n+1)-1)
                    k22 = k2 + 2 * abs(m2) * ((max_n+1)-1)
                    
                    coeff[k1-1][0], coeff[k1-1+dim][0], coeff[k1-1+2*dim][0], coeff[k1-1+3*dim][0] = m1, m1, m1, m1
                    coeff[k1-1][1], coeff[k1-1+dim][1], coeff[k1-1+2*dim][1], coeff[k1-1+3*dim][1] = n1, n1, n1, n1
                    coeff[k11-1][0], coeff[k11-1+dim][0], coeff[k11-1+2*dim][0], coeff[k11-1+3*dim][0] = -m1, -m1, -m1, -m1
                    coeff[k11-1][1], coeff[k11-1+dim][1], coeff[k11-1+2*dim][1], coeff[k11-1+3*dim][1] = n1, n1, n1, n1
                    
                    cs[k1-1], cs[k1-1+dim], cs[k1-1+2*dim], cs[k1-1+3*dim] = 'Ic', 'Is', 'Jc', 'Js'
                    cs[k11-1], cs[k11-1+dim], cs[k11-1+2*dim], cs[k11-1+3*dim] = 'Ic', 'Is', 'Jc', 'Js'
                    
                    mn = mn1(abs(m1), n1, abs(m2), n2, b, L, c)
                    user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)
                    
                    if m1 == m2:
                        
                        if V[k1-1, k2-1] == 0.0:
                            func = LowLevelCallable(V_minus_cfunc.f, user_data)
                            V[k1-1, k2-1], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100)                            
#                             V[k1-1, k2-1], error = quad (V_minus, 0, b, \
#                             args = (m1, n1, m2, n2, L, b, zeros, c), epsrel = 0, \
#                                     epsabs = 1e-12, limit = 100) 
                            kk += 1
                            V[k2-1, k1-1] = V[k1-1, k2-1] # IcIc
                            
                            if V[k1-1+dim, k2-1+dim] == 0.0:
                                V[k1-1+dim, k2-1+dim] = V[k1-1, k2-1]
                                V[k2-1+dim, k1-1+dim] = V[k1-1+dim, k2-1+dim]
                            
                        if V[k1-1+2*dim, k2-1+2*dim] == 0.0:
                            func = LowLevelCallable(V_plus_cfunc.f, user_data)
                            V[k1-1+2*dim, k2-1+2*dim], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c, delta), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100)  
#                             V[k1-1+2*dim, k2-1+2*dim], error = quad (V_plus, 0, b, \
#                             args = (m1, n1, m2, n2, L, b, zeros, c), \
#                                     epsrel = 0, epsabs = 1e-12, limit = 100)
                            kk += 1
                            V[k2-1+2*dim, k1-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim]
                            
                            if V[k1-1+3*dim, k2-1+3*dim] == 0.0:
                                V[k1-1+3*dim, k2-1+3*dim] = V[k1-1+2*dim, k2-1+2*dim]
                                V[k2-1+3*dim, k1-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim]

                        if k11 <= dim and k22 <= dim:
                            V[k22-1, k11-1] = V[k1-1, k2-1]
                            V[k11-1, k22-1] = V[k1-1, k2-1]
                                       
                            V[k22-1+dim, k11-1+dim] = V[k1-1+dim, k2-1+dim]
                            V[k11-1+dim, k22-1+dim] = V[k1-1+dim, k2-1+dim]    
                                      
                            V[k22-1+2*dim, k11-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim]
                            V[k11-1+2*dim, k22-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim]
                                      
                            V[k22-1+3*dim, k11-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim]
                            V[k11-1+3*dim, k22-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim]
                                                        
                    if abs(m2-m1) == 1:
                        
                        if V[k1-1, k2-1] == 0.0:
                            func = LowLevelCallable(V_minus_cfunc.f, user_data)
                            V[k1-1, k2-1], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100)                               
#                             V[k1-1, k2-1], error = quad (V_minus, 0, b, \
#                             args = (m1, n1, m2, n2, L, b, zeros, c), epsrel = 0, \
#                                     epsabs = 1e-12, limit = 100) 
                            kk += 1
                            V[k1-1, k2-1] = 1/2 * V[k1-1, k2-1] # IcIc
                            V[k2-1, k1-1] = V[k1-1, k2-1] # IcIc
                            
                            if V[k1-1+dim, k2-1+dim] == 0.0:
                                V[k1-1+dim, k2-1+dim] = -V[k1-1, k2-1] # IsIs
                                V[k2-1+dim, k1-1+dim] = V[k1-1+dim, k2-1+dim] # IsIs
                            
                            if V[k1-1, k2-1+dim] == 0.0:
                                V[k1-1, k2-1+dim] = V[k1-1, k2-1] # IcIs
                                V[k2-1, k1-1+dim] = -V[k1-1, k2-1+dim] # IcIs
                            
                            if V[k1-1+dim, k2-1] == 0.0:
                                V[k1-1+dim, k2-1] = -V[k1-1, k2-1] # IsIc
                                V[k2-1+dim, k1-1] = -V[k1-1+dim, k2-1] # IsIc
                                                            
                        if V[k1-1+2*dim, k2-1+2*dim] == 0.0:
                            func = LowLevelCallable(V_plus_cfunc.f, user_data)
                            V[k1-1+2*dim, k2-1+2*dim], error = quad (func, 0, b, \
                            args = (m1, n1, m2, n2, b, L, c, delta), epsrel = 0, \
                                    epsabs = 1e-12, limit = 100)
#                             V[k1-1+2*dim, k2-1+2*dim], error = quad (V_plus, 0, b, \
#                             args = (m1, n1, m2, n2, L, b, zeros, c), \
#                                     epsrel = 0, epsabs = 1e-12, limit = 100)   
                            kk += 1
                            V[k1-1+2*dim, k2-1+2*dim] = 1/2 * V[k1-1+2*dim, k2-1+2*dim] # JcJc
                            V[k2-1+2*dim, k1-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim] # JcJc
                            
                            if V[k1-1+3*dim, k2-1+3*dim] == 0.0:
                                V[k1-1+3*dim, k2-1+3*dim] = -V[k1-1+2*dim, k2-1+2*dim] # JsJs
                                V[k2-1+3*dim, k1-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim] # JsJs
                        
                            if V[k1-1+2*dim, k2-1+3*dim] == 0.0:
                                V[k1-1+2*dim, k2-1+3*dim] = V[k1-1+2*dim, k2-1+2*dim] # JcJs
                                V[k2-1+2*dim, k1-1+3*dim] = -V[k1-1+2*dim, k2-1+3*dim] # JcJs
                            
                            if V[k1-1+3*dim, k2-1+2*dim] == 0.0:
                                V[k1-1+3*dim, k2-1+2*dim] =  -V[k1-1+2*dim, k2-1+2*dim] # JsJc
                                V[k2-1+3*dim, k1-1+2*dim] = -V[k1-1+3*dim, k2-1+2*dim] # JsJc           

                        if k11 <= dim and k22 <= dim:
                            V[k22-1, k11-1] = V[k1-1, k2-1]
                            V[k11-1, k22-1] = V[k1-1, k2-1]
                               
                            V[k22-1, k11-1+dim] = V[k1-1, k2-1+dim]
                            V[k11-1, k22-1+dim] = -V[k1-1, k2-1+dim] 
                                  
                            V[k22-1+dim, k11-1] = V[k1-1+dim, k2-1]
                            V[k11-1+dim, k22-1] = -V[k1-1+dim, k2-1]
                            
                            V[k22-1+dim, k11-1+dim] = V[k1-1+dim, k2-1+dim]
                            V[k11-1+dim, k22-1+dim] = V[k1-1+dim, k2-1+dim]
                            
                            V[k22-1+2*dim, k11-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim]
                            V[k11-1+2*dim, k22-1+2*dim] = V[k1-1+2*dim, k2-1+2*dim]
                            
                            V[k22-1+3*dim, k11-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim]
                            V[k11-1+3*dim, k22-1+3*dim] = V[k1-1+3*dim, k2-1+3*dim]
   
                            V[k22-1+2*dim, k11-1+3*dim] = V[k1-1+2*dim, k2-1+3*dim]
                            V[k11-1+2*dim, k22-1+3*dim] = -V[k1-1+2*dim, k2-1+3*dim] 
                            
                            V[k22-1+3*dim, k11-1+2*dim] = V[k1-1+3*dim, k2-1+2*dim]
                            V[k11-1+3*dim, k22-1+2*dim] = -V[k1-1+3*dim, k2-1+2*dim]
    
    """V_A and V_B are the matrices obtained by adiabatic-to-diabatic transformation
    should get the same answer in both representations!"""
    
    for k1 in range(dim):
        psi_a1 = np.zeros((4*dim, 1))
        psi_a1[k1, 0] = 1
        psi_a1[k1 + 3*dim, 0] = 1    
        
        psi_b1 = np.zeros((4*dim, 1))
        psi_b1[k1 + dim, 0] = 1
        psi_b1[k1 + 2*dim, 0] = 1        
        
        for k2 in range(dim):
            psi_a2 = np.zeros((1, 4*dim))
            psi_a2[0, k2] = 1
            psi_a2[0, k2 + 3*dim] = 1

            psi_b2 = np.zeros((1, 4*dim))
            psi_b2[0, k2 + dim] = 1
            psi_b2[0, k2 + 2*dim] = 1
            
            V_T1 = np.dot(V, psi_a1)
            V_A[k1, k2] = np.dot(psi_a2, V_T1)
            
            V_T2 = np.dot(V, psi_b1)
            V_B[k1, k2] = np.dot(psi_b2, V_T2) 
            
#     print("V_A =")
#     printMatrix(V_A)
#     print("V_B =")
#     printMatrix(V_B)

    print("Total number of {} integrals was evaluated".format(kk))    
    return(V, coeff, cs)

def diabatizeStates(T, dim):
    T_A = np.zeros((dim, dim), dtype=np.complex128)
    T_B = np.zeros((dim, dim), dtype=np.complex128)
    for k1 in range(dim):
        """Transformation of adiabatic kinetic energy into diabatic:
        psiA = psiIc + psiJs
        psiB = psiIs + psiJc
        T_A and T_B must be diagonal! 
        WARNING: this does not work properly"""
        
        psi_a1 = np.zeros((4*dim, 1), dtype=np.complex128)
        psi_a1[k1, 0] = 1
        psi_a1[k1 + 3*dim, 0] = 1    
        
        psi_b1 = np.zeros((4*dim, 1), dtype=np.complex128)
        psi_b1[k1 + dim, 0] = 1j
        psi_b1[k1 + 2*dim, 0] = 1j        
        
        for k2 in range(dim):
            psi_a2 = np.zeros((1, 4*dim), dtype=np.complex128)
            psi_a2[0, k2] = 1
            psi_a2[0, k2 + 3*dim] = 1

            psi_b2 = np.zeros((1, 4*dim), dtype=np.complex128)
            psi_b2[0, k2 + dim] = -1j
            psi_b2[0, k2 + 2*dim] = -1j
            
            T_T1 = np.dot(T, psi_a1)
            T_A = np.dot(psi_a2, T_T1)
            
            T_T2 = np.dot(T, psi_b1)
            T_B = np.dot(psi_b2, T_T2)
    eigenvals, eigenvecs = lin.eigh(np.real(T_B)) 

if __name__ == "__calcKinEnAdiabaticRep__": calcKinEnAdiabaticRep()
if __name__ == "__main__": main()

