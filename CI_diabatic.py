'''
Created on Jul 31, 2017

@author: Dmitry
'''
#!/usr/local/bin/python3.6
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
from scipy.integrate import quad
from IPython.core.tests.test_formatters import numpy
from numpy.lib.scimath import *
from numpy.ma.bench import m1
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#===============================================================================
#                            Global parameters
#===============================================================================
global order_max
global root_max
global b
global zeros
global dim
global L
global k

ini = time.process_time()
max_m = 15 # m goes from 0 to max_m
max_n = 15 # n goes from 1 to max_n
order_max = max_m+1 #Maximum order m of the Bessel basis functions (angular momentum)
root_max = max_n+1  #Maximum number of roots n of the Bessel basis functions (number of nodes)
            #Basis will contain (m+1)*n total basis functions  
b = 10      #Limit of integration (at r=b all basis functions are zero)
L = 6       #Shift of the potential from the origin on x axis 
k = 0       # k parameter in V12 = k*y coupling term
dim = 2 * max_m * max_n + max_n 
print("dim={}".format(dim))
zeros = np.zeros((order_max+1, root_max+1)) # array with coordinates of zeros of Bessel functions we need 
for i in range(order_max): zeros[i,:] = sp.jn_zeros(i, root_max+1) #compute all zeros of bessel function for order j
print("\nGlobal parameters:\n\nMaximum m = {}\nMaximum n = {}\nb = {}\ndim = {}\nL = {}\n".format(max_m, max_n, b, dim,L))

def main():

    #for i in range(1,dim): print(k_to_mn(i))
    r=0
    KE = kinE_ME() # Building kinetic energy matrix
    start = time.process_time()
    print("Calculating potential energy:\n")
    V11, V12 = potE_ME() # Buidling potential energy matrix elements
    end = time.process_time()
    print("Potential Energy Matrix Elements were calculated in {0:.2f} s\n".format(end-start))
    
    #Building Hamiltonian
    H = np.zeros((2*dim, 2*dim), dtype=np.complex128) # without coupling should get a harmonic oscillator solutions
    for k1 in range(0,dim):
        for k2 in range(0,dim):
            #H[k1, k2] = 1j * V12[k1, k2]
            H[k1, k2] = KE[k1, k2] + V11[k1, k2]
            H[k1+dim, k2] = 1j * V12[k1, k2] 
            H[k2, k1+dim] = np.conjugate(H[k1+dim, k2])
            H[k1+dim, k2+dim] = H[k1, k2]
    
    #print(np.trace(H))            
    start = time.process_time()
    Energies, wf = lin.eigh(H)
    end = time.process_time()
    print("Diagonalization took {0:.2f} s\n".format(end-start))
    ix=np.argsort(Energies)
    E = Energies[ix] # Sorted energies
    Eigenvecs=wf[:,ix] # Sorted eigenvectors
    print("Energies=\n\n{}\n".format(E[0:20]))
    #print("\nEigenvectors=\n{}".format(Eigenvecs[:,0]))
    
    #===========================================================================
    #    Printing all arrays into .txt files
    #===========================================================================
    np.savetxt('V11.txt', V11, fmt='%1.8E' )
    np.savetxt('Energies.txt', E, fmt='%1.8E' )
    np.savetxt('Wavefunction.txt', wf, fmt='%1.8E' )
    np.savetxt('H.txt', H, fmt='%1.8E' )

    np.set_printoptions(precision=2)
    #print("V11=")
    #print_matrix(V11)
    #print("V12=")
    #print_matrix(V12)
    #print("KE=")
    #print_matrix(KE)
    #print("V12=")
    #print(H[0:6,6:12])
    #print("V21=")
    #print(H[6:12,0:6])
    last = time.process_time()
    print("\n -----Total execution time is {0:.2f} s----- ".format(last-ini)) 
    
    # plot 2D
    #n_grid=200 
    #prob_dens = np.zeros((3,n_grid))
    #r1, prob_dens[0,:] = calc_prob_dens(Eigenvecs[:,0], n_grid)
    #r2, prob_dens[1,:] = calc_prob_dens(Eigenvecs[:,1], n_grid)
    #r3, prob_dens[2,:] = calc_prob_dens(Eigenvecs[:,2], n_grid)
     
    #f, axarr = plt.subplots(3, sharex=True)
    #axarr[0].plot(r1, prob_dens[0,:])
    #axarr[0].set_title('r, au')
    #axarr[1].scatter(r2, prob_dens[1,:])
    #axarr[2].scatter(r3, prob_dens[2,:])
    #plt.show()
    
    # Plot 3D
    #x1 = np.zeros(( n_grid * 360))
    #y1 = np.zeros(( n_grid * 360))
    #prob_dens2D_1 = np.zeros((n_grid * 360))
    
    #for i in range(0,n_grid):
    #    for phi in range (1,360,1):
    #        x1[360*i + phi] = r1[i] * np.cos(phi/360*2*math.pi)
    #        y1[360*i + phi] = r1[i] * np.sin(phi/360*2*math.pi)
    #        prob_dens2D_1[360*i + phi] = prob_dens[0,i]
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(x1, y1, prob_dens2D_1,cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    #plt.show()
     
def k_to_mn(k):
    if k % max_n == 0: m = k//max_n-1; n = max_n
    elif k % max_n != 0: m = k//max_n; n= k % max_n
    return(m,n)
    
def calc_prob_dens(cmn, n_grid=300):
    r = np.arange(-b, b, 2*b / n_grid)
    prob_dens = np.zeros(n_grid)
    total_density = 0
    
    for i in range (0, n_grid-1):
        for k in range(0, dim-1):
            m,n = k_to_mn(k) 
            #print("k={},m={},n={}".format(k,m,n))
            prob_dens[i] += psi(r[i], m, n) * psi(r[i], m, n) * np.absolute(cmn[k])**2
            #prob_dens[i] += psi(r[i], m, n)  * cmn[k]
    
    return (r, prob_dens)

def psi (r, m, n):
    """Definition of the orthonormal set of Bessel functions"""
    return (math.sqrt(2) / (b * sp.jv(m+1, zeros[m, n-1])) * sp.jv(m, zeros[m, n-1] * r/b)) 

def potE_ME():
#        Integration of complex variables is not trivial using SciPy so the integration of angular part was performed in Mathematica,
#        here we only perform integration of the radial part: 
#        V11 matrix elements are non-zero in 2 cases:
#        1) m1=m2, any n1, n2. In this case the V11 function is used 
#        2) m1=m2+1, any n1, n2. V11_2 function is used 
#        This function returns two matrices: 
#        diagonal potential energy MEs V11=V22,
#        off-diagonal potential energy MEs V12
       
    def V11 (r, m1, n1, m2, n2): # Diagonal ME for m1=m2
        return (psi(r, m1, n1) * psi(r, m2, n2) * r * (L**2 + 4*r**2) / 4 / 2 )
    
    def V11_2(r, m1, n1, m2, n2): # Diagonal ME for m1=m2+1
        return (psi(r, m1, n1) * psi(r, m2, n2) * L * r**2 / 2 / 2)

    PE12 = np.zeros ((dim, dim))
    PE11 = np.zeros ((dim, dim))
    
    # Low-level callback routine which uses c library to speed up integration using quad
    # bessel function: psi(r, m1, n1) * psi(r, m2, n2) * r * (L^2/4 + r**2) / 2 (substitute for V11)
    # bessel1 function: psi(r, m1, n1) * psi(r, m2, n2) * L * r**2 / 2 (substitute for V11_2)
    
    V11_cfunc = ctypes.CDLL(os.path.abspath('V11.so'))
    V11_cfunc.f.restype = ctypes.c_double
    V11_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    V11_2_cfunc = ctypes.CDLL(os.path.abspath('V11_2.so'))
    V11_2_cfunc.f.restype = ctypes.c_double
    V11_2_cfunc.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    mn1 = ctypes.c_int * 6 # array to pass to the c library (m1, n1, m2, n2, b)   
    kk=0
    
    for m1 in range(-max_m, 1):
        for m2 in range(-max_m, 1):
            for n1 in range(1,root_max):
                for n2 in range(n1,root_max):
                    k1 = (m1+max_m) * (root_max-1) + n1
                    k2 = (m2+max_m) * (root_max-1) + n2
                    k11 = k1 + 2 * abs(m1) * (root_max-1)
                    k22 = k2 + 2 * abs(m2) * (root_max-1)
                    
                    if m1 == m2: # m1 = m2, using V11 integral (see Mathematica notebook) 
                        if PE11[k1-1, k2-1] == 0.0:
                            mn = mn1(abs(m1), n1, abs(m2), n2, b, L)
                            user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)
                            func = LowLevelCallable(V11_cfunc.f, user_data)
                            PE11[k1-1, k2-1], error = quad (func, 0, b, epsrel = 0, epsabs = 1e-10, limit = 100)
                            #PE11[k1-1, k2-1], error = quad (V11, 0, b, args = (m1, n1, m2, n2), epsrel = 0, epsabs = 1e-10, limit = 100)
                            PE11[k2-1, k1-1] = PE11[k1-1, k2-1]
                            
                            # Due to symmetry the MEs for m and -m are equal
                            if k11 <= dim and k22 <= dim:
                                PE11[k22-1, k11-1] = PE11[k1-1, k2-1]
                                PE11[k11-1, k22-1] = PE11[k1-1, k2-1]
                        kk = kk+1
                    
                    elif abs(m2-m1) == 1: # m2 = m1 + 1
                        if PE11[k1-1, k2-1] == 0.0:
                            kk = kk+1 
                            mn = mn1(abs(m1), n1, abs(m2), n2, b, L)
                            user_data = ctypes.cast(ctypes.pointer(mn), ctypes.c_void_p)
                            func = LowLevelCallable(V11_2_cfunc.f, user_data)
                            PE11[k1-1, k2-1], error = quad (func, 0, b, epsrel = 0, epsabs = 1e-10, limit = 100)
                            #PE11[k1-1, k2-1], error = quad (V11_2, 0, b, args = (m1, n1, m2, n2), epsrel = 0, epsabs = 1e-10, limit = 100)
                            PE11[k2-1, k1-1] = PE11[k1-1, k2-1]
                            
                            # Due to symmetry the MEs for m and -m are equal
                            if k11 <= dim and k22 <= dim:
                                PE11[k22-1, k11-1] = PE11[k1-1, k2-1]
                                PE11[k11-1, k22-1] = PE11[k1-1, k2-1]
                           
                            #Off-diagonal matrix elements, V12, are nonzero when m1 = m2+1 or m1 = m2-1 for any n1,n2
                            #V12= 2 * k / L i * V11_2, so we don't need to calculate it"""
                            PE12[k1-1, k2-1] = 2 * k /L * PE11[k1-1, k2-1] * (m2-m1)
                            PE12[k11-1, k22-1] = -PE12[k1-1, k2-1]
                            PE12[k22-1, k11-1] = PE12[k1-1, k2-1]
                            PE12[k2-1, k1-1] = -PE12[k1-1, k2-1]

    print('Total number of integrals evaluated = {}'.format(kk))
    return(PE11, PE12)

def kinE_ME():
#     Kinetic energy matrix elements are simply solutions of the Schrodinger equation for 
#     a free particle in the cylindrical potential T=jmn^2 / 2b^2,
#     jmn - nth zero of the Bessel function of order m
    
    T = np.zeros((dim, dim))
    for m in range(-max_m, order_max): 
        for n in range(1, root_max):
            k = (m+max_m) * (root_max-1) + n
            T[k-1, k-1] = zeros[abs(m), n-1]**2 / (2 * b**2)
    return(T)

def print_matrix(*args):
    for M in args:
        print("")
        fmt=' '.join(['%7.2f']*M.shape[1])
        for row in M:
            print(fmt%tuple(row))
        
if __name__ == "__kinE_ME__": kinE_ME()        
if __name__ == "__main__": main()
if __name__ == "__potE_ME__": potE_ME()
if __name__ == "__print_matrix__": print_matrix()
if __name__ == "__psi__": psi()
if __name__ == "__calc_prob_dens__": calc_prob_dens()
if __name__ == "__k_to_mn__": k_to_mn()

