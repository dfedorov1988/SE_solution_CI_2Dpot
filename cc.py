""" Created by Ben Levine, 2017
    Additions by Dmitry A. Fedorov 2018
    
    Solution of the SE around CI in the Harmonic oscillator basis"""
#from pyscf import lib
from numpy import *
import math
import numpy as np
from scipy import special as sp
#from sympy import *
from misc import transformMatrix, printMatrix
# from __future__ import division
import time
from numpy.linalg import norm
from misc import printMatrix
from bigfloat import *

def eval_psi( x, y, nx, ny, omega0, xdisp ):
    "This evaluates the wave function at cartesian coordinates x and y"
    pi = math.acos(-1.0)
    
    psix = 1.0 / math.sqrt( 2.0**nx * math.factorial(nx) )
    psix *= (omega0 / pi)**0.25
    psix *= math.exp( -0.5 * omega0 * (x - xdisp) * (x - xdisp) )
    psix *= sp.eval_hermite(nx, math.sqrt(omega0) * (x-xdisp) )

    psiy = 1.0 / math.sqrt( 2.0**ny * math.factorial(ny) )
    psiy *= (omega0 / pi) ** 0.25
    psiy *= math.exp( -0.5 * omega0 * y * y )
    psiy *= sp.eval_hermite(ny, math.sqrt(omega0) * y)
    
    psi = psix * psiy
    return psi

# the maximum total number of vibrational quanta in basis functions

def inverseIteration(A, eigenvalGuess, eigenVecGuess, maxIter=50, tol=1e-10):
    """This subroutine solves eigenvalue problem using inverse iteration method
    Parameters: A : (n, n) array
                    Matrix to solve
                eigenvalGuess : float
                    Initial guess for eigenvalue. Good guess helps to converge to the
                    right state
                eigenVecGuess : 1D array of size n 
                    Initial guess for eigenvector
                maxIter : int
                    Maximum number of iterations
                tol : float
                    Tolerance for convergence of the residual norm 
    Returns:    b_k : 1D array of size n
                    Final solution for the eigenvector
                """
    print("\nLooking for eigenvalue of the matrix close to {0:1.6f}:".format(eigenvalGuess))
    b_k = eigenVecGuess
    b_k_norm = np.linalg.norm(b_k)
    I = np.identity(np.shape(A)[0])
    Matrix = A - I * eigenvalGuess
    print("Iteration    normDiff")
    
    for n in range(maxIter): 
        
        b_k1 = np.linalg.solve(Matrix, b_k)
        b_k1_norm  = np.linalg.norm(b_k1)
        normDiff = abs(b_k1_norm - b_k_norm)
        print("{0:2d}           {1:1.2e}".format(n, float(normDiff)))
#         print(normResidual)
        if normDiff < tol: break
        
        b_k = b_k1 / b_k1_norm
        b_k_norm = b_k1_norm

    print("Converged")
    return(b_k)

def transformMatrix(transformationMatrix, M):       
    """ST * M * S transformation"""
    
    Q = np.dot(M, transformationMatrix)
    
    R = np.dot(transformationMatrix.transpose().conjugate(), Q)
    return(R)

ntotalmax = 40

# total dimension of basis
nbasis = (ntotalmax + 2) * ( ntotalmax + 1 )
# 2D Jahn-Teller Model
# V11 = omega^2 / 2 ( x + a/2 )^2 + omega^2 / 2 y^2
# V22 = omega^2 / 2 ( x - a/2 )^2 + omega^2 / 2 y^2 - Delta
# V12 = cy 

# parameters of model
omega = 1.0 
Delta = 0.01
# Yarkony model
a = 6.0
c = 3.0
# alternative
# a = 2.0
# c = 1.0

# dimensionless displacement
d = a * math.sqrt(omega/2)

# constants
pi = math.acos(-1.0)

# plot resolution (number of pixels per dimension)
npixels = 41

# initialize Hamiltonian
H = np.zeros((nbasis, nbasis))

# set diagonal
ibasis = 0

for intotal in range(ntotalmax + 1):
    for inx in range(intotal + 1):
        H[ibasis,ibasis] = omega * ( intotal + 1.0 )
        #print(ibasis,nbasis)
        H[ibasis + int(nbasis/2), ibasis + int(nbasis/2)] = omega * ( intotal + 1.0 ) - Delta
        ibasis += 1

# set off-diagonals
for inx in range (ntotalmax + 1):
    for jnx in range (ntotalmax + 1):
        # FC overlaps from http://gisslen.com/linus/vib.html
        Sij1D = 0.0
        for i in range(min(inx,jnx) + 1):
            tmp = math.factorial(inx) * math.factorial(jnx) / (math.factorial(i) * math.factorial(inx-i) * math.factorial(jnx-i))
            tmp *= (-1.0)**(jnx - i) * d**(inx + jnx - 2*i)
            Sij1D += tmp

#         Sij1D = np.longdouble(Sij1D)
        Sij1D *= math.exp(-0.5 * np.square(d)) 
        Sij1D /= math.sqrt(math.factorial(inx))
        Sij1D /= math.sqrt(math.factorial(jnx))

#         print(Sij1D)

        # transition dipoles from http://farside.ph.utexas.edu/teaching/qmech/Quantum/node120.html
        # d = c * sqrt ( n / (2 * omega)) delta(n,n'+1) (when n > n')
        for iny in range(ntotalmax - inx + 1):
            intotal = inx + iny
            ibasis = sum(range(intotal + 1)) + inx
                
            if ((iny - 1) <= (ntotalmax - jnx)) and ((iny - 1) >= 0):
                jny = iny - 1
                jntotal = jnx + jny
                jbasis = int(nbasis/2) + sum(range(jntotal + 1)) + jnx
                
                dij1D = c * np.sqrt(iny / (2 * omega))

                H[ibasis,jbasis] = Sij1D * dij1D
                H[jbasis,ibasis] = Sij1D * dij1D
                #print(ibasis,jbasis,Sij1D,dij1D,H[ibasis,jbasis])

            if ((iny + 1) <= (ntotalmax - jnx)) and ((iny + 1) >= 0):
                jny = iny + 1
                jntotal = jnx + jny
                jbasis = int(nbasis/2) + sum(range(jntotal + 1)) + jnx
                
                dij1D = c * np.sqrt(jny / (2 * omega))

                H[ibasis,jbasis] = Sij1D * dij1D
                H[jbasis,ibasis] = Sij1D * dij1D
                #print(ibasis,jbasis,Sij1D,dij1D,H[ibasis,jbasis])
    
#print(H)
# print(np.max(H, axis=0))
# print(np.max(H, axis=1))

np.save('H.npy', H)
print("Hamiltonian written into H.npy file")

# build delta(origin) operator
delta = np.zeros((nbasis, nbasis))
for intotal in range(ntotalmax + 1):
    for inx in range(intotal + 1):
        iny = intotal - inx
        ibasis = sum(range(intotal + 1)) + inx
         
        psii1 = eval_psi( 0.0, 0.0, inx, iny, omega, (-0.5*a) ) 
        psii2 = eval_psi( 0.0, 0.0, inx, iny, omega, (0.5*a) ) 
         
        for jntotal in range(ntotalmax + 1):
            for jnx in range(jntotal + 1):
                jny = jntotal - jnx
                jbasis = sum(range(jntotal + 1)) + jnx
 
                psij1 = eval_psi( 0.0, 0.0, jnx, jny, omega, (-0.5*a) ) 
                psij2 = eval_psi( 0.0, 0.0, jnx, jny, omega, ( 0.5*a) ) 
                 
                delta[ibasis, jbasis] = psii1 * psij1
                delta[jbasis, ibasis] = psii1 * psij1
                 
                delta[ibasis + int(nbasis/2), jbasis + int(nbasis/2)] = psii2 * psij2
                delta[jbasis + int(nbasis/2), ibasis + int(nbasis/2)] = psii2 * psij2


delta_eigenvals, delta_eigenvecs = np.linalg.eigh(delta)
ZeroEigenvalsIndex = np.abs(delta_eigenvals) < 1e-8
delta_eigenvecs_zero = delta_eigenvecs[:, ZeroEigenvalsIndex]

# Using the delta matrix throwing away basis functions that are zero at CI to simulate basis set
# that produces zero density at the CI 

HT = transformMatrix(delta_eigenvecs_zero, H)
# eigenval_2, eigenvect_2 = np.linalg.eigh(HT)
np.save('HT.npy', HT)
print("Transformed Hamiltonian written into HT.npy file")

# Solving the transformed Hamiltonian HT using inverse iteration method   
gsVec = inverseIteration(HT, 0.50, HT[:, 1], maxIter=50, tol=1e-10)
# print(ev_new)
gsEigenval = transformMatrix(gsVec, HT) / np.dot(np.transpose(gsVec), gsVec)
print("\nTransformed Hamiltonian GS Energy = {0:1.8f}".format(gsEigenval))


# Direct diagonalization of original Hamiltonian
eigenval, eigenvect = np.linalg.eigh(H)
transformed_delta = np.dot(np.transpose(eigenvect), np.dot(delta, eigenvect))
 
#print(transformed_delta)
 
origin_dens = np.zeros(nbasis)
 
for i in range(nbasis):
    origin_dens[i] = transformed_delta[i,i]
 
data = np.array([eigenval, origin_dens])
data = data.T

#outfile = open('dens_at_ci.dat', 'w')
#np.savetxt(outfile, data, fmt=['%f','%f'])

#outfile.close()

step = 3.0 * a / (npixels - 1)
xmin = -1.5 * a
ymin = -1.5 * a

#get index of ground state
iground = np.argmin(eigenval)

data = np.zeros((npixels * npixels, 5))
idata = 0

for ix in range(npixels):
    for iy in range(npixels):
        x = xmin + ix * step
        y = ymin + iy * step
        psi1 = 0.0
        psi2 = 0.0

        for intotal in range(ntotalmax + 1):
            for inx in range(intotal + 1):
                iny = intotal - inx
                ibasis = sum(range(intotal + 1)) + inx
                
                psii1 = eval_psi( x, y, inx, iny, omega, (-0.5*a) ) 
                psii2 = eval_psi( x, y, inx, iny, omega, (0.5*a) )

                psi1 += eigenvect[ibasis,iground] * psii1
                psi2 += eigenvect[ibasis+int(nbasis/2),iground] * psii2
        
        data[idata, 0] = x
        data[idata, 1] = y
        data[idata, 2] = psi1 * psi1 + psi2 * psi2
        data[idata, 3] = psi1 * psi1
        data[idata, 4] = psi2 * psi2

        idata += 1

outfile = open('ground_dens.dat', 'w')
np.savetxt(outfile, data, fmt = ['%f','%f','%f','%f','%f'])
#data = eigenvect[:,iground]

#outfile = open('ground_vect.dat', 'w')
#np.savetxt(outfile, data, fmt=['%f'])

print("Done")
