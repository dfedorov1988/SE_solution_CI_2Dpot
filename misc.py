import numpy as np
import scipy.special as sp 

def calculateDensity(b, dim, zeros, eigenvecs_T2, coeff, \
                    transformationAB, representation, nState, n_grid=50, limit_plot=6):
    """Calculates the density on the grid in Cartesian coordinates,
    prints out matrices x.dat, y.dat, Density_mat.dat, visualize elsewhere because
    matplotlib is not very good at 3D surfaces"""
    
    print("Calculating density distribution:")
    
    coord = np.arange(-limit_plot, limit_plot, 2*limit_plot / n_grid)
    prob_dens = np.zeros((n_grid*n_grid, 5))
    prob_dens_mat, x_mat, y_mat = np.zeros((3, n_grid, n_grid))
    inum = 0
    
    for ix in range(n_grid-1):
        x = coord[ix]
        for iy in range(n_grid-1):
            y = coord[iy]
            
            r = np.absolute(np.sqrt(np.square(x) + np.square(y)))
            phi = np.arctan2(x, y)
            if y <= 0:
                phi = phi + 2*np.pi
#             print(x,y,phi*360/2/np.pi)
            prob_dens[inum, 0] = x
            prob_dens[inum, 1] = y
            prob_dens[inum, 2]\
            = densityAtSinglePoint(r, phi, b, dim, zeros, eigenvecs_T2, coeff, transformationAB, representation, nState)
            
            x_mat[ix] = prob_dens[inum, 0]
            y_mat[iy] = prob_dens[inum, 1]
            prob_dens_mat[ix,iy] = prob_dens[inum, 2]
            
            inum += 1
            
    """Saving the coordinates and density"""
    np.savetxt('x.dat', coord, fmt='%1.8E' )
    np.savetxt('y.dat', coord, fmt='%1.8E' )
    np.savetxt('Density_mat.dat', prob_dens_mat, fmt='%1.8E' )
    np.savetxt('Density.dat', prob_dens, fmt='%1.8E' )
    
    return (prob_dens)

def densityAtSinglePoint(r, phi, b, dim, zeros, eigenvecs_T2, coeff, transformationAB, representation, nState):
    """Calculates density at single point using coeff array (1D array of m,n) and
    the wave function psi"""
    
    psi1 = 0.0
    psi2 = 0.0
    psiI= 0.0
    psiJ = 0.0
        
    totalDensity = 0.0
    
    if (representation == "diabatic") : 
        for k in range(0, dim):
            m, n = coeff[k][0], coeff[k][1]
            psi1 += psi(r, phi, m, n, zeros, b) * (eigenvecs_T2[k, nState])
            psi2 += psi(r, phi, m, n, zeros, b) * (eigenvecs_T2[k + dim, nState])
        totalDensity = np.dot(psi1, np.conjugate(psi1)) + np.dot(psi2, np.conjugate(psi2))
    else:
        for k in range(0, dim):
            m, n = coeff[k][0], coeff[k][1]

            psiI +=  psiC(r, phi, m, n, zeros, b) * (eigenvecs_T2[k, nState])\
            + psiS(r, phi, m, n, zeros, b) * (eigenvecs_T2[k+dim, nState])
            psiJ +=  psiC(r, phi, m, n, zeros, b) * (eigenvecs_T2[k+2*dim, nState])\
            + psiS(r, phi, m, n, zeros, b) * (eigenvecs_T2[k+3*dim, nState])
#             
        totalDensity = np.dot(psiI, np.conjugate(psiI)) + np.dot(psiJ, np.conjugate(psiJ)) 
    
    return(np.real(totalDensity))

def largeBasFuncComp(dim, coeff, eigenvecs_T2, cs, numRootsToShow=10):
    """Analysis of the components of the transformed Hamiltonian in terms of 
    original basis functions"""
    print("")
    np.set_printoptions(precision=2) 
    coeff1 = np.chararray(4*dim, itemsize = 22)
    for i in range(4*dim):
        sign = lambda a: '+' if a > 0 else '-' if a < 0 else ' '
        coeff1[i] = sign(eigenvecs_T2[i, 0]) + ' X' + cs[i] + "(" + str(coeff[i][0]) + "," +  str(coeff[i][1]) + ")"
    coeff1 = coeff1.decode("utf-8")
    print("\nThe following m, n contribute to the wave function:")       
    for j in range(numRootsToShow):
        significantIndices = np.abs(eigenvecs_T2[:,j]) > 0.1
        string = list((np.unique(coeff1[significantIndices])))
        joinedString = " ".join(string)
        print("state {}: {}".format(j, joinedString))

def psi(r, phi, m, n, zeros, b): 
    """Radial + angular parts of a nuclear basis function"""
    
    return (np.sqrt(1/np.pi) / (b * sp.jv(m+1, zeros[np.abs(m), n-1]))\
          * sp.jv(m, zeros[np.abs(m), n-1] * r/b) * np.exp(-1j * m * phi))

def psiC(r, phi, m, n, zeros, b): 
    """symmetric (cos) radial + angular parts of a nuclear basis function"""
    
    return (np.sqrt(1/np.pi) / (b * sp.jv(m+1, zeros[np.abs(m), n-1]))\
          * sp.jv(m, zeros[np.abs(m), n-1] * r/b) * np.exp(-1j * m * phi)\
          * np.cos(phi/2) * np.exp(-1j * phi / 2))

def psiS(r, phi, m, n, zeros, b): 
    """asymmetric (sin) radial + angular parts of a nuclear basis function"""
    
    return (np.sqrt(1/np.pi) / (b * sp.jv(m+1, zeros[np.abs(m), n-1]))\
          * sp.jv(m, zeros[np.abs(m), n-1] * r/b) * np.exp(-1j * m * phi)\
          * 1j * np.sin(phi/2) * np.exp(-1j * phi / 2))
       
def psiRadial(r, m, n, zeros, b): # Radial part of the basis function
    """Radial part of a basis function"""
    return ((-1)**m * np.sqrt(2) / (b * sp.jv(m+1, zeros[np.abs(m), n-1])) \
            * sp.jv(m, zeros[np.abs(m), n-1] * r/b)) 

def transformMatrix(transformationMatrix, M):       
    """ST * M * S transformation"""
    
    Q = np.dot(M, transformationMatrix)
    
    R = np.dot(transformationMatrix.transpose().conjugate(), Q)
    return(R)
        
def printMatrix(*args):
    """prints matrix with real values row by row"""
    
    for M in args:
        print("")
        fmt=' '.join(['%7.5f']*M.shape[1])
        for row in M:
            print(fmt%tuple(row))
 
def isHermitian(M):
    """checks if a Matrix is Hermitian"""
    
    if (np.allclose(np.real(M), np.real(np.transpose(M)), rtol = 1e-04, atol = 1e-06) and \
        np.allclose(np.imag(M), -np.imag(np.transpose(M))))  :
        return True
    else:
        print("Maximum discrepancy in real part os symmetric matrix elements: {}".\
              format(max(np.amax(np.real(H_T2) - np.real(np.transpose(H_T2)), axis=1))))
        print("Maximum discrepancy in imaginary part os symmetric matrix elements: {}".\
              format(max(np.amax(np.imag(H_T2) + np.imag(np.transpose(H_T2)), axis=1)))) 
        return False

def V11(r, m1, n1, m2, n2, L, b, zeros): 
    return (psiRadial(r, m1, n1, zeros, b) * psiRadial(r, m2, n2, zeros, b) \
   * r * (L**2 + 4*r**2) / 4 / 2 )
  
def V11_2(r, m1, n1, m2, n2, L, b, zeros): 
    return (psiRadial(r, m1, n1, zeros, b) * psiRadial(r, m2, n2, zeros, b) * L * r**2 / 2 / 2)
  
def V22(r, m1, n1, m2, n2, L, b, zeros, delta): 
    return (psiRadial(r, m1, n1, zeros, b) * psiRadial(r, m2, n2, zeros, b) \
   * r * (L**2 + 4*r**2 - 8*delta ) / 4 / 2 )

def V_plus(r, m1, n1, m2, n2, L, b, zeros, c):
    """For test purposes: V+ potential energy matrix elements (adiabatic rep), much slower
    than using functions precompiled in c"""
    
    result = np.square(r+c) * r * 1/2 / np.square(b) \
        * sp.jv(m1, zeros[np.abs(m1), n1-1] * r/b) \
        * sp.jv(m2, zeros[np.abs(m2), n2-1] * r/b) \
        / sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
        / sp.jv(m2+1, zeros[np.abs(m2), n2-1])
    return(result)

def V_minus(r, m1, n1, m2, n2, L, b, zeros, c):
    """For test purposes: V- potential energy matrix elements (adiabatic rep), much slower
    than using functions precompiled in c"""
    
    result = np.square(r-c) * r * 1/2 / np.square(b) \
        * sp.jv(m1, zeros[np.abs(m1), n1-1] * r/b) \
        * sp.jv(m2, zeros[np.abs(m2), n2-1] * r/b) \
        / sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
        / sp.jv(m2+1, zeros[np.abs(m2), n2-1])
    return(result)
def T1Int(r, m1, n1, m2, n2, L, b, zeros, c): 
    """T1 component of kinetic energy (d/dr^2)"""
    
    result = 1/4 * r * sp.jv(m1, r/b * zeros[np.abs(m1), n1-1])\
            * (sp.jv(m2-2, r/b * zeros[np.abs(m2), n2-1]) \
            - 2* sp.jv(m2, r/b * zeros[np.abs(m2), n2-1]) \
            + sp.jv(m2+2, r/b * zeros[np.abs(m2), n2-1])) \
            * np.square(zeros[np.abs(m2), n2-1]) \
            / (b**4 * sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
            * sp.jv(m2+1, zeros[np.abs(m2), n2-1]))
    return(result)

def T2Int(r, m1, n1, m2, n2, L, b, zeros, c): 
    """T2 component of kinetic energy (1/r d/dr)"""
    result = 1/2 * sp.jv(m1, r/b * zeros[np.abs(m1), n1-1]) \
            * (sp.jv(m2-1, r/b * zeros[np.abs(m2), n2-1]) \
            - sp.jv(m2+1, r/b * zeros[np.abs(m2), n2-1])) * zeros[np.abs(m2), n2-1] \
            / (b**3 * sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
            * sp.jv(m2+1, zeros[np.abs(m2), n2-1]))
    return(result)

def mainInt(r, m1, n1, m2, n2, L, b, zeros, c):
    """T3, DBOC and DC integrals are different only by a constant, so we calculate
    mainInt only once to make things faster"""
    result = sp.jv(m1, r/b * zeros[np.abs(m1), n1-1]) \
    * sp.jv(m2, r/b * zeros[np.abs(m2), n2-1]) \
    /  (b**2 * r * sp.jv(m1+1, zeros[np.abs(m1), n1-1]) \
    * sp.jv(m2+1, zeros[np.abs(m2), n2-1]))
    return(result)

if __name__ == "__isHermitian__": isHermitian()
if __name__ == "__printMatrix__": printMatrix()
if __name__ == "__psi__": psi()        
if __name__ == "__psiRadialc__": psiRadial()   

