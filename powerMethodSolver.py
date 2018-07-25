from numpy import *
import numpy as np
from misc import transformMatrix, printMatrix
from misc import printMatrix
import scipy

def power_iteration(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = A[:, 1]
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k
    
def inverseIteration(A, eigenvalGuess, eigenVecGuess, maxIter, tol):
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
                
    b_k = eigenVecGuess
    b_k_norm = np.linalg.norm(b_k)
    I = np.identity(np.shape(A)[0])
    Matrix = A - I * eigenvalGuess

    for n in range(maxIter): 
        
        b_k1 = np.linalg.solve(Matrix, b_k)
        b_k1_norm  = np.linalg.norm(b_k1)
        normResidual = abs(b_k1_norm - b_k_norm)
        print("residual = {0:1.8e}".format(normResidual))
        
        if normResidual < tol: break
        
        b_k = b_k1 / b_k1_norm
        b_k_norm = b_k1_norm

    return(b_k)

H = np.load('HT.npy')
#ev_new = power_iteration(H,1000)
gsVec = inverseIteration(H, 0.50, H[:, 1], 50, 1e-10)
# print(ev_new)
gsEigenval = transformMatrix(gsVec, H) / np.dot(np.transpose(gsVec), gsVec)
print("\nEnergy = {}".format(gsEigenval))

# eigenvals, eigenvecs = np.linalg.eigh(H)
# print(eigenvals)