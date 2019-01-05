# Wave function continuity around conical intersections
This program solves the Schrodinger equation at the conical intersection using the basis of
cylindrical basis functions in both diabatic and adiabatic representations. The model system is 
the 2D Jahn-Teller model (similar to C. Xie et al., Phys. Rev. A 95, 022104, (2017)).
The main goal of the project is to prove that the electron density at the conical intersection doesn't have 
to be 0 in both diabatic and adiabatic representations. The introduction to this problem can be found in:
G. Meek and B. Levine J. Chem. Phys. 144, 184109 (2016).
# Citations
The paper based on this work is currently under review in the Journal of Chemical Physics, citation will be added as soon as it is available.
# Features
Solves Schrodinger equation around conical intersection using the direct Hamiltonian diagonalization, basis set is represented 
by Bessel functions to take advantage of the radial symmetry of the system. 
1. In diabatic representation the solution is pretty straightforward. 
2. In adiabatic representation we use the discontinuous basis set
Integration of matrix elements over angle is performed in Mathematica (see notebook for 
derivation of integrals), integration over r is performed numerically.
# How to Run
The program is written in python 3, file to run is adiab.py, where you can also change parameters for potential and basis set.
Because we need to calculate a lot of integrals numerically, the radial parts of integrals are written and precompiled in C language to spped up calculations. Gives a 5X speedup. Requires V11.so,
V2.so, T1.so, etc. in the working directory (need to be compiled on your machine).
